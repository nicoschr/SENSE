"""Prepare ecoinvent geographies, priced activities, and embedding stores."""

from __future__ import annotations

import json
import logging
import math
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from sense.artifacts import write_json_atomic
from sense.brightway.repository import empty_inventory_keys, reference_production_amounts
from sense.data.embeddings import (
    BGE_QUERY_PROMPT,
    _sha,
    load_embeddings,
    prep_query_bge,
    save_embeddings,
)
from sense.data.price_inference import build_product_graph, run_product_inference
from sense.data.prices import (
    BrightwayPriceCatalog,
    build_brightway_price_catalog,
    prod_key,
    reference_price_property,
)
from sense.data.seed_prices import (
    build_minimal_seed_worklist,
    load_seed_template,
    write_or_merge_seed_template,
)
from sense.eligibility import exclusion_reason, is_treatment
from sense.exceptions import SenseInputRequiredError
from sense.geography import read_ecoinvent_geographies
from sense.utils.helpers import norm_code, safe_name

logger = logging.getLogger(__name__)


def _embedding_store_matches_content(
    out_dir: str,
    base_name: str,
    activities: pd.DataFrame,
    expected_meta: dict[str, Any],
) -> bool:
    """Validate an embedding store by ID-to-text content, independent of row order."""

    try:
        vectors, stored_ids_series, manifest = load_embeddings(out_dir, base_name)
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False

    stored_ids = [str(value) for value in stored_ids_series.tolist()]
    expected_text = dict(
        zip(
            activities["code"].astype(str),
            activities["text"].astype(str),
            strict=True,
        )
    )
    if (
        len(stored_ids) != len(set(stored_ids))
        or set(stored_ids) != set(expected_text)
        or vectors.ndim != 2
        or vectors.shape[0] != len(stored_ids)
        or manifest.get("rows") != vectors.shape[0]
        or manifest.get("dim") != vectors.shape[1]
    ):
        return False

    stored_meta = manifest.get("meta") or {}
    stable_meta = {key: value for key, value in expected_meta.items() if key != "input_hash"}
    if any(stored_meta.get(key) != value for key, value in stable_meta.items()):
        return False

    texts_in_stored_order = [prep_query_bge(expected_text[code]) for code in stored_ids]
    content_hash = _sha(json.dumps(texts_in_stored_order, ensure_ascii=False))
    return stored_meta.get("input_hash") == content_hash


def map_ecoinvent_locations(
    ecoinvent_excel_path: str,
    psilca_sector_json_path: str | None = None,
) -> pd.DataFrame:
    """
    Map ecoinvent geographies to country sets used by the selected workflow.

    This function translates all ecoinvent locations into sets of PSILCA-compatible
    ISO2 country codes. The mapping depends on geography classification:

    - Countries present in PSILCA stay as-is (US -> US)
    - Subnational regions get assigned to their parent country (US-CO -> US)
    - Supranational regions get assigned all countries they contain (RER -> DE;FR;IT;...)
    - Countries not in PSILCA get assigned to the smallest overlapping region
    - Some locations require manual overrides

    Args:
        ecoinvent_excel_path: Path to ecoinvent Database Overview Excel file
        psilca_sector_json_path: PSILCA country-sector JSON. When omitted, retain
            every country resolved from the ecoinvent geography workbook.

    Returns:
        DataFrame with columns including 'Shortname', 'simplified_ISO2', 'PSILCA_mapping'
    """
    ecoinvent_locations_df = read_ecoinvent_geographies(Path(ecoinvent_excel_path))

    if psilca_sector_json_path is None:
        ecoinvent_locations_df["PSILCA_mapping"] = ecoinvent_locations_df["simplified_ISO2"]
        ecoinvent_locations_df["PSILCA_mapping_info"] = None
        return ecoinvent_locations_df

    # Load PSILCA ISO2 -> sectors mapping from JSON
    with open(psilca_sector_json_path, encoding="utf-8") as f:
        psilca_sector_dict = json.load(f)

    # Get PSILCA countries
    psilca_iso2_set = set(psilca_sector_dict.keys())

    # Build region map for multi-country regions
    region_map = {
        idx: {code.strip() for code in str(row["simplified_ISO2"]).split(";")}
        for idx, row in ecoinvent_locations_df.iterrows()
        if row["simplified_ISO2"] and ";" in str(row["simplified_ISO2"])
    }

    def map_to_psilca(row):
        info: str | None
        iso_string = row["simplified_ISO2"]
        if not iso_string or pd.isna(iso_string):
            return None, None

        iso_list = [code.strip() for code in iso_string.split(";") if code.strip()]
        iso_set = set(iso_list)

        # Case: single country
        if len(iso_list) == 1:
            iso2 = iso_list[0]
            if iso2 in psilca_iso2_set:
                return iso2, None

            # Find smallest region containing this country + at least one PSILCA country
            candidates = [(idx, region_map[idx]) for idx in region_map if iso2 in region_map[idx]]
            candidates.sort(key=lambda x: len(x[1]))

            for idx, region in candidates:
                psilca_countries = sorted(code for code in region if code in psilca_iso2_set)
                if psilca_countries:
                    removed = sorted(code for code in region if code not in psilca_iso2_set)
                    region_name = ecoinvent_locations_df.loc[idx, "Shortname"]
                    info = f"Used region: {region_name}, removed: {', '.join(removed)}"
                    return "; ".join(psilca_countries), info

            return None, f"No PSILCA region found for {iso2}"

        # Case: region (multiple countries)
        else:
            filtered = [code for code in iso_list if code in psilca_iso2_set]
            if filtered:
                removed = [code for code in iso_list if code not in psilca_iso2_set]
                info = f"Removed: {', '.join(removed)}" if removed else None
                return "; ".join(filtered), info
            else:
                candidates = [
                    (idx, region_map[idx])
                    for idx in region_map
                    if iso_set & region_map[idx]
                    and any(code in psilca_iso2_set for code in region_map[idx])
                ]
                candidates.sort(key=lambda x: len(x[1]))

                for idx, region in candidates:
                    psilca_countries = sorted(code for code in region if code in psilca_iso2_set)
                    if psilca_countries:
                        removed = sorted(code for code in region if code not in psilca_iso2_set)
                        region_name = ecoinvent_locations_df.loc[idx, "Shortname"]
                        info = (
                            f"Used overlapping region: {region_name}, removed: {', '.join(removed)}"
                        )
                        return "; ".join(psilca_countries), info

                return None, f"No overlapping PSILCA region found for region: {iso_string}"

    ecoinvent_locations_df[["PSILCA_mapping", "PSILCA_mapping_info"]] = (
        ecoinvent_locations_df.apply(map_to_psilca, axis=1, result_type="expand")
    )

    return ecoinvent_locations_df


# ---------------------------------------------------------------------------
# Location concordance: detect unmapped locations, template write/merge/load
# ---------------------------------------------------------------------------


def collect_database_locations(technosphere: str) -> set[str]:
    """Collect all unique location codes from a Brightway database.

    Lightweight metadata-only scan (no exchange traversal).

    Args:
        technosphere: Name of the Brightway technosphere database.

    Returns:
        Non-empty location codes used by its activities.
    """
    from bw2data import Database as BWDatabase

    locations: set[str] = set()
    for act in BWDatabase(technosphere):
        loc = act.get("location")
        if loc:
            locations.add(str(loc).strip())
    return locations


def detect_unmapped_locations(
    db_locations: set[str],
    mapping_df: pd.DataFrame,
) -> list[str]:
    """Return location codes present in the database but absent from the mapping."""
    usable = mapping_df["PSILCA_mapping"].fillna("").astype(str).str.strip().ne("")
    known = set(mapping_df.loc[usable, "Shortname"].dropna().astype(str).str.strip())
    known.update({"RoW", "RoE"})
    return sorted(db_locations - known)


def write_or_merge_location_concordance(
    concordance_path: str,
    unmapped: list[str],
    database_name: str,
) -> tuple[int, int, int]:
    """Write a location concordance template or merge new unmapped locations into an existing one.

    If the file does not exist, creates it with all locations from *unmapped*.
    If it exists, loads it, keeps all existing entries (preserving user-filled
    countries), and appends any new unmapped locations not yet in the file.

    Args:
        concordance_path: JSON template to create or update.
        unmapped: Location codes requiring a country mapping in the current run.
        database_name: Database label recorded in template metadata.

    Returns:
        (n_loaded, n_new, n_still_missing): count of entries with valid countries,
        count of newly added template entries, count of entries still missing.
    """
    existing: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {}

    if os.path.exists(concordance_path):
        with open(concordance_path, encoding="utf-8") as f:
            data = json.load(f)
        existing = data.get("locations", [])
        metadata = data.get("metadata", {})

    # Index existing entries by location_code for dedup
    existing_keys: dict[str, int] = {}
    required_codes = {str(code).strip() for code in unmapped}
    for i, entry in enumerate(existing):
        code = str(entry.get("location_code", "")).strip()
        if code:
            existing_keys[code] = i
            entry["required"] = code in required_codes

    # Merge new unmapped locations
    n_new = 0
    for code in unmapped:
        if code in existing_keys:
            continue
        existing.append(
            {
                "location_code": code,
                "countries": None,
                "comment": None,
                "required": True,
            }
        )
        existing_keys[code] = len(existing) - 1
        n_new += 1

    # Count loaded vs still missing
    n_loaded = 0
    n_still_missing = 0
    for entry in existing:
        if not entry.get("required", False):
            continue
        countries = entry.get("countries")
        if isinstance(countries, str) and countries.strip():
            n_loaded += 1
        else:
            n_still_missing += 1

    # Update metadata
    metadata.update(
        {
            "database": database_name,
            "note": (
                "Fill in ISO2 country codes for each unmapped location "
                "(semicolon-separated). Provide the FULL list of constituent "
                "countries — SENSE will automatically filter to PSILCA-available "
                "countries. Leave null to halt the pipeline."
            ),
            "example": {
                "location_code": "EXAMPLE_REGION",
                "countries": "DE; FR; IT; ES; NL; BE; AT; PT",
                "comment": "Example: list all countries in the region, SENSE filters to PSILCA",
            },
        }
    )

    write_json_atomic(Path(concordance_path), {"metadata": metadata, "locations": existing})

    return n_loaded, n_new, n_still_missing


def load_location_concordance(concordance_path: str) -> dict[str, str]:
    """Load filled location concordance entries from a JSON file.

    Args:
        concordance_path: Concordance JSON previously written by SENSE.

    Returns:
        Dict mapping location_code -> semicolon-separated ISO2 countries string.
        Only entries where countries is non-null and non-empty are returned.
    """
    if not concordance_path or not os.path.exists(concordance_path):
        return {}

    with open(concordance_path, encoding="utf-8") as f:
        data = json.load(f)

    out: dict[str, str] = {}
    for entry in data.get("locations", []):
        code = str(entry.get("location_code", "")).strip()
        countries = entry.get("countries")
        if code and isinstance(countries, str) and countries.strip():
            out[code] = countries.strip()
    return out


def fetch_ecoinvent_activities(
    mapping_df: pd.DataFrame,
    output_path: str,
    technosphere: str,
    exclude_terms: Sequence[str] | None = None,
    exclude_terms_starting_with: Sequence[str] | None = None,
    price_floor_percentile: float = 20.0,
    *,
    exclude_exact: Sequence[str] | None = None,
    zero_price_policy: str = "treatment_working_hours",
    treatment_working_hours_percentile: float = 20.0,
    auto_seed_prices_path: str | None = None,
    database_label: str | None = None,
    price_currency: str = "EUR",
    price_year: int = 2005,
    price_source_database: str | None = None,
    activities: Sequence[Any] | None = None,
    exclude_empty_inventories: bool = True,
    empty_keys: set[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """
    Build an activity table with explicit monetary, zero, and treatment routes.

    Prices are seeded from reference-exchange properties retained by the
    Brightway ecoinvent importer, activity-level values, and the managed seed
    template, then propagated through the technosphere input graph.

    Args:
        mapping_df: DataFrame mapping ISO2 codes to PSILCA regions
        output_path: Path for the resulting JSON output
        technosphere: Name of the technosphere database to analyze
        exclude_terms: Terms that exclude activities from output if found anywhere
        exclude_terms_starting_with: Terms that exclude if found as prefix
        price_floor_percentile: Percentile used only by the legacy price-floor policy.
        zero_price_policy: Treatment hours, legacy price replacement, or zero direct scores.
        treatment_working_hours_percentile: Zero disables unpriced treatment inventories.
        exclude_exact: Exact activity names to exclude from final output.
        auto_seed_prices_path: Managed seed template written and read by SENSE.
        database_label: Human-readable database name stored in seed metadata.
        price_currency: Currency expected for manual seed prices.
        price_year: Reference year expected for manual seed prices.
        price_source_database: Ecoinvent Brightway database containing source
            prices. Defaults to ``technosphere``.
        activities: Optional preloaded Brightway activity snapshot.
        exclude_empty_inventories: Exclude empty physical inventories in any
            database from direct social mapping and treatment-hour references.
        empty_keys: Optional precomputed empty-inventory exclusion set.

    Returns:
        DataFrame with activity data including prices

    Note:
        Requires Brightway2 (`bw2data`) to be installed and a project to be active.
    """
    from bw2data import Database
    from tqdm import tqdm

    if zero_price_policy not in {"treatment_working_hours", "legacy_price_floor", "zero"}:
        raise ValueError("Unknown zero_price_policy")
    if not math.isfinite(treatment_working_hours_percentile) or not (
        0 <= treatment_working_hours_percentile <= 100
    ):
        raise ValueError("Treatment percentile must be between 0 and 100")
    exclude_terms = exclude_terms or []
    exclude_terms_starting_with = exclude_terms_starting_with or []
    pricing_audit: list[dict[str, Any]] = []

    # PSILCA mappings
    iso2_to_psilca: dict[str, str] = mapping_df.set_index("Shortname")["PSILCA_mapping"].to_dict()
    # Geographic membership is distinct from the countries supplying a social
    # proxy. A territory may use global PSILCA data without covering the globe.
    iso2_to_geography = (
        mapping_df.set_index("Shortname")["simplified_ISO2"].to_dict()
        if "simplified_ISO2" in mapping_df
        else {}
    )
    base_psilca_sets: dict[str, set[str]] = {
        "RoW": {c.strip() for c in str(iso2_to_psilca.get("GLO") or "").split(";") if c.strip()},
        "RoE": {c.strip() for c in str(iso2_to_psilca.get("RER") or "").split(";") if c.strip()},
    }

    # Load database
    technosphere_activities = sorted(
        activities if activities is not None else Database(technosphere),
        key=lambda activity: (str(activity.key[0]), str(activity.key[1])),
    )
    from sense.eligibility import validate_activity_locations

    validate_activity_locations(
        technosphere_activities,
        prefixes=exclude_terms_starting_with,
        terms=exclude_terms,
        exact=exclude_exact or (),
    )

    activity_meta: dict[str, dict[str, Any]] = {}
    estimated_prices: dict[str, float | None] = {}
    n_not_found: int = 0

    source_database = price_source_database or technosphere
    source_activities = (
        technosphere_activities
        if source_database == technosphere
        else sorted(
            Database(source_database),
            key=lambda activity: (str(activity.key[0]), str(activity.key[1])),
        )
    )
    price_catalog: BrightwayPriceCatalog = build_brightway_price_catalog(
        source_activities,
        source_database=source_database,
        price_currency=price_currency,
        price_year=price_year,
        price_floor_percentile=(
            price_floor_percentile if zero_price_policy == "legacy_price_floor" else 0.0
        ),
    )
    logger.info(
        "Loaded %d exact and %d product-level prices from Brightway database %r",
        len(price_catalog.by_code),
        len(price_catalog.by_product),
        source_database,
    )

    # Load the managed seed template before building the graph. This lets graph
    # construction stop at already priced suppliers instead of recursively
    # expanding their databases.
    _price_currency = price_currency
    _price_year = int(price_year)
    _auto_seed_path = auto_seed_prices_path
    auto_map = (
        load_seed_template(
            str(_auto_seed_path),
            expected_currency=_price_currency,
            expected_year=_price_year,
        )
        if _auto_seed_path and os.path.exists(_auto_seed_path)
        else {}
    )

    resolved_prices: dict[tuple, float | None] = {}
    price_sources: dict[tuple, str] = {}

    def resolve_price(activity: Any) -> float | None:
        """Resolve once per activity: user seed, own price, exact source, product."""
        key = tuple(activity.key)
        if key in resolved_prices:
            return resolved_prices[key]
        product_key = prod_key(
            str(activity.get("unit") or ""),
            str(activity.get("reference product") or activity.get("name") or ""),
        )
        price = auto_map.get(product_key)
        source = "user_seed" if price is not None else "unresolved"
        if price is None:
            raw_own_price = activity.get("price")
            # Brightway can expose imported production properties directly on
            # the activity as {amount, unit}, alongside numeric custom prices.
            imported_property = isinstance(raw_own_price, dict)
            if imported_property:
                monetary_unit = str(raw_own_price.get("unit") or "").replace(" ", "").casefold()
                if monetary_unit != f"{_price_currency}{_price_year}".casefold():
                    raise ValueError(
                        f"Activity {activity.key!r} has an incompatible activity price unit"
                    )
                raw_own_price = raw_own_price.get("amount")
            try:
                own_price = float(raw_own_price)
            except (TypeError, ValueError) as error:
                if activity.get("price") is not None:
                    raise ValueError(
                        f"Activity {activity.key!r} has a non-numeric activity price"
                    ) from error
                own_price = None
            if own_price is not None:
                if not math.isfinite(own_price) or own_price < 0:
                    raise ValueError(f"Activity {activity.key!r} has an invalid activity price")
                if (
                    activity.get("sense_paired_price_basis") is None
                    or activity.get("sense_paired_price_policy") == zero_price_policy
                ):
                    # The legacy catalog already replaces imported zeros.
                    if not (
                        imported_property
                        and own_price == 0
                        and zero_price_policy == "legacy_price_floor"
                    ):
                        price = own_price
                        source = "activity_price"
        if price is None:
            # Source reference prices were already read, validated and floored
            # when building the catalog; avoid a second exchange query per row.
            code = norm_code(activity.get("code")) or str(activity.key[1])
            exact = price_catalog.by_code.get(code)
            if key[0] == source_database and exact is not None and exact[0] == product_key:
                price = exact[1]
                source = "source_reference_price"
        if price is None:
            reference_price = reference_price_property(activity)
            if reference_price is not None:
                expected = f"{_price_currency}{_price_year}"
                if reference_price[1].replace(" ", "").casefold() != expected.casefold():
                    raise ValueError(
                        f"Activity {activity.key!r} uses price unit {reference_price[1]!r}; "
                        f"expected {expected}"
                    )
                if reference_price[0] > 0 or zero_price_policy != "legacy_price_floor":
                    price = reference_price[0]
                    source = "reference_price"
        if price is None:
            code = norm_code(activity.get("code")) or str(activity.key[1])
            exact = price_catalog.by_code.get(code)
            if exact is not None and exact[0] == product_key:
                price = exact[1]
                source = "exact_background_price"
        if price is None:
            price = price_catalog.by_product.get(product_key)
            if price is not None:
                source = "product_background_price"
        price_sources[key] = source
        resolved_prices[key] = price
        return price

    # ------------------ Build metadata and resolve direct prices ------------------
    unresolved_activities: list[Any] = []
    empty_inventory = set()
    if exclude_empty_inventories:
        empty_inventory = (
            empty_keys if empty_keys is not None else empty_inventory_keys(technosphere_activities)
        )
    exclusions = {
        tuple(act.key): exclusion_reason(
            act,
            prefixes=exclude_terms_starting_with,
            terms=exclude_terms,
            exact=exclude_exact or [],
            empty_inventory=tuple(act.key) in empty_inventory,
        )
        for act in technosphere_activities
    }
    # Use the writer's reference validation, reading SQLite in batches rather
    # than issuing one production query for every regionalized activity.
    reference_amounts = reference_production_amounts(
        {
            tuple(act.key): act
            for act in technosphere_activities
            if exclusions[tuple(act.key)] is None
        },
        preserve_sign=True,
    )
    for act in tqdm(
        technosphere_activities,
        desc=f"Preparing activities: {technosphere}",
        unit="activity",
        dynamic_ncols=True,
    ):
        reason = exclusions[tuple(act.key)]
        if reason:
            pricing_audit.append(
                {
                    "code": str(act["code"]),
                    "name": act.get("name"),
                    "status": "excluded",
                    "reason": reason,
                }
            )
            continue
        code = norm_code(act["code"]) or str(act.key[1])

        # Extract ISIC from Brightway classifications (authoritative source)
        isic_bw = None
        for cls in act.get("classifications") or []:
            if isinstance(cls, (list, tuple)) and len(cls) >= 2 and "ISIC" in str(cls[0]):
                isic_bw = str(cls[1])
                break

        activity_meta[code] = {
            "name": act["name"],
            "reference product": act["reference product"],
            "unit": (act["unit"] or "").lower(),
            "location": act["location"],
            "isic": isic_bw,
        }

        direct_price = resolve_price(act)
        reference_amount = reference_amounts[tuple(act.key)]
        treatment = is_treatment(act, reference_amount)
        if (
            treatment
            and price_sources[tuple(act.key)] == "product_background_price"
            and zero_price_policy == "treatment_working_hours"
        ):
            # A generic product average is not a supplied treatment-service price.
            direct_price = None
            price_sources[tuple(act.key)] = "unresolved_treatment_price"
        activity_meta[code]["reference amount"] = reference_amount
        activity_meta[code]["is treatment"] = treatment
        activity_meta[code]["price source"] = price_sources[tuple(act.key)]
        needs_fallback = (
            treatment
            and (direct_price is None or direct_price == 0)
            and zero_price_policy == "treatment_working_hours"
        )
        route = "monetary"
        if needs_fallback:
            route = "zero" if treatment_working_hours_percentile == 0 else "treatment_fallback"
        activity_meta[code]["calculation route"] = route
        if route in {"zero", "treatment_fallback"}:
            # Disabled treatments retain zero/missing price provenance and never
            # acquire an inferred price or enter the donor-reference calculation.
            estimated_prices[code] = direct_price
        elif direct_price is None:
            unresolved_activities.append(act)
        elif direct_price >= 0:
            estimated_prices[code] = float(direct_price)

    # ------------------ Product-graph price inference ------------------
    graph = None
    product_prices: dict[int, float] = {}
    if unresolved_activities:
        logger.info(
            "Running product-graph inference for %d/%d activities without a direct price...",
            len(unresolved_activities),
            len(activity_meta),
        )

        # Determine biosphere DB name — check for spatialized version first
        from bw2data import databases as bw_databases

        _bio_candidates = ["biosphere3_spatialized_flows", "biosphere3"]
        _biosphere_db_name = "biosphere3"
        for _bc in _bio_candidates:
            if _bc in bw_databases:
                _biosphere_db_name = _bc
                break

        graph = build_product_graph(
            target_database=technosphere,
            biosphere_database=_biosphere_db_name,
            activities=unresolved_activities,
            should_expand_external=lambda activity: resolve_price(activity) is None,
            resolve_price=resolve_price,
            expected_price_unit=f"{_price_currency}{_price_year}",
        )

        initial_prices = graph.pid_activity_price_seed
        initial_sources = dict.fromkeys(initial_prices, "resolved_activity_price")

        logger.info(
            "  Seeds: %d / %d products priced before inference",
            len(initial_prices),
            len(graph.id_to_product),
        )

        product_prices, _ = run_product_inference(
            cache=graph,
            initial_prices=initial_prices,
            initial_sources=initial_sources,
        )

        for act in unresolved_activities:
            code = norm_code(act.get("code")) or str(act.key[1])
            activity_pid = graph.activity_to_id.get((technosphere, code))
            if activity_pid is not None and activity_pid in product_prices:
                price_val = product_prices[activity_pid]
                if price_val >= 0:
                    estimated_prices[code] = price_val
                    activity_meta[code]["price source"] = "inferred_from_inputs"
    else:
        logger.info(
            "Direct Brightway prices resolved all %d eligible activities; "
            "product-graph inference is not needed.",
            len(activity_meta),
        )

    n_not_found = len(activity_meta) - len(estimated_prices)

    # Log seed worklist and write/merge auto seed template if there are unresolved products
    n_unresolved = len(activity_meta) - len(estimated_prices)
    seed_worklist = (
        build_minimal_seed_worklist(cache=graph, priced=product_prices)
        if n_unresolved > 0 and graph is not None
        else pd.DataFrame()
    )
    if not seed_worklist.empty:
        logger.info(
            "  %d activities remain unpriced; the minimal seed worklist has %d products.",
            n_unresolved,
            len(seed_worklist),
        )

    # Keep an existing template synchronized even when every product now
    # resolves, so stale null entries cannot block a later pipeline stage.
    if _auto_seed_path and (not seed_worklist.empty or os.path.exists(_auto_seed_path)):
        _db_name = database_label or technosphere
        n_loaded, n_new, n_still_missing = write_or_merge_seed_template(
            seed_json_path=str(_auto_seed_path),
            worklist_df=seed_worklist,
            database_name=_db_name,
            price_currency=_price_currency,
            price_year=_price_year,
        )
        if n_new > 0:
            logger.info(
                "  📝 Wrote %d new entries to seed price template: %s",
                n_new,
                _auto_seed_path,
            )
        if n_loaded > 0:
            logger.info(
                "  ✅ %d seed prices loaded from template.",
                n_loaded,
            )
        if n_still_missing > 0:
            raise SenseInputRequiredError(
                f"{n_still_missing} seed prices require user input",
                Path(_auto_seed_path),
            )

    rows: list[dict[str, Any]] = []
    for code, meta in activity_meta.items():
        price = estimated_prices.get(code)
        route = meta["calculation route"]
        if route not in {"zero", "treatment_fallback"}:
            if price is None:
                pricing_audit.append({"code": code, "name": meta["name"], "status": "unresolved"})
                continue
            route = "monetary" if price > 0 else "zero"
        pricing_audit.append(
            {
                "code": code,
                "name": meta["name"],
                "status": route,
                "price": price,
                "price_source": meta["price source"],
                "reference_amount": meta["reference amount"],
                "treatment_fallback_disabled": (
                    meta["is treatment"]
                    and route == "zero"
                    and zero_price_policy == "treatment_working_hours"
                    and treatment_working_hours_percentile == 0
                ),
            }
        )
        rows.append(
            {
                "ISO2": meta["location"],
                "name": meta["name"],
                "reference product": meta["reference product"],
                "unit": meta["unit"],
                "code": code,
                "ISIC rev.4 ecoinvent": meta.get("isic"),
                "price": price,
                "price source": meta["price source"],
                "reference amount": meta["reference amount"],
                "is treatment": meta["is treatment"],
                "calculation route": route,
            }
        )
    df = pd.DataFrame(
        rows,
        columns=[
            "ISO2",
            "name",
            "reference product",
            "unit",
            "code",
            "ISIC rev.4 ecoinvent",
            "price",
            "price source",
            "reference amount",
            "is treatment",
            "calculation route",
        ],
    )
    if df.empty and n_not_found:
        raise ValueError(f"No resolved activities remain in {technosphere!r}")
    write_json_atomic(
        Path(output_path).with_suffix(".pricing.json"),
        {
            "schema": 1,
            "policy": zero_price_policy,
            "activities": pricing_audit,
        },
    )

    # Normalize output types. Name exclusions were applied before graph-root
    # selection; excluded activities can still be traversed as dependencies.
    if not df.empty:
        df["name"] = df["name"].astype(str)
        df["code"] = df["code"].astype(str)
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        df = df.sort_values(["ISO2", "code"], kind="stable").reset_index(drop=True)

    # Map to PSILCA & save JSON
    if not df.empty:
        key_iso2_map = (
            df[["name", "reference product", "ISO2"]]
            .drop_duplicates()
            .groupby(["name", "reference product"])["ISO2"]
            .agg(set)
            .to_dict()
        )
    else:
        key_iso2_map = {}

    final_output: list[dict[str, Any]] = []
    for iso2, group in df.groupby("ISO2"):
        if iso2 in base_psilca_sets:
            base = base_psilca_sets[iso2]
            activities = []

            for _, row in group.iterrows():
                key = (row["name"], row["reference product"])
                all_iso2s = key_iso2_map.get(key, set())
                others = all_iso2s - {iso2}
                excluded_countries: set[str] = set()
                for other in others:
                    # Residual labels have activity-dependent membership, not a
                    # fixed concordance entry. Preserve their existing handling;
                    # this correction concerns ordinary geographic exclusions.
                    if other in base_psilca_sets:
                        continue
                    geography = iso2_to_geography.get(other)
                    if not isinstance(geography, str) or not geography.strip():
                        raise ValueError(
                            f"Location {other!r} has no actual country membership "
                            f"(simplified_ISO2) for {iso2} exclusions; "
                            "PSILCA proxy countries cannot define geographic coverage"
                        )
                    excluded_countries.update(c.strip() for c in geography.split(";") if c.strip())
                psilca = "; ".join(sorted(base - excluded_countries))
                if not psilca:
                    raise ValueError(
                        f"Activity {row['code']!r} has an empty {iso2} country mapping"
                    )

                row_dict = row.drop(labels="ISO2").to_dict()
                row_dict["PSILCA_mapping"] = psilca
                activities.append(row_dict)

            final_output.append({"ISO2": iso2, "Activities": activities})
        else:
            mapped_psilca = iso2_to_psilca.get(iso2)
            if not isinstance(mapped_psilca, str) or not mapped_psilca.strip():
                raise ValueError(
                    f"Location {iso2!r} has no usable country mapping for activities "
                    f"{group['code'].tolist()[:8]}"
                )
            if mapped_psilca:
                activities = group.drop(columns="ISO2").to_dict(orient="records")
                final_output.append(
                    {
                        "ISO2": iso2,
                        "PSILCA_mapping": mapped_psilca,
                        "Activities": activities,
                    }
                )

    for entry in final_output:
        for activity in entry["Activities"]:
            if pd.isna(activity.get("price")):
                activity["price"] = None
    write_json_atomic(Path(output_path), final_output)
    logger.info("Saved enriched activity data to %s", output_path)

    # Log unresolved activities (if any remain after accepting price=0)
    if n_not_found:
        logger.info("  %d activities have no price (not even zero).", n_not_found)

    df.attrs["pricing_dependency_databases"] = sorted(
        {
            source_database,
            technosphere,
            *(
                str(meta.get("database"))
                for meta in (graph.id_to_meta if graph is not None else [])
                if meta.get("database")
            ),
        }
    )
    return df


def embed_ecoinvent_to_store(
    technosphere_json: str,
    out_dir: str,
    model_name: str = "BAAI/bge-base-en-v1.5",
    batch_size: int = 256,
    *,
    target_iso2_list: list[str],
    force: bool = False,
    revision: str | None = None,
) -> None:
    """
    Embed ecoinvent activities and save to disk.

    Creates embedding stores for ecoinvent activities per ISO2 region:
      - {out_dir}/ecoinvent_{ISO2}.npy
      - {out_dir}/ecoinvent_{ISO2}.manifest.json

    Skips already-existing stores to avoid redundant computation.

    Args:
        technosphere_json: Path to the activity list JSON file
        out_dir: Directory to save embedding stores
        model_name: Name of the sentence-transformer model to use
        batch_size: Batch size for embedding computation
        target_iso2_list: ISO2 codes to embed.
        force: Rebuild stores even when their manifests match.

    Note:
        Requires the `sentence-transformers` package.
    """
    os.makedirs(out_dir, exist_ok=True)

    if not target_iso2_list:
        raise ValueError("target_iso2_list must contain at least one region")

    with open(technosphere_json, encoding="utf-8") as input_handle:
        data = json.load(input_handle)

    # Build the exact input rows before checking caches. A store is reusable
    # only when model, prompt, identifiers, and embedded text are unchanged.
    from sense.data.embeddings import model_revision

    resolved_revision = model_revision(model_name, revision)
    prepared: dict[str, tuple[pd.DataFrame, dict[str, Any]]] = {}
    missing_iso2s: list[str] = []
    for iso2 in target_iso2_list:
        base_name = f"ecoinvent_{safe_name(iso2)}"
        rows = []
        for loc in data:
            if loc.get("ISO2") != iso2:
                continue
            for act in loc.get("Activities", []):
                if act.get("calculation route") == "zero":
                    continue
                code = (act.get("code") or "").strip()
                name = (act.get("name") or "").strip()
                ref = (act.get("reference product") or "").strip()
                isic_raw = act.get("ISIC rev.4 ecoinvent")
                isic = isic_raw.strip() if isinstance(isic_raw, str) else ""
                text = f"{name}; {ref}; isic rev.4: {isic}".lower().strip()
                rows.append((code, text))

        if not rows:
            continue
        df = (
            pd.DataFrame(rows, columns=["code", "text"])
            .drop_duplicates("code")
            .sort_values("code", kind="stable")
            .reset_index(drop=True)
        )
        encoded_texts = df["text"].map(prep_query_bge).tolist()
        meta = {
            "model": model_name,
            "revision": resolved_revision,
            "prompt": BGE_QUERY_PROMPT,
            "normalized": True,
            "source": "ecoinvent_activities",
            "iso2": iso2,
            "input_hash": _sha(json.dumps(encoded_texts, ensure_ascii=False)),
        }
        prepared[iso2] = (df, meta)
        if force or not _embedding_store_matches_content(out_dir, base_name, df, meta):
            missing_iso2s.append(iso2)

    if not missing_iso2s:
        logger.info("All ecoinvent embedding stores are current in %s", out_dir)
        return

    logger.info("Creating %d ecoinvent embedding stores", len(missing_iso2s))
    logger.info("Loading embedding model '%s'...", model_name)
    from tqdm import tqdm

    from sense.data.embeddings import load_encoder

    model = load_encoder(model_name, revision=resolved_revision)

    # Embed missing ISO2s
    for iso2 in tqdm(
        missing_iso2s,
        desc="Embedding ecoinvent regions",
        unit="region",
        dynamic_ncols=True,
    ):
        base_name = f"ecoinvent_{safe_name(iso2)}"

        df, meta = prepared[iso2]

        embeddings = model.encode(
            df["text"].map(prep_query_bge).tolist(),
            normalize_embeddings=True,
            convert_to_numpy=True,
            batch_size=batch_size,
            show_progress_bar=False,
        ).astype("float32")

        save_embeddings(out_dir, base_name, embeddings, df["code"], meta)

    logger.info("Finished ecoinvent embedding stores in %s", out_dir)
