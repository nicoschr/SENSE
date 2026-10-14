"""Prepare and validate source data, geography, prices and embedding caches."""

from __future__ import annotations

import hashlib
import json
import logging
from difflib import get_close_matches
from pathlib import Path
from typing import Any

import pandas as pd
from bw2data import Database, databases

from sense.artifacts import atomic_output_path
from sense.config import SenseConfig
from sense.data.ecoinvent import (
    collect_database_locations,
    detect_unmapped_locations,
    embed_ecoinvent_to_store,
    fetch_ecoinvent_activities,
    load_location_concordance,
    map_ecoinvent_locations,
    write_or_merge_location_concordance,
)
from sense.data.price_inference import PRICE_INFERENCE_VERSION
from sense.data.psilca import (
    PSILCA_CACHE_SCHEMA_VERSION,
    combine_psilca_css_and_extract_social_exchanges,
    embed_psilca_to_store,
)
from sense.exceptions import SenseInputRequiredError
from sense.utils.cache import (
    directory_fingerprint,
    file_hash,
    read_manifest,
    stable_hash,
    write_manifest,
)
from sense.utils.helpers import slug

logger = logging.getLogger(__name__)


def _load_technosphere_activities(database_name: str) -> list[Any]:
    """Load a non-empty Brightway technosphere database with a useful error."""
    if database_name not in databases:
        available = sorted(str(name) for name in databases)
        close = get_close_matches(database_name, available, n=1, cutoff=0.6)
        hint = f" Did you mean {close[0]!r}?" if close else ""
        raise ValueError(
            f"Brightway database {database_name!r} does not exist in the current project." f"{hint}"
        )

    activities = sorted(
        Database(database_name),
        key=lambda activity: (str(activity.key[0]), str(activity.key[1])),
    )
    if not activities:
        raise ValueError(f"Brightway database {database_name!r} exists but contains no activities.")
    return activities


def _brightway_technosphere_fingerprint(database_name: str) -> str:
    """Hash the pricing-relevant portion of a Brightway technosphere.

    Biosphere exchanges are deliberately excluded so writing SENSE results
    does not invalidate the price cache on the next run.
    """

    digest = hashlib.sha256()
    activities = sorted(
        Database(database_name),
        key=lambda activity: (str(activity.key[0]), str(activity.key[1])),
    )
    for activity in activities:
        technosphere = sorted(
            (
                str(exchange.input.key[0]),
                str(exchange.input.key[1]),
                float(exchange["amount"]),
            )
            for exchange in activity.technosphere()
        )
        production = sorted(
            (
                str(exchange.input.key[0]),
                str(exchange.input.key[1]),
                float(exchange["amount"]),
                json.dumps(
                    (exchange.get("properties") or {}).get("price"),
                    sort_keys=True,
                    default=str,
                ),
            )
            for exchange in activity.production()
        )
        payload = {
            "key": [str(activity.key[0]), str(activity.key[1])],
            "name": activity.get("name"),
            "reference_product": activity.get("reference product"),
            "unit": activity.get("unit"),
            "location": activity.get("location"),
            "classifications": activity.get("classifications"),
            "price": activity.get("price"),
            "comment": activity.get("comment"),
            "activity_type": activity.get("activity type"),
            "sense_activity_role": activity.get("sense_activity_role"),
            "sense_paired_price_policy": activity.get("sense_paired_price_policy"),
            "production_amount": activity.get("production amount"),
            "technosphere": technosphere,
            "production": production,
        }
        digest.update(
            json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
        )
    return digest.hexdigest()


def _brightway_database_token(database_name: str) -> str:
    """Return a fast cache token for a Brightway database.

    Brightway updates its ``modified`` metadata whenever exchanges change. Older
    or custom backends without that metadata fall back to the exact, slower
    technosphere fingerprint.
    """

    metadata = databases.get(database_name) or {}
    modified = metadata.get("modified")
    if modified is None:
        return _brightway_technosphere_fingerprint(database_name)
    return stable_hash(
        {
            "database": database_name,
            "backend": metadata.get("backend"),
            "modified": str(modified),
        }
    )


def _refresh_activity_cache_token(manifest_path: Path, database_name: str) -> None:
    """Record the post-run database token after SENSE-only biosphere writes."""

    manifest = read_manifest(manifest_path)
    if manifest is None:
        return
    refreshed_token = _brightway_database_token(database_name)
    manifest["brightway_database_token"] = refreshed_token
    if manifest.get("price_source_database") == database_name:
        manifest["price_source_database_token"] = refreshed_token
    dependency_tokens = manifest.get("pricing_dependency_tokens")
    if isinstance(dependency_tokens, dict) and database_name in dependency_tokens:
        dependency_tokens[database_name] = refreshed_token
    eligibility = manifest.get("eligibility_dependency_tokens")
    if isinstance(eligibility, dict):
        manifest["eligibility_dependency_tokens"] = _eligibility_dependency_tokens(set(eligibility))
    write_manifest(manifest_path, manifest)


def _eligibility_dependency_tokens(names: set[str]) -> dict[str, str] | None:
    """Use tracked database changes, conservatively declining untracked backends.

    The ordinary fallback fingerprint omits environmental exchanges. It cannot
    establish empty-inventory cache validity when ``modified`` is unavailable.
    """
    if any((databases.get(name) or {}).get("modified") is None for name in names):
        return None
    return {name: _brightway_database_token(name) for name in sorted(names)}


def _activity_cache_matches(
    manifest: dict[str, Any] | None,
    expected: dict[str, Any],
) -> bool:
    """Validate static inputs and every Brightway database used for pricing."""
    if manifest is None or any(manifest.get(key) != value for key, value in expected.items()):
        return False
    dependency_tokens = manifest.get("pricing_dependency_tokens")
    if not isinstance(dependency_tokens, dict) or not dependency_tokens:
        return False
    return all(
        database_name in databases and _brightway_database_token(database_name) == stored_token
        for database_name, stored_token in dependency_tokens.items()
    )


def _prepare_psilca_data(
    psilca_dir: Path,
    cfg: SenseConfig,
    combined_path: Path,
    css_list_path: Path,
    zero_report_path: Path,
) -> None:
    """Create the PSILCA assessment cache and sector list when inputs changed."""
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = combined_path.with_suffix(".manifest.json")
    native = psilca_dir.is_file()
    if native:
        from sense.data.zolca_reader import archive_fingerprint

    expected_manifest: dict[str, Any] = {
        "schema": 3,
        "parser": (
            f"psilca-zolca-v1-schema{PSILCA_CACHE_SCHEMA_VERSION}"
            if native
            else f"psilca-ecospold01-v{PSILCA_CACHE_SCHEMA_VERSION}"
        ),
        "input_fingerprint": (
            archive_fingerprint(psilca_dir)
            if native
            else directory_fingerprint(psilca_dir, {".xml"})
        ),
        **(
            {
                "reader_sha256": file_hash(Path(__file__).parent / "resources/zolca/reader.jar"),
                "normalizer_sha256": stable_hash(
                    {
                        name: file_hash(Path(__file__).parent / "data" / name)
                        for name in ("psilca.py", "psilca_zolca.py")
                    }
                ),
            }
            if native
            else {}
        ),
    }

    manifest = read_manifest(manifest_path)
    if (
        not cfg.rebuild_cache
        and combined_path.is_file()
        and css_list_path.is_file()
        and manifest is not None
        and all(manifest.get(key) == value for key, value in expected_manifest.items())
        and manifest.get("artifact_hashes")
        == {
            "assessments": file_hash(combined_path),
            "sectors": file_hash(css_list_path),
        }
    ):
        logger.info("PSILCA: sectors and exchanges already exist in %s", combined_path.parent)
        return

    logger.info("Step 1/5: Extract PSILCA sectors and social exchanges...")
    if native:
        from sense.data.psilca_zolca import extract_zolca

        extract_zolca(psilca_dir, combined_path, css_list_path, zero_report_path)
    else:
        combine_psilca_css_and_extract_social_exchanges(
            input_folder=str(psilca_dir),
            combined_output_path=str(combined_path),
            css_list_output_path=str(css_list_path),
            zero_report_path=str(zero_report_path),
        )
    expected_manifest["artifact_hashes"] = {
        "assessments": file_hash(combined_path),
        "sectors": file_hash(css_list_path),
    }
    write_manifest(manifest_path, expected_manifest)
    logger.info(
        "PSILCA: wrote combined file '%s' and CSS list '%s'",
        combined_path.name,
        css_list_path.name,
    )


def _prepare_location_mapping(
    cfg: SenseConfig,
    css_list_path: Path | None,
    location_mapping_xlsx: Path,
    ecoinvent_geographies_excel: Path,
    output_dir: Path,
) -> pd.DataFrame:
    """Map ecoinvent geographies to the workflow's eligible countries."""
    output_dir.mkdir(parents=True, exist_ok=True)

    destination = "PSILCA countries" if css_list_path is not None else "constituent countries"
    logger.info("Step 2/5: Mapping ecoinvent locations to %s...", destination)
    loc_df = map_ecoinvent_locations(
        ecoinvent_excel_path=str(ecoinvent_geographies_excel),
        psilca_sector_json_path=str(css_list_path) if css_list_path is not None else None,
    )

    if cfg.save_intermediate:
        with atomic_output_path(location_mapping_xlsx) as temporary:
            loc_df.to_excel(temporary, index=False)
        logger.info("Saved location mapping to: %s", location_mapping_xlsx)

    return loc_df


def _check_location_concordance(
    technosphere_db: str,
    loc_df: pd.DataFrame,
    concordance_dir: Path,
    database_label: str,
    psilca_css_list_path: Path | None,
    database_locations: set[str] | None = None,
) -> pd.DataFrame:
    """Detect unmapped locations, manage concordance template, inject filled entries.

    The user provides the full list of constituent countries for each unmapped
    location. SENSE filters these to PSILCA coverage in the integrated workflow
    and retains them all in the productivity-only workflow.

    Returns augmented loc_df if all locations are resolved.
    """
    import pandas as pd

    db_locations = (
        collect_database_locations(technosphere_db)
        if database_locations is None
        else database_locations
    )
    unmapped = detect_unmapped_locations(db_locations, loc_df)

    if not unmapped:
        return loc_df  # fast path — nothing to do

    concordance_dir.mkdir(parents=True, exist_ok=True)
    concordance_path = concordance_dir / f"location_concordance_{slug(database_label)}.json"

    n_loaded, n_new, n_still_missing = write_or_merge_location_concordance(
        concordance_path=str(concordance_path),
        unmapped=unmapped,
        database_name=database_label,
    )

    if n_new > 0:
        logger.info(
            "  %d new unmapped location(s) added to concordance template.",
            n_new,
        )
    if n_loaded > 0:
        logger.info(
            "  %d location concordance(s) already filled.",
            n_loaded,
        )

    if n_still_missing > 0:
        raise SenseInputRequiredError(
            f"{n_still_missing} location concordance entries are required",
            concordance_path,
        )

    psilca_iso2_set: set[str] | None = None
    if psilca_css_list_path is not None:
        with open(psilca_css_list_path, encoding="utf-8") as f:
            psilca_iso2_set = set(json.load(f).keys())

    # All filled — load, filter to PSILCA countries, and inject into loc_df
    concordance = load_location_concordance(str(concordance_path))
    loc_df = loc_df[~loc_df["Shortname"].isin(unmapped)].copy()
    existing_codes = set(loc_df["Shortname"].dropna().astype(str).str.strip())

    new_rows = []
    for code, countries_raw in concordance.items():
        if code not in unmapped or code in existing_codes:
            continue
        all_codes = [c.strip() for c in countries_raw.split(";") if c.strip()]
        if psilca_iso2_set is None:
            new_rows.append(
                {
                    "Shortname": code,
                    "simplified_ISO2": "; ".join(all_codes),
                    "PSILCA_mapping": "; ".join(sorted(all_codes)),
                    "PSILCA_mapping_info": "From location concordance",
                }
            )
            continue
        psilca_codes = [c for c in all_codes if c in psilca_iso2_set]
        dropped = [c for c in all_codes if c not in psilca_iso2_set]

        if not psilca_codes:
            raise SenseInputRequiredError(
                f"Location {code!r} has no countries available in PSILCA",
                concordance_path,
            )

        if dropped:
            logger.info(
                "  Location '%s': filtered %d/%d countries to PSILCA-available (dropped: %s).",
                code,
                len(psilca_codes),
                len(all_codes),
                ", ".join(dropped),
            )

        psilca_str = "; ".join(sorted(psilca_codes))
        new_rows.append(
            {
                "Shortname": code,
                "simplified_ISO2": "; ".join(all_codes),
                "PSILCA_mapping": psilca_str,
                "PSILCA_mapping_info": f"From location concordance (filtered {len(psilca_codes)}/{len(all_codes)} to PSILCA)",
            }
        )

    if new_rows:
        loc_df = pd.concat([loc_df, pd.DataFrame(new_rows)], ignore_index=True)
        logger.info(
            "  Injected %d location(s) from concordance into mapping.",
            len(new_rows),
        )

    return loc_df


def _prepare_activity_list(
    technosphere_db: str,
    price_source_database: str,
    loc_df: pd.DataFrame,
    cfg: SenseConfig,
    activity_list_path: Path,
    seed_prices_dir: Path,
    database_label: str,
    concordance_path: Path | None = None,
    technosphere_activities: list[Any] | None = None,
) -> list[dict]:
    """Build the JSON list of activities with prices + PSILCA mappings."""
    activity_list_path.parent.mkdir(parents=True, exist_ok=True)

    # Auto seed template path (one per database)
    seed_prices_dir.mkdir(parents=True, exist_ok=True)
    auto_seed_path = seed_prices_dir / f"seed_prices_{slug(database_label)}.json"

    activity_manifest_path = activity_list_path.with_suffix(".manifest.json")
    expected_manifest: dict[str, Any] = {
        "schema": 4,
        "activity_file_hash": (
            file_hash(activity_list_path) if activity_list_path.exists() else None
        ),
        "pricing_audit_hash": (
            file_hash(activity_list_path.with_suffix(".pricing.json"))
            if activity_list_path.with_suffix(".pricing.json").exists()
            else None
        ),
        "price_inference_version": PRICE_INFERENCE_VERSION,
        "rest_of_region_exclusions": "actual_country_membership_v1",
        "eligibility_version": "empty_physical_inventory_v2",
        "brightway_database": technosphere_db,
        "brightway_database_token": _brightway_database_token(technosphere_db),
        "price_source_database": price_source_database,
        "price_source_database_token": _brightway_database_token(price_source_database),
        "location_mapping_hash": stable_hash(loc_df.fillna("").to_dict(orient="records")),
        "seed_file_hash": file_hash(auto_seed_path) if auto_seed_path.exists() else None,
        "concordance_file_hash": (
            file_hash(concordance_path)
            if concordance_path is not None and concordance_path.exists()
            else None
        ),
        "pricing_config_hash": stable_hash(
            {
                "exclude_terms": cfg.exclude_terms,
                "exclude_terms_starting_with": cfg.exclude_terms_starting_with,
                "exclude_exact": cfg.exclude_exact,
                "exclude_empty_inventories": cfg.exclude_empty_inventories,
                "price_floor_percentile": cfg.price_floor_percentile,
                "zero_price_policy": cfg.zero_price_policy,
                "treatment_fallback_enabled": cfg.treatment_working_hours_percentile > 0,
                "price_currency": cfg.ecoinvent_price_currency,
                "price_year": cfg.ecoinvent_price_year,
            }
        ),
    }

    cached_manifest = read_manifest(activity_manifest_path)
    cache_inputs_match = (
        not cfg.rebuild_cache
        and activity_list_path.exists()
        and _activity_cache_matches(cached_manifest, expected_manifest)
    )
    stored_dependencies = (cached_manifest or {}).get("eligibility_dependency_tokens")
    eligibility_is_current = not cfg.exclude_empty_inventories or (
        isinstance(stored_dependencies, dict)
        and technosphere_db in stored_dependencies
        and _eligibility_dependency_tokens(set(stored_dependencies)) == stored_dependencies
    )

    def read_cached_activities() -> list[dict]:
        logger.info("Step 3/5: Using validated activity cache: %s", activity_list_path)
        with open(activity_list_path, encoding="utf-8") as f:
            cached_activities = json.load(f)
        if not isinstance(cached_activities, list):
            raise ValueError(f"Invalid activity cache structure: {activity_list_path}")
        return cached_activities

    if cache_inputs_match and eligibility_is_current:
        return read_cached_activities()

    from sense.brightway.repository import empty_inventory_keys

    flow_dependencies: set[str] = set()
    empty_inventory = set()
    if cfg.exclude_empty_inventories:
        logger.info("Checking physical inventory eligibility for %r...", technosphere_db)
        empty_inventory = empty_inventory_keys(
            (
                technosphere_activities
                if technosphere_activities is not None
                else _load_technosphere_activities(technosphere_db)
            ),
            flow_dependencies=flow_dependencies,
        )
        logger.info("Excluded %d empty physical inventories", len(empty_inventory))
    expected_manifest["empty_inventory_hash"] = stable_hash(sorted(empty_inventory))
    expected_manifest["eligibility_dependency_tokens"] = _eligibility_dependency_tokens(
        flow_dependencies | {technosphere_db}
    )
    # Upgrade an older manifest, or one invalidated only by flow metadata, without
    # repricing or changing mapping outputs when the eligibility set is identical.
    if (
        cache_inputs_match
        and cached_manifest is not None
        and cached_manifest.get("empty_inventory_hash") == expected_manifest["empty_inventory_hash"]
    ):
        cached_manifest["eligibility_dependency_tokens"] = expected_manifest[
            "eligibility_dependency_tokens"
        ]
        write_manifest(activity_manifest_path, cached_manifest)
        return read_cached_activities()

    logger.info("Step 3/5: Extract ecoinvent activities and product prices...")
    activity_df = fetch_ecoinvent_activities(
        mapping_df=loc_df,
        output_path=str(activity_list_path),
        technosphere=technosphere_db,
        exclude_terms=cfg.exclude_terms,
        exclude_terms_starting_with=cfg.exclude_terms_starting_with,
        exclude_exact=cfg.exclude_exact,
        price_floor_percentile=cfg.price_floor_percentile,
        zero_price_policy=cfg.zero_price_policy,
        treatment_working_hours_percentile=cfg.treatment_working_hours_percentile,
        auto_seed_prices_path=str(auto_seed_path),
        database_label=database_label,
        price_currency=cfg.ecoinvent_price_currency,
        price_year=cfg.ecoinvent_price_year,
        price_source_database=price_source_database,
        activities=technosphere_activities,
        exclude_empty_inventories=cfg.exclude_empty_inventories,
        empty_keys=empty_inventory,
    )
    # The seed template may have been created or extended during extraction;
    # record its final state so the next unchanged run can reuse this cache.
    expected_manifest["seed_file_hash"] = (
        file_hash(auto_seed_path) if auto_seed_path.exists() else None
    )
    expected_manifest["activity_file_hash"] = file_hash(activity_list_path)
    expected_manifest["pricing_audit_hash"] = file_hash(
        activity_list_path.with_suffix(".pricing.json")
    )
    dependency_databases = activity_df.attrs.get("pricing_dependency_databases", [])
    expected_manifest.update(
        pricing_dependency_tokens={
            name: _brightway_database_token(name) for name in dependency_databases
        }
    )
    write_manifest(activity_manifest_path, expected_manifest)

    with open(activity_list_path, encoding="utf-8") as f:
        activities = json.load(f)
    if not isinstance(activities, list):
        raise ValueError(f"Invalid activity output structure: {activity_list_path}")
    return activities


def _prepare_embedding_stores(
    iso2_data: dict[str, dict],
    cfg: SenseConfig,
    css_list_path: Path,
    activity_list_path: Path,
    psilca_emb_dir: Path,
    ecoinvent_emb_dir: Path,
) -> None:
    """Make sure PSILCA + ecoinvent embedding stores exist."""
    psilca_emb_dir.mkdir(parents=True, exist_ok=True)
    ecoinvent_emb_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Step 4/5: Creating/using PSILCA embeddings (per country)...")
    embed_psilca_to_store(
        psilca_sector_json=str(css_list_path),
        out_dir=str(psilca_emb_dir),
        model_name=cfg.embedding_model,
        revision=cfg.embedding_revision,
        batch_size=cfg.embedding_batch_size,
        force=cfg.rebuild_cache,
    )

    logger.info("Step 5/5: Creating/using ecoinvent activity embeddings (per ISO2)...")
    target_iso2_list = list(iso2_data.keys())
    embed_ecoinvent_to_store(
        technosphere_json=str(activity_list_path),
        out_dir=str(ecoinvent_emb_dir),
        model_name=cfg.embedding_model,
        revision=cfg.embedding_revision,
        batch_size=cfg.embedding_batch_size,
        target_iso2_list=target_iso2_list,
        force=cfg.rebuild_cache,
    )
