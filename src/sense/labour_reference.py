"""Cache physical labour references without modifying Brightway inventories."""

import gzip
import json
import logging
from collections.abc import Callable
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from sense.artifacts import atomic_output_path, write_json_atomic
from sense.core.treatment import reference_distribution
from sense.utils.cache import file_hash, read_manifest, stable_hash
from sense.utils.helpers import slug


def active_regions(entries: list[dict]) -> dict[str, dict]:
    """Keep eligible positive-price activities and explicit treatment recipients.

    The priced input snapshot retains zero prices. They are omitted only from
    matching and exchange generation, not classified as excluded activities.
    """
    return {
        entry["ISO2"]: {
            **entry,
            "Activities": [
                activity
                for activity in entry.get("Activities", [])
                if activity.get("calculation route") != "zero"
            ],
        }
        for entry in entries
    }


def load_or_build_reference(
    *,
    cache_directory: Path,
    identity: dict[str, Any],
    regions: list[str],
    prepare_region: Callable[[str], dict],
    percentile: float,
    prepared_directory: Path | None = None,
    retain_regions: set[str] | None = None,
    rebuild: bool = False,
) -> tuple[dict, Path, dict[str, Path]]:
    """Use all reference regions, independent of target selection and order.

    Only one region's social profiles remains in memory. Optional disk payloads
    allow an initial reference-background run to write its results without
    repeating sector matching. A failed build never publishes a valid cache.
    """
    identity = {"schema": 1, "percentile": percentile, **identity}
    path = cache_directory / f"{stable_hash(identity)}.json"
    cached = read_manifest(path) if not rebuild else None
    if (
        cached is not None
        and cached.get("identity") == identity
        and isinstance(cached.get("reference"), dict)
        and cached.get("reference_hash") == stable_hash(cached["reference"])
    ):
        return cached["reference"], path, {}

    observations: list[dict[str, Any]] = []
    prepared_paths = {}
    for region in sorted(regions):
        prepared = prepare_region(region)
        observations.extend(
            {
                key: record.get(key)
                for key in (
                    "Activity code",
                    "Unit",
                    "Price",
                    "Working_hours",
                    "Working_hours_source",
                )
            }
            for record in prepared["records"]
        )
        if prepared_directory is not None and (retain_regions is None or region in retain_regions):
            prepared_path = (
                prepared_directory / f"{slug(region)}-{stable_hash(region)[:12]}.json.gz"
            )
            with atomic_output_path(prepared_path) as temporary:
                with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=1) as handle:
                    json.dump(prepared, handle, ensure_ascii=False)
            prepared_paths[region] = prepared_path
    reference = reference_distribution(observations, percentile)
    write_json_atomic(
        path,
        {
            "identity": identity,
            "reference": reference,
            "reference_hash": stable_hash(reference),
        },
    )
    return reference, path, prepared_paths


def load_prepared(paths: dict[str, Path], region: str) -> dict | None:
    """Read one prepared region only when this run built the reference."""
    if region not in paths:
        return None
    with gzip.open(paths[region], "rt", encoding="utf-8") as handle:
        prepared = json.load(handle)
    if not isinstance(prepared, dict):
        raise ValueError(f"Invalid prepared-region cache for {region!r}")
    return prepared


def prepare_labour_reference(
    *,
    config,
    target_database,
    source_database,
    target_regions,
    selected_regions,
    target_paths,
    technosphere_lookup,
    location_mapping,
    concordance_directory,
    seed_directory,
    economic_context,
    psilca_store,
    evidence,
    ecoinvent_version,
    psilca_version,
) -> tuple[dict, dict[str, Path]]:
    """Resolve a complete background reference before any inventory mutation."""
    from sense.preparation import (
        _check_location_concordance,
        _load_technosphere_activities,
        _prepare_activity_list,
        _prepare_embedding_stores,
    )
    from sense.regional import (
        _preflight_productivity_coverage,
        _run_productivity_for_iso2,
        _run_social_for_iso2,
    )
    from sense.run_context import RunPaths
    from sense.utils.helpers import norm_code

    same_database = target_database == source_database
    cfg = replace(
        config,
        iso2_selection=None,
        iso2_start_at=None,
        enable_working_hours_flows=True,
        save_intermediate=config.save_intermediate if same_database else False,
    )
    if same_database:
        paths, regions, lookup = target_paths, target_regions, technosphere_lookup
    else:
        paths = RunPaths.create(
            data_dir=target_paths.output_dir.parent,
            technosphere_database=source_database,
            ecoinvent_version=ecoinvent_version,
            psilca_version=psilca_version,
            use_psilca=cfg.use_psilca,
        )
        activities = _load_technosphere_activities(source_database)
        mapping = _check_location_concordance(
            technosphere_db=source_database,
            loc_df=location_mapping,
            concordance_dir=concordance_directory,
            database_label=source_database,
            psilca_css_list_path=paths.psilca_sector_list if cfg.use_psilca else None,
            database_locations={str(act.get("location") or "").strip() for act in activities},
        )
        entries = _prepare_activity_list(
            technosphere_db=source_database,
            price_source_database=source_database,
            loc_df=mapping,
            cfg=cfg,
            activity_list_path=paths.ecoinvent_activity_list,
            seed_prices_dir=seed_directory,
            database_label=source_database,
            concordance_path=concordance_directory
            / f"location_concordance_{slug(source_database)}.json",
            technosphere_activities=activities,
        )
        regions = active_regions(entries)
        lookup = {norm_code(act.get("code")): act for act in activities}
    selected = set(selected_regions)
    regions = {
        region: {
            **entry,
            "Activities": [
                activity
                for activity in entry.get("Activities", [])
                if activity.get("calculation route") != "treatment_fallback"
                or (same_database and region in selected)
            ],
        }
        for region, entry in regions.items()
    }
    reference_regions = [region for region, entry in regions.items() if entry.get("Activities")]

    # These affect neither direct ordinary hours nor the retained sector set.
    ignored = {
        "iso2_selection",
        "iso2_start_at",
        "save_intermediate",
        "rebuild_cache",
        "level_weights",
        "social_flow_mean",
        "rounding_digits",
        "minimum_similarity",
        "enable_social_flows",
        "enable_working_hours_flows",
        "enable_isic_working_hours_methods",
        "continue_on_iso2_error",
        "precreated_flows",
        "update_tol",
    }
    files = {
        "priced_activities": paths.ecoinvent_activity_list,
        "psilca_manifest": paths.psilca_assessments.with_suffix(".manifest.json"),
    }
    for field in ("productivity_baseline_path", "country_gdp_path"):
        value = getattr(cfg, field, None)
        if value:
            files[field] = Path(value)
    identity = {
        "source_database": source_database,
        "versions": {"ecoinvent": ecoinvent_version, "psilca": psilca_version},
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(cfg).items()
            if key not in ignored
        },
        "inputs": {name: file_hash(path) for name, path in files.items() if path.is_file()},
        "economic_assumptions": {
            "psilca_factor": economic_context.psilca_conversion.factor if cfg.use_psilca else None,
            "productivity_factor": economic_context.productivity_price_factor,
            "country_weights": {
                key: value
                for key, value in economic_context.country_weights.audit_dict().items()
                if key
                not in {
                    "country_observations",
                    "data_source_counts",
                    "resolution_counts",
                    "data_year_counts",
                    "missing_country_counts",
                }
            },
        },
        "implementation": evidence.identity["source_hash"],
    }
    # JSON normalizes tuples and paths consistently before cache comparison.
    identity = json.loads(json.dumps(identity, default=str))
    ready = False
    productivity = {}

    def prepare(region):
        nonlocal ready, productivity
        if not ready:
            logging.getLogger(__name__).info(
                "Building physical working-hour reference from %s (%d regions)",
                source_database,
                len(reference_regions),
            )
            if cfg.use_psilca:
                _prepare_embedding_stores(
                    iso2_data=regions,
                    cfg=cfg,
                    css_list_path=paths.psilca_sector_list,
                    activity_list_path=paths.ecoinvent_activity_list,
                    psilca_emb_dir=paths.psilca_embeddings,
                    ecoinvent_emb_dir=paths.ecoinvent_embeddings,
                )
            if not cfg.use_psilca:
                productivity = _preflight_productivity_coverage(
                    regions=reference_regions,
                    iso2_data=regions,
                    config=cfg,
                    baseline=economic_context.productivity,
                )
            ready = True
        common = {
            "target_iso2": region,
            "iso2_data": regions,
            "technosphere_db": source_database,
            "sociosphere_db": evidence.identity["sociosphere"],
            "cfg": (
                cfg
                if same_database and region in selected
                else replace(cfg, save_intermediate=False)
            ),
            "techno_lookup": lookup,
            "sociosphere_lookup": {},
            "ecoinvent_version_label": ecoinvent_version,
            "output_dir": evidence.directory,
            "prepare_only": True,
            "productivity_price_factor": economic_context.productivity_price_factor,
        }
        if cfg.use_psilca:
            return _run_social_for_iso2(
                **common,
                psilca_store=psilca_store,
                ecoinvent_emb_dir=paths.ecoinvent_embeddings,
                psilca_emb_dir=paths.psilca_embeddings,
                productivity_baseline=economic_context.productivity,
                country_weight_provider=economic_context.country_weights,
                psilca_price_factor=economic_context.psilca_conversion.factor,
            )
        values, audit = productivity[region]
        return _run_productivity_for_iso2(
            **common, productivity_values=values, productivity_audit=audit
        )

    reference, path, prepared = load_or_build_reference(
        cache_directory=target_paths.output_dir / "WorkingHoursReferences",
        identity=identity,
        regions=reference_regions,
        prepare_region=prepare,
        percentile=cfg.treatment_working_hours_percentile,
        rebuild=cfg.rebuild_cache,
        prepared_directory=evidence.directory / "PreparedReference" if same_database else None,
        retain_regions=selected,
    )
    evidence.snapshot_input("treatment_working_hours_reference", path)
    if not same_database:
        evidence.snapshot_input(
            "treatment_reference_priced_activities", paths.ecoinvent_activity_list
        )
    return reference, prepared
