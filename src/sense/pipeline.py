"""Orchestrate a SENSE run and publish its completion evidence."""

from __future__ import annotations

import logging
import math
from collections import Counter
from pathlib import Path
from typing import Any

from bw2data import Database, databases, projects

from sense.artifacts import read_json, write_json_atomic
from sense.brightway.methods import (
    add_social_impact_categories,
    add_working_hours_impact_categories_by_isic,
    add_working_hours_impact_category,
)
from sense.config import SenseConfig
from sense.core.matching import PSILCALookupStore
from sense.core.treatment import fallback_overrides
from sense.economics import (
    resolve_economic_context,
    write_economic_assumptions,
    write_productivity_audit,
)
from sense.eligibility import validate_activity_locations
from sense.exceptions import SenseInputRequiredError, SenseRegionError
from sense.labour_reference import active_regions, load_prepared, prepare_labour_reference
from sense.run_context import RunEvidence, RunPaths
from sense.utils.helpers import norm_code, slug
from sense.utils.locking import exclusive_run
from sense.weighting import (
    BUNDLED_IMF_GDP_FILE,
    BUNDLED_IMF_METADATA_FILE,
)

from .preparation import (
    _check_location_concordance,
    _load_technosphere_activities,
    _prepare_activity_list,
    _prepare_embedding_stores,
    _prepare_location_mapping,
    _prepare_psilca_data,
    _refresh_activity_cache_token,
)
from .regional import (
    _clear_managed_exchanges,
    _preflight_productivity_coverage,
    _run_productivity_for_iso2,
    _run_social_for_iso2,
)

logger = logging.getLogger(__name__)


def _get_package_data_dir() -> Path:
    """Get the data directory relative to where SENSE is run from."""
    return Path.cwd() / "Data"


def _select_iso2_regions(iso2_data: dict[str, dict], config: SenseConfig) -> list[str]:
    """Apply optional selection and restart controls while preserving source order."""

    regions = list(dict.fromkeys(iso2_data))
    if config.iso2_selection is not None:
        allowed = set(config.iso2_selection)
        missing = allowed - set(regions)
        if missing:
            raise ValueError(
                f"iso2_selection contains regions absent from the database: {sorted(missing)}"
            )
        regions = [region for region in regions if region in allowed]

    if config.iso2_start_at:
        if config.iso2_start_at not in regions:
            raise ValueError(
                f"iso2_start_at={config.iso2_start_at!r} was not found in the region list"
            )
        regions = regions[regions.index(config.iso2_start_at) :]
        logger.info(
            "Starting from iso2_start_at='%s'; %d ISO2s will be processed",
            config.iso2_start_at,
            len(regions),
        )
    return regions


def _create_impact_methods(sociosphere_db: str, config: SenseConfig) -> None:
    """Update enabled methods, retaining methods shared with other databases."""

    if config.enable_social_flows:
        logger.info(
            "Creating/updating social impact methods from sociosphere DB '%s'...",
            sociosphere_db,
        )
        add_social_impact_categories(biosphere_db_name=sociosphere_db)

    if config.enable_working_hours_flows:
        logger.info(
            "Creating/updating working-hours impact method from sociosphere DB '%s'...",
            sociosphere_db,
        )
        add_working_hours_impact_category(biosphere_db_name=sociosphere_db)

    if config.enable_isic_working_hours_methods:
        logger.info(
            "Creating/updating per-ISIC working-hours methods from sociosphere DB '%s'...",
            sociosphere_db,
        )
        add_working_hours_impact_categories_by_isic(biosphere_db_name=sociosphere_db)


@exclusive_run
def run_sense(
    *,
    project: str,
    technosphere_db: str,
    sociosphere_db: str = "sociosphere",
    price_source_db: str | None = None,
    psilca_path: Path | None = None,
    psilca_dir: Path | None = None,
    ecoinvent_version: str,
    psilca_version: str | None = None,
    config: SenseConfig,
    data_dir: Path | None = None,
    ecoinvent_geographies_excel: Path | None = None,
) -> list[dict[str, Any]]:
    """Run the SENSE preparation, matching, aggregation, and Brightway stages.

    Args:
        project: Brightway project name.
        technosphere_db: Technosphere database to process.
        sociosphere_db: Database that owns SENSE social flows.
        price_source_db: Reference background supplying fallback prices and the
            complete direct working-hour distribution for eligible treatments.
            Defaults to the target database. Set explicitly for transformed or
            foreground databases; names and system models are not inferred.
        psilca_path: Native PSILCA .zolca backup (recommended), or EcoSpold01 XML
            directory. Required only when ``config.use_psilca`` is true.
        psilca_dir: Legacy alias for ``psilca_path``. Do not supply both.
        ecoinvent_version: Ecoinvent label used in cache and audit names.
        psilca_version: PSILCA label used for caches and monetary defaults.
            Required only when ``config.use_psilca`` is true.
        config: Validated pipeline configuration.
        data_dir: Artifact root; defaults to ``./Data``.
        ecoinvent_geographies_excel: Ecoinvent Database Overview workbook. If omitted,
            SENSE accepts exactly one matching workbook in ``data_dir``.

    Returns:
        One processing summary per selected ecoinvent region.

    Raises:
        FileNotFoundError: If a required input file or directory is absent.
        SenseInputRequiredError: If generated seed or concordance entries need input.
        SenseRegionError: If a region fails and continuation is disabled.
        ValueError: If input data or monetary assumptions are invalid.
    """
    if not isinstance(config, SenseConfig):
        raise ValueError("config must be a SenseConfig")
    # Config objects are mutable; revalidate changes made after construction.
    config.__post_init__()
    for name, value in (
        ("project", project),
        ("technosphere_db", technosphere_db),
        ("sociosphere_db", sociosphere_db),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be non-empty text")
    if sociosphere_db in {technosphere_db, price_source_db}:
        raise ValueError(
            "sociosphere_db must differ from the technosphere and price source databases"
        )
    if psilca_path is not None and psilca_dir is not None:
        raise ValueError("Supply psilca_path or the legacy psilca_dir alias, not both")
    psilca_source = psilca_path if psilca_path is not None else psilca_dir
    if config.use_psilca:
        if psilca_source is None:
            raise ValueError("psilca_path is required when use_psilca=True")
        from sense.data.psilca import validate_psilca_source

        psilca_source = validate_psilca_source(psilca_source)
    if config.use_psilca and not psilca_version:
        raise ValueError("psilca_version is required when use_psilca=True")
    active_psilca_version = psilca_version if config.use_psilca else None
    # Set up paths
    if data_dir is None:
        data_dir = _get_package_data_dir()
    data_dir = Path(data_dir).expanduser()

    run_paths = RunPaths.create(
        data_dir=data_dir,
        technosphere_database=technosphere_db,
        ecoinvent_version=ecoinvent_version,
        psilca_version=active_psilca_version,
        use_psilca=config.use_psilca,
    )
    output_dir = run_paths.output_dir
    ecoinvent_out_dir = run_paths.ecoinvent_output_dir

    # Resolve the version-specific ecoinvent geography workbook explicitly.
    if ecoinvent_geographies_excel is None:
        candidates = sorted(data_dir.glob("Database-Overview-for-ecoinvent*.xlsx"))
        if len(candidates) != 1:
            raise FileNotFoundError(
                "Set ecoinvent_geographies_excel explicitly, or place exactly one "
                "Database-Overview-for-ecoinvent*.xlsx workbook in "
                f"{data_dir}. Found {len(candidates)}."
            )
        ecoinvent_geographies_excel = candidates[0]
    ecoinvent_geographies_excel = Path(ecoinvent_geographies_excel)
    if not ecoinvent_geographies_excel.is_file():
        raise FileNotFoundError(
            f"Ecoinvent geographies workbook not found: {ecoinvent_geographies_excel}"
        )

    economic_context = resolve_economic_context(
        config=config,
        psilca_version=active_psilca_version,
        ecoinvent_geographies_excel=ecoinvent_geographies_excel,
    )
    psilca_price_conversion = economic_context.psilca_conversion
    country_weight_provider = economic_context.country_weights
    pb = economic_context.productivity
    productivity_price_factor = economic_context.productivity_price_factor

    logger.info("Starting SENSE pipeline in Brightway project '%s'", project)
    if project not in projects:
        raise ValueError(
            f"Brightway project {project!r} does not exist; import the technosphere first"
        )
    projects.set_current(project)

    # Validate the requested database before parsing source data or building caches.
    # Brightway otherwise permits ``Database(unknown_name)`` and exposes an empty
    # iterator, which hides naming mistakes until a much later processing stage.
    technosphere_activities = _load_technosphere_activities(technosphere_db)
    validate_activity_locations(
        technosphere_activities,
        prefixes=config.exclude_terms_starting_with,
        terms=config.exclude_terms,
        exact=config.exclude_exact,
    )
    price_source_database = price_source_db or technosphere_db
    if price_source_database not in databases:
        raise ValueError(f"Price source database {price_source_database!r} does not exist")

    base_label = technosphere_db
    ei_tag = run_paths.ecoinvent_tag
    ps_tag = run_paths.psilca_tag
    psilca_assessments = run_paths.psilca_assessments
    psilca_css_list_json = run_paths.psilca_sector_list
    psilca_zero_report = run_paths.psilca_nonpositive_report
    eco_activity_list_json = run_paths.ecoinvent_activity_list
    location_mapping_xlsx = run_paths.location_mapping
    emb_eco_dir = run_paths.ecoinvent_embeddings
    emb_psilca_dir = run_paths.psilca_embeddings

    # 1) Optional PSILCA data + location mapping + activity list
    if config.use_psilca:
        assert psilca_source is not None
        _prepare_psilca_data(
            psilca_dir=psilca_source,
            cfg=config,
            combined_path=psilca_assessments,
            css_list_path=psilca_css_list_json,
            zero_report_path=psilca_zero_report,
        )
    else:
        logger.info("PSILCA disabled: parsing and licensed inputs are skipped")

    loc_df = _prepare_location_mapping(
        cfg=config,
        css_list_path=psilca_css_list_json if config.use_psilca else None,
        location_mapping_xlsx=location_mapping_xlsx,
        ecoinvent_geographies_excel=ecoinvent_geographies_excel,
        output_dir=ecoinvent_out_dir,
    )

    # Check for unmapped locations (e.g. REMIND regions from premise)
    concordance_dir = (
        config.location_concordances_dir
        if config.location_concordances_dir is not None
        else output_dir / "LocationConcordances"
    )
    loc_df = _check_location_concordance(
        technosphere_db=technosphere_db,
        loc_df=loc_df,
        concordance_dir=concordance_dir,
        database_label=base_label,
        psilca_css_list_path=psilca_css_list_json if config.use_psilca else None,
        database_locations={
            str(activity.get("location") or "").strip()
            for activity in technosphere_activities
            if str(activity.get("location") or "").strip()
        },
    )
    # Resolve seed prices directory
    if config.seed_prices_dir is not None:
        seed_prices_dir = config.seed_prices_dir
    else:
        seed_prices_dir = output_dir / "SeedPrices"

    auto_seed_path = seed_prices_dir / f"seed_prices_{slug(base_label)}.json"

    auto_concordance_path = concordance_dir / f"location_concordance_{slug(base_label)}.json"

    _activity_list_raw = _prepare_activity_list(
        technosphere_db=technosphere_db,
        price_source_database=price_source_database,
        loc_df=loc_df,
        cfg=config,
        activity_list_path=eco_activity_list_json,
        seed_prices_dir=seed_prices_dir,
        database_label=base_label,
        concordance_path=auto_concordance_path,
        technosphere_activities=technosphere_activities,
    )
    # Index by ISO2 for O(1) lookups (replaces repeated linear scans)
    iso2_data = active_regions(_activity_list_raw)

    # Check if seed prices are still missing — stop early if so
    if auto_seed_path.exists():
        _seed_data = read_json(auto_seed_path)

        def _required_seed_is_missing(seed: dict[str, Any]) -> bool:
            if not seed.get("required", False):
                return False
            raw_price = seed.get("price")
            if raw_price is None:
                return True
            try:
                value = float(raw_price)
            except (TypeError, ValueError):
                return True
            return not math.isfinite(value) or value < 0

        _n_missing = sum(_required_seed_is_missing(seed) for seed in _seed_data.get("seeds", []))
        if _n_missing > 0:
            raise SenseInputRequiredError(
                f"{_n_missing} seed prices are required",
                auto_seed_path,
            )

    # 3) Build BW lookups
    techno_lookup = {}

    for act in technosphere_activities:
        c = norm_code(act.get("code"))
        if c:
            techno_lookup[c] = act

    # Include regions containing only excluded or zero-priced activities so a
    # rerun can remove their obsolete inventory within the requested scope.
    for act in techno_lookup.values():
        location = str(act.get("location") or "").strip()
        iso2_data.setdefault(location, {"ISO2": location, "Activities": []})

    # 4) Decide which ISO2s to run
    iso2_ordered = _select_iso2_regions(iso2_data, config)

    productivity_preflight: dict[str, tuple[dict[str, float], list[dict[str, Any]]]] = {}
    if (
        not config.use_psilca and config.enable_working_hours_flows
    ) or config.productivity_gdp_policy == "available_countries":
        if pb is None:  # Defensive boundary; SenseConfig rejects this combination.
            raise ValueError("PSILCA-free working hours require a productivity baseline")
        productivity_preflight = _preflight_productivity_coverage(
            regions=iso2_ordered,
            iso2_data=iso2_data,
            config=config,
            baseline=pb,
        )

    evidence = RunEvidence.start(
        output_dir,
        project,
        technosphere_db,
        sociosphere_db,
        config,
        {
            "geographies": ecoinvent_geographies_excel,
            "priced_activities": eco_activity_list_json,
            "pricing_audit": eco_activity_list_json.with_suffix(".pricing.json"),
            "seed_prices": auto_seed_path,
            "location_concordance": auto_concordance_path,
            "country_gdp": (
                Path(config.country_gdp_path)
                if config.country_gdp_path
                else Path(__file__).parent / "resources/world_bank_gdp_current_usd_2000_2024.csv"
            ),
            **(
                {
                    "country_gdp_imf": Path(__file__).parent / "resources" / BUNDLED_IMF_GDP_FILE,
                    "country_gdp_imf_metadata": Path(__file__).parent
                    / "resources"
                    / BUNDLED_IMF_METADATA_FILE,
                }
                if config.country_weighting == "gdp" and not config.country_gdp_path
                else {}
            ),
            "activity_manifest": eco_activity_list_json.with_suffix(".manifest.json"),
            "psilca_manifest": psilca_assessments.with_suffix(".manifest.json"),
            **(
                {"productivity": Path(config.productivity_baseline_path)}
                if config.productivity_baseline_path
                else {}
            ),
        },
        snapshot_inputs={
            "priced_activities",
            "pricing_audit",
            "activity_manifest",
            "psilca_manifest",
        },
        source_directories=(
            {"psilca": psilca_source}
            if config.use_psilca and psilca_source is not None and psilca_source.is_dir()
            else {}
        ),
        source_archives=(
            {"psilca": psilca_source}
            if config.use_psilca and psilca_source is not None and psilca_source.is_file()
            else {}
        ),
        versions={"ecoinvent": ecoinvent_version, "psilca": active_psilca_version},
    )

    psilca_store = (
        PSILCALookupStore(psilca_assessments, process_overrides=config.psilca_process_overrides)
        if config.use_psilca
        else None
    )
    working_hours_reference: dict[str, Any] = {}
    prepared_regions: dict[str, Path] = {}
    if any(
        activity.get("calculation route") == "treatment_fallback"
        for region in iso2_ordered
        for activity in iso2_data[region].get("Activities", [])
    ):
        working_hours_reference, prepared_regions = prepare_labour_reference(
            config=config,
            target_database=technosphere_db,
            source_database=price_source_database,
            target_regions=iso2_data,
            selected_regions=iso2_ordered,
            target_paths=run_paths,
            technosphere_lookup=techno_lookup,
            location_mapping=loc_df,
            concordance_directory=concordance_dir,
            seed_directory=seed_prices_dir,
            economic_context=economic_context,
            psilca_store=psilca_store,
            evidence=evidence,
            ecoinvent_version=ecoinvent_version,
            psilca_version=active_psilca_version,
        )
        for region in iso2_ordered:
            fallback_overrides(iso2_data[region].get("Activities", []), working_hours_reference)

    # 2) Embeddings exist only to match ecoinvent activities to PSILCA sectors.
    if (
        config.use_psilca
        and not prepared_regions
        and any(
            entry.get("Activities") for region, entry in iso2_data.items() if region in iso2_ordered
        )
    ):
        _prepare_embedding_stores(
            iso2_data={region: iso2_data[region] for region in iso2_ordered},
            cfg=config,
            css_list_path=psilca_css_list_json,
            activity_list_path=eco_activity_list_json,
            psilca_emb_dir=emb_psilca_dir,
            ecoinvent_emb_dir=emb_eco_dir,
        )

    # Reference and productivity preflight must succeed before inventory mutation.
    if sociosphere_db not in databases:
        logger.info("Creating new sociosphere database: %s", sociosphere_db)
        Database(sociosphere_db).write({})
        logger.info("Sociosphere database created")

    sociosphere_db_obj = Database(sociosphere_db)
    sociosphere_lookup = {
        (f.get("name"), f.get("unit"), f.get("categories")): f for f in sociosphere_db_obj
    }

    selected = {
        code: act
        for code, act in techno_lookup.items()
        if str(act.get("location") or "").strip() in iso2_ordered
    }
    retained_codes = {
        norm_code(act.get("code"))
        for region in iso2_ordered
        for act in iso2_data[region].get("Activities", [])
    }
    omitted = {code: act for code, act in selected.items() if code not in retained_codes}
    all_kinds = {"social", "working_hours", "working_hours_isic"}
    _clear_managed_exchanges(
        technosphere_db=technosphere_db,
        sociosphere_db=sociosphere_db,
        techno_lookup=omitted,
        sociosphere_lookup=sociosphere_lookup,
        kinds=all_kinds,
    )
    disabled = set()
    if not config.enable_social_flows:
        disabled.add("social")
    if not config.enable_working_hours_flows:
        disabled.add("working_hours")
    if not config.enable_isic_working_hours_methods:
        disabled.add("working_hours_isic")
    _clear_managed_exchanges(
        technosphere_db=technosphere_db,
        sociosphere_db=sociosphere_db,
        techno_lookup=selected,
        sociosphere_lookup=sociosphere_lookup,
        kinds=disabled,
    )

    # Main loop over ISO2s
    from tqdm import tqdm

    results: list[dict[str, Any]] = []
    region_progress = tqdm(
        iso2_ordered,
        desc="Processing SENSE regions",
        unit="region",
        dynamic_ncols=True,
    )

    try:
        for iso2 in region_progress:

            def progress_cb(step: int, total_steps: int, message: str, iso2=iso2):
                region_progress.set_postfix_str(
                    f"{iso2} · step {step}/{total_steps}: {message}",
                    refresh=True,
                )

            try:
                if not iso2_data[iso2].get("Activities"):
                    # Exclusions and zero prices can leave a valid empty region.
                    # Its obsolete exchanges were reconciled before this loop.
                    summary = {
                        "ISO2": iso2,
                        "num_activities": 0,
                        "num_matches": 0,
                        "social_added": 0,
                        "wh_added": 0,
                        "issues": [],
                        "intentionally_empty": True,
                    }
                elif config.use_psilca:
                    assert psilca_store is not None
                    assert psilca_price_conversion is not None
                    summary = _run_social_for_iso2(
                        target_iso2=iso2,
                        iso2_data=iso2_data,
                        technosphere_db=technosphere_db,
                        sociosphere_db=sociosphere_db,
                        cfg=config,
                        techno_lookup=techno_lookup,
                        sociosphere_lookup=sociosphere_lookup,
                        ecoinvent_version_label=ecoinvent_version,
                        psilca_store=psilca_store,
                        output_dir=evidence.directory,
                        ecoinvent_emb_dir=emb_eco_dir,
                        psilca_emb_dir=emb_psilca_dir,
                        progress_cb=progress_cb,
                        productivity_baseline=pb,
                        country_weight_provider=country_weight_provider,
                        psilca_price_factor=psilca_price_conversion.factor,
                        productivity_price_factor=productivity_price_factor,
                        working_hours_reference=working_hours_reference,
                        prepared=load_prepared(prepared_regions, iso2),
                    )
                else:
                    values, audit = productivity_preflight.get(iso2, ({}, []))
                    summary = _run_productivity_for_iso2(
                        target_iso2=iso2,
                        iso2_data=iso2_data,
                        technosphere_db=technosphere_db,
                        sociosphere_db=sociosphere_db,
                        cfg=config,
                        techno_lookup=techno_lookup,
                        sociosphere_lookup=sociosphere_lookup,
                        ecoinvent_version_label=ecoinvent_version,
                        output_dir=evidence.directory,
                        productivity_values=values,
                        productivity_audit=audit,
                        productivity_price_factor=productivity_price_factor,
                        progress_cb=progress_cb,
                        working_hours_reference=working_hours_reference,
                        prepared=load_prepared(prepared_regions, iso2),
                    )
            except Exception as exc:
                if not config.continue_on_iso2_error:
                    if isinstance(exc, SenseRegionError):
                        raise
                    raise SenseRegionError(iso2, str(exc)) from exc
                logger.exception("ISO2 %s: unhandled error — continuing by request.", iso2)
                summary = {
                    "ISO2": iso2,
                    "num_activities": 0,
                    "num_matches": 0,
                    "social_added": 0,
                    "wh_added": 0,
                    "issues": ["unhandled_exception"],
                }
            finally:
                # SENSE writes only biosphere exchanges, but Brightway's broad
                # modification token changes after each region. Keep the pricing
                # cache aligned even when a notebook run is interrupted later.
                _refresh_activity_cache_token(
                    eco_activity_list_json.with_suffix(".manifest.json"),
                    technosphere_db,
                )
            results.append(summary)

    finally:
        region_progress.close()
        # Prepared profiles are transient and can contain unit-hour placeholders.
        # Only finalized selected-region inventories belong in completed evidence.
        for prepared_path in prepared_regions.values():
            prepared_path.unlink(missing_ok=True)
        prepared_directory = evidence.directory / "PreparedReference"
        if prepared_directory.is_dir():
            prepared_directory.rmdir()

    low_similarity_count = sum(
        result.get("low_similarity_activity_country_pairs", 0) for result in results
    )
    low_similarity_regions = sum(
        bool(result.get("low_similarity_activity_country_pairs")) for result in results
    )
    run_input_diagnostic_counts: Counter[str] = Counter()
    for result in results:
        run_input_diagnostic_counts.update(result.get("input_diagnostic_counts") or {})
    issue_regions = [
        {"region": result["ISO2"], "issues": result["issues"]}
        for result in results
        if result.get("issues")
    ]
    missing_isic_count = sum(int(result.get("isic_no_isic_activities", 0)) for result in results)

    pricing_path = eco_activity_list_json.with_suffix(".pricing.json")
    pricing_records = (
        read_json(pricing_path).get("activities", []) if pricing_path.is_file() else []
    )

    diagnostic_report = {
        "run_directory": str(evidence.directory),
        "project": project,
        "schema_version": 1,
        "technosphere_database": technosphere_db,
        "ecoinvent_version": ecoinvent_version,
        "psilca_version": active_psilca_version,
        "regions_processed": len(results),
        "processing_issues": issue_regions,
        "semantic_matching": {
            "minimum_similarity": config.minimum_similarity,
            "low_similarity_activity_country_pairs": low_similarity_count,
            "regions_affected": low_similarity_regions,
            "details": [
                {"region": result["ISO2"], **detail}
                for result in results
                for detail in result.get("low_similarity_details", [])
            ],
        },
        "source_data": {
            "counts": dict(sorted(run_input_diagnostic_counts.items())),
            "activities_affected": sum(
                int(result.get("activities_with_input_diagnostics", 0)) for result in results
            ),
            "details": [
                {"region": result["ISO2"], **detail}
                for result in results
                for detail in result.get("input_diagnostic_details", [])
            ],
        },
        "isic_coverage": {
            "activities_without_isic": missing_isic_count,
            "details": [
                {"region": result["ISO2"], **detail}
                for result in results
                for detail in result.get("isic_no_isic_details", [])
            ],
        },
        "country_weighting": economic_context.country_weights.audit_dict(),
        "activity_pricing": {
            "scope": "entire target database before ISO2 selection",
            "policy": config.zero_price_policy,
            "status_counts": dict(Counter(row["status"] for row in pricing_records)),
            "exclusion_reasons": dict(
                Counter(row["reason"] for row in pricing_records if row.get("reason"))
            ),
            "treatment_reference_database": (
                price_source_database if working_hours_reference else None
            ),
            "treatment_working_hours_percentile": config.treatment_working_hours_percentile,
        },
    }
    diagnostic_deviation_count = (
        len(issue_regions)
        + low_similarity_count
        + sum(run_input_diagnostic_counts.values())
        + missing_isic_count
    )

    # Country-weight coverage statistics are populated during regional
    # processing, so persist the economic audit only after the loop.
    write_economic_assumptions(
        output_dir=evidence.directory,
        ecoinvent_tag=ei_tag,
        psilca_tag=ps_tag,
        ecoinvent_version=ecoinvent_version,
        psilca_version=active_psilca_version,
        config=config,
        context=economic_context,
    )

    if pb is not None:
        write_productivity_audit(
            output_dir=evidence.directory,
            technosphere_database=technosphere_db,
            results=results,
        )

    # 6) Create impact methods
    _create_impact_methods(sociosphere_db, config)

    write_json_atomic(evidence.directory / "diagnostics.json", diagnostic_report)
    from sense.brightway.state import capture_database_state

    logger.info(
        "Impact methods ready. Verifying inventory and social data for the completed run..."
    )
    evidence.complete(
        database_state=capture_database_state(
            technosphere_db, sociosphere_db, extra_databases=(price_source_database,)
        )
    )
    # Retain the historical latest-diagnostics location for interactive users.
    # Immutable evidence and exact project pointers remain authoritative.
    write_json_atomic(run_paths.run_diagnostics, diagnostic_report)
    for result in results:
        result["run_directory"] = str(evidence.directory)
    if diagnostic_deviation_count:
        logger.warning(
            "SENSE completed with diagnostics: %d region issues, %d low-confidence "
            "activity-country matches, and %d source-data omissions. Details: %s",
            len(issue_regions),
            low_similarity_count,
            sum(run_input_diagnostic_counts.values()) + missing_isic_count,
            run_paths.run_diagnostics,
        )
    else:
        logger.info("SENSE diagnostics: no deviations detected")

    # SENSE changes only biosphere exchanges in the technosphere database. Save
    # Brightway's resulting modification token so those owned writes do not
    # invalidate the pricing cache on the next otherwise-identical run.
    _refresh_activity_cache_token(
        eco_activity_list_json.with_suffix(".manifest.json"),
        technosphere_db,
    )

    logger.info("SENSE pipeline finished. Processed %d ISO2 regions.", len(results))
    return results
