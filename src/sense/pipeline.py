"""
Main SENSE pipeline orchestrator.

This module contains the primary `run_sense()` function that orchestrates
the full social-LCA integration pipeline.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
from tqdm import tqdm
from bw2data import projects, Database, databases

from sense.config import SenseConfig
from sense.utils.helpers import norm_code, slug
from sense.data.psilca import (
    combine_psilca_css_and_extract_social_exchanges,
    embed_psilca_to_store,
)
from sense.data.ecoinvent import (
    map_ecoinvent_locations,
    fetch_ecoinvent_activities_from_ecospold_files,
    embed_ecoinvent_to_store,
)
from sense.core.matching import match_from_stores, add_social_exchanges
from sense.core.averaging import average_sector_mappings_w_prices
from sense.brightway.flows import (
    add_social_biosphere_flows_and_exchanges,
    add_working_hours_flows_and_exchanges,
)
from sense.brightway.methods import (
    add_social_impact_categories,
    add_working_hours_impact_category,
)
from sense.diagnostics import reattach_prices_safely, summarize_and_optionally_save


logger = logging.getLogger(__name__)


def _get_package_data_dir() -> Path:
    """Get the data directory relative to where SENSE is run from."""
    return Path.cwd() / "Data"


def psilca_iso3_for_iso2(extract_data: List[dict], target_iso2: str) -> List[str]:
    """
    Figure out which PSILCA country codes should be considered for a given ecoinvent ISO2 region.
    
    For "normal" ISO2: uses top-level PSILCA_mapping.
    For RoW / RoE: unions all per-activity PSILCA_mappings.
    
    Args:
        extract_data: List of region entries with Activities
        target_iso2: Target ISO2 code
        
    Returns:
        List of PSILCA ISO3 country codes
    """
    entry = next((e for e in extract_data if e.get("ISO2") == target_iso2), None)
    if not entry:
        return []

    top = entry.get("PSILCA_mapping")
    if isinstance(top, str) and top.strip():
        return [c.strip() for c in top.split(";") if c.strip()]

    iso = set()
    for act in entry.get("Activities", []):
        m = act.get("PSILCA_mapping")
        if isinstance(m, str) and m.strip():
            iso.update(c.strip() for c in m.split(";") if c.strip())
    return sorted(iso)


def _prepare_psilca_data(
    psilca_dir: Path,
    cfg: SenseConfig,
    combined_path: Path,
    css_list_path: Path,
    zero_report_path: Path,
) -> None:
    """Create combined PSILCA JSON + CSS list JSON if missing."""
    combined_path.parent.mkdir(parents=True, exist_ok=True)

    if combined_path.exists() and css_list_path.exists():
        logger.info("PSILCA: sectors and exchanges already exist in %s", combined_path.parent)
        return

    logger.info("Step 1/5: Extract PSILCA sectors and social exchanges...")
    combine_psilca_css_and_extract_social_exchanges(
        input_folder=str(psilca_dir),
        combined_output_path=str(combined_path),
        css_list_output_path=str(css_list_path),
        zero_report_path=str(zero_report_path),
    )
    logger.info(
        "PSILCA: wrote combined file '%s' and CSS list '%s'",
        combined_path.name,
        css_list_path.name,
    )


def _prepare_location_mapping(
    cfg: SenseConfig,
    css_list_path: Path,
    location_mapping_xlsx: Path,
    ecoinvent_geographies_excel: Path,
    output_dir: Path,
) -> pd.DataFrame:
    """Map ecoinvent geographies to PSILCA countries."""
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Step 2/5: Mapping ecoinvent locations to PSILCA countries...")
    loc_df = map_ecoinvent_locations(
        ecoinvent_excel_path=str(ecoinvent_geographies_excel),
        psilca_sector_json_path=str(css_list_path),
    )

    if cfg.save_intermediate:
        location_mapping_xlsx.parent.mkdir(parents=True, exist_ok=True)
        loc_df.to_excel(location_mapping_xlsx, index=False)
        logger.info("Saved location mapping to: %s", location_mapping_xlsx)

    return loc_df


def _prepare_activity_list(
    ecoinvent_dir: Path,
    technosphere_db: str,
    background_databases: List[str],
    foreground_active: bool,
    foreground_database: Optional[str],
    loc_df: pd.DataFrame,
    cfg: SenseConfig,
    activity_list_path: Path,
    missing_price_report: Path,
) -> List[dict]:
    """Build the JSON list of activities with prices + PSILCA mappings."""
    activity_list_path.parent.mkdir(parents=True, exist_ok=True)

    if activity_list_path.exists():
        logger.info("Step 3/5: Ecoinvent activities already exist in: %s", activity_list_path)
        with open(activity_list_path, encoding="utf-8") as f:
            return json.load(f)

    if foreground_active:
        if not foreground_database:
            raise ValueError("foreground_active=True but no foreground_database provided.")
        techno_name = foreground_database
    else:
        techno_name = technosphere_db

    existing_dbs = set(databases)
    bg_dbs = [db for db in background_databases if db in existing_dbs]

    if not bg_dbs:
        logger.warning(
            "No background_databases found in this project; using '%s' only for pricing.",
            techno_name,
        )
        bg_dbs = [techno_name]

    logger.info("Step 3/5: Extract ecoinvent activities and product prices...")
    fetch_ecoinvent_activities_from_ecospold_files(
        folder_path=str(ecoinvent_dir),
        mapping_df=loc_df,
        output_path=str(activity_list_path),
        technosphere=techno_name,
        background_databases=bg_dbs,
        foreground=foreground_active,
        missing_price_report_path=str(missing_price_report),
        exclude_terms=cfg.exclude_terms,
        exclude_terms_starting_with=cfg.exclude_terms_starting_with,
        price_floor_percentile=cfg.price_floor_percentile,
    )

    with open(activity_list_path, encoding="utf-8") as f:
        return json.load(f)


def _prepare_embedding_stores(
    extract_country_iso2s_data: List[dict],
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
        batch_size=cfg.embedding_batch_size,
        per_country=True,
    )

    logger.info("Step 5/5: Creating/using ecoinvent activity embeddings (per ISO2)...")
    target_iso2_list = [entry["ISO2"] for entry in extract_country_iso2s_data]
    embed_ecoinvent_to_store(
        technosphere_json=str(activity_list_path),
        out_dir=str(ecoinvent_emb_dir),
        model_name=cfg.embedding_model,
        batch_size=cfg.embedding_batch_size,
        target_iso2_list=target_iso2_list,
    )


def _run_social_for_iso2(
    target_iso2: str,
    extract_country_iso2s_data: List[dict],
    technosphere_db: str,
    biosphere_db: str,
    sociosphere_db: str,
    foreground_active: bool,
    foreground_database: Optional[str],
    cfg: SenseConfig,
    techno_lookup: Dict[str, Any],
    biosphere_lookup: Dict[tuple, Any],
    sociosphere_lookup: Dict[tuple, Any],
    ecoinvent_version_label: str,
    activity_list_path: Path,
    psilca_combined_path: Path,
    output_dir: Path,
    ecoinvent_emb_dir: Path,
    psilca_emb_dir: Path,
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
) -> Dict[str, Any]:
    """Match, enrich, average and write social flows for one ISO2 region."""
    TOTAL_STEPS = 7
    step = 1

    # Build per-activity metadata table
    if progress_cb:
        progress_cb(step, TOTAL_STEPS, "building metadata")
    entry = next((e for e in extract_country_iso2s_data if e.get("ISO2") == target_iso2), None)
    acts = entry.get("Activities", []) if entry else []

    meta_rows = []
    for a in acts:
        meta_rows.append({
            "Activity code": str(a.get("code") or "").strip(),
            "Activity name": str(a.get("name") or "").strip(),
            "Reference product": str(a.get("reference product") or "").strip(),
            "ISIC": str(a.get("ISIC rev.4 ecoinvent") or "").strip()
            if isinstance(a.get("ISIC rev.4 ecoinvent"), str) else "",
            "Unit": str(a.get("unit") or "").strip(),
            "Price": a.get("price", None),
            "ISO2": target_iso2,
        })
    meta_df = pd.DataFrame(meta_rows)

    # Semantic matching
    step += 1
    if progress_cb:
        progress_cb(step, TOTAL_STEPS, "semantic matching")
    iso3_list = psilca_iso3_for_iso2(extract_country_iso2s_data, target_iso2)

    sector_mapping_df = match_from_stores(
        ecoinvent_store_dir=str(ecoinvent_emb_dir),
        psilca_store_dir=str(psilca_emb_dir),
        target_iso2=target_iso2,
        psilca_iso3_list=iso3_list,
        threshold=cfg.similarity_delta,
        ecoinvent_list_of_activities_path=str(activity_list_path),
    )

    if sector_mapping_df.empty:
        logger.warning("ISO2 %s: no matched sectors found; skipping social pipeline.", target_iso2)
        return {
            "ISO2": target_iso2,
            "num_activities": len(acts),
            "num_matches": 0,
            "social_added": 0,
            "wh_added": 0,
            "issues": ["no matched sectors found"],
        }

    # Add social exchanges
    step += 1
    if progress_cb:
        progress_cb(step, TOTAL_STEPS, "attaching social exchanges")
    enriched_df = add_social_exchanges(
        sector_mapping_df=sector_mapping_df,
        combined_json_path=str(psilca_combined_path),
    )

    # Reattach prices
    step += 1
    if progress_cb:
        progress_cb(step, TOTAL_STEPS, "reattaching prices")
    enriched_df, price_updates = reattach_prices_safely(enriched_df, meta_df)

    # Averaging
    step += 1
    if progress_cb:
        progress_cb(step, TOTAL_STEPS, "averaging indicators")
    averaged_exchanges = average_sector_mappings_w_prices(
        enriched_df=enriched_df,
        level_weights=cfg.level_weights,
        tiny_threshold=cfg.tiny_threshold,
        rounding=cfg.rounding_digits,
        mean_method=cfg.social_flow_mean,
    )

    social_results = {"added": 0, "updated": 0}
    wh_results = {"added": 0, "updated": 0}

    # BW integration: social flows
    step += 1
    if cfg.enable_social_flows:
        if progress_cb:
            progress_cb(step, TOTAL_STEPS, "updating social flows in Brightway")
        social_results = add_social_biosphere_flows_and_exchanges(
            data=averaged_exchanges,
            database=technosphere_db,
            foreground_database=foreground_database,
            biosphere_db_name=sociosphere_db,
            foreground_active=foreground_active,
            biosphere_lookup=sociosphere_lookup,
            techno_lookup=techno_lookup,
            precreated_flows=cfg.precreated_flows,
            update_tol=cfg.update_tol,
        )

    # BW integration: working-hours flows
    step += 1
    if cfg.enable_working_hours_flows:
        if progress_cb:
            progress_cb(step, TOTAL_STEPS, "updating working-hours flows & diagnostics")
        wh_results = add_working_hours_flows_and_exchanges(
            data=averaged_exchanges,
            database=technosphere_db,
            foreground_database=foreground_database,
            biosphere_db_name=sociosphere_db,
            foreground_active=foreground_active,
            biosphere_lookup=sociosphere_lookup,
            techno_lookup=techno_lookup,
            precreated_flows=cfg.precreated_flows,
            update_tol=cfg.update_tol,
        )

    # Diagnostics + optional saving
    issues = summarize_and_optionally_save(
        target_iso2=target_iso2,
        enriched_df=enriched_df,
        averaged_exchanges=averaged_exchanges,
        output_dir=str(output_dir),
        ecoinvent_version=ecoinvent_version_label,
        save_switch=cfg.save_intermediate,
    )

    if issues:
        logger.warning("ISO2 %s: diagnostics reported %d issues", target_iso2, len(issues))

    return {
        "ISO2": target_iso2,
        "num_activities": len(acts),
        "num_matches": int(sector_mapping_df["Activity code"].nunique()),
        "price_updates": int(price_updates),
        "social_added": social_results.get("added", 0),
        "social_updated": social_results.get("updated", 0),
        "wh_added": wh_results.get("added", 0),
        "wh_updated": wh_results.get("updated", 0),
        "issues": issues,
    }


def run_sense(
    *,
    project: str,
    technosphere_db: str,
    biosphere_db: str,
    sociosphere_db: str = "sociosphere",
    foreground_active: bool,
    foreground_database: Optional[str],
    background_databases: List[str],
    ecoinvent_dir: Path,
    psilca_dir: Path,
    ecoinvent_version: str,
    psilca_version: str,
    config: SenseConfig,
    data_dir: Optional[Path] = None,
    ecoinvent_geographies_excel: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """
    Run the full SENSE pipeline.
    
    This function orchestrates the complete social-LCA integration process:
    1. Parse PSILCA data and extract social exchanges
    2. Map ecoinvent locations to PSILCA countries
    3. Extract ecoinvent activities with prices
    4. Create embeddings for matching
    5. Match ecoinvent activities to PSILCA sectors
    6. Average social indicators across matches
    7. Write social flows to Brightway databases
    8. Create impact categories
    
    Args:
        project: Brightway project name
        technosphere_db: Name of the main technosphere DB (e.g., "ecoinvent-3.11-cutoff")
        biosphere_db: Name of the biosphere DB (usually "biosphere3")
        sociosphere_db: Name of the sociosphere DB for social flows (default "sociosphere")
        foreground_active: If True, attach flows to `foreground_database` instead
        foreground_database: Name of the foreground DB (if used)
        background_databases: Other BW databases used for price estimation
        ecoinvent_dir: Path to folder with EcoSpold02 .spold files
        psilca_dir: Path to folder with PSILCA EcoSpold01 .xml files
        ecoinvent_version: Version label for ecoinvent (e.g., "3.11")
        psilca_version: Version label for PSILCA (e.g., "4.0")
        config: SenseConfig instance with tuning options
        data_dir: Optional base data directory (defaults to ./Data)
        ecoinvent_geographies_excel: Optional path to ecoinvent Database Overview Excel file
            with Geographies sheet (defaults to Database-Overview-for-ecoinvent-v3.10_29.04.24.xlsx
            in data_dir)
        
    Returns:
        A list of per-ISO2 summary dicts with processing results
        
    Examples:
        >>> from sense import run_sense, SenseConfig
        >>> results = run_sense(
        ...     project="my_project",
        ...     technosphere_db="ecoinvent-3.11-cutoff",
        ...     biosphere_db="biosphere3",
        ...     foreground_active=False,
        ...     foreground_database=None,
        ...     background_databases=["ecoinvent-3.11-cutoff"],
        ...     ecoinvent_dir=Path("Data/ecoinvent 3.11_cutoff_ecoSpold02/datasets"),
        ...     psilca_dir=Path("Data/PSILCA/EcoSpold01"),
        ...     ecoinvent_version="3.11",
        ...     psilca_version="4.0",
        ...     config=SenseConfig(),
        ...     ecoinvent_geographies_excel=Path("Data/Database-Overview-for-ecoinvent-v3.10.xlsx"),
        ... )
    """
    logger.info("Starting SENSE pipeline in BW project '%s'", project)
    projects.set_current(project)

    # Set up paths
    if data_dir is None:
        data_dir = _get_package_data_dir()
    
    output_dir = data_dir / "Output"
    psilca_out_dir = output_dir / "PSILCA"
    ecoinvent_out_dir = output_dir / "Ecoinvent"
    emb_base_dir = output_dir / "EmbeddingStore"
    
    # Look for ecoinvent geographies Excel
    if ecoinvent_geographies_excel is None:
        ecoinvent_geographies_excel = data_dir / "Database-Overview-for-ecoinvent-v3.10_29.04.24.xlsx"

    # Build tags & per-run paths
    if foreground_active and foreground_database:
        base_label = foreground_database
    else:
        base_label = technosphere_db

    safe_techno = slug(base_label)
    ei_tag = f"{safe_techno}_v{ecoinvent_version}"
    ps_tag = f"v{psilca_version}"

    # PSILCA JSONs
    psilca_combined_json = psilca_out_dir / f"psilca_css_combined_{ps_tag}.json"
    psilca_css_list_json = psilca_out_dir / f"psilca_css_list_{ps_tag}.json"
    psilca_zero_report = psilca_out_dir / f"psilca_zero_or_negative_exchanges_{ps_tag}.json"

    # Ecoinvent files
    eco_activity_list_json = ecoinvent_out_dir / f"ecoinvent_activities_with_prices_{ei_tag}.json"
    missing_price_report = ecoinvent_out_dir / f"missing_prices_{ei_tag}.csv"
    location_mapping_xlsx = ecoinvent_out_dir / f"ecoinvent_location_mapping_v{ecoinvent_version}_{ps_tag}.xlsx"

    # Embedding stores
    emb_eco_dir = emb_base_dir / "ecoinvent" / ei_tag
    emb_psilca_dir = emb_base_dir / "psilca" / ps_tag

    # 1) PSILCA data + location mapping + activity list
    _prepare_psilca_data(
        psilca_dir=psilca_dir,
        cfg=config,
        combined_path=psilca_combined_json,
        css_list_path=psilca_css_list_json,
        zero_report_path=psilca_zero_report,
    )

    loc_df = _prepare_location_mapping(
        cfg=config,
        css_list_path=psilca_css_list_json,
        location_mapping_xlsx=location_mapping_xlsx,
        ecoinvent_geographies_excel=ecoinvent_geographies_excel,
        output_dir=ecoinvent_out_dir,
    )

    extract_country_iso2s_data = _prepare_activity_list(
        ecoinvent_dir=ecoinvent_dir,
        technosphere_db=technosphere_db,
        background_databases=background_databases,
        foreground_active=foreground_active,
        foreground_database=foreground_database,
        loc_df=loc_df,
        cfg=config,
        activity_list_path=eco_activity_list_json,
        missing_price_report=missing_price_report,
    )

    # 2) Embedding stores
    _prepare_embedding_stores(
        extract_country_iso2s_data=extract_country_iso2s_data,
        cfg=config,
        css_list_path=psilca_css_list_json,
        activity_list_path=eco_activity_list_json,
        psilca_emb_dir=emb_psilca_dir,
        ecoinvent_emb_dir=emb_eco_dir,
    )

    # 3) Build BW lookups
    techno_db_name = foreground_database if foreground_active else technosphere_db
    techno_db = Database(techno_db_name)
    techno_lookup = {}

    for act in techno_db:
        c = norm_code(act.get("code"))
        if c:
            techno_lookup[c] = act

    biosphere_db_obj = Database(biosphere_db)
    biosphere_lookup = {
        (f.get("name"), f.get("unit"), f.get("categories")): f for f in biosphere_db_obj
    }

    # Create sociosphere database if needed
    if sociosphere_db not in databases:
        logger.info("Creating new sociosphere database: %s", sociosphere_db)
        Database(sociosphere_db).write({})
        logger.info("✅ Sociosphere database created")

    sociosphere_db_obj = Database(sociosphere_db)
    sociosphere_lookup = {
        (f.get("name"), f.get("unit"), f.get("categories")): f for f in sociosphere_db_obj
    }

    # 4) Decide which ISO2s to run
    iso2_list = [entry["ISO2"] for entry in extract_country_iso2s_data]

    seen = set()
    iso2_ordered: List[str] = []
    for code in iso2_list:
        if code not in seen:
            seen.add(code)
            iso2_ordered.append(code)

    if config.iso2_selection:
        allowed = set(config.iso2_selection)
        iso2_ordered = [c for c in iso2_ordered if c in allowed]

    if config.iso2_start_at:
        if config.iso2_start_at in iso2_ordered:
            start_idx = iso2_ordered.index(config.iso2_start_at)
            iso2_ordered = iso2_ordered[start_idx:]
            logger.info(
                "Starting from iso2_start_at='%s'; %d ISO2s will be processed",
                config.iso2_start_at,
                len(iso2_ordered),
            )
        else:
            logger.warning(
                "iso2_start_at='%s' not found in ISO2 list; nothing will be processed.",
                config.iso2_start_at,
            )
            iso2_ordered = []

    # 5) Main loop over ISO2s
    results: List[Dict[str, Any]] = []
    total_iso2 = len(iso2_ordered)
    logger.info("Beginning ISO2 loop over %d regions...", total_iso2)

    pbar = tqdm(iso2_ordered, desc="🌍 SENSE social pipeline", unit="ISO2")

    for idx, iso2 in enumerate(pbar, start=1):
        def progress_cb(step: int, total_steps: int, message: str, iso2=iso2, idx=idx):
            pbar.set_description(
                f"[{idx}/{total_iso2}] {iso2} ({step}/{total_steps}) {message}"
            )

        summary = _run_social_for_iso2(
            target_iso2=iso2,
            extract_country_iso2s_data=extract_country_iso2s_data,
            technosphere_db=technosphere_db,
            biosphere_db=biosphere_db,
            sociosphere_db=sociosphere_db,
            foreground_active=foreground_active,
            foreground_database=foreground_database,
            cfg=config,
            techno_lookup=techno_lookup,
            biosphere_lookup=biosphere_lookup,
            sociosphere_lookup=sociosphere_lookup,
            ecoinvent_version_label=ecoinvent_version,
            activity_list_path=eco_activity_list_json,
            psilca_combined_path=psilca_combined_json,
            output_dir=output_dir,
            ecoinvent_emb_dir=emb_eco_dir,
            psilca_emb_dir=emb_psilca_dir,
            progress_cb=progress_cb,
        )
        results.append(summary)

        if summary.get("issues"):
            logger.warning(
                "ISO2 %s: issues encountered: %s",
                iso2,
                " | ".join(summary["issues"]),
            )

    # 6) Create impact methods
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

    logger.info("SENSE pipeline finished. Processed %d ISO2 regions.", len(results))
    return results
