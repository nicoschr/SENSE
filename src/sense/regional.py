"""Calculate and persist one region, including productivity preflight."""

from __future__ import annotations

import logging
import math
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

from sense.brightway.flows import (
    add_social_biosphere_flows_and_exchanges,
    add_working_hours_flows_and_exchanges,
)
from sense.brightway.repository import synchronize_exchanges
from sense.config import SenseConfig
from sense.core.averaging import average_sector_mappings_w_prices
from sense.core.matching import PSILCALookupStore, add_social_exchanges, match_from_stores
from sense.core.treatment import annotate_fallbacks, fallback_overrides, finalize_profiles
from sense.core.working_hours import build_productivity_working_hour_records
from sense.diagnostics import (
    collect_pipeline_issues,
    reattach_prices_safely,
    save_intermediate_artifacts,
    summarize_and_optionally_save,
)
from sense.exceptions import SenseRegionError
from sense.productivity.geography import iso2_to_iso3
from sense.productivity.loader import (
    LOOKUP_STATUS_OVERRIDE_PARTIAL_AGG,
    OVERRIDE_APPLIED_STATUSES,
    ProductivityBaseline,
)
from sense.utils.helpers import norm_code
from sense.weighting import (
    CountryWeightProvider,
    attach_country_weights,
)

logger = logging.getLogger(__name__)


def _build_activity_metadata(
    target_iso2: str,
    entry: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], pd.DataFrame, list[dict[str, Any]]]:
    """Return source activities, matching metadata, and empty flow records."""

    activities = entry.get("Activities", []) if entry else []
    rows = [
        {
            "Activity code": str(activity.get("code") or "").strip(),
            "Activity name": str(activity.get("name") or "").strip(),
            "Reference product": str(activity.get("reference product") or "").strip(),
            "ISIC": (
                str(activity.get("ISIC rev.4 ecoinvent") or "").strip()
                if isinstance(activity.get("ISIC rev.4 ecoinvent"), str)
                else ""
            ),
            "Unit": str(activity.get("unit") or "").strip(),
            "Price": activity.get("price"),
            "ISO2": target_iso2,
        }
        for activity in activities
    ]
    empty_records = [
        {
            **row,
            "Averaged_Exchanges": [],
            "Working_hours": None,
        }
        for row in rows
        if row["Activity code"]
    ]
    return activities, pd.DataFrame(rows), empty_records


def _build_productivity_overrides(
    *,
    activities: list[dict[str, Any]],
    target_iso2: str,
    config: SenseConfig,
    baseline: ProductivityBaseline | None,
) -> tuple[dict[str, float] | None, list[dict[str, Any]]]:
    """Resolve working hours; audit rejected lookups and accepted GDP omissions."""

    if baseline is None:
        return None, []

    overrides: dict[str, float] = {}
    audit_misses: list[dict[str, Any]] = []
    for activity in activities:
        if activity.get("calculation route") == "treatment_fallback":
            continue
        code = str(activity.get("code") or "").strip()
        isic_raw = activity.get("ISIC rev.4 ecoinvent", "")
        if not code:
            continue

        if target_iso2 in {"RoW", "RoE"}:
            mappings = str(activity.get("PSILCA_mapping") or "").split(";")
            iso3_codes = [
                iso3
                for iso2 in mappings
                if (country_code := iso2.strip())
                if (iso3 := iso2_to_iso3(country_code)) is not None
            ]
            result = baseline.lookup_row_with_status(
                iso3_codes,
                isic_raw,
                config.productivity_year,
            )
        else:
            result = baseline.lookup_with_status(
                target_iso2,
                isic_raw,
                config.productivity_year,
            )

        accepted_gdp_omissions = (
            config.productivity_gdp_policy == "available_countries"
            and result.status == LOOKUP_STATUS_OVERRIDE_PARTIAL_AGG
            and result.coverage is not None
            and result.coverage.gdp_omissions_only
            and result.value is not None
            and math.isfinite(result.value)
            and result.value > 0
        )
        if result.value is not None and (
            result.status != LOOKUP_STATUS_OVERRIDE_PARTIAL_AGG or accepted_gdp_omissions
        ):
            overrides[code] = result.value
        if (
            result.status not in OVERRIDE_APPLIED_STATUSES
            or result.status == LOOKUP_STATUS_OVERRIDE_PARTIAL_AGG
        ):
            audit_misses.append(
                {
                    "ISO2": target_iso2,
                    "activity_code": code,
                    "activity_name": str(activity.get("name") or ""),
                    "reference_product": str(activity.get("reference product") or ""),
                    "isic_raw": str(isic_raw or ""),
                    "isic2": result.used_isic2 or "",
                    "year_requested": config.productivity_year,
                    "status": result.status,
                    "n_countries_attempted": result.n_countries_attempted,
                    "n_countries_used": result.n_countries_used,
                    "accepted_gdp_omissions": accepted_gdp_omissions,
                    "weight_method": result.coverage.weight_method if result.coverage else None,
                    "omitted_gdp_countries": (
                        list(result.coverage.missing_gdp_countries) if result.coverage else []
                    ),
                    "missing_productivity_countries": (
                        list(result.coverage.missing_productivity_countries)
                        if result.coverage
                        else []
                    ),
                }
            )

    if overrides:
        logger.debug(
            "ISO2 %s: productivity override for %d / %d activities",
            target_iso2,
            len(overrides),
            len(activities),
        )
    return overrides, audit_misses


def _preflight_productivity_coverage(
    *,
    regions: list[str],
    iso2_data: dict[str, dict],
    config: SenseConfig,
    baseline: ProductivityBaseline,
) -> dict[str, tuple[dict[str, float], list[dict[str, Any]]]]:
    """Resolve all productivity values before any Brightway mutation.

    Returns:
        Region-keyed productivity values and audit rows.

    Raises:
        ValueError: If any selected activity lacks a productivity value.
    """

    resolved: dict[str, tuple[dict[str, float], list[dict[str, Any]]]] = {}
    misses: list[dict[str, Any]] = []
    for region in regions:
        entry = iso2_data.get(region)
        activities = entry.get("Activities", []) if entry else []
        overrides, audit_rows = _build_productivity_overrides(
            activities=activities,
            target_iso2=region,
            config=config,
            baseline=baseline,
        )
        resolved[region] = (overrides or {}, audit_rows)
        misses.extend(row for row in audit_rows if not row.get("accepted_gdp_omissions", False))

    if misses:
        status_counts = Counter(row["status"] for row in misses)
        examples = "; ".join(
            f"{row['ISO2']}:{row['activity_code']} ({row['status']})" for row in misses[:8]
        )
        suffix = "; ..." if len(misses) > 8 else ""
        raise ValueError(
            "Working hours require complete productivity coverage of retained countries. "
            f"Unresolved activities: {len(misses)}; statuses={dict(status_counts)}; "
            f"examples: {examples}{suffix}. Add baseline data, correct ISIC/location "
            "metadata, or declare an explicit productivity_location_proxies entry."
        )
    return resolved


def _run_productivity_for_iso2(
    *,
    target_iso2: str,
    iso2_data: dict[str, dict],
    technosphere_db: str,
    sociosphere_db: str,
    cfg: SenseConfig,
    techno_lookup: dict[str, Any],
    sociosphere_lookup: dict[tuple, Any],
    ecoinvent_version_label: str,
    output_dir: Path,
    productivity_values: dict[str, float],
    productivity_audit: list[dict[str, Any]],
    productivity_price_factor: float,
    progress_cb: Callable[[int, int, str], None] | None = None,
    working_hours_reference: dict | None = None,
    prepare_only: bool = False,
    prepared: dict | None = None,
) -> dict[str, Any]:
    """Calculate and write one region without PSILCA matching or exchanges."""

    if prepared is not None:
        records = finalize_profiles(
            prepared["records"], working_hours_reference or {}, cfg.rounding_digits
        )
        return _write_prepared_region(
            prepared["summary"],
            records,
            technosphere_db,
            sociosphere_db,
            cfg,
            techno_lookup,
            sociosphere_lookup,
            output_dir,
            ecoinvent_version_label,
        )

    entry = iso2_data.get(target_iso2)
    activities, metadata, empty_records = _build_activity_metadata(target_iso2, entry)
    if progress_cb:
        progress_cb(1, 2, "calculating productivity-based working hours")

    records = (
        build_productivity_working_hour_records(
            metadata.to_dict(orient="records"),
            productivity_values,
            productivity_price_factor,
            direct_hours_override=fallback_overrides(
                activities, working_hours_reference or {}, profiles_only=prepare_only
            ),
        )
        if cfg.enable_working_hours_flows
        else empty_records
    )

    if not prepare_only:
        annotate_fallbacks(records, working_hours_reference or {})
    working_results = {"added": 0, "updated": 0, "removed": 0}
    if cfg.enable_working_hours_flows and not prepare_only:
        if progress_cb:
            progress_cb(2, 2, "updating working-hours flows in Brightway")
        working_results = add_working_hours_flows_and_exchanges(
            data=records,
            database=technosphere_db,
            biosphere_db_name=sociosphere_db,
            biosphere_lookup=sociosphere_lookup,
            techno_lookup=techno_lookup,
            precreated_flows=cfg.precreated_flows,
            update_tol=cfg.update_tol,
            add_isic_flows=cfg.enable_isic_working_hours_methods,
        )

    issues = summarize_and_optionally_save(
        target_iso2=target_iso2,
        enriched_df=metadata,
        averaged_exchanges=records,
        output_dir=str(output_dir),
        ecoinvent_version=ecoinvent_version_label,
        save_switch=cfg.save_intermediate,
        expect_social_exchanges=False,
        save_averaged=not prepare_only,
    )
    summary = {
        "ISO2": target_iso2,
        "num_activities": len(activities),
        "num_matches": 0,
        "price_updates": 0,
        "social_added": 0,
        "social_updated": 0,
        "social_removed": 0,
        "wh_added": working_results.get("added", 0),
        "wh_updated": working_results.get("updated", 0),
        "wh_removed": working_results.get("removed", 0),
        "isic_no_isic_activities": working_results.get("isic_no_isic_activities", 0),
        "isic_no_isic_details": working_results.get("isic_no_isic_details", []),
        "input_diagnostic_counts": {},
        "activities_with_input_diagnostics": 0,
        "input_diagnostic_details": [],
        "issues": issues,
        "productivity_override_applied": len(productivity_values),
        "productivity_audit_misses": [
            row for row in productivity_audit if not row.get("accepted_gdp_omissions", False)
        ],
        "productivity_audit_gdp_omissions": [
            row for row in productivity_audit if row.get("accepted_gdp_omissions", False)
        ],
    }
    return {"summary": summary, "records": records} if prepare_only else summary


def _clear_managed_exchanges(
    *,
    technosphere_db: str,
    sociosphere_db: str,
    techno_lookup: dict[str, Any],
    sociosphere_lookup: dict[tuple, Any],
    kinds: set[str],
) -> int:
    """Remove selected owned exchange families without deleting shared methods."""
    owned = {
        tuple(flow.key)
        for flow in sociosphere_lookup.values()
        if flow.key[0] == sociosphere_db and flow.get("sense_kind") in kinds
    }
    if not owned or not techno_lookup:
        return 0
    _, _, removed = synchronize_exchanges(
        database_name=technosphere_db,
        grouped={code: [] for code in techno_lookup},
        techno_lookup=techno_lookup,
        resolve_flow=lambda _: (None, None),
        managed_flow=lambda flow: tuple(flow.key) in owned,
        managed_flow_keys=owned,
        update_tol=0.0,
    )
    return removed


def _run_social_for_iso2(
    target_iso2: str,
    iso2_data: dict[str, dict],
    technosphere_db: str,
    sociosphere_db: str,
    cfg: SenseConfig,
    techno_lookup: dict[str, Any],
    sociosphere_lookup: dict[tuple, Any],
    ecoinvent_version_label: str,
    psilca_store: PSILCALookupStore,
    output_dir: Path,
    ecoinvent_emb_dir: Path,
    psilca_emb_dir: Path,
    progress_cb: Callable[[int, int, str], None] | None = None,
    productivity_baseline: ProductivityBaseline | None = None,
    country_weight_provider: CountryWeightProvider | None = None,
    psilca_price_factor: float = 1.0,
    productivity_price_factor: float = 1.0,
    working_hours_reference: dict | None = None,
    prepare_only: bool = False,
    prepared: dict | None = None,
) -> dict[str, Any]:
    """Match, enrich, average and write social flows for one ISO2 region."""
    if prepared is not None:
        records = finalize_profiles(
            prepared["records"], working_hours_reference or {}, cfg.rounding_digits
        )
        return _write_prepared_region(
            prepared["summary"],
            records,
            technosphere_db,
            sociosphere_db,
            cfg,
            techno_lookup,
            sociosphere_lookup,
            output_dir,
            ecoinvent_version_label,
        )
    TOTAL_STEPS = 7
    step = 1

    # Build per-activity metadata table
    if progress_cb:
        progress_cb(step, TOTAL_STEPS, "building metadata")
    entry = iso2_data.get(target_iso2)
    if entry is None:
        raise ValueError(f"Activity cache has no entry for region {target_iso2!r}")
    acts, meta_df, empty_records = _build_activity_metadata(target_iso2, entry)

    # Semantic matching
    step += 1
    if progress_cb:
        progress_cb(step, TOTAL_STEPS, "semantic matching")
    sector_mapping_df = match_from_stores(
        ecoinvent_store_dir=str(ecoinvent_emb_dir),
        psilca_store_dir=str(psilca_emb_dir),
        target_iso2=target_iso2,
        threshold=cfg.similarity_delta,
        minimum_similarity=cfg.minimum_similarity,
        activity_entry=entry,
    )

    low_similarity_details: list[dict[str, Any]] = []
    low_similarity_column = "Below minimum similarity"
    if low_similarity_column in sector_mapping_df:
        low_similarity_rows = sector_mapping_df[sector_mapping_df[low_similarity_column]]
        if not low_similarity_rows.empty:
            best_row_indices = low_similarity_rows.groupby(
                ["Activity code", "PSILCA_country"], sort=False
            )["Semantic_score"].idxmax()
            detail_columns = [
                "Activity code",
                "Activity name",
                "Reference product",
                "PSILCA_country",
                "Matched_sector",
                "Semantic_score",
            ]
            low_similarity_details = low_similarity_rows.loc[
                best_row_indices, detail_columns
            ].to_dict(orient="records")

    if sector_mapping_df.empty:
        # A failed match is not an authoritative empty inventory. Preserve the
        # previous region until its replacement has been successfully calculated.
        raise SenseRegionError(target_iso2, "no matched PSILCA sectors found")

    # Add social exchanges
    step += 1
    if progress_cb:
        progress_cb(step, TOTAL_STEPS, "attaching social exchanges")
    enriched_df = add_social_exchanges(
        sector_mapping_df=sector_mapping_df,
        psilca_lookup=psilca_store.for_countries(
            sorted(sector_mapping_df["PSILCA_country"].dropna().astype(str).unique())
        ),
    )

    # Reattach prices
    step += 1
    if progress_cb:
        progress_cb(step, TOTAL_STEPS, "reattaching prices")
    enriched_df, price_updates = reattach_prices_safely(enriched_df, meta_df)

    if country_weight_provider is not None:
        enriched_df = attach_country_weights(enriched_df, country_weight_provider)

    hours_override, audit_misses = _build_productivity_overrides(
        activities=acts,
        target_iso2=target_iso2,
        config=cfg,
        baseline=productivity_baseline,
    )

    # Averaging
    step += 1
    if progress_cb:
        progress_cb(step, TOTAL_STEPS, "averaging indicators")
    averaged_exchanges = average_sector_mappings_w_prices(
        enriched_df=enriched_df,
        level_weights=cfg.level_weights,
        rounding=None if prepare_only else cfg.rounding_digits,
        mean_method=cfg.social_flow_mean,
        hours_override=hours_override,
        psilca_price_factor=psilca_price_factor,
        productivity_price_factor=productivity_price_factor,
        direct_hours_override=fallback_overrides(
            acts, working_hours_reference or {}, profiles_only=prepare_only
        ),
    )
    if not prepare_only:
        annotate_fallbacks(averaged_exchanges, working_hours_reference or {})

    issues = collect_pipeline_issues(
        target_iso2,
        enriched_df,
        averaged_exchanges,
        expect_social_exchanges=bool(cfg.enable_social_flows),
    )
    if cfg.enable_working_hours_flows:
        for record in averaged_exchanges:
            hours = record.get("Working_hours")
            if hours is None or not math.isfinite(float(hours)) or float(hours) <= 0:
                issues.append(
                    f"Activity {record.get('Activity code')!r} has no valid working hours"
                )
    if issues:
        raise SenseRegionError(target_iso2, "; ".join(issues))

    matched_activity_codes = {
        norm_code(activity.get("Activity code")) for activity in averaged_exchanges
    }
    unmatched_records = [
        record
        for record in empty_records
        if norm_code(record["Activity code"]) not in matched_activity_codes
    ]
    if unmatched_records:
        raise SenseRegionError(
            target_iso2,
            f"{len(unmatched_records)} retained activities have no matched social inventory",
        )

    social_results = {"added": 0, "updated": 0, "removed": 0}
    wh_results = {"added": 0, "updated": 0, "removed": 0}

    if not prepare_only:
        # BW integration: social flows
        step += 1
        if cfg.enable_social_flows:
            if progress_cb:
                progress_cb(step, TOTAL_STEPS, "updating social flows in Brightway")
            social_results = add_social_biosphere_flows_and_exchanges(
                data=averaged_exchanges,
                database=technosphere_db,
                biosphere_db_name=sociosphere_db,
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
                biosphere_db_name=sociosphere_db,
                biosphere_lookup=sociosphere_lookup,
                techno_lookup=techno_lookup,
                precreated_flows=cfg.precreated_flows,
                update_tol=cfg.update_tol,
                add_isic_flows=cfg.enable_isic_working_hours_methods,
            )

    # Diagnostics + optional saving
    issues = summarize_and_optionally_save(
        target_iso2=target_iso2,
        enriched_df=enriched_df,
        averaged_exchanges=averaged_exchanges,
        output_dir=str(output_dir),
        ecoinvent_version=ecoinvent_version_label,
        save_switch=cfg.save_intermediate,
        save_averaged=not prepare_only,
    )
    if unmatched_records:
        issues.append(f"{len(unmatched_records)} activities could not be semantically matched")

    input_diagnostic_counts: Counter[str] = Counter()
    input_diagnostic_details: list[dict[str, Any]] = []
    for activity in averaged_exchanges:
        activity_diagnostics = activity.get("Input_diagnostics") or {}
        if not activity_diagnostics:
            continue
        input_diagnostic_counts.update(activity_diagnostics)
        input_diagnostic_details.append(
            {
                "activity_code": activity.get("Activity code"),
                "activity_name": activity.get("Activity name"),
                "causes": dict(sorted(activity_diagnostics.items())),
            }
        )

    summary = {
        "ISO2": target_iso2,
        "num_activities": len(acts),
        "num_matches": int(sector_mapping_df["Activity code"].nunique()),
        "price_updates": int(price_updates),
        "social_added": social_results.get("added", 0),
        "social_updated": social_results.get("updated", 0),
        "social_removed": social_results.get("removed", 0),
        "wh_added": wh_results.get("added", 0),
        "wh_updated": wh_results.get("updated", 0),
        "wh_removed": wh_results.get("removed", 0),
        "isic_no_isic_activities": wh_results.get("isic_no_isic_activities", 0),
        "isic_no_isic_details": wh_results.get("isic_no_isic_details", []),
        "low_similarity_activity_country_pairs": len(low_similarity_details),
        "low_similarity_activities": len(
            {detail["Activity code"] for detail in low_similarity_details}
        ),
        "low_similarity_details": low_similarity_details,
        "input_diagnostic_counts": dict(sorted(input_diagnostic_counts.items())),
        "activities_with_input_diagnostics": len(input_diagnostic_details),
        "input_diagnostic_details": input_diagnostic_details,
        "issues": issues,
        "productivity_override_applied": len(hours_override) if hours_override is not None else 0,
        "productivity_audit_misses": [
            row for row in audit_misses if not row.get("accepted_gdp_omissions", False)
        ],
        "productivity_audit_gdp_omissions": [
            row for row in audit_misses if row.get("accepted_gdp_omissions", False)
        ],
    }
    if prepare_only:
        return {"summary": summary, "records": averaged_exchanges}
    return summary


def _write_prepared_region(
    summary,
    records,
    database,
    sociosphere,
    cfg,
    techno_lookup,
    sociosphere_lookup,
    output_dir,
    version,
) -> dict[str, Any]:
    """Write validated profiles after the reference population is complete."""
    social = {"added": 0, "updated": 0, "removed": 0}
    hours = dict(social)
    common = {
        "database": database,
        "biosphere_db_name": sociosphere,
        "biosphere_lookup": sociosphere_lookup,
        "techno_lookup": techno_lookup,
        "precreated_flows": cfg.precreated_flows,
        "update_tol": cfg.update_tol,
    }
    if cfg.enable_social_flows:
        social = add_social_biosphere_flows_and_exchanges(records, **common)
    if cfg.enable_working_hours_flows:
        hours = add_working_hours_flows_and_exchanges(
            records, **common, add_isic_flows=cfg.enable_isic_working_hours_methods
        )
    save_intermediate_artifacts(
        summary["ISO2"],
        None,
        records,
        str(output_dir),
        version,
        cfg.save_intermediate,
        include_social_data=cfg.use_psilca,
    )
    result = dict(summary)
    for prefix, stats in (("social", social), ("wh", hours)):
        for action in ("added", "updated", "removed"):
            result[f"{prefix}_{action}"] = stats.get(action, 0)
    result["isic_no_isic_activities"] = hours.get("isic_no_isic_activities", 0)
    result["isic_no_isic_details"] = hours.get("isic_no_isic_details", [])
    return result
