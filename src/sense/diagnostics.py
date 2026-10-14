"""Collect pipeline diagnostics and write optional intermediate artifacts."""

from __future__ import annotations

import gzip
import os
from pathlib import Path
from typing import Any

import pandas as pd

from sense.artifacts import atomic_output_path, write_json_atomic
from sense.utils.helpers import norm_code

__all__ = [
    "reattach_prices_safely",
    "collect_pipeline_issues",
    "save_intermediate_artifacts",
    "summarize_and_optionally_save",
]


def reattach_prices_safely(
    enriched_df: pd.DataFrame,
    meta_df: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    """Fill absent match prices from activity metadata, preserving explicit zeros.

    Fills NaN prices from the activity list metadata
    using normalized 'Activity code' as the join key.

    Args:
        enriched_df: Matched activities that may have missing prices.
        meta_df: Authoritative activity metadata and prices.

    Returns:
        Updated rows and the number of replaced prices.
    """
    if "Activity code" not in enriched_df or "Price" not in enriched_df:
        return enriched_df, 0

    df = enriched_df.copy()
    df["Activity code"] = df["Activity code"].apply(norm_code)

    tmp = meta_df.copy()
    if "Activity code" in tmp:
        tmp["Activity code"] = tmp["Activity code"].apply(norm_code)
        fetched_map = tmp.set_index("Activity code")["Price"]
    else:
        fetched_map = pd.Series(dtype=float)

    price_cur = pd.to_numeric(df["Price"], errors="coerce")
    price_new = pd.to_numeric(df["Activity code"].map(fetched_map), errors="coerce")

    need_fill = price_cur.isna()
    have_new = price_new.notna()
    to_update = need_fill & have_new

    updated_count = int(to_update.sum())
    if updated_count:
        df.loc[to_update, "Price"] = price_new[to_update]

    return df, updated_count


def collect_pipeline_issues(
    target_iso2: str,
    enriched_df: pd.DataFrame,
    averaged_exchanges: list[dict[str, Any]],
    *,
    expect_social_exchanges: bool = True,
) -> list[str]:
    """Check the post-enrichment schema for one region.

    Args:
        target_iso2: Region being processed.
        enriched_df: Matched and enriched activities.
        averaged_exchanges: Matched activity records to validate before writing.
        expect_social_exchanges: Check PSILCA-specific schema columns.

    Returns:
        Issue descriptions; empty when no checks fail.
    """
    _ = target_iso2
    issues: list[str] = []

    # Schema checks
    if "Price" not in enriched_df.columns:
        issues.append("enriched_df missing 'Price' column")
    if expect_social_exchanges and "risk level" not in enriched_df.columns:
        issues.append("no 'risk level' column after enrichment")
    if expect_social_exchanges and "PSILCA_Exchanges" not in enriched_df.columns:
        issues.append("enriched_df missing 'PSILCA_Exchanges'")

    for record in averaged_exchanges:
        if expect_social_exchanges and not record.get("Averaged_Exchanges"):
            issues.append(f"Activity {record.get('Activity code')!r} has no valid social exchanges")
        counters = record.get("Input_diagnostics") or {}
        if counters.get("working_hours_negative") or counters.get("working_hours_missing_or_nan"):
            issues.append(f"Activity {record.get('Activity code')!r} has invalid worker hours")
    return issues


def save_intermediate_artifacts(
    target_iso2: str,
    enriched_df: pd.DataFrame | None,
    averaged_exchanges: list[dict[str, Any]],
    output_dir: str,
    ecoinvent_version: str,
    save_switch: bool = True,
    include_social_data: bool = True,
    save_averaged: bool = True,
) -> None:
    """Save optional enriched gzip JSON and averaged JSON artifacts.

    Args:
        target_iso2: Region being processed.
        enriched_df: Matched and enriched activities.
        averaged_exchanges: Averaged activity records.
        output_dir: Artifact root.
        ecoinvent_version: Version label used in directory names.
        save_switch: Whether to write the artifacts.
        include_social_data: Use integrated-workflow artifact names.
    """
    if not save_switch:
        return

    # Save enriched activities JSON
    raw_name = (
        f"matched_{target_iso2}_prices_w_social_exchanges.json.gz"
        if include_social_data
        else f"priced_{target_iso2}_productivity_input.json.gz"
    )
    enriched_output_path = os.path.join(
        output_dir,
        "Sector mappings",
        "Raw",
        f"{ecoinvent_version}",
        raw_name,
    )
    if enriched_df is not None:
        _save_enriched(enriched_df, Path(enriched_output_path))

    if not save_averaged:
        return

    # Save averaged results
    averaged_slim = []
    for activity in averaged_exchanges or []:
        slim = activity.copy()
        if isinstance(slim.get("PSILCA_countries_used"), (list, set, tuple)):
            slim["PSILCA_countries_used"] = "; ".join(slim["PSILCA_countries_used"])
        if isinstance(slim.get("PSILCA_sectors_used"), dict):
            slim["PSILCA_sectors_used"] = {
                k: "; ".join(v) for k, v in slim["PSILCA_sectors_used"].items()
            }
        averaged_slim.append(slim)

    averaged_name = (
        f"{target_iso2}_averaged.json"
        if include_social_data
        else f"{target_iso2}_productivity_working_hours.json"
    )
    averaged_output_path = (
        Path(output_dir) / "Sector mappings" / "Averaged" / ecoinvent_version / averaged_name
    )
    write_json_atomic(averaged_output_path, averaged_slim)


def _save_enriched(enriched_df: pd.DataFrame, path: Path) -> None:
    """Stream large repeated sector assessments without a full JSON copy."""
    with atomic_output_path(path) as temporary:
        # pandas builds the entire serialized document before writing it.
        # Large RoW tables repeat full assessment lists across matches, so that
        # transient string can exceed RAM even when the calculation fits.
        # Keep the same JSON-array schema and pandas value conversions while
        # bounding serialization memory to a small number of rows.
        with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=1) as handle:
            handle.write("[")
            for start in range(0, len(enriched_df), 64):
                payload = enriched_df.iloc[start : start + 64].to_json(
                    orient="records", force_ascii=False
                )
                if start:
                    handle.write(",")
                handle.write(payload[1:-1])
            handle.write("]\n")


def summarize_and_optionally_save(
    target_iso2: str,
    enriched_df: pd.DataFrame,
    averaged_exchanges: list[dict[str, Any]],
    *,
    output_dir: str,
    ecoinvent_version: str,
    save_switch: bool,
    expect_social_exchanges: bool = True,
    save_averaged: bool = True,
) -> list[str]:
    """Collect diagnostics and optionally save intermediate artifacts.

    Args:
        target_iso2: Region being processed.
        enriched_df: Matched and enriched activities.
        averaged_exchanges: Averaged activity records.
        output_dir: Artifact root.
        ecoinvent_version: Version label used in directory names.
        save_switch: Whether to write intermediate artifacts.
        expect_social_exchanges: Run PSILCA-specific diagnostic checks.

    Returns:
        Issue descriptions; empty when no checks fail.
    """
    issues = collect_pipeline_issues(
        target_iso2,
        enriched_df,
        averaged_exchanges,
        expect_social_exchanges=expect_social_exchanges,
    )
    save_intermediate_artifacts(
        target_iso2=target_iso2,
        enriched_df=enriched_df,
        averaged_exchanges=averaged_exchanges,
        output_dir=output_dir,
        ecoinvent_version=ecoinvent_version,
        save_switch=save_switch,
        include_social_data=expect_social_exchanges,
        save_averaged=save_averaged,
    )
    return issues
