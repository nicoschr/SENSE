"""
Diagnostics and reporting for SENSE pipeline.

This module provides functions for:
- Reattaching prices to enriched data
- Collecting pipeline issues
- Saving intermediate artifacts
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

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
) -> Tuple[pd.DataFrame, int]:
    """
    Fill enriched_df['Price'] from meta_df where needed.
    
    Fills NaN or non-positive prices from the activity list metadata
    using normalized 'Activity code' as the join key.
    
    Args:
        enriched_df: DataFrame with matched activities (may have missing prices)
        meta_df: DataFrame with activity metadata including prices
        
    Returns:
        Tuple of (updated_df, count_of_updated_prices)
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

    need_fill = price_cur.isna() | (price_cur <= 0)
    have_new = price_new.notna()
    to_update = need_fill & have_new

    updated_count = int(to_update.sum())
    if updated_count:
        df.loc[to_update, "Price"] = price_new[to_update]

    return df, updated_count


def collect_pipeline_issues(
    target_iso2: str,
    enriched_df: pd.DataFrame,
    averaged_exchanges: List[Dict[str, Any]],
) -> List[str]:
    """
    Build a list of human-readable issue strings.
    
    Args:
        target_iso2: The ISO2 region being processed
        enriched_df: DataFrame with matched and enriched activities
        averaged_exchanges: List of averaged activity dictionaries
        
    Returns:
        List of issue description strings
    """
    issues: List[str] = []

    # Schema checks
    if "Price" not in enriched_df.columns:
        issues.append("enriched_df missing 'Price' column")
    if "risk level" not in enriched_df.columns:
        issues.append("no 'risk level' column after enrichment")
    if "PSILCA_Exchanges" not in enriched_df.columns:
        issues.append("enriched_df missing 'PSILCA_Exchanges'")

    # Missing exchanges per sector
    if "PSILCA_Exchanges" in enriched_df.columns:
        try:
            missing_exchs = enriched_df[enriched_df["PSILCA_Exchanges"].apply(len) == 0]
            if not missing_exchs.empty:
                issues.append(f"{len(missing_exchs)} matched sectors have NO social exchanges")
        except Exception:
            issues.append("could not evaluate PSILCA_Exchanges lengths")

    # Zero / NaN meanValues after averaging
    zero_count = 0
    nan_count = 0
    for act in (averaged_exchanges or []):
        for ex in act.get("Averaged_Exchanges", []) or []:
            mv = ex.get("meanValue", np.nan)
            if isinstance(mv, (int, float)):
                if mv == 0.0:
                    zero_count += 1
                elif (not np.isfinite(mv)) or np.isnan(mv):
                    nan_count += 1
            else:
                nan_count += 1
    if zero_count > 0:
        issues.append(f"{zero_count} zero meanValues after averaging")
    if nan_count > 0:
        issues.append(f"{nan_count} NaN meanValues after averaging")

    # Missing/zero prices
    if "Price" in enriched_df.columns:
        zp = pd.to_numeric(enriched_df["Price"], errors="coerce").fillna(0)
        cnt = int((zp <= 0).sum())
        if cnt > 0:
            issues.append(f"{cnt} activities have missing/zero Price pre-averaging")

    return issues


def save_intermediate_artifacts(
    target_iso2: str,
    enriched_df: pd.DataFrame,
    averaged_exchanges: List[Dict[str, Any]],
    output_dir: str,
    ecoinvent_version: str,
    save_switch: bool = True,
) -> None:
    """
    Save intermediate JSON artifacts for debugging.
    
    Args:
        target_iso2: The ISO2 region being processed
        enriched_df: DataFrame with matched and enriched activities
        averaged_exchanges: List of averaged activity dictionaries
        output_dir: Base directory for output files
        ecoinvent_version: Version string for subdirectory naming
        save_switch: If False, skip saving entirely
    """
    if not save_switch:
        return

    # Save enriched activities JSON
    enriched_output_path = os.path.join(
        output_dir, "Sector mappings", "Raw", f"{ecoinvent_version}",
        f"matched_{target_iso2}_prices_w_social_exchanges.json"
    )
    os.makedirs(os.path.dirname(enriched_output_path), exist_ok=True)
    enriched_df.to_json(enriched_output_path, orient="records", indent=2, force_ascii=False)

    # Save averaged results
    averaged_slim = []
    for activity in averaged_exchanges or []:
        slim = activity.copy()
        if isinstance(slim.get("PSILCA_countries_used"), (list, set, tuple)):
            slim["PSILCA_countries_used"] = '; '.join(slim["PSILCA_countries_used"])
        if isinstance(slim.get("PSILCA_sectors_used"), dict):
            slim["PSILCA_sectors_used"] = {k: '; '.join(v) for k, v in slim["PSILCA_sectors_used"].items()}
        averaged_slim.append(slim)

    averaged_exchanges_path = os.path.join(
        output_dir, "Sector mappings", "Averaged", f"{ecoinvent_version}",
        f"{target_iso2}_averaged.json"
    )
    os.makedirs(os.path.dirname(averaged_exchanges_path), exist_ok=True)
    with open(averaged_exchanges_path, "w", encoding="utf-8") as f:
        json.dump(averaged_slim, f, indent=2, ensure_ascii=False)


def summarize_and_optionally_save(
    target_iso2: str,
    enriched_df: pd.DataFrame,
    averaged_exchanges: List[Dict[str, Any]],
    *,
    output_dir: str,
    ecoinvent_version: str,
    save_switch: bool,
) -> List[str]:
    """
    Full diagnostics pass with optional artifact saving.
    
    Args:
        target_iso2: The ISO2 region being processed
        enriched_df: DataFrame with matched and enriched activities
        averaged_exchanges: List of averaged activity dictionaries
        output_dir: Base directory for output files
        ecoinvent_version: Version string for subdirectory naming
        save_switch: If True, save intermediate artifacts
        
    Returns:
        List of issue description strings
    """
    issues = collect_pipeline_issues(target_iso2, enriched_df, averaged_exchanges)
    save_intermediate_artifacts(
        target_iso2=target_iso2,
        enriched_df=enriched_df,
        averaged_exchanges=averaged_exchanges,
        output_dir=output_dir,
        ecoinvent_version=ecoinvent_version,
        save_switch=save_switch,
    )
    return issues
