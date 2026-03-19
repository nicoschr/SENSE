"""
Semantic matching between ecoinvent activities and PSILCA sectors.

This module provides functions for:
- Matching ecoinvent activities to PSILCA sectors using embeddings
- Attaching PSILCA social exchanges to matched activities
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from sense.data.embeddings import load_embeddings
from sense.utils.helpers import norm_code


def match_from_stores(
    ecoinvent_store_dir: str,
    psilca_store_dir: str,
    target_iso2: str,
    psilca_iso3_list: Optional[List[str]] = None,
    threshold: float = 0.015,
    ecoinvent_list_of_activities_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Match ecoinvent activities to PSILCA sectors using precomputed embeddings.
    
    Uses cosine similarity between sentence embeddings to find the best matching
    PSILCA sectors for each ecoinvent activity. Supports per-activity allow-lists
    for complex regions like RoW (Rest of World).
    
    Args:
        ecoinvent_store_dir: Directory containing ecoinvent embeddings
        psilca_store_dir: Directory containing PSILCA embeddings
        target_iso2: ISO2 code to match (e.g., 'BR', 'RoW')
        psilca_iso3_list: Fallback list of ISO3 countries if not using activity allow-lists
        threshold: Cosine similarity tolerance for multiple matches
        ecoinvent_list_of_activities_path: Optional path to JSON with per-activity metadata
        
    Returns:
        DataFrame with columns:
        ["ISO2", "Activity code", "Activity name", "Reference product", "ISIC",
         "Unit", "Price", "PSILCA_country", "Matched_sector", "Semantic_score"]
         
    Examples:
        >>> df = match_from_stores(
        ...     ecoinvent_store_dir="./embeddings/ecoinvent",
        ...     psilca_store_dir="./embeddings/psilca",
        ...     target_iso2="CH",
        ...     psilca_iso3_list=["CHE"],
        ... )
    """
    def _safe_name(name: str) -> str:
        return (str(name)
                .replace("/", "_").replace("\\", "_").replace(":", "_")
                .replace("-", "_").replace(" ", "_"))

    # Load metadata & per-activity PSILCA allow-lists
    act_allowed: Dict[str, set] = {}
    meta_df = pd.DataFrame(columns=[
        "Activity code", "Activity name", "Reference product", "ISIC", "Unit", "Price", "ISO2"
    ])
    all_iso3: List[str] = []

    if ecoinvent_list_of_activities_path:
        with open(ecoinvent_list_of_activities_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        entry = next((e for e in data if e.get("ISO2") == target_iso2), None)
        activities = entry.get("Activities", []) if entry else []
        default_mapping = entry.get("PSILCA_mapping", "") if target_iso2 not in ("RoW", "RoE") else None

        meta_rows = []
        for a in activities:
            raw_code = a.get("code")
            code = norm_code((raw_code or "").strip())
            name = (a.get("name") or "").strip()
            ref = (a.get("reference product") or "").strip()
            isic_raw = a.get("ISIC rev.4 ecoinvent")
            isic = isic_raw.strip() if isinstance(isic_raw, str) else ""
            unit = (a.get("unit") or "").strip()
            price = a.get("price", None)

            mapping = a.get("PSILCA_mapping", "") if target_iso2 in ("RoW", "RoE") else (default_mapping or "")
            allowed = set([c.strip() for c in mapping.split(";") if c.strip()])
            act_allowed[code] = allowed

            meta_rows.append({
                "Activity code": code,
                "Activity name": name,
                "Reference product": ref,
                "ISIC": isic,
                "Unit": unit,
                "Price": price,
                "ISO2": target_iso2,
            })

        meta_df = pd.DataFrame(meta_rows)
        all_iso3 = sorted({iso3 for s in act_allowed.values() for iso3 in s})
    else:
        all_iso3 = psilca_iso3_list or []

    # Load ecoinvent embeddings for this ISO2
    base_name = f"ecoinvent_{_safe_name(target_iso2)}"
    act_emb, act_ids, _ = load_embeddings(ecoinvent_store_dir, base_name)

    # Build normalized-code -> column-index map
    act_ids_list = act_ids.tolist()
    act_ids_norm = [norm_code(c) for c in act_ids_list]
    code_to_col_norm = {c_norm: i for i, c_norm in enumerate(act_ids_norm)}

    # Choose activity codes to consider
    if not meta_df.empty:
        wanted_codes_norm = [c for c in meta_df["Activity code"].tolist() if c in code_to_col_norm]
    else:
        wanted_codes_norm = act_ids_norm

    if not wanted_codes_norm:
        return pd.DataFrame(columns=[
            "ISO2", "Activity code", "Activity name", "Reference product", "ISIC", "Unit", "Price",
            "PSILCA_country", "Matched_sector", "Semantic_score"
        ])

    col_idx = np.array([code_to_col_norm[c] for c in wanted_codes_norm], dtype=int)
    inv_codes_norm = np.array(wanted_codes_norm, dtype=object)

    # Match for each PSILCA ISO3
    rows: List[Dict[str, Any]] = []

    for iso3 in all_iso3:
        try:
            sec_emb, sec_ids, _ = load_embeddings(psilca_store_dir, f"psilca_{iso3}")
        except FileNotFoundError:
            continue

        if sec_emb.shape[1] != act_emb.shape[1]:
            continue

        # Activity-level filtering (for RoW/RoE allow-lists)
        if ecoinvent_list_of_activities_path:
            mask_cols = np.array([iso3 in act_allowed.get(code, set()) for code in inv_codes_norm], dtype=bool)
            if not mask_cols.any():
                continue
            sub_cols = col_idx[mask_cols]
            sub_codes = inv_codes_norm[mask_cols]
        else:
            sub_cols = col_idx
            sub_codes = inv_codes_norm

        if sub_cols.size == 0:
            continue

        # Cosine similarities
        sims = sec_emb @ act_emb[sub_cols].T
        best = sims.max(axis=0)
        keep = sims >= (best - threshold)[None, :]
        si, ai = keep.nonzero()

        sec_ids_np = sec_ids.to_numpy()
        for s, a in zip(si, ai):
            try:
                _, sector_name = sec_ids_np[s].split("::", 1)
            except ValueError:
                sector_name = str(sec_ids_np[s])

            rows.append({
                "ISO2": target_iso2,
                "Activity code": sub_codes[a],
                "PSILCA_country": iso3,
                "Matched_sector": sector_name,
                "Semantic_score": float(np.round(sims[s, a], 4)),
            })

    df_map = pd.DataFrame(rows)

    if df_map.empty:
        return pd.DataFrame(columns=[
            "ISO2", "Activity code", "Activity name", "Reference product", "ISIC", "Unit", "Price",
            "PSILCA_country", "Matched_sector", "Semantic_score"
        ])

    # Merge metadata
    df_map["Activity code"] = df_map["Activity code"].astype(str)

    if meta_df.empty:
        df_map["Activity name"] = np.nan
        df_map["Reference product"] = np.nan
        df_map["ISIC"] = np.nan
        df_map["Unit"] = np.nan
        df_map["Price"] = np.nan
        df_out = df_map[[
            "ISO2", "Activity code", "Activity name", "Reference product", "ISIC", "Unit", "Price",
            "PSILCA_country", "Matched_sector", "Semantic_score"
        ]].copy()
        return df_out

    meta_df = meta_df.copy()
    meta_df["Activity code"] = meta_df["Activity code"].astype(str)
    if "Price" in meta_df.columns:
        meta_df["Price"] = pd.to_numeric(meta_df["Price"], errors="coerce")

    df_out = df_map.merge(
        meta_df,
        on=["ISO2", "Activity code"],
        how="left",
        validate="m:1"
    )

    df_out = df_out[[
        "ISO2", "Activity code", "Activity name", "Reference product", "ISIC", "Unit", "Price",
        "PSILCA_country", "Matched_sector", "Semantic_score"
    ]].copy()

    return df_out


def add_social_exchanges(
    sector_mapping_df: pd.DataFrame,
    combined_json_path: str,
) -> pd.DataFrame:
    """
    Add PSILCA social exchanges (risks and opportunities) to mapped ecoinvent activities.
    
    Attaches social exchange data from PSILCA to the matched sector mappings,
    including risk levels, working hours, and indicator types.
    
    Args:
        sector_mapping_df: DataFrame with ecoinvent-PSILCA mappings
        combined_json_path: Path to the combined PSILCA JSON file
        
    Returns:
        DataFrame with new columns:
        - PSILCA_Exchanges: list of dicts with parsed social exchanges
        - risk/opportunity types: aggregated per activity
        - risk level: aggregated per activity
        
    Examples:
        >>> enriched_df = add_social_exchanges(
        ...     sector_mapping_df=matches_df,
        ...     combined_json_path="./data/psilca_combined.json",
        ... )
    """
    def _norm(s):
        """Normalize keys for reliable matching."""
        return (s or "").strip().lower()

    # Load combined PSILCA data
    with open(combined_json_path, "r", encoding="utf-8") as f:
        psilca_data = json.load(f)

    # Build lookup dictionary
    psilca_lookup: Dict[tuple, List[Dict]] = {}
    for entry in psilca_data:
        location = _norm(entry.get("Location"))
        refname = _norm(entry.get("Reference name"))
        exchanges = []

        for exch in entry.get("FilteredExchanges", []):
            mv_str = exch.get("@meanValue", "")
            if mv_str in (None, "", "nan"):
                continue

            try:
                mv = float(mv_str)
            except Exception:
                continue

            if exch.get("@unit") != "h":
                continue

            name_field = (exch.get("@name") or "").strip()
            indicator = name_field
            level = "no data"
            kind = "risk"

            if ";" in name_field:
                left, right = [x.strip() for x in name_field.split(";", 1)]
                indicator = left
                right_lower = right.lower()
                parts = right_lower.split()
                if parts:
                    if parts[-1] in {"risk", "opportunity"}:
                        kind = parts[-1]
                        level = " ".join(parts[:-1]).strip() or "no data"
                    else:
                        level = right_lower
                else:
                    level = "no data"

            exchanges.append({
                "indicator": indicator,
                "type": kind,
                "risk level": level,
                "meanValue": mv,
                "unit": exch.get("@unit", ""),
                "category": exch.get("@category", ""),
                "subcategory": exch.get("@subCategory", ""),
            })

        psilca_lookup[(location, refname)] = exchanges

    # Attach exchanges to each mapped activity
    exchanges_list = []
    all_levels, all_types = set(), set()
    missing_keys = []

    for _, row in sector_mapping_df.iterrows():
        key = (_norm(row.get("PSILCA_country")), _norm(row.get("Matched_sector")))
        exchs = psilca_lookup.get(key)

        if exchs is None or len(exchs) == 0:
            missing_keys.append(key)
            exchs = []

        exchanges_list.append(exchs)
        all_levels.update(ex["risk level"] for ex in exchs)
        all_types.update(ex["type"] for ex in exchs)

    # Attach results and summaries
    sector_mapping_df = sector_mapping_df.copy()
    sector_mapping_df["PSILCA_Exchanges"] = exchanges_list

    sector_mapping_df["risk/opportunity types"] = [
        "; ".join(sorted(set(ex["type"] for ex in exchs))) if exchs else ""
        for exchs in exchanges_list
    ]
    sector_mapping_df["risk level"] = [
        "; ".join(sorted(set(ex["risk level"] for ex in exchs))) if exchs else ""
        for exchs in exchanges_list
    ]

    # Diagnostics
    if missing_keys:
        uniq_missing = sorted(set(missing_keys))
        print(f"⚠️  {len(uniq_missing)} unmapped (Location, Reference name) pairs "
              f"— possible normalization or country mismatch.")
        for k in uniq_missing[:10]:
            print(f"   → Missing PSILCA sector: {k}")
        if len(uniq_missing) > 10:
            print(f"   ... and {len(uniq_missing)-10} more.")

    return sector_mapping_df
