"""
Brightway2 biosphere flow management for SENSE.

This module provides functions for creating and managing social biosphere flows
in Brightway2 databases, including social risk/opportunity flows and working hours.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from bw2data import Database

from sense.utils.helpers import norm_code, slug


def _build_biosphere_lookup(
    biosphere_db: Database,
    biosphere_lookup: Optional[Dict[tuple, Any]],
) -> Tuple[Dict[tuple, Any], Set[tuple]]:
    """
    Ensure we have a dict[(name, unit, (cat, subcat))] -> flow.
    Returns (lookup_dict, existing_keys_set).
    """
    if biosphere_lookup is not None:
        return biosphere_lookup, set(biosphere_lookup.keys())
    lookup = {
        (f.get("name"), f.get("unit"), f.get("categories")): f
        for f in biosphere_db
    }
    return lookup, set(lookup.keys())


def _build_techno_lookup(
    techno_db: Database,
    techno_lookup: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Ensure we have a dict[activity_code] -> activity.
    """
    if techno_lookup is not None:
        return techno_lookup
    lookup = {}
    for act in techno_db:
        c = norm_code(act.get("code"))
        if c:
            lookup[c] = act
    return lookup


def add_social_biosphere_flows_and_exchanges(
    data: Union[pd.DataFrame, List[Dict[str, Any]]],
    database: str,
    foreground_database: Optional[str],
    biosphere_db_name: str,
    foreground_active: bool = False,
    *,
    biosphere_lookup: Optional[Dict[tuple, Any]] = None,
    techno_lookup: Optional[Dict[str, Any]] = None,
    precreated_flows: bool = False,
    update_tol: float = 0.0,
) -> Dict[str, int]:
    """
    Create and attach PSILCA-derived social biosphere flows to Brightway2 activities.
    
    Creates country-resolved social biosphere flows (if needed) and attaches them
    as exchanges to technosphere activities.
    
    Args:
        data: Averaged exchanges data (DataFrame or list of dicts)
        database: Name of the main technosphere database
        foreground_database: Name of the foreground database (if used)
        biosphere_db_name: Name of the sociosphere database for social flows
        foreground_active: If True, attach to foreground_database instead
        biosphere_lookup: Optional pre-built lookup dict for reuse
        techno_lookup: Optional pre-built lookup dict for reuse
        precreated_flows: Skip flow creation if flows already exist
        update_tol: Minimum difference to trigger exchange update
        
    Returns:
        Dict with counts: {"added", "updated", "missing", "missing_indicators", "failed_flows"}
        
    Note:
        Requires Brightway2 (`bw2data`) to be installed and a project to be active.
    """
    def _gx(exch):
        """Polyglot extractor for exchange fields."""
        get = exch.get
        ind = get("indicator") or get("@name") or get("name")
        unit = get("unit") or get("@unit") or ""
        cat = get("category") or get("@category") or "Unknown"
        subcat = get("subcategory") or get("@subCategory") or "Unknown"
        typ = (get("type") or get("@type") or "risk").strip().lower()
        val = get("meanValue")
        if val is None:
            val = get("mean value")
        return ind, unit, cat, subcat, typ, val

    def _pack_dataframe_to_activity_records(df: pd.DataFrame) -> List[Dict]:
        """Convert a wide DataFrame into list-of-activity dicts."""
        if df.empty:
            return []

        code_col = "Activity code" if "Activity code" in df.columns else (
            "code" if "code" in df.columns else None
        )
        if code_col is None:
            return []

        iso2_col = "ISO2" if "ISO2" in df.columns else None
        ind_cols = [c for c in ("indicator", "@name", "name") if c in df.columns]
        unit_cols = [c for c in ("unit", "@unit") if c in df.columns]
        cat_cols = [c for c in ("category", "@category") if c in df.columns]
        subcat_cols = [c for c in ("subcategory", "@subCategory") if c in df.columns]
        type_cols = [c for c in ("type", "@type") if c in df.columns]
        val_cols = [c for c in ("meanValue", "mean value") if c in df.columns]

        def first_nonnull(rowdict, cols, default=None):
            for c in cols:
                v = rowdict.get(c)
                if pd.notna(v):
                    return v
            return default

        group_cols = [code_col] + ([iso2_col] if iso2_col else [])
        records = []

        for keys, g in df.groupby(group_cols, dropna=False, sort=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            code = str(keys[0]).strip() if keys and keys[0] is not None else None
            iso2_val = (
                keys[1]
                if len(keys) > 1
                else (g[iso2_col].iloc[0] if iso2_col else None)
            ) or "GLO"

            exchs = []
            for _, row in g.iterrows():
                rd = row.to_dict()
                exchs.append({
                    "indicator": first_nonnull(rd, ind_cols, None),
                    "unit": first_nonnull(rd, unit_cols, ""),
                    "category": first_nonnull(rd, cat_cols, "Unknown"),
                    "subcategory": first_nonnull(rd, subcat_cols, "Unknown"),
                    "type": (first_nonnull(rd, type_cols, "risk") or "risk"),
                    "meanValue": first_nonnull(rd, val_cols, None),
                })

            records.append({
                "Activity code": code,
                "ISO2": iso2_val,
                "Averaged_Exchanges": exchs,
            })
        return records

    # Accept DataFrame or list
    if isinstance(data, pd.DataFrame):
        data = _pack_dataframe_to_activity_records(data)

    # Guards & setup
    if not data or not isinstance(data, list):
        print("⚠️ No input data provided to add_social_biosphere_flows_and_exchanges().")
        return {
            "added": 0,
            "updated": 0,
            "missing": 0,
            "missing_indicators": 0,
            "failed_flows": 0,
        }

    techno_db_name = foreground_database if foreground_active else database
    techno_db = Database(techno_db_name)
    biosphere_db = Database(biosphere_db_name)

    # Build lookups
    biosphere_lookup, existing_flows = _build_biosphere_lookup(biosphere_db, biosphere_lookup)
    techno_lookup = _build_techno_lookup(techno_db, techno_lookup)

    # STEP 1: collect unique flows needed
    missing_indicators: List[Dict] = []
    unique_flows: Dict[tuple, Dict] = {}

    if not precreated_flows:
        for activity in data:
            iso2_here = activity.get("ISO2") or "GLO"
            for exch in (activity.get("Averaged_Exchanges", []) or []):
                ind_name, unit, cat, subcat, exch_type, _ = _gx(exch)
                if not ind_name:
                    missing_indicators.append(exch)
                    ind_name = "Unnamed indicator"

                flow_name = f"{ind_name} ({exch_type}), {iso2_here}"
                key = (flow_name, unit, (cat, subcat))
                if key not in unique_flows:
                    unique_flows[key] = {
                        "name": flow_name,
                        "unit": unit,
                        "categories": (cat, subcat),
                        "comment": "Imported from PSILCA sector matching (country-resolved, typed)",
                    }

        if missing_indicators:
            print(f"⚠️ {len(missing_indicators)} exchanges missing indicator names. Example:")
            print(missing_indicators[0])

    # STEP 2: create missing flows
    failed_flows: List[str] = []
    if not precreated_flows and unique_flows:
        to_create = [k for k in unique_flows.keys() if k not in existing_flows]

        for (name, unit, categories) in to_create:
            try:
                code = f"social_{slug(name)}_{slug(unit)}"
                new_flow = biosphere_db.new_activity(
                    name=name,
                    unit=unit,
                    categories=categories,
                    type="emission",
                    code=code,
                    comment=unique_flows[(name, unit, categories)]["comment"],
                )
                new_flow.save()
                biosphere_lookup[(name, unit, categories)] = new_flow
                existing_flows.add((name, unit, categories))
            except Exception as e:
                failed_flows.append(f"{name} [{unit}] {categories}: {e}")
                print(f"⚠️ Failed to create biosphere flow '{name}': {e}")

    # STEP 3: map exchanges -> flows
    exch_to_flow: Dict[tuple, Any] = {}
    not_found: List[tuple] = []

    for activity in data:
        iso2_here = activity.get("ISO2") or "GLO"
        for exch in (activity.get("Averaged_Exchanges", []) or []):
            ind_name, unit, cat, subcat, exch_type, _ = _gx(exch)
            if not ind_name:
                ind_name = "Unnamed indicator"

            flow_name = f"{ind_name} ({exch_type}), {iso2_here}"
            key_flow = (flow_name, unit, (cat, subcat))
            flow = biosphere_lookup.get(key_flow)

            if flow:
                exch_to_flow[(ind_name, exch_type, unit, cat, subcat, iso2_here)] = flow
            else:
                not_found.append(key_flow)
                print(f"❌ Biosphere flow not found for: {key_flow}")

    # STEP 4: attach exchanges
    added, updated = 0, 0

    grouped: Dict[str, List[Dict]] = {}
    for activity in data:
        raw_code = activity.get("code") or activity.get("Activity code")
        code = norm_code(raw_code)
        if not code:
            print("⚠️ No activity code found in entry.")
            continue
        grouped.setdefault(code, []).append(activity)

    for code, rows in grouped.items():
        tech_act = techno_lookup.get(code)
        if not tech_act:
            print(f"❌ Technosphere activity not found for code: {code}")
            continue

        existing = {
            (exc.input.key, exc["type"]): exc
            for exc in tech_act.exchanges()
            if exc.get("type") == "biosphere"
        }
        to_save_this_act = []

        for activity in rows:
            iso2_here = activity.get("ISO2") or "GLO"
            for exch in (activity.get("Averaged_Exchanges", []) or []):
                ind_name, unit, cat, subcat, exch_type, val = _gx(exch)
                if not ind_name:
                    ind_name = "Unnamed indicator"

                try:
                    amount = float(val)
                except Exception:
                    print(f"⚠️ Invalid meanValue for {ind_name}: {val}")
                    continue
                if not np.isfinite(amount) or amount <= 0:
                    continue

                flow = exch_to_flow.get((ind_name, exch_type, unit, cat, subcat, iso2_here))
                if not flow:
                    continue

                exchange_key = (flow.key, "biosphere")
                if exchange_key in existing:
                    exc_obj = existing[exchange_key]
                    if update_tol == 0.0 or abs(exc_obj["amount"] - amount) > update_tol:
                        exc_obj["amount"] = amount
                        to_save_this_act.append(exc_obj)
                        updated += 1
                else:
                    try:
                        new_exc = tech_act.new_exchange(
                            input=flow.key,
                            amount=amount,
                            type="biosphere",
                        )
                        to_save_this_act.append(new_exc)
                        added += 1
                    except Exception as e:
                        print(f"⚠️ Failed to create exchange for ({ind_name}, {exch_type}): {e}")

        for exc in to_save_this_act:
            try:
                exc.save()
            except Exception as e:
                print(f"⚠️ Failed to save exchange for act {code}: {e}")

    return {
        "added": added,
        "updated": updated,
        "missing": len(not_found),
        "missing_indicators": len(missing_indicators),
        "failed_flows": len(failed_flows),
    }


def add_working_hours_flows_and_exchanges(
    data: Union[pd.DataFrame, List[Dict[str, Any]]],
    database: str,
    foreground_database: Optional[str],
    biosphere_db_name: str,
    foreground_active: bool = False,
    *,
    biosphere_lookup: Optional[Dict[tuple, Any]] = None,
    techno_lookup: Optional[Dict[str, Any]] = None,
    precreated_flows: bool = False,
    update_tol: float = 0.0,
) -> Dict[str, int]:
    """
    Create and attach country-resolved working hours biosphere flows.
    
    Creates working hours flows for each ISO2 region and attaches them
    to technosphere activities.
    
    Args:
        data: Activity data with Working_hours values (DataFrame or list)
        database: Name of the main technosphere database
        foreground_database: Name of the foreground database (if used)
        biosphere_db_name: Name of the sociosphere database
        foreground_active: If True, attach to foreground_database instead
        biosphere_lookup: Optional pre-built lookup dict for reuse
        techno_lookup: Optional pre-built lookup dict for reuse
        precreated_flows: Skip flow creation if flows already exist
        update_tol: Minimum difference to trigger exchange update
        
    Returns:
        Dict with counts: {"added", "updated", "missing", "failed_flows"}
    """
    def _pack_dataframe_to_working_records(df: pd.DataFrame) -> List[Dict]:
        if df.empty:
            return []

        code_col = "Activity code" if "Activity code" in df.columns else (
            "code" if "code" in df.columns else None
        )
        if code_col is None:
            return []

        records = []
        for _, row in df.iterrows():
            code = row.get(code_col)
            iso2_val = row.get("ISO2") or "GLO"
            wh_val = row.get("Working_hours")
            records.append({
                "Activity code": norm_code(code),
                "ISO2": iso2_val,
                "Working_hours": wh_val,
            })
        return records

    # Accept DataFrame or list
    if isinstance(data, pd.DataFrame):
        data = _pack_dataframe_to_working_records(data)

    # Guards & setup
    if not data or not isinstance(data, list):
        print("⚠️ No input data provided to add_working_hours_flows_and_exchanges().")
        return {"added": 0, "updated": 0, "missing": 0, "failed_flows": 0}

    techno_db_name = foreground_database if foreground_active else database
    techno_db = Database(techno_db_name)
    biosphere_db = Database(biosphere_db_name)

    # Build lookups
    biosphere_lookup, existing_flows = _build_biosphere_lookup(biosphere_db, biosphere_lookup)
    techno_lookup = _build_techno_lookup(techno_db, techno_lookup)

    # STEP 1: collect unique working-hours flows needed
    unit = "working hour"
    categories = ("Social", "Working hours")
    iso2_set = {rec.get("ISO2") or "GLO" for rec in data}

    unique_flows: Dict[tuple, Dict] = {}
    if not precreated_flows:
        for iso2_here in iso2_set:
            flow_name = f"working hours, {iso2_here}"
            key = (flow_name, unit, categories)
            if key not in unique_flows:
                unique_flows[key] = {
                    "name": flow_name,
                    "unit": unit,
                    "categories": categories,
                    "comment": "Total working hours from PSILCA sector mapping",
                }

    # STEP 2: create missing flows
    failed_flows: List[str] = []
    if not precreated_flows and unique_flows:
        to_create = [k for k in unique_flows.keys() if k not in existing_flows]

        for (name, unit_, categories_) in to_create:
            try:
                code = f"working_hours_{slug(name)}"
                new_flow = biosphere_db.new_activity(
                    name=name,
                    unit=unit_,
                    categories=categories_,
                    type="emission",
                    code=code,
                    comment=unique_flows[(name, unit_, categories_)]["comment"],
                )
                new_flow.save()
                biosphere_lookup[(name, unit_, categories_)] = new_flow
                existing_flows.add((name, unit_, categories_))
            except Exception as e:
                failed_flows.append(f"{name} [{unit_}] {categories_}: {e}")
                print(f"⚠️ Failed to create working-hours flow '{name}': {e}")

    # STEP 3: map ISO2 -> flows
    iso2_to_flow: Dict[str, Any] = {}
    not_found: List[tuple] = []

    for iso2_here in iso2_set:
        flow_name = f"working hours, {iso2_here}"
        key_flow = (flow_name, unit, categories)
        flow = biosphere_lookup.get(key_flow)
        if flow:
            iso2_to_flow[iso2_here] = flow
        else:
            not_found.append(key_flow)
            print(f"❌ Working-hours biosphere flow not found for: {key_flow}")

    # STEP 4: attach exchanges
    added, updated = 0, 0

    grouped: Dict[str, List[Dict]] = {}
    for rec in data:
        raw_code = rec.get("Activity code") or rec.get("code")
        code = norm_code(raw_code)
        if not code:
            print("⚠️ No activity code found in working-hours entry.")
            continue
        grouped.setdefault(code, []).append(rec)

    for code, rows in grouped.items():
        tech_act = techno_lookup.get(code)
        if not tech_act:
            print(f"❌ Technosphere activity not found for code: {code}")
            continue

        existing = {
            (exc.input.key, exc["type"]): exc
            for exc in tech_act.exchanges()
            if exc.get("type") == "biosphere"
        }
        to_save_this_act = []

        for rec in rows:
            iso2_here = rec.get("ISO2") or "GLO"
            wh_val = rec.get("Working_hours")

            try:
                amount = float(wh_val)
            except Exception:
                print(f"⚠️ Invalid Working_hours value for act {code}: {wh_val}")
                continue
            if not np.isfinite(amount) or amount <= 0:
                continue

            flow = iso2_to_flow.get(iso2_here)
            if not flow:
                continue

            exchange_key = (flow.key, "biosphere")
            if exchange_key in existing:
                exc_obj = existing[exchange_key]
                if update_tol == 0.0 or abs(exc_obj["amount"] - amount) > update_tol:
                    exc_obj["amount"] = amount
                    to_save_this_act.append(exc_obj)
                    updated += 1
            else:
                try:
                    new_exc = tech_act.new_exchange(
                        input=flow.key,
                        amount=amount,
                        type="biosphere",
                    )
                    to_save_this_act.append(new_exc)
                    added += 1
                except Exception as e:
                    print(f"⚠️ Failed to create working-hours exchange for act {code}: {e}")

        for exc in to_save_this_act:
            try:
                exc.save()
            except Exception as e:
                print(f"⚠️ Failed to save working-hours exchange for act {code}: {e}")

    return {
        "added": added,
        "updated": updated,
        "missing": len(not_found),
        "failed_flows": len(failed_flows),
    }
