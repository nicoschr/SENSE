"""Create SENSE sociosphere flows and synchronize their Brightway exchanges."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from bw2data import Database

from sense.brightway.repository import create_missing_flows, synchronize_exchanges
from sense.utils.helpers import extract_isic2, norm_code

logger = logging.getLogger(__name__)


def _validate_amount(value: Any, label: str) -> None:
    """Reject malformed amounts before creating flows or removing old exchanges."""
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be finite and non-negative") from error
    if isinstance(value, bool) or not np.isfinite(number) or number < 0:
        raise ValueError(f"{label} must be finite and non-negative")


def _build_biosphere_lookup(
    biosphere_db: Database,
    biosphere_lookup: dict[tuple, Any] | None,
) -> tuple[dict[tuple, Any], set[tuple]]:
    """
    Ensure we have a dict[(name, unit, (cat, subcat))] -> flow.
    Returns (lookup_dict, existing_keys_set).
    """
    if biosphere_lookup is not None:
        return biosphere_lookup, set(biosphere_lookup.keys())
    lookup = {(f.get("name"), f.get("unit"), f.get("categories")): f for f in biosphere_db}
    return lookup, set(lookup.keys())


def _build_techno_lookup(
    techno_db: Database,
    techno_lookup: dict[str, Any] | None,
) -> dict[str, Any]:
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


def _create_missing_flows(
    unique_flows: dict[tuple, dict],
    existing_flows: set[tuple],
    biosphere_db: Any,
    biosphere_lookup: dict[tuple, Any],
    code_prefix: str = "social",
) -> list[str]:
    """Create missing biosphere flows using the configured repository backend."""

    return create_missing_flows(
        flow_specs=unique_flows,
        existing_keys=existing_flows,
        database=biosphere_db,
        lookup=biosphere_lookup,
        code_prefix=code_prefix,
    )


def _attach_exchanges(
    grouped: dict[str, list[dict]],
    techno_lookup: dict[str, Any],
    resolve_flow,
    managed_flow=None,
    managed_flow_keys: set[tuple[str, str]] | None = None,
    update_tol: float = 0.0,
) -> tuple[int, int, int]:
    """Synchronize one family of SENSE-managed biosphere exchanges."""

    database_name = ""
    for activity in techno_lookup.values():
        key = getattr(activity, "key", None)
        if key:
            database_name = str(key[0])
            break
    return synchronize_exchanges(
        database_name=database_name,
        grouped=grouped,
        techno_lookup=techno_lookup,
        resolve_flow=resolve_flow,
        managed_flow=managed_flow,
        managed_flow_keys=managed_flow_keys,
        update_tol=update_tol,
    )


def add_social_biosphere_flows_and_exchanges(
    data: pd.DataFrame | list[dict[str, Any]],
    database: str,
    biosphere_db_name: str,
    *,
    biosphere_lookup: dict[tuple, Any] | None = None,
    techno_lookup: dict[str, Any] | None = None,
    precreated_flows: bool = False,
    update_tol: float = 0.0,
) -> dict[str, int]:
    """
    Create and attach PSILCA-derived social biosphere flows to Brightway2 activities.

    Creates country-resolved social biosphere flows (if needed) and attaches them
    as exchanges to technosphere activities.

    Args:
        data: Averaged exchanges data (DataFrame or list of dicts)
        database: Name of the technosphere database to attach flows to
        biosphere_db_name: Name of the sociosphere database for social flows
        biosphere_lookup: Optional pre-built lookup dict for reuse
        techno_lookup: Optional pre-built lookup dict for reuse
        precreated_flows: Skip flow creation if flows already exist
        update_tol: Minimum difference to trigger exchange update

    Returns:
        Counts for added, updated, removed, missing, unnamed, and failed flows.

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

    def _pack_dataframe_to_activity_records(df: pd.DataFrame) -> list[dict]:
        """Convert a wide DataFrame into list-of-activity dicts."""
        if df.empty:
            return []

        code_col = (
            "Activity code"
            if "Activity code" in df.columns
            else ("code" if "code" in df.columns else None)
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
                keys[1] if len(keys) > 1 else (g[iso2_col].iloc[0] if iso2_col else None)
            ) or "GLO"

            exchs = []
            for _, row in g.iterrows():
                rd = row.to_dict()
                exchs.append(
                    {
                        "indicator": first_nonnull(rd, ind_cols, None),
                        "unit": first_nonnull(rd, unit_cols, ""),
                        "category": first_nonnull(rd, cat_cols, "Unknown"),
                        "subcategory": first_nonnull(rd, subcat_cols, "Unknown"),
                        "type": (first_nonnull(rd, type_cols, "risk") or "risk"),
                        "meanValue": first_nonnull(rd, val_cols, None),
                    }
                )

            records.append(
                {
                    "Activity code": code,
                    "ISO2": iso2_val,
                    "Averaged_Exchanges": exchs,
                }
            )
        return records

    # Accept DataFrame or list
    if isinstance(data, pd.DataFrame):
        data = _pack_dataframe_to_activity_records(data)

    # Guards & setup
    if not data or not isinstance(data, list):
        logger.info("No social activity data supplied; no exchanges were written")
        return {
            "added": 0,
            "updated": 0,
            "removed": 0,
            "missing": 0,
            "missing_indicators": 0,
            "failed_flows": 0,
        }

    for activity in data:
        for exchange in activity.get("Averaged_Exchanges", []) or []:
            indicator, _, _, _, _, amount = _gx(exchange)
            if not indicator:
                raise ValueError("Social exchanges are missing indicator names")
            _validate_amount(amount, "Social exchange amount")

    techno_db = Database(database)
    biosphere_db = Database(biosphere_db_name)

    # Build lookups
    biosphere_lookup, existing_flows = _build_biosphere_lookup(biosphere_db, biosphere_lookup)
    techno_lookup = _build_techno_lookup(techno_db, techno_lookup)

    # STEP 1: collect unique flows needed
    missing_indicators: list[dict] = []
    unique_flows: dict[tuple, dict] = {}

    if not precreated_flows:
        for activity in data:
            iso2_here = activity.get("ISO2") or "GLO"
            for exch in activity.get("Averaged_Exchanges", []) or []:
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
                        "sense_kind": "social",
                    }

        if missing_indicators:
            raise ValueError(
                f"{len(missing_indicators)} social exchanges are missing indicator names"
            )

    # STEP 2: create missing flows
    failed_flows = (
        _create_missing_flows(
            unique_flows, existing_flows, biosphere_db, biosphere_lookup, "social"
        )
        if not precreated_flows and unique_flows
        else []
    )
    if failed_flows:
        raise RuntimeError(
            f"Failed to create {len(failed_flows)} social flows: " + "; ".join(failed_flows[:5])
        )

    # STEP 3: map exchanges -> flows
    exch_to_flow: dict[tuple, Any] = {}
    not_found: list[tuple] = []

    for activity in data:
        iso2_here = activity.get("ISO2") or "GLO"
        for exch in activity.get("Averaged_Exchanges", []) or []:
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

    if not_found:
        raise RuntimeError(
            f"{len(not_found)} required social flows are missing from "
            f"{biosphere_db_name!r}; examples: {not_found[:5]}"
        )

    # STEP 4: group by activity and attach exchanges
    grouped: dict[str, list[dict]] = {}
    for activity in data:
        raw_code = activity.get("code") or activity.get("Activity code")
        code = norm_code(raw_code)
        if code:
            grouped.setdefault(code, []).append(activity)

    def _resolve_social(rec):
        iso2_here = rec.get("ISO2") or "GLO"
        results = []
        for exch in rec.get("Averaged_Exchanges", []) or []:
            ind_name, unit, cat, subcat, exch_type, val = _gx(exch)
            if not ind_name:
                ind_name = "Unnamed indicator"
            try:
                amount = float(val)
            except Exception:
                continue
            flow = exch_to_flow.get((ind_name, exch_type, unit, cat, subcat, iso2_here))
            results.append((flow, amount))
        return results

    # Flatten: one record per (activity, exchange) for _attach_exchanges
    flat_grouped: dict[str, list[dict]] = {}
    for code, rows in grouped.items():
        flat_records = []
        for rec in rows:
            for flow, amount in _resolve_social(rec):
                flat_records.append({"_flow": flow, "_amount": amount})
        flat_grouped[code] = flat_records

    social_flow_keys = {
        tuple(flow.key) for flow in biosphere_lookup.values() if flow.get("sense_kind") == "social"
    }
    added, updated, removed = _attach_exchanges(
        flat_grouped,
        techno_lookup,
        resolve_flow=lambda rec: (rec["_flow"], rec["_amount"]),
        managed_flow=lambda flow: flow.key[0] == biosphere_db_name
        and flow.get("sense_kind") == "social",
        managed_flow_keys=social_flow_keys,
        update_tol=update_tol,
    )

    return {
        "added": added,
        "updated": updated,
        "removed": removed,
        "missing": len(not_found),
        "missing_indicators": len(missing_indicators),
        "failed_flows": len(failed_flows),
    }


def _get_isic2_from_activity(act: Any) -> str | None:
    """Extract 2-digit ISIC code from a BW Activity proxy's classifications."""
    for cls in act.get("classifications") or []:
        if isinstance(cls, (list, tuple)) and len(cls) >= 2 and "ISIC" in str(cls[0]):
            return extract_isic2(str(cls[1]))
    return None


def add_working_hours_flows_and_exchanges(
    data: pd.DataFrame | list[dict[str, Any]],
    database: str,
    biosphere_db_name: str,
    *,
    biosphere_lookup: dict[tuple, Any] | None = None,
    techno_lookup: dict[str, Any] | None = None,
    precreated_flows: bool = False,
    update_tol: float = 0.0,
    add_isic_flows: bool = False,
) -> dict[str, Any]:
    """
    Create and attach country-resolved working hours biosphere flows.

    Creates working hours flows for each ISO2 region and attaches them
    to technosphere activities.

    When ``add_isic_flows=True``, also creates ISIC-specific biosphere flows
    (``"working hours, isic{XX}, {iso2}"``) and attaches parallel exchanges
    for each activity that has an ISIC classification.

    Args:
        data: Activity data with Working_hours values (DataFrame or list)
        database: Name of the technosphere database to attach flows to
        biosphere_db_name: Name of the sociosphere database
        biosphere_lookup: Optional pre-built lookup dict for reuse
        techno_lookup: Optional pre-built lookup dict for reuse
        precreated_flows: Skip flow creation if flows already exist
        update_tol: Minimum difference to trigger exchange update
        add_isic_flows: If True, create per-ISIC-2-digit flows and exchanges

    Returns:
        Counts for added, updated, removed, missing, and failed flows.
        When add_isic_flows=True, also includes:
        {"isic_added", "isic_updated", "isic_flows_created", "isic_no_isic_activities"}
    """

    def _pack_dataframe_to_working_records(df: pd.DataFrame) -> list[dict]:
        if df.empty:
            return []

        code_col = (
            "Activity code"
            if "Activity code" in df.columns
            else ("code" if "code" in df.columns else None)
        )
        if code_col is None:
            return []

        records = []
        for _, row in df.iterrows():
            code = row.get(code_col)
            iso2_val = row.get("ISO2") or "GLO"
            wh_val = row.get("Working_hours")
            records.append(
                {
                    "Activity code": norm_code(code),
                    "ISO2": iso2_val,
                    "Working_hours": wh_val,
                }
            )
        return records

    # Accept DataFrame or list
    if isinstance(data, pd.DataFrame):
        data = _pack_dataframe_to_working_records(data)

    # Guards & setup
    if not data or not isinstance(data, list):
        logger.info("No working-hours activity data supplied; no exchanges were written")
        return {"added": 0, "updated": 0, "removed": 0, "missing": 0, "failed_flows": 0}

    for record in data:
        if record.get("Working_hours") is not None:
            _validate_amount(record["Working_hours"], "Working_hours")

    techno_db = Database(database)
    biosphere_db = Database(biosphere_db_name)

    # Build lookups
    biosphere_lookup, existing_flows = _build_biosphere_lookup(biosphere_db, biosphere_lookup)
    techno_lookup = _build_techno_lookup(techno_db, techno_lookup)

    # STEP 1: collect unique working-hours flows needed
    wh_unit = "working hour"
    wh_categories = ("Social", "Working hours")

    def _has_positive_working_hours(record: dict[str, Any]) -> bool:
        raw_value = record.get("Working_hours")
        if raw_value is None:
            return False
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            return False
        return bool(np.isfinite(value) and value > 0)

    iso2_set = {rec.get("ISO2") or "GLO" for rec in data if _has_positive_working_hours(rec)}

    unique_flows: dict[tuple, dict] = {}
    if not precreated_flows:
        for iso2_here in iso2_set:
            flow_name = f"working hours, {iso2_here}"
            key = (flow_name, wh_unit, wh_categories)
            unique_flows[key] = {
                "name": flow_name,
                "unit": wh_unit,
                "categories": wh_categories,
                "comment": "Total working hours calculated by SENSE",
                "sense_kind": "working_hours",
            }

    # STEP 2: create missing flows
    failed_flows = (
        _create_missing_flows(
            unique_flows, existing_flows, biosphere_db, biosphere_lookup, "working_hours"
        )
        if not precreated_flows and unique_flows
        else []
    )
    if failed_flows:
        raise RuntimeError(
            f"Failed to create {len(failed_flows)} working-hours flows: "
            + "; ".join(failed_flows[:5])
        )

    # STEP 3: map ISO2 -> flows
    iso2_to_flow: dict[str, Any] = {}
    not_found: list[tuple] = []

    for iso2_here in iso2_set:
        flow_name = f"working hours, {iso2_here}"
        key_flow = (flow_name, wh_unit, wh_categories)
        flow = biosphere_lookup.get(key_flow)
        if flow:
            iso2_to_flow[iso2_here] = flow
        else:
            not_found.append(key_flow)

    if not_found:
        raise RuntimeError(
            f"{len(not_found)} required working-hours flows are missing from "
            f"{biosphere_db_name!r}; examples: {not_found[:5]}"
        )

    # STEP 4: group by activity and attach exchanges
    grouped: dict[str, list[dict]] = {}
    for rec in data:
        raw_code = rec.get("Activity code") or rec.get("code")
        code = norm_code(raw_code)
        if code:
            grouped.setdefault(code, []).append(rec)

    def _resolve_wh(rec):
        iso2_here = rec.get("ISO2") or "GLO"
        flow = iso2_to_flow.get(iso2_here)
        try:
            amount = float(rec.get("Working_hours"))
        except Exception:
            return None, None
        return flow, amount

    added, updated, removed = _attach_exchanges(
        grouped,
        techno_lookup,
        resolve_flow=_resolve_wh,
        managed_flow=lambda flow: (
            flow.key[0] == biosphere_db_name and flow.get("sense_kind") == "working_hours"
        ),
        managed_flow_keys={
            tuple(flow.key)
            for flow in biosphere_lookup.values()
            if flow.get("sense_kind") == "working_hours"
        },
        update_tol=update_tol,
    )

    result: dict[str, Any] = {
        "added": added,
        "updated": updated,
        "removed": removed,
        "missing": len(not_found),
        "failed_flows": len(failed_flows),
    }

    # ── ISIC-specific flows and exchanges ────────────────────────────
    if add_isic_flows:
        # Pre-check: audit ISIC coverage across activities with working hours
        activities_with_wh = set()
        for rec in data:
            wh = rec.get("Working_hours")
            if wh is None:
                continue
            try:
                float(wh)
            except (TypeError, ValueError):
                continue
            code = norm_code(rec.get("Activity code") or rec.get("code"))
            if code:
                activities_with_wh.add(code)

        no_isic_activities: list[tuple[str, str]] = []
        code_to_isic2: dict[str, str] = {}
        for code in activities_with_wh:
            act = techno_lookup.get(code)
            if not act:
                no_isic_activities.append((code, "<not in database>"))
                continue
            isic2 = _get_isic2_from_activity(act)
            if isic2:
                code_to_isic2[code] = isic2
            else:
                no_isic_activities.append((code, str(act.get("name", ""))))

        # Collect unique (isic2, iso2) pairs
        isic_iso2_pairs: set[tuple[str, str]] = set()
        for rec in data:
            code = norm_code(rec.get("Activity code") or rec.get("code"))
            if not code:
                continue
            isic2 = code_to_isic2.get(code)
            if not isic2:
                continue
            iso2_here = rec.get("ISO2") or "GLO"
            isic_iso2_pairs.add((isic2, iso2_here))

        # Create ISIC-specific flows
        isic_unique_flows: dict[tuple, dict] = {}
        if not precreated_flows:
            for isic2, iso2_here in isic_iso2_pairs:
                flow_name = f"working hours, isic{isic2}, {iso2_here}"
                key = (flow_name, wh_unit, wh_categories)
                isic_unique_flows[key] = {
                    "name": flow_name,
                    "unit": wh_unit,
                    "categories": wh_categories,
                    "comment": f"Working hours for ISIC 2-digit division {isic2}",
                    "sense_kind": "working_hours_isic",
                }

        isic_failed = (
            _create_missing_flows(
                isic_unique_flows,
                existing_flows,
                biosphere_db,
                biosphere_lookup,
                "working_hours_isic",
            )
            if not precreated_flows and isic_unique_flows
            else []
        )
        if isic_failed:
            raise RuntimeError(
                f"Failed to create {len(isic_failed)} ISIC working-hours flows: "
                + "; ".join(isic_failed[:5])
            )

        # Build (isic2, iso2) -> flow mapping
        isic_iso2_to_flow: dict[tuple[str, str], Any] = {}
        isic_not_found: list[tuple] = []
        for isic2, iso2_here in isic_iso2_pairs:
            flow_name = f"working hours, isic{isic2}, {iso2_here}"
            key_flow = (flow_name, wh_unit, wh_categories)
            flow = biosphere_lookup.get(key_flow)
            if flow:
                isic_iso2_to_flow[(isic2, iso2_here)] = flow
            else:
                isic_not_found.append(key_flow)

        if isic_not_found:
            raise RuntimeError(
                f"{len(isic_not_found)} required ISIC working-hours flows are missing "
                f"from {biosphere_db_name!r}; examples: {isic_not_found[:5]}"
            )

        # Attach ISIC-specific exchanges (second pass over same grouped data)
        def _resolve_wh_isic(rec):
            code = norm_code(rec.get("Activity code") or rec.get("code"))
            if not code:
                return None, None
            isic2 = code_to_isic2.get(code)
            if not isic2:
                return None, None
            iso2_here = rec.get("ISO2") or "GLO"
            flow = isic_iso2_to_flow.get((isic2, iso2_here))
            try:
                amount = float(rec.get("Working_hours"))
            except Exception:
                return None, None
            return flow, amount

        isic_added, isic_updated, isic_removed = _attach_exchanges(
            grouped,
            techno_lookup,
            resolve_flow=_resolve_wh_isic,
            managed_flow=lambda flow: (
                flow.key[0] == biosphere_db_name and flow.get("sense_kind") == "working_hours_isic"
            ),
            managed_flow_keys={
                tuple(flow.key)
                for flow in biosphere_lookup.values()
                if flow.get("sense_kind") == "working_hours_isic"
            },
            update_tol=update_tol,
        )

        result.update(
            {
                "isic_added": isic_added,
                "isic_updated": isic_updated,
                "isic_removed": isic_removed,
                "isic_failed_flows": len(isic_failed),
                "isic_flows_created": len(isic_unique_flows),
                "isic_no_isic_activities": len(no_isic_activities),
                "isic_no_isic_details": [
                    {"activity_code": code, "activity_name": name}
                    for code, name in no_isic_activities
                ],
            }
        )

    return result
