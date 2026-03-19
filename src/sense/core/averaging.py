"""
Sector averaging with price weighting for SENSE.

This module provides functions for averaging social indicators across
matched PSILCA sectors, applying risk-level weights and price scaling.
"""

from __future__ import annotations

from collections import defaultdict, Counter
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def _geometric_mean_strict(values: List[float], context: str = "") -> float:
    """
    Compute robust geometric mean.
    
    - Returns NaN if values is empty after filtering
    - Returns NaN if any negatives
    - Drops zeros (treated as missing data)
    
    Args:
        values: List of values to average
        context: Context string for warning messages
        
    Returns:
        Geometric mean of positive values, or NaN if invalid
    """
    arr = np.asarray(values, dtype=float)
    ctx = f" ({context})" if context else ""

    arr = arr[~np.isnan(arr)]

    if arr.size == 0:
        return float("nan")

    if np.any(arr < 0):
        print(f"⚠️ Geometric mean received negative values{ctx}. Returning NaN.")
        return float("nan")

    if np.any(arr == 0):
        print(f"⚠️ Geometric mean received zero values{ctx}. Treating as missing (NaN).")
        arr = arr[arr > 0]
        if arr.size == 0:
            return float("nan")

    return float(np.exp(np.log(arr).mean()))

def _arithmetic_mean_strict(values: List[float], context: str = "") -> float:
    """
    Compute robust arithmetic mean.
    - Returns NaN if values is empty after filtering
    - Returns NaN if any negatives
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    if np.any(arr < 0):
        print(f"[WARN] Negative value(s) encountered in arithmetic mean: {context}")
        return np.nan
    return float(arr.mean())

def _aggregate_strict(values: List[float], method: str, context: str = "") -> float:
    method = (method or "").strip().lower()
    if method == "geometric":
        return _geometric_mean_strict(values, context=context)
    if method == "arithmetic":
        return _arithmetic_mean_strict(values, context=context)
    raise ValueError(f"Unknown averaging method: {method!r} (use 'geometric' or 'arithmetic')")

def average_sector_mappings_w_prices(
    enriched_df: pd.DataFrame,
    level_weights: Dict[str, float],
    tiny_threshold: float = 1e-9,
    rounding: Optional[int] = None,
    mean_method: str = "geometric",
) -> List[Dict[str, Any]]:
    """
    Compute final 'medium risk/opportunity hours' per activity.
    
    The averaging process:
    1) Risk-weighted working hours: r_ik = hours_per_USD * price * c(level)
    2) Geometric or arithmetic mean across PSILCA sectors (per country)
    3) Geometric or arithmetic across PSILCA countries (per activity)
    
    Also computes per-activity 'Working_hours' as arithmetic mean of
    unweighted working hours across countries/sectors.
    
    Args:
        enriched_df: DataFrame with matched sectors and PSILCA exchanges
        level_weights: Mapping from risk level strings to numeric weights
        tiny_threshold: Threshold for counting "tiny" values in diagnostics
        rounding: Number of decimal digits to round to (None for no rounding)
        
    Returns:
        List of activity dictionaries with 'Averaged_Exchanges' and 'Working_hours'
        
    Examples:
        >>> from sense.config import SenseConfig
        >>> cfg = SenseConfig()
        >>> results = average_sector_mappings_w_prices(
        ...     enriched_df=enriched_df,
        ...     level_weights=cfg.level_weights,
        ... )
    """
    final_activities: List[Dict[str, Any]] = []
    type_map: Dict[str, set] = defaultdict(set)
    count_map: Dict[str, int] = defaultdict(int)

    # Price handling
    enriched_df = enriched_df.copy()
    enriched_df.columns = enriched_df.columns.str.strip()
    if "Price" in enriched_df.columns:
        enriched_df["Price"] = pd.to_numeric(enriched_df["Price"], errors="coerce")
        price_available = enriched_df["Price"].notna().any()
    else:
        price_available = False
        enriched_df["Price"] = np.nan

    if not price_available:
        print("⚠️ No price information found. Proceeding without price weighting.")

    # Diagnostics containers
    per_activity_causes: Dict[str, Counter] = defaultdict(Counter)
    per_activity_meta: Dict[str, Dict] = {}
    global_cause_counts: Counter = Counter()
    global_examples: List[Dict] = []
    MAX_ACTIVITY_EXAMPLES = 12
    MAX_GLOBAL_EXAMPLES = 20

    diag_empty_country_sets = 0
    diag_missing_indicator_country = 0
    diag_missing_indicator_activity = 0
    diag_examples_missing: List[Dict] = []
    tiny_count = 0

    def _record_missing(activity_code, activity_name, country, sector, indicator,
                        qual_level, raw_wh, price, cause, extra=None):
        per_activity_causes[activity_code][cause] += 1
        per_activity_meta[activity_code] = {"name": activity_name}
        global_cause_counts[cause] += 1
        if len(global_examples) < MAX_GLOBAL_EXAMPLES:
            global_examples.append({
                "activity_code": activity_code,
                "activity_name": activity_name,
                "country": country,
                "sector": sector,
                "indicator": indicator,
                "risk_level": qual_level,
                "raw_wh": raw_wh,
                "price": (float(price) if (price is not None and np.isfinite(price)) else np.nan),
                "cause": cause,
                **(extra or {})
            })

    grouped_by_activity = enriched_df.groupby(["Activity code"], sort=False)

    for activity_code, group in grouped_by_activity:
        activity_meta = group.iloc[0]
        psilca_countries_used: set = set()
        psilca_sectors_used: Dict[str, set] = defaultdict(set)
        country_level_aggregates: List[Dict] = []
        wh_country_values: List[float] = []

        # Activity price
        price = pd.to_numeric(activity_meta.get("Price", np.nan), errors="coerce")
        has_price = np.isfinite(price)
        price_positive = bool(has_price and price > 0)
        use_price = price_available and price_positive

        if not has_price:
            _record_missing(
                activity_code, activity_meta.get("Activity name"), country=None, sector=None,
                indicator=None, qual_level=None, raw_wh=None, price=price,
                cause="price_missing_or_nan"
            )
        elif not price_positive:
            _record_missing(
                activity_code, activity_meta.get("Activity name"), country=None, sector=None,
                indicator=None, qual_level=None, raw_wh=None, price=price,
                cause="price_non_positive"
            )

        # Countries mapped to this activity
        country_groups = list(group.groupby("PSILCA_country", sort=False))
        if not country_groups:
            diag_empty_country_sets += 1

        for psilca_country, sub_df in country_groups:
            psilca_countries_used.add(psilca_country)
            exchange_tracker: Dict[tuple, List[float]] = defaultdict(list)
            wh_values_this_country: List[float] = []

            for _, row in sub_df.iterrows():
                matched_sector = row.get("Matched_sector")
                psilca_sectors_used[psilca_country].add(matched_sector)

                sector_exchanges = row.get("PSILCA_Exchanges")
                if not sector_exchanges or not isinstance(sector_exchanges, list):
                    _record_missing(
                        activity_code, activity_meta.get("Activity name"),
                        country=psilca_country, sector=matched_sector,
                        indicator=None, qual_level=None, raw_wh=None, price=price,
                        cause="no_exchanges_for_matched_sector"
                    )
                    continue

                for exch in sector_exchanges:
                    indicator = (exch.get("indicator") or "").strip()
                    qual_level_raw = (exch.get("risk level") or "").strip()
                    qual_level = qual_level_raw.lower() if qual_level_raw else ""
                    exch_type = (exch.get("type") or "risk").strip().lower()
                    unit = exch.get("unit", "h")
                    stakeholder_group = exch.get("category", "Unknown")
                    subcategory = exch.get("subcategory", "Unknown")

                    if not indicator:
                        _record_missing(
                            activity_code, activity_meta.get("Activity name"),
                            country=psilca_country, sector=matched_sector,
                            indicator=None, qual_level=qual_level_raw, raw_wh=None, price=price,
                            cause="indicator_missing"
                        )
                        continue

                    raw_wh_val = exch.get("meanValue", np.nan)
                    try:
                        raw_wh = float(raw_wh_val)
                    except Exception:
                        raw_wh = np.nan

                    if not np.isfinite(raw_wh):
                        _record_missing(
                            activity_code, activity_meta.get("Activity name"),
                            country=psilca_country, sector=matched_sector,
                            indicator=indicator, qual_level=qual_level_raw, raw_wh=raw_wh_val, price=price,
                            cause="working_hours_missing_or_nan"
                        )
                        continue
                    if raw_wh <= 0:
                        _record_missing(
                            activity_code, activity_meta.get("Activity name"),
                            country=psilca_country, sector=matched_sector,
                            indicator=indicator, qual_level=qual_level_raw, raw_wh=raw_wh, price=price,
                            cause="working_hours_non_positive"
                        )
                        continue

                    # Risk level / weight mapping
                    qual_missing = (qual_level in ("", "?", "unknown", None))
                    weight = level_weights.get(qual_level) if not qual_missing else None
                    if qual_missing:
                        _record_missing(
                            activity_code, activity_meta.get("Activity name"),
                            country=psilca_country, sector=matched_sector,
                            indicator=indicator, qual_level=qual_level_raw, raw_wh=raw_wh, price=price,
                            cause="risk_level_missing_or_unknown"
                        )
                        weight = float(level_weights.get("no data", 1.0))
                    elif weight is None:
                        _record_missing(
                            activity_code, activity_meta.get("Activity name"),
                            country=psilca_country, sector=matched_sector,
                            indicator=indicator, qual_level=qual_level_raw, raw_wh=raw_wh, price=price,
                            cause="risk_level_unmapped"
                        )
                        weight = float(level_weights.get("no data", 1.0))
                    else:
                        weight = float(weight)

                    if weight <= 0:
                        _record_missing(
                            activity_code, activity_meta.get("Activity name"),
                            country=psilca_country, sector=matched_sector,
                            indicator=indicator, qual_level=qual_level_raw, raw_wh=raw_wh, price=price,
                            cause="risk_weight_non_positive"
                        )
                        continue

                    price_factor = float(price) if use_price else 1.0

                    # Working-hours contribution (no risk weighting)
                    wh_val = raw_wh * price_factor
                    if np.isfinite(wh_val) and wh_val > 0:
                        wh_values_this_country.append(wh_val)

                    # Weighted value for social risk/opportunity
                    weighted = raw_wh * weight * price_factor
                    if not np.isfinite(weighted) or weighted <= 0:
                        _record_missing(
                            activity_code, activity_meta.get("Activity name"),
                            country=psilca_country, sector=matched_sector,
                            indicator=indicator, qual_level=qual_level_raw, raw_wh=raw_wh, price=price,
                            cause="weighted_non_positive_or_nan",
                            extra={"weight": weight, "price_factor": price_factor}
                        )
                        continue

                    key = (indicator, unit, exch_type, stakeholder_group, subcategory)
                    type_map[indicator].add(exch_type)
                    count_map[indicator] += 1
                    exchange_tracker[key].append(weighted)

            # Mean within country (geometric or arithmetic, depending on config)
            country_avg: Dict[tuple, float] = {}
            for key, vals in exchange_tracker.items():
                avg_val = _aggregate_strict(
                    vals,
                    method=mean_method,
                    context=f"activity={activity_code}, country={psilca_country}, key={key}"
                )
                if np.isfinite(avg_val) and avg_val > 0:
                    country_avg[key] = avg_val
            country_level_aggregates.append(country_avg)

            # Arithmetic mean of working-hours in this country (keep arithmetic)
            if wh_values_this_country:
                country_mean_wh = float(np.mean(wh_values_this_country))
                if np.isfinite(country_mean_wh) and country_mean_wh > 0:
                    wh_country_values.append(country_mean_wh)

        # Combine across countries
        combined: Dict[tuple, List[float]] = defaultdict(list)
        for cdict in country_level_aggregates:
            for key, val in cdict.items():
                if np.isfinite(val) and val > 0:
                    combined[key].append(val)

        final_exchanges: List[Dict[str, Any]] = []
        for (name, unit, exch_type, stakeholder_group, subcategory), vals in combined.items():
            vals = [v for v in vals if np.isfinite(v) and v > 0]
            if len(vals) == 0:
                diag_missing_indicator_country += 1
                if len(diag_examples_missing) < 15:
                    diag_examples_missing.append({
                        "activity_code": activity_code,
                        "activity_name": activity_meta.get("Activity name"),
                        "indicator": name,
                        "type": exch_type,
                        "reason": "empty_after_country_aggregation",
                    })
                continue

            avg_val = _aggregate_strict(
                vals,
                method=mean_method,
                context=f"activity={activity_code}, countries={len(vals)}, key={(name, unit, exch_type, stakeholder_group, subcategory)}"
            )
            if not np.isfinite(avg_val) or avg_val <= 0:
                diag_missing_indicator_activity += 1
                if len(diag_examples_missing) < 15:
                    diag_examples_missing.append({
                        "activity_code": activity_code,
                        "activity_name": activity_meta.get("Activity name"),
                        "indicator": name,
                        "type": exch_type,
                        "reason": "geometric_mean_nan_or_nonpositive",
                    })
                continue

            # Optional rounding
            if isinstance(rounding, int):
                rounded = round(avg_val, rounding)
                value_out = avg_val if (rounded == 0.0 and avg_val > 0.0) else rounded
            else:
                value_out = avg_val

            if value_out < tiny_threshold:
                tiny_count += 1

            # Type mismatch safeguard
            if exch_type == "risk" and "opportunity" in type_map[name] and count_map[name] > 1:
                exch_type = "opportunity"
                out_unit = "med opportunity hours"
            else:
                out_unit = "med opportunity hours" if exch_type == "opportunity" else "med risk hours"

            final_exchanges.append({
                "indicator": name,
                "type": exch_type,
                "category": stakeholder_group,
                "subcategory": subcategory,
                "unit": out_unit,
                "meanValue": float(value_out),
                "origin_matches": len(vals),
            })

        # Compute working-hours aggregate
        working_hours_value = None
        if wh_country_values:
            wh_vals = [v for v in wh_country_values if np.isfinite(v) and v > 0]
            if wh_vals:
                working_hours_value = float(np.mean(wh_vals))

        # Store activity record
        final_activities.append({
            "Activity code": activity_code,
            "Activity name": activity_meta.get("Activity name"),
            "Reference product": activity_meta.get("Reference product"),
            "ISIC": activity_meta.get("ISIC"),
            "Unit": activity_meta.get("Unit"),
            "ISO2": activity_meta.get("ISO2"),
            "Price": float(price) if use_price else "N/A",
            "PSILCA_countries_used": sorted(list(psilca_countries_used)),
            "PSILCA_sectors_used": {k: sorted(list(v)) for k, v in psilca_sectors_used.items()},
            "Averaged_Exchanges": final_exchanges,
            "Working_hours": working_hours_value,
        })

    # Diagnostics output
    if per_activity_causes:
        print("\n🔎 Missing input diagnostics (per activity; limited to a few examples):")
        shown = 0
        for act_code, causes in per_activity_causes.items():
            name = per_activity_meta.get(act_code, {}).get("name", "")
            cause_str = ", ".join(f"{k}={v}" for k, v in causes.most_common())
            print(f" - {act_code} | {name} → {cause_str}")
            shown += 1
            if shown >= MAX_ACTIVITY_EXAMPLES:
                print(f"   … and {len(per_activity_causes) - shown} more activities with missing inputs.")
                break

    if global_cause_counts:
        print("\n📊 Missing input summary (global):")
        for cause, n in global_cause_counts.most_common():
            print(f" - {cause}: {n}")

    if tiny_count:
        print(f"ℹ️ {tiny_threshold:g}-tiny outputs: {tiny_count} (positive but very small).")

    return final_activities
