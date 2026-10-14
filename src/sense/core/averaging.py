"""Convert and aggregate PSILCA sector assessments for ecoinvent activities."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from sense.utils.helpers import norm_code

IndicatorKey = tuple[str, Any, str, Any, Any]


@dataclass(slots=True)
class _SectorProfile:
    """Reusable numerical representation of one PSILCA country-sector."""

    psilca_totals: dict[IndicatorKey, float]
    psilca_counts: Counter[IndicatorKey]
    override_totals: dict[IndicatorKey, float]
    override_counts: Counter[IndicatorKey]
    psilca_diagnostics: Counter[str]
    override_diagnostics: Counter[str]
    working_hours: float | None
    inconsistent_hours: tuple[float, ...]


@dataclass(slots=True)
class _ActivityState:
    """Metadata and audit state retained while activities are aggregated."""

    meta: dict[str, Any]
    price: float
    override_hours: float | None
    direct_hours: float | None
    conversion_factor: float
    country_weights: dict[str, float]
    weighting_methods: list[str]
    reference_year: int | None
    data_years: dict[str, int]
    countries: set[Any]
    sectors: dict[Any, set[Any]]
    processes: dict[Any, set[str]]
    diagnostics: Counter[str]


@dataclass(frozen=True, slots=True)
class _InputRow:
    """Only the enriched columns required by aggregation."""

    activity_code: Any
    activity_name: Any
    reference_product: Any
    isic: Any
    unit: Any
    iso2: Any
    price: Any
    country: Any
    sector: Any
    process: Any
    exchanges: Any
    country_weight: Any
    weighting_method: Any
    weight_reference_year: Any
    weight_data_year: Any


@dataclass(slots=True)
class _CrossCountryAggregation:
    """Weighted and fallback accumulators for one indicator matrix."""

    weighted: np.ndarray
    weight_sums: np.ndarray
    unweighted: np.ndarray
    country_counts: np.ndarray

    @classmethod
    def zeros(cls, shape: tuple[int, int]) -> _CrossCountryAggregation:
        return cls(
            weighted=np.zeros(shape, dtype=np.float64),
            weight_sums=np.zeros(shape, dtype=np.float64),
            unweighted=np.zeros(shape, dtype=np.float64),
            country_counts=np.zeros(shape, dtype=np.float64),
        )

    def add(
        self,
        activity_ids: np.ndarray,
        country_means: np.ndarray,
        valid: np.ndarray,
        country_weights: np.ndarray,
    ) -> None:
        weighted_values = country_means * country_weights[:, None] * valid
        self.weighted[activity_ids] += weighted_values
        self.weight_sums[activity_ids] += country_weights[:, None] * valid
        self.unweighted[activity_ids] += country_means * valid
        self.country_counts[activity_ids] += valid
        if (
            np.isinf(self.weighted[activity_ids]).any()
            or np.isinf(self.unweighted[activity_ids]).any()
        ):
            raise ValueError(
                "Non-finite computed country aggregation; reduce overflowing input values"
            )

    def finish(self, *, exponentiate: bool) -> np.ndarray:
        result = np.full(self.weighted.shape, np.nan, dtype=np.float64)
        weighted_mask = self.weight_sums > 0
        result[weighted_mask] = self.weighted[weighted_mask] / self.weight_sums[weighted_mask]
        fallback_mask = ~weighted_mask & (self.country_counts > 0)
        result[fallback_mask] = self.unweighted[fallback_mask] / self.country_counts[fallback_mask]
        if exponentiate:
            with np.errstate(over="ignore"):
                np.exp(result, out=result, where=np.isfinite(result))
        if np.isinf(result).any():
            raise ValueError("Non-finite computed aggregation result")
        return result


def _build_sector_profile(
    exchanges: Any,
    level_weights: dict[str, float],
    *,
    geometric: bool,
) -> _SectorProfile:
    """Parse one sector once into additive totals used by matrix aggregation."""

    psilca_totals: dict[IndicatorKey, float] = defaultdict(float)
    psilca_counts: Counter[IndicatorKey] = Counter()
    override_totals: dict[IndicatorKey, float] = defaultdict(float)
    override_counts: Counter[IndicatorKey] = Counter()
    psilca_diagnostics: Counter[str] = Counter()
    override_diagnostics: Counter[str] = Counter()

    if not exchanges or not isinstance(exchanges, list):
        return _SectorProfile(
            {},
            Counter(),
            {},
            Counter(),
            psilca_diagnostics,
            override_diagnostics,
            None,
            (),
        )

    sector_hours: list[float] = []
    for exchange in exchanges:
        try:
            candidate = float(exchange.get("meanValue", np.nan))
        except (TypeError, ValueError):
            continue
        if math.isfinite(candidate) and candidate > 0:
            sector_hours.append(candidate)

    working_hours = sector_hours[0] if sector_hours else None
    inconsistent_hours: tuple[float, ...] = ()
    if working_hours is not None and not np.allclose(
        sector_hours,
        working_hours,
        rtol=1e-9,
        atol=0.0,
    ):
        inconsistent_hours = tuple(sorted(set(sector_hours))[:10])

    no_data_weight = float(level_weights.get("no data", 1.0))

    for exchange in exchanges:
        indicator = (exchange.get("indicator") or "").strip()
        exchange_type = (exchange.get("type") or "risk").strip().lower()
        if exchange_type not in {"risk", "opportunity"}:
            raise ValueError(
                f"Unsupported social exchange type {exchange_type!r} "
                f"for indicator {indicator!r}"
            )
        if not indicator:
            psilca_diagnostics["indicator_missing"] += 1
            override_diagnostics["indicator_missing"] += 1
            continue

        key: IndicatorKey = (
            indicator,
            exchange.get("unit", "h"),
            exchange_type,
            exchange.get("category", "Unknown"),
            exchange.get("subcategory", "Unknown"),
        )
        level_raw = (exchange.get("risk level") or "").strip()
        level = level_raw.lower() if level_raw else ""
        level_missing = level in {"", "?", "unknown"}
        configured_weight = level_weights.get(level) if not level_missing else None
        if level_missing:
            weight = no_data_weight
            weight_diagnostic = "risk_level_missing_or_unknown"
        elif configured_weight is None:
            weight = no_data_weight
            weight_diagnostic = "risk_level_unmapped"
        else:
            weight = float(configured_weight)
            weight_diagnostic = None

        # Productivity overrides replace PSILCA hours, so only the qualitative
        # level multiplier enters their reusable sector profile.
        if weight_diagnostic is not None:
            override_diagnostics[weight_diagnostic] += 1
        if weight <= 0:
            override_diagnostics["risk_weight_non_positive"] += 1
        elif not math.isfinite(weight):
            override_diagnostics["weighted_non_positive_or_nan"] += 1
        else:
            transformed = math.log(weight) if geometric else weight
            override_totals[key] += transformed
            override_counts[key] += 1

        try:
            raw_hours = float(exchange.get("meanValue", np.nan))
        except (TypeError, ValueError):
            raw_hours = float("nan")
        if not math.isfinite(raw_hours):
            psilca_diagnostics["working_hours_missing_or_nan"] += 1
            continue
        if raw_hours <= 0:
            psilca_diagnostics[
                "working_hours_negative" if raw_hours < 0 else "working_hours_non_positive"
            ] += 1
            continue
        if weight_diagnostic is not None:
            psilca_diagnostics[weight_diagnostic] += 1
        if weight <= 0:
            psilca_diagnostics["risk_weight_non_positive"] += 1
            continue
        weighted = raw_hours * weight
        if math.isinf(weighted):
            raise ValueError(f"Non-finite computed risk-weighted hours for {indicator!r}")
        if not math.isfinite(weighted) or weighted <= 0:
            psilca_diagnostics["weighted_non_positive_or_nan"] += 1
            continue
        transformed = math.log(weighted) if geometric else weighted
        psilca_totals[key] += transformed
        psilca_counts[key] += 1

    return _SectorProfile(
        dict(psilca_totals),
        psilca_counts,
        dict(override_totals),
        override_counts,
        psilca_diagnostics,
        override_diagnostics,
        working_hours,
        inconsistent_hours,
    )


def average_sector_mappings_w_prices(
    enriched_df: pd.DataFrame,
    level_weights: dict[str, float],
    rounding: int | None = None,
    mean_method: str = "geometric",
    hours_override: dict[str, float] | None = None,
    psilca_price_factor: float = 1.0,
    productivity_price_factor: float | None = None,
    direct_hours_override: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Compute medium-equivalent risk or opportunity hours per activity.

    The averaging process:
    1) Risk-weighted working hours: r_ik = hours_per_USD * price * c(level)
    2) Geometric or arithmetic mean across PSILCA sectors (per country)
    3) Configured country-weighted geometric or arithmetic mean across PSILCA
       countries (per activity; equal weights when no Country_weight column is
       present)

    Also computes per-activity 'Working_hours' as arithmetic mean of
    working hours across sectors, followed by a weighted arithmetic mean
    across countries.

    When hours_override is provided, activities with an entry in the dict
    use the override value (hours per USD from an external productivity
    baseline) instead of PSILCA's meanValue for both risk scaling and
    working hours. PSILCA risk levels are still used for weighting.
    Activities without an override fall back to PSILCA hours.

    Args:
        enriched_df: Matched sectors with PSILCA exchanges and positive prices.
        level_weights: Numeric multiplier for each qualitative assessment level.
        rounding: Decimal places for results; ``None`` preserves full precision.
        mean_method: ``"geometric"`` or ``"arithmetic"`` for social flows.
        hours_override: Optional dict mapping Activity code to hours-per-USD
            from an external productivity baseline. When present, replaces
            PSILCA's meanValue for that activity.
        direct_hours_override: Direct hours per physical reference unit. Bypasses
            prices, monetary conversion and labour intensities, retaining risk
            factors and geographic/sector aggregation. Never a monetary seed.
        psilca_price_factor: Multiplier converting activity prices into the
            monetary basis of PSILCA worker-hour intensities.
        productivity_price_factor: Multiplier converting activity prices into
            the external productivity baseline's monetary basis. Defaults to
            1.0 when no external conversion is configured.

    Returns:
        Activity records containing ``Averaged_Exchanges`` and ``Working_hours``.
    """

    def _validated_factor(value: float, name: str) -> float:
        try:
            factor = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a positive finite number") from exc
        if not math.isfinite(factor) or factor <= 0:
            raise ValueError(f"{name} must be a positive finite number")
        return factor

    psilca_price_factor = _validated_factor(psilca_price_factor, "psilca_price_factor")
    direct_hours_override = {
        code: _validated_factor(value, f"direct_hours_override[{code!r}]")
        for code, value in (direct_hours_override or {}).items()
    }
    if set(direct_hours_override) & set(hours_override or {}):
        raise ValueError("Direct hours and monetary labour overrides cannot overlap")
    if hours_override is not None:
        hours_override = {
            code: _validated_factor(value, f"hours_override[{code!r}]")
            for code, value in hours_override.items()
        }
    if productivity_price_factor is None:
        productivity_price_factor = 1.0
    else:
        productivity_price_factor = _validated_factor(
            productivity_price_factor, "productivity_price_factor"
        )

    # Price handling. PSILCA intensities are expressed per monetary unit, so
    # calculating an amount per reference product without a positive product
    # price is dimensionally invalid. Upstream extraction normally guarantees
    # this; keep the validation here as the public-function safety boundary.
    enriched_df = enriched_df.copy()
    enriched_df.columns = enriched_df.columns.str.strip()
    if "Price" in enriched_df.columns:
        originally_missing_price = enriched_df["Price"].isna()
        enriched_df["Price"] = pd.to_numeric(enriched_df["Price"], errors="coerce")
    else:
        enriched_df["Price"] = np.nan
        originally_missing_price = pd.Series(True, index=enriched_df.index)

    direct_mask = enriched_df["Activity code"].map(
        lambda code: (norm_code(code) or str(code)) in direct_hours_override
    )
    # Missing monetary prices are permitted only with explicit physical hours.
    enriched_df.loc[direct_mask & originally_missing_price, "Price"] = 0.0
    invalid_price = (
        ~np.isfinite(enriched_df["Price"]) | (enriched_df["Price"] <= 0)
    ) & ~direct_mask
    invalid_price |= ~np.isfinite(enriched_df["Price"]) | (enriched_df["Price"] < 0)
    if invalid_price.any():
        invalid_codes = (
            enriched_df.loc[invalid_price, "Activity code"]
            .astype(str)
            .drop_duplicates()
            .head(10)
            .tolist()
        )
        suffix = "..." if invalid_price.sum() > len(invalid_codes) else ""
        raise ValueError(
            "All activities require a positive finite price before social "
            f"aggregation. Invalid activity codes: {invalid_codes}{suffix}"
        )

    method = (mean_method or "").strip().lower()
    if method not in {"geometric", "arithmetic"}:
        raise ValueError(f"Unknown averaging method: {method!r} (use 'geometric' or 'arithmetic')")
    geometric = method == "geometric"

    column_positions = {name: index for index, name in enumerate(enriched_df.columns)}

    def _optional_value(values: tuple[Any, ...], column: str) -> Any:
        position = column_positions.get(column)
        return values[position] if position is not None else None

    records = [
        _InputRow(
            activity_code=_optional_value(values, "Activity code"),
            activity_name=_optional_value(values, "Activity name"),
            reference_product=_optional_value(values, "Reference product"),
            isic=_optional_value(values, "ISIC"),
            unit=_optional_value(values, "Unit"),
            iso2=_optional_value(values, "ISO2"),
            price=_optional_value(values, "Price"),
            country=_optional_value(values, "PSILCA_country"),
            sector=_optional_value(values, "Matched_sector"),
            process=_optional_value(values, "PSILCA_process"),
            exchanges=_optional_value(values, "PSILCA_Exchanges"),
            country_weight=_optional_value(values, "Country_weight"),
            weighting_method=_optional_value(values, "Country_weighting_method"),
            weight_reference_year=_optional_value(values, "Country_weight_reference_year"),
            weight_data_year=_optional_value(values, "Country_weight_data_year"),
        )
        for values in enriched_df.itertuples(index=False, name=None)
    ]
    activity_groups: dict[Any, list[_InputRow]] = {}
    for row in records:
        activity_code = row.activity_code
        if pd.isna(activity_code):
            continue
        activity_groups.setdefault(activity_code, []).append(row)

    activity_codes = list(activity_groups)
    activity_index = {code: index for index, code in enumerate(activity_codes)}
    activity_states: list[_ActivityState] = []

    for activity_code, group_rows in activity_groups.items():
        first_row = group_rows[0]
        activity_meta = {
            "Activity name": first_row.activity_name,
            "Reference product": first_row.reference_product,
            "ISIC": first_row.isic,
            "Unit": first_row.unit,
            "ISO2": first_row.iso2,
        }
        price = float(pd.to_numeric(first_row.price, errors="coerce"))
        if any(float(row.price) != price for row in group_rows):
            raise ValueError(f"Conflicting prices for activity {activity_code!r}")
        normalized_code = norm_code(activity_code) or str(activity_code)
        direct_hours = direct_hours_override.get(normalized_code)
        override_hours = None
        if hours_override is not None:
            candidate = hours_override.get(normalized_code)
            if candidate is not None and candidate > 0:
                override_hours = float(candidate)
        if direct_hours is not None:
            override_hours = 1.0  # Select the risk-factor-only sector profiles.

        country_rows: dict[Any, list[_InputRow]] = {}
        for row in group_rows:
            country = row.country
            if pd.isna(country):
                continue
            country_rows.setdefault(country, []).append(row)

        country_weight_map: dict[str, float] = {}
        for country, rows in country_rows.items():
            if "Country_weight" not in column_positions:
                continue
            valid_weights = {
                numeric_weight
                for row in rows
                if math.isfinite(
                    numeric_weight := float(pd.to_numeric(row.country_weight, errors="coerce"))
                )
                and numeric_weight >= 0
            }
            if len(valid_weights) > 1:
                raise ValueError(
                    f"Inconsistent country weights for activity {activity_code!r}, "
                    f"country {country!r}"
                )
            if valid_weights:
                country_weight_map[str(country)] = next(iter(valid_weights))

        total_country_weight = sum(country_weight_map.values())
        if total_country_weight > 0:
            country_weight_map = {
                country: weight / total_country_weight
                for country, weight in country_weight_map.items()
            }
        elif country_rows:
            equal_weight = 1.0 / len(country_rows)
            country_weight_map = {str(country): equal_weight for country in country_rows}

        weighting_methods = ["equal"]
        if "Country_weighting_method" in column_positions:
            weighting_methods = sorted(
                {
                    str(method_value)
                    for row in group_rows
                    if pd.notna(method_value := row.weighting_method) and str(method_value)
                }
            ) or ["equal"]

        reference_year = None
        if "Country_weight_reference_year" in column_positions:
            reference_years = {
                numeric_reference_year
                for row in group_rows
                if math.isfinite(
                    numeric_reference_year := float(
                        pd.to_numeric(row.weight_reference_year, errors="coerce")
                    )
                )
            }
            if len(reference_years) == 1:
                reference_year = int(next(iter(reference_years)))

        data_years: dict[str, int] = {}
        if "Country_weight_data_year" in column_positions:
            for country, rows in country_rows.items():
                years = {
                    numeric_data_year
                    for row in rows
                    if math.isfinite(
                        numeric_data_year := float(
                            pd.to_numeric(row.weight_data_year, errors="coerce")
                        )
                    )
                }
                if len(years) == 1:
                    data_years[str(country)] = int(next(iter(years)))

        conversion_factor = (
            productivity_price_factor if override_hours is not None else psilca_price_factor
        )
        if direct_hours is not None:
            conversion_factor = 1.0
        activity_states.append(
            _ActivityState(
                meta=activity_meta,
                price=price,
                override_hours=override_hours,
                direct_hours=direct_hours,
                conversion_factor=conversion_factor,
                country_weights=country_weight_map,
                weighting_methods=weighting_methods,
                reference_year=reference_year,
                data_years=data_years,
                countries=set(),
                sectors=defaultdict(set),
                processes=defaultdict(set),
                diagnostics=Counter(),
            )
        )
    del activity_groups

    profiles: dict[tuple[Any, Any, int], _SectorProfile] = {}
    matches_by_country: dict[Any, list[tuple[int, _SectorProfile]]] = {}
    indicator_keys: list[IndicatorKey] = []
    indicator_index: dict[IndicatorKey, int] = {}

    for row in records:
        activity_code = row.activity_code
        activity_id = activity_index.get(activity_code)
        country = row.country
        if activity_id is None or pd.isna(country):
            continue
        state = activity_states[activity_id]
        sector = row.sector
        state.countries.add(country)
        state.sectors[country].add(sector)
        if row.process is not None and not pd.isna(row.process):
            state.processes[country].add(str(row.process))

        exchanges = row.exchanges
        if not exchanges or not isinstance(exchanges, list):
            state.diagnostics["no_exchanges_for_matched_sector"] += 1
            continue

        # The normal pipeline shares one exchange list for every occurrence of
        # a source process. Its identity keeps distinct processes with the same
        # country-sector label separate (also for direct API callers).
        profile_key = (country, sector, id(exchanges))
        profile = profiles.get(profile_key)
        if profile is None:
            profile = _build_sector_profile(
                exchanges,
                level_weights,
                geometric=geometric,
            )
            profiles[profile_key] = profile
            for key in (*profile.psilca_totals, *profile.override_totals):
                if key not in indicator_index:
                    indicator_index[key] = len(indicator_keys)
                    indicator_keys.append(key)

        if state.override_hours is None:
            if any(
                profile.psilca_diagnostics.get(key, 0)
                for key in (
                    "working_hours_negative",
                    "working_hours_missing_or_nan",
                )
            ):
                raise ValueError(
                    f"Invalid PSILCA worker-hour intensity for activity {activity_code!r}, "
                    f"country {country!r}, sector {sector!r}"
                )
            if profile.inconsistent_hours:
                raise ValueError(
                    "Inconsistent PSILCA worker-hour intensities for "
                    f"activity {activity_code!r}, country {country!r}, "
                    f"sector {sector!r}: {list(profile.inconsistent_hours)}"
                )
            state.diagnostics.update(profile.psilca_diagnostics)
        else:
            state.diagnostics.update(profile.override_diagnostics)
        matches_by_country.setdefault(country, []).append((activity_id, profile))
    del records

    activity_count = len(activity_states)
    indicator_count = len(indicator_keys)
    shape = (activity_count, indicator_count)
    needs_psilca_profiles = any(state.override_hours is None for state in activity_states)
    needs_override_profiles = any(state.override_hours is not None for state in activity_states)

    psilca_aggregation = _CrossCountryAggregation.zeros(shape)
    override_aggregation = _CrossCountryAggregation.zeros(shape)
    hours_weighted = np.zeros(activity_count, dtype=np.float64)
    hours_weight_sums = np.zeros(activity_count, dtype=np.float64)
    hours_unweighted = np.zeros(activity_count, dtype=np.float64)
    hours_country_counts = np.zeros(activity_count, dtype=np.float64)

    def _country_profile_means(
        *,
        grouped_rows: np.ndarray,
        group_starts: np.ndarray,
        sorted_sector_rows: np.ndarray,
        sector_profiles: list[_SectorProfile],
        totals_attribute: str,
        counts_attribute: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        sector_totals = np.zeros((len(sector_profiles), indicator_count), dtype=np.float64)
        sector_counts = np.zeros_like(sector_totals)
        for sector_id, profile in enumerate(sector_profiles):
            totals: dict[IndicatorKey, float] = getattr(profile, totals_attribute)
            counts: Counter[IndicatorKey] = getattr(profile, counts_attribute)
            for key, total in totals.items():
                column = indicator_index[key]
                sector_totals[sector_id, column] = total
                sector_counts[sector_id, column] = counts[key]

        country_totals = np.zeros((len(grouped_rows), indicator_count), dtype=np.float64)
        country_counts = np.zeros_like(country_totals)
        country_totals[grouped_rows] = np.add.reduceat(
            sector_totals[sorted_sector_rows],
            group_starts,
            axis=0,
        )
        country_counts[grouped_rows] = np.add.reduceat(
            sector_counts[sorted_sector_rows],
            group_starts,
            axis=0,
        )
        valid = country_counts > 0
        country_means = np.zeros_like(country_totals)
        country_means[valid] = country_totals[valid] / country_counts[valid]
        return country_means, valid

    for country, matches in matches_by_country.items():
        active_ids = np.array(list(dict.fromkeys(item[0] for item in matches)), dtype=np.intp)
        local_index = {activity_id: index for index, activity_id in enumerate(active_ids)}
        sector_profiles = list(dict.fromkeys(id(item[1]) for item in matches))
        profile_by_identity = {id(profile): profile for _, profile in matches}
        profiles_in_order = [profile_by_identity[identity] for identity in sector_profiles]
        profile_index = {identity: index for index, identity in enumerate(sector_profiles)}
        local_rows = np.fromiter(
            (local_index[activity_id] for activity_id, _ in matches),
            dtype=np.intp,
            count=len(matches),
        )
        sector_rows = np.fromiter(
            (profile_index[id(profile)] for _, profile in matches),
            dtype=np.intp,
            count=len(matches),
        )
        stable_order = np.argsort(local_rows, kind="stable")
        sorted_local_rows = local_rows[stable_order]
        group_starts = np.flatnonzero(
            np.concatenate(([True], sorted_local_rows[1:] != sorted_local_rows[:-1]))
        )
        grouped_rows = sorted_local_rows[group_starts]
        sorted_sector_rows = sector_rows[stable_order]
        country_weight_vector = np.fromiter(
            (
                activity_states[activity_id].country_weights.get(str(country), 0.0)
                for activity_id in active_ids
            ),
            dtype=np.float64,
            count=len(active_ids),
        )

        if needs_psilca_profiles:
            country_means, valid = _country_profile_means(
                grouped_rows=grouped_rows,
                group_starts=group_starts,
                sorted_sector_rows=sorted_sector_rows,
                sector_profiles=profiles_in_order,
                totals_attribute="psilca_totals",
                counts_attribute="psilca_counts",
            )
            psilca_aggregation.add(active_ids, country_means, valid, country_weight_vector)
        if needs_override_profiles:
            country_means, valid = _country_profile_means(
                grouped_rows=grouped_rows,
                group_starts=group_starts,
                sorted_sector_rows=sorted_sector_rows,
                sector_profiles=profiles_in_order,
                totals_attribute="override_totals",
                counts_attribute="override_counts",
            )
            override_aggregation.add(active_ids, country_means, valid, country_weight_vector)

        sector_hours = np.array(
            [
                profile.working_hours if profile.working_hours is not None else 0.0
                for profile in profiles_in_order
            ],
            dtype=np.float64,
        )
        sector_hours_valid = sector_hours > 0
        country_hours = np.zeros(len(active_ids), dtype=np.float64)
        country_hours_counts = np.zeros(len(active_ids), dtype=np.float64)
        country_hours[grouped_rows] = np.add.reduceat(
            sector_hours[sorted_sector_rows],
            group_starts,
        )
        country_hours_counts[grouped_rows] = np.add.reduceat(
            sector_hours_valid[sorted_sector_rows].astype(np.float64),
            group_starts,
        )
        hours_valid = country_hours_counts > 0
        country_hours[hours_valid] /= country_hours_counts[hours_valid]
        hours_weighted[active_ids] += country_hours * country_weight_vector * hours_valid
        hours_weight_sums[active_ids] += country_weight_vector * hours_valid
        hours_unweighted[active_ids] += country_hours * hours_valid
        hours_country_counts[active_ids] += hours_valid

    psilca_means = psilca_aggregation.finish(exponentiate=geometric)
    override_means = override_aggregation.finish(exponentiate=geometric)
    working_hour_means = _CrossCountryAggregation(
        weighted=hours_weighted,
        weight_sums=hours_weight_sums,
        unweighted=hours_unweighted,
        country_counts=hours_country_counts,
    ).finish(exponentiate=False)

    final_activities: list[dict[str, Any]] = []
    for activity_id, activity_code in enumerate(activity_codes):
        state = activity_states[activity_id]
        meta = state.meta
        price = state.price
        override_hours = state.override_hours
        conversion_factor = state.conversion_factor
        price_factor = price * conversion_factor
        if not math.isfinite(price_factor):
            raise ValueError(
                f"Non-finite computed monetary conversion for activity {activity_code!r}"
            )
        means = override_means if override_hours is not None else psilca_means
        country_counts = (
            override_aggregation.country_counts
            if override_hours is not None
            else psilca_aggregation.country_counts
        )
        activity_scale = (
            price_factor * override_hours if override_hours is not None else price_factor
        )
        if state.direct_hours is not None:
            activity_scale = state.direct_hours
        if not math.isfinite(activity_scale) or np.isinf(means[activity_id]).any():
            raise ValueError(f"Non-finite computed intensity for activity {activity_code!r}")

        final_exchanges: list[dict[str, Any]] = []
        for column in np.flatnonzero(np.isfinite(means[activity_id])):
            mean_value = float(means[activity_id, column]) * activity_scale
            if not math.isfinite(mean_value):
                raise ValueError(
                    f"Non-finite computed indicator {indicator_keys[column][0]!r} "
                    f"for activity {activity_code!r}"
                )
            if not math.isfinite(mean_value) or mean_value <= 0:
                continue
            if isinstance(rounding, int):
                rounded = round(mean_value, rounding)
                value_out = mean_value if rounded == 0.0 else rounded
            else:
                value_out = mean_value
            name, _unit, exchange_type, stakeholder_group, subcategory = indicator_keys[column]
            final_exchanges.append(
                {
                    "indicator": name,
                    "type": exchange_type,
                    "category": stakeholder_group,
                    "subcategory": subcategory,
                    "unit": (
                        "med opportunity hours"
                        if exchange_type == "opportunity"
                        else "med risk hours"
                    ),
                    "meanValue": float(value_out),
                    "origin_matches": int(country_counts[activity_id, column]),
                }
            )

        working_hours: float | None
        if state.direct_hours is not None:
            working_hours = state.direct_hours
        elif override_hours is not None:
            working_hours = override_hours * price_factor
        else:
            working_hours = float(working_hour_means[activity_id]) * price_factor
        if math.isinf(working_hours):
            raise ValueError(f"Non-finite computed working hours for activity {activity_code!r}")
        if not math.isfinite(working_hours) or working_hours <= 0:
            working_hours = None

        final_activities.append(
            {
                "Activity code": activity_code,
                "Activity name": meta.get("Activity name"),
                "Reference product": meta.get("Reference product"),
                "ISIC": meta.get("ISIC"),
                "Unit": meta.get("Unit"),
                "ISO2": meta.get("ISO2"),
                "Price": price,
                "Price_conversion_factor": conversion_factor,
                "Converted_price": price_factor,
                "PSILCA_countries_used": sorted(state.countries),
                "PSILCA_sectors_used": {
                    country: sorted(sectors) for country, sectors in state.sectors.items()
                },
                "PSILCA_processes_used": {
                    country: sorted(processes) for country, processes in state.processes.items()
                },
                "PSILCA_country_weights": dict(sorted(state.country_weights.items())),
                "Country_weighting_method": "; ".join(state.weighting_methods),
                "Country_weight_reference_year": state.reference_year,
                "Country_weight_data_years": dict(sorted(state.data_years.items())),
                "Input_diagnostics": dict(sorted(state.diagnostics.items())),
                "Averaged_Exchanges": final_exchanges,
                "Working_hours": working_hours,
                "Working_hours_source": (
                    "treatment_fallback"
                    if state.direct_hours is not None
                    else "external_productivity" if override_hours is not None else "psilca"
                ),
            }
        )

    return final_activities
