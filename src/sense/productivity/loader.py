"""Load hours-per-USD productivity and resolve it by geography, ISIC, and year."""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from sense.utils.helpers import extract_isic2
from sense.weighting import CountryWeightProvider

logger = logging.getLogger(__name__)


# Resolution statuses written to the productivity audit. ``override_*`` means
# a baseline value was applied; all other statuses retain PSILCA working hours.
LOOKUP_STATUS_OVERRIDE_EXACT = "override_exact"
LOOKUP_STATUS_OVERRIDE_YEAR_FB = "override_year_fallback"
LOOKUP_STATUS_OVERRIDE_FULL_AGG = "override_full_aggregate"
LOOKUP_STATUS_OVERRIDE_PARTIAL_AGG = "override_partial_aggregate"
LOOKUP_STATUS_NO_ISIC = "no_isic"
LOOKUP_STATUS_ISIC_NOT_IN_BASELINE = "isic_not_in_baseline"
LOOKUP_STATUS_LOCATION_NOT_MAPPED = "location_not_mapped"
LOOKUP_STATUS_NO_COUNTRY_DATA = "no_country_data"

OVERRIDE_APPLIED_STATUSES = {
    LOOKUP_STATUS_OVERRIDE_EXACT,
    LOOKUP_STATUS_OVERRIDE_YEAR_FB,
    LOOKUP_STATUS_OVERRIDE_FULL_AGG,
    LOOKUP_STATUS_OVERRIDE_PARTIAL_AGG,
}


@dataclass(frozen=True)
class ProductivityCoverage:
    """Separate unavailable GDP weights from missing retained coefficients."""

    weight_method: str
    missing_gdp_countries: tuple[str, ...]
    missing_productivity_countries: tuple[str, ...]

    @property
    def gdp_omissions_only(self) -> bool:
        return (
            self.weight_method == "gdp_available_countries"
            and bool(self.missing_gdp_countries)
            and not self.missing_productivity_countries
        )


@dataclass
class LookupResult:
    """Result of a productivity lookup with coverage metadata."""

    value: float | None
    status: str
    used_isic2: str | None = None
    used_year: int | None = None
    n_countries_attempted: int = 0
    n_countries_used: int = 0
    coverage: ProductivityCoverage | None = None


class ProductivityBaseline:
    """Resolve hours-per-USD by ecoinvent geography, ISIC division, and year.

    The Parquet input must contain ``country``, ``year``, ``isic4_code``, and
    ``productivity_hrs_per_usd``.
    """

    def __init__(
        self,
        parquet_path: Path,
        location_mapping: dict[str, list[tuple[str, float]]],
        *,
        weight_provider: CountryWeightProvider,
        reference_year: int = 2020,
    ):
        """Load and validate a productivity baseline.

        Args:
            parquet_path: Path to the productivity baseline parquet file.
            location_mapping: Output of build_location_mapping() — maps
                ecoinvent shortnames to weighted ISO3 lists.
            reference_year: Default year for lookups when year is not specified.
            weight_provider: Shared SENSE country-weight provider. Its fixed
                reference year is also used for dynamic RoW/RoE aggregation.
        """
        logger.info("Loading productivity baseline from %s", parquet_path)
        df = pd.read_parquet(
            parquet_path,
            columns=["country", "year", "isic4_code", "productivity_hrs_per_usd"],
        )

        if (
            not df["country"]
            .map(lambda value: isinstance(value, str) and bool(re.fullmatch(r"[A-Z]{3}", value)))
            .all()
        ):
            raise ValueError("Productivity country codes must be uppercase ISO3 strings.")
        if (
            not df["isic4_code"]
            .map(lambda value: isinstance(value, str) and bool(re.fullmatch(r"[0-9]{2}", value)))
            .all()
        ):
            raise ValueError("isic4_code must contain two-digit ISIC Rev.4 division strings.")
        years = pd.to_numeric(df["year"], errors="coerce")
        if not years.map(
            lambda value: pd.notna(value)
            and math.isfinite(value)
            and value > 0
            and value == int(value)
        ).all():
            raise ValueError("Productivity years must be positive finite integers.")
        values = pd.to_numeric(df["productivity_hrs_per_usd"], errors="raise")
        if not values.map(lambda value: pd.isna(value) or math.isfinite(value)).all():
            raise ValueError("Productivity observations must be finite or missing.")
        df["year"] = years.map(int)
        df["productivity_hrs_per_usd"] = values
        if df.duplicated(["country", "isic4_code", "year"]).any():
            raise ValueError("Productivity baseline has duplicate country/ISIC/year keys.")

        # Build lookup: (ISO3, ISIC2, year) → productivity
        self._lookup: dict[tuple[str, str, int], float] = {}
        for _, row in df.iterrows():
            key = (row["country"], row["isic4_code"], int(row["year"]))
            val = row["productivity_hrs_per_usd"]
            if pd.notna(val) and val > 0:
                self._lookup[key] = float(val)

        self._location_mapping = {key: list(value) for key, value in location_mapping.items()}
        self._weight_provider = weight_provider
        self._reference_year = reference_year
        self._available_iso3: set[str] = set(df["country"].unique())
        self._available_isic: set[str] = set(df["isic4_code"].unique())
        self._available_years: set[int] = set(df["year"].unique())

        logger.info(
            "  Loaded %d productivity values (%d countries, %d sectors, %d years)",
            len(self._lookup),
            len(self._available_iso3),
            len(self._available_isic),
            len(self._available_years),
        )

    def _lookup_single_with_year(
        self,
        iso3: str,
        isic2: str,
        year: int,
    ) -> tuple[float | None, int | None]:
        """Look up productivity for one country, falling back to nearest year.

        Tries the exact year first, then searches within ±5 years for the
        nearest available data point for this specific country × sector.

        Returns:
            (value, year_used). Both None if nothing found within ±5 years.
        """
        val = self._lookup.get((iso3, isic2, year))
        if val is not None:
            return val, year

        # Search nearby years for this specific country × sector
        best_val: float | None = None
        best_year: int | None = None
        best_dist = 999
        for delta in range(1, 6):
            for candidate in (year - delta, year + delta):
                v = self._lookup.get((iso3, isic2, candidate))
                if v is not None and delta < best_dist:
                    best_val = v
                    best_year = candidate
                    best_dist = delta
                    break  # Equal-distance ties retain the earlier year.
            if best_val is not None and best_dist <= delta:
                break
        return best_val, best_year

    def _weighted_productivity_detailed(
        self,
        country_weights: list[tuple[str, float]],
        isic2: str,
        year: int,
    ) -> tuple[float | None, int, int, bool]:
        """Compute country-weighted productivity across multiple countries.

        For each country, falls back to the nearest available year if the
        exact year is missing.

        Returns:
            (value, n_countries_used, n_countries_attempted, exact_year_for_all)
            value is None if no country has data. exact_year_for_all is True
            iff every contributing country had a value at the exact requested
            year (no year fallback).
        """
        total_weight = 0.0
        weighted_sum = 0.0
        n_used = 0
        n_attempted = len(country_weights)
        any_year_fallback = False

        for iso3, weight in country_weights:
            if weight <= 0:
                # Unavailable GDP weights remain in n_attempted, but cannot
                # count as contributing productivity observations.
                continue
            val, year_used = self._lookup_single_with_year(iso3, isic2, year)
            if val is not None:
                weighted_sum += val * weight
                total_weight += weight
                n_used += 1
                if year_used != year:
                    any_year_fallback = True

        if total_weight > 0:
            return weighted_sum / total_weight, n_used, n_attempted, not any_year_fallback
        return None, 0, n_attempted, False

    def _coverage_details(
        self, countries: list[str], isic2: str, year: int
    ) -> ProductivityCoverage:
        resolution = self._weight_provider.resolve_iso3(countries)
        missing_productivity = tuple(
            country
            for country, weight in resolution.weights.items()
            if weight > 0 and self._lookup_single_with_year(country, isic2, year)[0] is None
        )
        return ProductivityCoverage(
            resolution.method, resolution.missing_countries, missing_productivity
        )

    def apply_location_proxies(
        self,
        proxies: dict[str, list[str]],
    ) -> dict[str, list[str]]:
        """Add explicit donor mappings for locations absent from the baseline.

        Existing mappings are never overridden. Every requested donor must be
        present in the productivity baseline; multi-country donors use the same
        fixed country-weight provider as ordinary composite geographies.

        Args:
            proxies: Ecoinvent location to ISO3 donor-country lists.

        Returns:
            The normalized proxy mappings that were applied.

        Raises:
            ValueError: If a donor is absent or receives no usable weight.
        """

        applied: dict[str, list[str]] = {}
        for raw_location, raw_donors in proxies.items():
            location = str(raw_location).strip()
            if location in self._location_mapping:
                continue
            donors = list(dict.fromkeys(str(donor).strip().upper() for donor in raw_donors))
            missing = [donor for donor in donors if donor not in self._available_iso3]
            if missing:
                raise ValueError(
                    f"Productivity proxy {location!r} uses donor countries absent from "
                    f"the baseline: {missing}"
                )
            weights = self._weight_provider.resolve_iso3(donors).weights
            weighted = [(donor, weights.get(donor, 0.0)) for donor in donors]
            total = sum(weight for _, weight in weighted)
            if total <= 0:
                raise ValueError(f"Productivity proxy {location!r} has no usable donor weights")
            self._location_mapping[location] = [
                (donor, weight / total) for donor, weight in weighted
            ]
            applied[location] = donors
        return applied

    def lookup_with_status(
        self,
        location: str,
        raw_isic: str,
        year: int | None = None,
    ) -> LookupResult:
        """Look up productivity and report how the value was resolved.

        Status values follow the LOOKUP_STATUS_* constants at module top:
            override_exact / override_year_fallback / override_full_aggregate /
            override_partial_aggregate (override applied) vs.
            no_isic / isic_not_in_baseline / location_not_mapped / no_country_data
            (override NOT applied).

        Args:
            location: Ecoinvent location shortname.
            raw_isic: ISIC code as stored on the ecoinvent activity.
            year: Requested year; defaults to the baseline reference year.

        Returns:
            Value, resolution status, and coverage metadata.
        """
        isic2 = extract_isic2(raw_isic)
        if isic2 is None:
            return LookupResult(value=None, status=LOOKUP_STATUS_NO_ISIC)
        if isic2 not in self._available_isic:
            return LookupResult(
                value=None, status=LOOKUP_STATUS_ISIC_NOT_IN_BASELINE, used_isic2=isic2
            )

        yr = year or self._reference_year
        country_weights = self._location_mapping.get(location)
        if country_weights is None:
            return LookupResult(
                value=None, status=LOOKUP_STATUS_LOCATION_NOT_MAPPED, used_isic2=isic2
            )

        coverage = self._coverage_details([country for country, _ in country_weights], isic2, yr)
        val, n_used, n_attempted, exact_year = self._weighted_productivity_detailed(
            country_weights,
            isic2,
            yr,
        )
        if val is None:
            return LookupResult(
                value=None,
                status=LOOKUP_STATUS_NO_COUNTRY_DATA,
                used_isic2=isic2,
                n_countries_attempted=n_attempted,
                coverage=coverage,
            )

        # Decide which override status to report
        if n_attempted == 1:
            # Single-country (most common — specific country code, e.g. DE).
            status = LOOKUP_STATUS_OVERRIDE_EXACT if exact_year else LOOKUP_STATUS_OVERRIDE_YEAR_FB
        else:
            # Composite location (e.g. RER): shared country-weighted aggregate.
            resolution = self._weight_provider.resolve_iso3(
                [country for country, _ in country_weights]
            )
            status = (
                LOOKUP_STATUS_OVERRIDE_FULL_AGG
                if n_used == n_attempted and not resolution.missing_countries
                else LOOKUP_STATUS_OVERRIDE_PARTIAL_AGG
            )

        return LookupResult(
            value=val,
            status=status,
            used_isic2=isic2,
            used_year=yr if exact_year else None,
            n_countries_attempted=n_attempted,
            n_countries_used=n_used,
            coverage=coverage,
        )

    def lookup(
        self,
        location: str,
        raw_isic: str,
        year: int | None = None,
    ) -> float | None:
        """Return a productivity value without resolution metadata.

        Use ``lookup_with_status()`` when audit information is required.

        Args:
            location: Ecoinvent location shortname (e.g., "DE", "RER", "CA-QC").
                RoW/RoE are NOT handled here — use lookup_row() for those.
            raw_isic: ISIC code string as it appears in ecoinvent
                (e.g., "2410:Manufacture of basic iron and steel").
            year: Year to look up. Defaults to reference_year. Falls back
                to nearest available year if exact year is missing.

        Returns:
            Hours per USD of gross output, or None if unavailable.
        """
        return self.lookup_with_status(location, raw_isic, year).value

    def lookup_row_with_status(
        self,
        iso3_countries: list[str],
        raw_isic: str,
        year: int | None = None,
    ) -> LookupResult:
        """Look up an activity-specific RoW/RoE productivity aggregate.

        Args:
            iso3_countries: Countries constituting the activity's rest-of region.
            raw_isic: ISIC code as stored on the ecoinvent activity.
            year: Requested year; defaults to the baseline reference year.

        Returns:
            Value, resolution status, and country-coverage metadata.
        """
        isic2 = extract_isic2(raw_isic)
        if isic2 is None:
            return LookupResult(value=None, status=LOOKUP_STATUS_NO_ISIC)
        if isic2 not in self._available_isic:
            return LookupResult(
                value=None, status=LOOKUP_STATUS_ISIC_NOT_IN_BASELINE, used_isic2=isic2
            )

        yr = year or self._reference_year

        # Filter to countries in the baseline
        valid = sorted(set(iso3_countries))
        n_attempted_total = len(valid)
        if not valid:
            return LookupResult(
                value=None,
                status=LOOKUP_STATUS_NO_COUNTRY_DATA,
                used_isic2=isic2,
                n_countries_attempted=n_attempted_total,
            )

        coverage = self._coverage_details(valid, isic2, yr)

        # Use the same fixed country weights as the social aggregation.
        resolution = self._weight_provider.resolve_iso3(valid)
        weights = resolution.weights

        country_weights = [(iso3, w) for iso3, w in weights.items() if w > 0]
        if not country_weights:
            return LookupResult(
                value=None,
                status=LOOKUP_STATUS_NO_COUNTRY_DATA,
                used_isic2=isic2,
                n_countries_attempted=n_attempted_total,
                coverage=coverage,
            )

        val, n_used, _n_attempted_inner, exact_year = self._weighted_productivity_detailed(
            country_weights,
            isic2,
            yr,
        )
        if val is None:
            return LookupResult(
                value=None,
                status=LOOKUP_STATUS_NO_COUNTRY_DATA,
                used_isic2=isic2,
                n_countries_attempted=n_attempted_total,
                coverage=coverage,
            )

        status = (
            LOOKUP_STATUS_OVERRIDE_FULL_AGG
            if n_used == n_attempted_total and not resolution.missing_countries
            else LOOKUP_STATUS_OVERRIDE_PARTIAL_AGG
        )
        return LookupResult(
            value=val,
            status=status,
            used_isic2=isic2,
            used_year=yr if exact_year else None,
            n_countries_attempted=n_attempted_total,
            n_countries_used=n_used,
            coverage=coverage,
        )

    def lookup_row(
        self,
        iso3_countries: list[str],
        raw_isic: str,
        year: int | None = None,
    ) -> float | None:
        """Return a RoW/RoE productivity value without resolution metadata.

        Use ``lookup_row_with_status()`` when audit information is required.

        For RoW/RoE, the set of countries is activity-specific (it excludes
        countries where a location-specific process exists). The caller
        provides this list, and this method applies the shared country weights.

        Args:
            iso3_countries: ISO3 country codes that constitute
                the "rest of" region for this specific activity.
            raw_isic: ISIC code string from ecoinvent.
            year: Year to look up.

        Returns:
            Hours per USD, or None if unavailable.
        """
        return self.lookup_row_with_status(iso3_countries, raw_isic, year).value

    @property
    def available_iso3(self) -> set[str]:
        """Set of ISO3 country codes available in the baseline."""
        return set(self._available_iso3)

    @property
    def available_isic(self) -> set[str]:
        """Set of ISIC 2-digit codes available in the baseline."""
        return set(self._available_isic)

    @property
    def available_years(self) -> set[int]:
        """Set of years available in the baseline."""
        return set(self._available_years)
