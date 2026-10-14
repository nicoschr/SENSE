"""Shared country weighting for regional SENSE calculations.

The default provider uses World Bank GDP with an IMF WEO fallback. Both the
PSILCA country aggregation and the optional productivity baseline consume the
same provider, so enabling productivity cannot change the regional weights.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import cache
from importlib import resources
from pathlib import Path

import pandas as pd

GDP_COLUMNS = ("country_code", "year", "gdp_current_usd")
BUNDLED_GDP_FILE = "world_bank_gdp_current_usd_2000_2024.csv"
BUNDLED_IMF_GDP_FILE = "imf_weo_gdp_current_usd_2000_2024.csv"
BUNDLED_IMF_METADATA_FILE = "imf_weo_gdp_current_usd_2000_2024.json"
WORLD_BANK_INDICATOR = "NY.GDP.MKTP.CD"
WORLD_BANK_SOURCE_URL = "https://api.worldbank.org/v2/country/all/indicator/NY.GDP.MKTP.CD"


@cache
def iso2_to_iso3(iso2: str) -> str | None:
    """Convert an ISO 3166-1 alpha-2 code to alpha-3."""

    code = str(iso2 or "").strip().upper()
    if code == "XK":
        return "XKX"  # Kosovo: commonly used user-assigned codes.

    # Keep the dependency local so ISO3-only uses of this module do not need
    # to import the full country catalogue.
    import pycountry

    country = pycountry.countries.get(alpha_2=code)
    return country.alpha_3 if country else None


@dataclass(frozen=True)
class WeightResolution:
    """Weights and provenance for one set of countries."""

    weights: dict[str, float]
    method: str
    reference_year: int
    data_years: dict[str, int]
    missing_countries: tuple[str, ...] = ()
    data_sources: dict[str, str] = field(default_factory=dict)


class CountryWeightProvider:
    """Resolve normalized country weights from one validated data source."""

    def __init__(
        self,
        *,
        method: str,
        reference_year: int,
        values: dict[str, dict[int, float]] | None = None,
        source: str = "equal weights",
        max_year_gap: int = 5,
    ) -> None:
        method = str(method).strip().lower()
        if method not in {"gdp", "equal"}:
            raise ValueError("country weighting must be 'gdp' or 'equal'")
        if type(reference_year) is not int or reference_year <= 0:
            raise ValueError("reference_year must be a positive integer")
        if type(max_year_gap) is not int or max_year_gap < 0:
            raise ValueError("max_year_gap must be non-negative")

        self.method = method
        self.reference_year = int(reference_year)
        self.source = source
        self._is_bundled = False
        self._fallback_values: dict[str, dict[int, float]] = {}
        self._primary_source_id = "user_provided"
        self._sources: list[dict[str, object]] = []
        self._observations_used: dict[str, dict[str, object]] = {}
        self._data_source_counts: Counter = Counter()
        self.max_year_gap = int(max_year_gap)
        self._values = values or {}
        self._cache: dict[tuple[str, tuple[str, ...]], WeightResolution] = {}
        self._method_counts: Counter = Counter()
        self._missing_counts: Counter = Counter()
        self._data_year_counts: Counter = Counter()

    @classmethod
    def from_dataframe(
        cls,
        data: pd.DataFrame,
        *,
        reference_year: int,
        source: str = "user-provided GDP data",
        max_year_gap: int = 5,
    ) -> CountryWeightProvider:
        """Build a GDP provider from the established SENSE GDP schema."""

        missing = set(GDP_COLUMNS).difference(data.columns)
        if missing:
            raise ValueError(
                "GDP data is missing required column(s): " + ", ".join(sorted(missing))
            )

        clean = data[list(GDP_COLUMNS)].copy()
        clean["country_code"] = clean["country_code"].astype(str).str.strip().str.upper()
        clean["year"] = pd.to_numeric(clean["year"], errors="coerce")
        clean["gdp_current_usd"] = pd.to_numeric(clean["gdp_current_usd"], errors="coerce")
        if (
            not clean["gdp_current_usd"]
            .map(lambda value: pd.isna(value) or math.isfinite(value))
            .all()
        ):
            raise ValueError("GDP observations must be finite or missing")
        if (
            not clean["year"]
            .map(
                lambda value: pd.isna(value)
                or (math.isfinite(value) and value > 0 and value == int(value))
            )
            .all()
        ):
            raise ValueError("GDP years must be positive finite integers or missing")
        clean = clean[
            clean["country_code"].ne("")
            & clean["year"].notna()
            & clean["gdp_current_usd"].notna()
            & clean["gdp_current_usd"].gt(0)
        ].copy()
        clean["year"] = clean["year"].astype(int)

        duplicates = clean.duplicated(["country_code", "year"], keep=False)
        if duplicates.any():
            duplicate_keys = (
                clean.loc[duplicates, ["country_code", "year"]]
                .drop_duplicates()
                .head(5)
                .itertuples(index=False, name=None)
            )
            raise ValueError(
                "GDP data contains duplicate country/year rows: "
                + ", ".join(f"{code}/{year}" for code, year in duplicate_keys)
            )
        if clean.empty:
            raise ValueError("GDP data contains no positive numeric observations")

        values: dict[str, dict[int, float]] = {}
        for row in clean.itertuples(index=False):
            values.setdefault(row.country_code, {})[int(row.year)] = float(row.gdp_current_usd)

        return cls(
            method="gdp",
            reference_year=reference_year,
            values=values,
            source=source,
            max_year_gap=max_year_gap,
        )

    @classmethod
    def from_csv(
        cls,
        path: Path,
        *,
        reference_year: int,
        max_year_gap: int = 5,
    ) -> CountryWeightProvider:
        """Build a GDP provider from a CSV using the SENSE GDP schema."""

        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Country GDP data not found: {path}")
        data = pd.read_csv(path, dtype={"country_code": str})
        return cls.from_dataframe(
            data,
            reference_year=reference_year,
            source=str(path.resolve()),
            max_year_gap=max_year_gap,
        )

    @classmethod
    def bundled_gdp(
        cls,
        *,
        reference_year: int,
        max_year_gap: int = 5,
    ) -> CountryWeightProvider:
        """Load World Bank first, using IMF only outside World Bank coverage.

        Source priority precedes nearest-year selection: a usable World Bank
        observation within the configured window always takes precedence.
        This does not change the semantics of explicit custom GDP inputs.
        """

        resource = resources.files("sense.resources").joinpath(BUNDLED_GDP_FILE)
        with resource.open("r", encoding="utf-8") as handle:
            data = pd.read_csv(handle, dtype={"country_code": str})
        provider = cls.from_dataframe(
            data,
            reference_year=reference_year,
            source=f"bundled:{BUNDLED_GDP_FILE}; fallback:{BUNDLED_IMF_GDP_FILE}",
            max_year_gap=max_year_gap,
        )
        provider._is_bundled = True
        provider._primary_source_id = "world_bank"
        root = resources.files("sense.resources")
        metadata = json.loads(root.joinpath(BUNDLED_IMF_METADATA_FILE).read_text(encoding="utf-8"))
        raw = root.joinpath(BUNDLED_IMF_GDP_FILE).read_bytes()
        if hashlib.sha256(raw).hexdigest() != metadata["sha256"]:
            raise ValueError("Bundled IMF GDP snapshot does not match its provenance hash")
        with root.joinpath(BUNDLED_IMF_GDP_FILE).open("r", encoding="utf-8") as handle:
            fallback = cls.from_dataframe(
                pd.read_csv(handle), reference_year=reference_year, max_year_gap=max_year_gap
            )
        provider._fallback_values = fallback._values
        provider._sources = [
            {
                "source_id": "world_bank",
                "indicator": WORLD_BANK_INDICATOR,
                "source_url": WORLD_BANK_SOURCE_URL,
                "license": "CC BY 4.0",
                "file": BUNDLED_GDP_FILE,
                "sha256": hashlib.sha256(resource.read_bytes()).hexdigest(),
            },
            {**metadata, "file": BUNDLED_IMF_GDP_FILE},
        ]
        return provider

    @classmethod
    def equal(cls, *, reference_year: int) -> CountryWeightProvider:
        """Build a provider that assigns equal shares to requested countries."""

        return cls(method="equal", reference_year=reference_year)

    def _observation(self, iso3: str) -> tuple[float | None, int | None]:
        value, year, _source = self._observation_with_source(iso3)
        return value, year

    def _nearest_observation(
        self,
        observations: dict[int, float] | None,
    ) -> tuple[float | None, int | None]:
        if not observations:
            return None, None
        if self.reference_year in observations:
            return observations[self.reference_year], self.reference_year

        candidates = [
            year for year in observations if abs(year - self.reference_year) <= self.max_year_gap
        ]
        if not candidates:
            return None, None
        # Prefer the more recent observation when two years are equally close.
        year = min(
            candidates,
            key=lambda value: (abs(value - self.reference_year), -value),
        )
        return observations[year], year

    def _observation_with_source(self, iso3: str) -> tuple[float | None, int | None, str | None]:
        for values, source_id in [
            (self._values, self._primary_source_id),
            (self._fallback_values, "imf_weo"),
        ]:
            value, year = self._nearest_observation(values.get(iso3))
            if value is not None:
                return value, year, source_id
        return None, None, None

    def _resolve(
        self,
        countries: Iterable[str],
        *,
        code_system: str,
    ) -> WeightResolution:
        unique = tuple(dict.fromkeys(str(code).strip() for code in countries if str(code).strip()))
        cache_key = (code_system, unique)
        cached = self._cache.get(cache_key)
        if cached is not None:
            self._record(cached)
            return cached

        if not unique:
            result = WeightResolution({}, "empty", self.reference_year, {})
        elif len(unique) == 1:
            result = WeightResolution({unique[0]: 1.0}, "single_country", self.reference_year, {})
        elif self.method == "equal":
            equal = 1.0 / len(unique)
            result = WeightResolution(
                dict.fromkeys(unique, equal),
                "equal",
                self.reference_year,
                {},
            )
        else:
            raw_values: dict[str, float] = {}
            data_years: dict[str, int] = {}
            data_sources: dict[str, str] = {}
            missing_countries: list[str] = []
            for code in unique:
                iso3 = code.upper() if code_system == "iso3" else iso2_to_iso3(code)
                value, year, source_id = (
                    self._observation_with_source(iso3) if iso3 else (None, None, None)
                )
                if value is None or year is None:
                    missing_countries.append(code)
                    continue
                raw_values[code] = value
                data_years[code] = year
                data_sources[code] = str(source_id)
                self._observations_used[str(iso3)] = {
                    "source_id": source_id,
                    "year": year,
                    "gdp_current_usd": value,
                }

            total = sum(raw_values.values())
            if total > 0:
                method = "gdp" if not missing_countries else "gdp_available_countries"
                result = WeightResolution(
                    {code: value / total for code, value in raw_values.items()},
                    method,
                    self.reference_year,
                    data_years,
                    tuple(missing_countries),
                    data_sources,
                )
            else:
                equal = 1.0 / len(unique)
                result = WeightResolution(
                    dict.fromkeys(unique, equal),
                    "equal_fallback_no_gdp",
                    self.reference_year,
                    {},
                    tuple(unique),
                )

        self._cache[cache_key] = result
        self._record(result)
        return result

    def _record(self, result: WeightResolution) -> None:
        self._method_counts[result.method] += 1
        self._missing_counts.update(result.missing_countries)
        self._data_year_counts.update(result.data_years.values())
        self._data_source_counts.update(result.data_sources.values())

    def resolve_iso2(self, countries: Iterable[str]) -> WeightResolution:
        """Resolve weights for ISO 3166-1 alpha-2 country codes."""

        return self._resolve(countries, code_system="iso2")

    def resolve_iso3(self, countries: Iterable[str]) -> WeightResolution:
        """Resolve weights for ISO 3166-1 alpha-3 country codes."""

        return self._resolve(countries, code_system="iso3")

    def audit_dict(self) -> dict[str, object]:
        """Return run-level provenance and fallback statistics."""

        return {
            "schema_version": 2,
            "method": self.method,
            "reference_year": self.reference_year,
            "nearest_year_window": self.max_year_gap,
            "source": self.source,
            # Mixed source rights and indicators are reported per dataset.
            "indicator": None,
            "source_url": None,
            "license": None,
            "source_policy": (
                "World Bank within nearest-year window, then IMF WEO within the same window"
                if self._is_bundled
                else "Explicit input only; no bundled supplementation"
            ),
            "sources": self._sources,
            "country_observations": dict(sorted(self._observations_used.items())),
            "data_source_counts": dict(sorted(self._data_source_counts.items())),
            "aggregation_note": (
                "GDP is a country-size proxy, not activity-specific production. "
                "Countries without usable GDP are omitted and the available "
                "weights are renormalized; if none are available, equal weights "
                "are used."
                if self.method == "gdp"
                else "All mapped countries contribute equally."
            ),
            "resolution_counts": dict(sorted(self._method_counts.items())),
            "data_year_counts": {
                str(year): count for year, count in sorted(self._data_year_counts.items())
            },
            "missing_country_counts": dict(sorted(self._missing_counts.items())),
        }


def attach_country_weights(
    enriched_df: pd.DataFrame,
    provider: CountryWeightProvider,
) -> pd.DataFrame:
    """Attach one country weight to every matched activity/country row."""

    required = {"Activity code", "PSILCA_country"}
    missing = required.difference(enriched_df.columns)
    if missing:
        raise ValueError(
            "Cannot attach country weights; missing column(s): " + ", ".join(sorted(missing))
        )

    result = enriched_df.copy()
    result["Country_weight"] = 0.0
    result["Country_weighting_method"] = ""
    result["Country_weight_reference_year"] = provider.reference_year
    result["Country_weight_data_year"] = pd.NA
    result["Country_weight_data_source"] = ""

    for _, indices in result.groupby("Activity code", sort=False).groups.items():
        raw_countries = result.loc[indices, "PSILCA_country"]
        countries = list(
            dict.fromkeys(
                str(value).strip()
                for value in raw_countries
                if pd.notna(value) and str(value).strip()
            )
        )
        resolution = provider.resolve_iso2(countries)
        country_series = raw_countries.map(
            lambda value: str(value).strip() if pd.notna(value) else ""
        )
        result.loc[indices, "Country_weight"] = (
            country_series.map(resolution.weights).fillna(0.0).astype(float)
        )
        result.loc[indices, "Country_weighting_method"] = resolution.method
        result.loc[indices, "Country_weight_data_year"] = country_series.map(resolution.data_years)
        result.loc[indices, "Country_weight_data_source"] = country_series.map(
            resolution.data_sources
        ).fillna("")

    return result
