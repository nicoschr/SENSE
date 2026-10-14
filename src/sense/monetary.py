"""Monetary-basis resolution for ecoinvent and PSILCA price scaling.

SENSE combines ecoinvent product prices with labour intensities expressed per
monetary unit.  This module keeps the necessarily approximate temporal
harmonisation explicit, deterministic, and independent of network access.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

EUROSTAT_HICP_SOURCE = "https://doi.org/10.2908/PRC_HICP_AIND"
ECB_EXCHANGE_RATE_SOURCE = "https://data.ecb.europa.eu/data/datasets/EXR"
DATA_ACCESS_DATE = "2026-09-03"


@dataclass(frozen=True)
class MonetaryBasis:
    """Currency and representative price year used by a monetary quantity."""

    currency: str
    year: int
    approximate: bool = False
    note: str = ""

    def __post_init__(self) -> None:
        currency = self.currency.strip().upper()
        if not currency:
            raise ValueError("Monetary currency must not be empty")
        if self.year <= 0:
            raise ValueError("Monetary year must be a positive integer")
        object.__setattr__(self, "currency", currency)

    @property
    def label(self) -> str:
        """Return the compact currency-year label used in audits."""

        return f"{self.currency}{self.year}"


@dataclass(frozen=True)
class MonetaryConversion:
    """A reproducible multiplier between two monetary bases."""

    source: MonetaryBasis
    target: MonetaryBasis
    factor: float
    method: str
    user_supplied: bool = False

    def __post_init__(self) -> None:
        if not math.isfinite(self.factor) or self.factor <= 0:
            raise ValueError("Monetary conversion factor must be positive and finite")

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation for audit artifacts."""

        return {
            "source_basis": asdict(self.source),
            "target_basis": asdict(self.target),
            "factor": self.factor,
            "operation": "target_price = source_price * factor",
            "method": self.method,
            "user_supplied": self.user_supplied,
        }


# PSILCA does not expose a single machine-readable monetary reference year in
# its EcoSpold01 processes. These are documented representative years for the
# worker-hours-per-USD activity variable, not years for every social indicator.
_PSILCA_PROFILES: dict[str, MonetaryBasis] = {
    "3.0": MonetaryBasis(
        "USD",
        2011,
        approximate=True,
        note=(
            "Representative year: PSILCA v3 documents 2011 gross output, "
            "labour-cost extrapolation, and normally 2011 exchange rates for "
            "its worker-hour calculation. Individual inputs can differ."
        ),
    ),
    "3.1": MonetaryBasis(
        "USD",
        2011,
        approximate=True,
        note=(
            "Representative year: PSILCA v3 documents 2011 gross output, "
            "labour-cost extrapolation, and normally 2011 exchange rates for "
            "its worker-hour calculation. Individual inputs can differ."
        ),
    ),
    "3.1.1": MonetaryBasis(
        "USD",
        2011,
        approximate=True,
        note=(
            "Representative year: PSILCA v3 documents 2011 gross output, "
            "labour-cost extrapolation, and normally 2011 exchange rates for "
            "its worker-hour calculation. Individual inputs can differ."
        ),
    ),
    "4.0": MonetaryBasis(
        "USD",
        2015,
        approximate=True,
        note=(
            "Representative proxy: the PSILCA v4 manual identifies 2015 as "
            "the reference year for its Eora economic structure. Worker-hour "
            "input years are not uniform: the most recent available values are "
            "used by country-sector."
        ),
    ),
}

_VERSION_ALIASES = {
    "3": "3.0",
    "4": "4.0",
}

# Euro-area all-items HICP annual-average index (EA20, historical series). The
# absolute index base is irrelevant because only target/2005 ratios are used.
_EURO_AREA_HICP = {
    2005: 84.64,
    2011: 95.66,
    2015: 100.0,
    2019: 104.79,
}

# ECB annual-average reference rate, expressed as USD per EUR.
_ECB_USD_PER_EUR = {
    2011: 1.3919552529183,
    2015: 1.109512890625,
    2019: 1.1194745098039,
}


def normalize_psilca_version(version: str) -> str:
    """Return a stable PSILCA version key (for example ``v4`` -> ``4.0``)."""

    normalized = str(version).strip().lower()
    if normalized.startswith("v"):
        normalized = normalized[1:]
    return _VERSION_ALIASES.get(normalized, normalized)


def resolve_psilca_basis(
    psilca_version: str,
    *,
    currency_override: str | None = None,
    year_override: int | None = None,
) -> MonetaryBasis:
    """Resolve PSILCA's representative worker-hour monetary basis.

    An explicit year takes priority. Unknown versions intentionally require an
    explicit year instead of silently guessing.

    Args:
        psilca_version: Version label used to select a registered profile.
        currency_override: Currency declared for a custom PSILCA export.
        year_override: Representative price year selected by the user.

    Returns:
        The declared or version-derived PSILCA monetary basis.
    """

    version_key = normalize_psilca_version(psilca_version)
    profile = _PSILCA_PROFILES.get(version_key)

    if year_override is None and profile is None:
        supported = ", ".join(sorted(_PSILCA_PROFILES))
        raise ValueError(
            f"No representative monetary year is registered for PSILCA "
            f"{psilca_version!r}. Set SenseConfig.psilca_price_year explicitly. "
            f"Known versions: {supported}."
        )

    currency = currency_override or (profile.currency if profile else "USD")
    if year_override is not None:
        return MonetaryBasis(
            currency=currency,
            year=int(year_override),
            approximate=True,
            note=(
                "User-selected representative year. PSILCA EcoSpold01 does not "
                "provide a uniform machine-readable monetary year."
            ),
        )

    assert profile is not None  # narrowed above
    if currency_override is None:
        return profile
    return MonetaryBasis(
        currency=currency,
        year=profile.year,
        approximate=profile.approximate,
        note=profile.note,
    )


def resolve_price_conversion(
    source: MonetaryBasis,
    target: MonetaryBasis,
    *,
    factor_override: float | None = None,
) -> MonetaryConversion:
    """Resolve the multiplier that converts ``source`` prices to ``target``.

    Built-in automatic conversion deliberately covers only SENSE's documented
    defaults: EUR2005 to USD2011, USD2015 or USD2019. Other bases require an explicit
    factor, which keeps unsupported assumptions visible and reproducible.

    Args:
        source: Monetary basis of the input prices.
        target: Monetary basis required by the intensity data.
        factor_override: Explicit multiplier from ``source`` to ``target``.

    Returns:
        Validated conversion factor with provenance.
    """

    if factor_override is not None:
        factor = float(factor_override)
        if not math.isfinite(factor) or factor <= 0:
            raise ValueError("ecoinvent_to_psilca_price_factor must be positive and finite")
        return MonetaryConversion(
            source=source,
            target=target,
            factor=factor,
            method="Explicit user-supplied conversion factor.",
            user_supplied=True,
        )

    if source.currency == target.currency and source.year == target.year:
        return MonetaryConversion(
            source=source,
            target=target,
            factor=1.0,
            method="Identical source and target monetary bases; no conversion.",
        )

    is_supported_default = (
        source.currency == "EUR"
        and source.year == 2005
        and target.currency == "USD"
        and target.year in _ECB_USD_PER_EUR
        and source.year in _EURO_AREA_HICP
        and target.year in _EURO_AREA_HICP
    )
    if not is_supported_default:
        raise ValueError(
            f"No built-in conversion is available from {source.label} to "
            f"{target.label}. Set SenseConfig.ecoinvent_to_psilca_price_factor "
            "to the multiplier that converts source prices into the target basis."
        )

    inflation_factor = _EURO_AREA_HICP[target.year] / _EURO_AREA_HICP[source.year]
    exchange_rate = _ECB_USD_PER_EUR[target.year]
    factor = inflation_factor * exchange_rate
    method = (
        "Inflate EUR2005 with the Eurostat euro-area all-items HICP annual-average "
        f"index ({_EURO_AREA_HICP[source.year]} to {_EURO_AREA_HICP[target.year]}), "
        f"then apply the ECB {target.year} annual-average reference rate "
        f"({exchange_rate} USD/EUR). Derived by SENSE from Eurostat series "
        f"PRC_HICP_AIND and ECB series EXR.A.USD.EUR.SP00.A, accessed "
        f"{DATA_ACCESS_DATE}. Sources: {EUROSTAT_HICP_SOURCE}; "
        f"{ECB_EXCHANGE_RATE_SOURCE}."
    )
    return MonetaryConversion(
        source=source,
        target=target,
        factor=factor,
        method=method,
    )


def resolve_psilca_price_conversion(
    psilca_version: str,
    *,
    ecoinvent_currency: str = "EUR",
    ecoinvent_year: int = 2005,
    psilca_currency: str | None = None,
    psilca_year: int | None = None,
    factor_override: float | None = None,
) -> MonetaryConversion:
    """Resolve the ecoinvent-price conversion for a PSILCA version."""

    source = MonetaryBasis(ecoinvent_currency, int(ecoinvent_year))
    target = resolve_psilca_basis(
        psilca_version,
        currency_override=psilca_currency,
        year_override=psilca_year,
    )
    return resolve_price_conversion(source, target, factor_override=factor_override)
