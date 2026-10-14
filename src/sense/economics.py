"""Monetary, country-weight, and productivity setup for a SENSE run."""

from __future__ import annotations

import logging
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from sense.artifacts import atomic_output_path, write_json_atomic
from sense.config import SenseConfig
from sense.monetary import MonetaryConversion, resolve_psilca_price_conversion
from sense.productivity.geography import build_location_mapping
from sense.productivity.loader import ProductivityBaseline
from sense.weighting import CountryWeightProvider

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EconomicContext:
    """Resolved economic assumptions and optional productivity data."""

    psilca_conversion: MonetaryConversion | None
    country_weights: CountryWeightProvider
    productivity: ProductivityBaseline | None
    productivity_price_factor: float
    productivity_conversion_method: str | None


def _country_weight_provider(cfg: SenseConfig, reference_year: int) -> CountryWeightProvider:
    gdp_path = cfg.country_gdp_path
    if cfg.country_weighting == "equal":
        if gdp_path is not None:
            logger.warning("GDP input is ignored because country_weighting='equal'")
        return CountryWeightProvider.equal(reference_year=reference_year)
    if gdp_path is not None:
        return CountryWeightProvider.from_csv(Path(gdp_path), reference_year=reference_year)
    return CountryWeightProvider.bundled_gdp(reference_year=reference_year)


def resolve_economic_context(
    *,
    config: SenseConfig,
    psilca_version: str | None,
    ecoinvent_geographies_excel: Path,
) -> EconomicContext:
    """Resolve and validate all economic assumptions before database mutation."""

    conversion: MonetaryConversion | None = None
    if config.use_psilca:
        if not psilca_version:
            raise ValueError("psilca_version is required when use_psilca=True")
        conversion = resolve_psilca_price_conversion(
            psilca_version,
            ecoinvent_currency=config.ecoinvent_price_currency,
            ecoinvent_year=config.ecoinvent_price_year,
            psilca_currency=config.psilca_price_currency,
            psilca_year=config.psilca_price_year,
            factor_override=config.ecoinvent_to_psilca_price_factor,
        )
        logger.info(
            "Monetary alignment: %s to %s; price factor %.9g",
            conversion.source.label,
            conversion.target.label,
            conversion.factor,
        )
        if conversion.target.approximate:
            logger.info("PSILCA monetary year is representative: %s", conversion.target.note)

    reference_year = config.country_weight_year or (
        conversion.target.year if conversion is not None else config.productivity_year
    )
    weight_provider = _country_weight_provider(config, int(reference_year))
    productivity, factor, method = _load_productivity(
        config=config,
        workbook=ecoinvent_geographies_excel,
        weight_provider=weight_provider,
    )
    return EconomicContext(conversion, weight_provider, productivity, factor, method)


def _load_productivity(
    *,
    config: SenseConfig,
    workbook: Path,
    weight_provider: CountryWeightProvider,
) -> tuple[ProductivityBaseline | None, float, str | None]:
    if config.productivity_baseline_path is None:
        return None, 1.0, None
    baseline_path = Path(config.productivity_baseline_path)
    if not baseline_path.is_file():
        raise FileNotFoundError(f"Productivity baseline not found: {baseline_path}")
    countries = pd.read_parquet(baseline_path, columns=["country"])
    location_mapping = build_location_mapping(
        ecoinvent_geographies_excel=workbook,
        available_iso3=set(countries["country"].unique()),
        weight_provider=weight_provider,
    )
    baseline = ProductivityBaseline(
        parquet_path=baseline_path,
        location_mapping=location_mapping,
        reference_year=config.productivity_year,
        weight_provider=weight_provider,
    )
    applied_proxies = baseline.apply_location_proxies(config.productivity_location_proxies)
    for location, donors in sorted(applied_proxies.items()):
        logger.info("Productivity location proxy: %s -> %s", location, ", ".join(donors))

    divisor: float | None = None
    method: str | None = None
    if config.productivity_eur_per_usd is not None:
        divisor = float(config.productivity_eur_per_usd)
        method = "Explicit productivity_eur_per_usd divisor supplied by the user."
    if divisor is not None and (not math.isfinite(divisor) or divisor <= 0):
        raise ValueError("Productivity price divisor must be positive and finite")
    if divisor is None:
        raise ValueError(
            "External productivity requires an explicit productivity monetary "
            "conversion. Set productivity_eur_per_usd (use 1.0 only when the "
            "activity prices already match the baseline)."
        )
    return baseline, 1.0 / divisor, method


def write_economic_assumptions(
    *,
    output_dir: Path,
    ecoinvent_tag: str,
    psilca_tag: str,
    ecoinvent_version: str,
    psilca_version: str | None,
    config: SenseConfig,
    context: EconomicContext,
) -> None:
    """Persist monetary and country-weight assumptions for reproducibility."""

    monetary_path = (
        output_dir / "Monetary" / f"monetary_conversion_{ecoinvent_tag}_{psilca_tag}.json"
    )
    write_json_atomic(
        monetary_path,
        {
            "schema_version": 1,
            "ecoinvent_version": ecoinvent_version,
            "psilca_enabled": config.use_psilca,
            "psilca_version": psilca_version,
            "psilca_price_conversion": (
                context.psilca_conversion.as_dict()
                if context.psilca_conversion is not None
                else None
            ),
            "limitation": (
                "PSILCA EcoSpold01 does not expose one uniform, machine-readable "
                "monetary year; the selected target year is representative."
                if config.use_psilca
                else None
            ),
            "external_productivity_conversion": (
                {
                    "configured": True,
                    "factor": context.productivity_price_factor,
                    "operation": "target_price = ecoinvent_price * factor",
                    "method": context.productivity_conversion_method,
                    "target_year": config.productivity_year,
                    "location_proxies": config.productivity_location_proxies,
                }
                if context.productivity is not None
                else {"configured": False}
            ),
        },
    )
    weighting_path = (
        output_dir / "Weighting" / f"country_weighting_{ecoinvent_tag}_{psilca_tag}.json"
    )
    write_json_atomic(weighting_path, context.country_weights.audit_dict())


def write_productivity_audit(
    *,
    output_dir: Path,
    technosphere_database: str,
    results: list[dict[str, Any]],
) -> None:
    """Write rejected overrides and accepted GDP omissions to a run-level CSV."""

    misses = [row for result in results for row in result.get("productivity_audit_misses", [])]
    accepted = [
        row for result in results for row in result.get("productivity_audit_gdp_omissions", [])
    ]
    total = sum(int(result.get("num_activities") or 0) for result in results)
    applied = sum(int(result.get("productivity_override_applied") or 0) for result in results)
    logger.info(
        "Productivity override applied to %d/%d activities; misses=%s",
        applied,
        total,
        dict(Counter(row["status"] for row in misses)),
    )
    columns = [
        "ISO2",
        "activity_code",
        "activity_name",
        "reference_product",
        "isic_raw",
        "isic2",
        "year_requested",
        "status",
        "n_countries_attempted",
        "n_countries_used",
        "accepted_gdp_omissions",
        "weight_method",
        "omitted_gdp_countries",
        "missing_productivity_countries",
    ]
    path = (
        output_dir
        / "ProductivityAudit"
        / f"productivity_audit_{technosphere_database.replace(' ', '_')}.csv"
    )
    with atomic_output_path(path) as temporary:
        pd.DataFrame(misses + accepted, columns=columns).to_csv(temporary, index=False)
