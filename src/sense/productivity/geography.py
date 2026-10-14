"""Map ecoinvent geographies to weighted ISO3 productivity countries."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from sense.geography import read_ecoinvent_geographies
from sense.weighting import CountryWeightProvider, iso2_to_iso3

logger = logging.getLogger(__name__)


def _parse_geographies(
    ecoinvent_geographies_excel: Path,
) -> pd.DataFrame:
    """Load the Geographies sheet and resolve each location to ISO2 codes.

    Returns a DataFrame with columns:
        Shortname, Name, Geography Classification, simplified_ISO2
    """
    return read_ecoinvent_geographies(ecoinvent_geographies_excel)


def build_location_mapping(
    ecoinvent_geographies_excel: Path,
    available_iso3: set[str],
    weight_provider: CountryWeightProvider,
) -> dict[str, list[tuple[str, float]]]:
    """Map each ecoinvent geography to weighted ISO3 codes from the baseline.

    Args:
        ecoinvent_geographies_excel: Path to ecoinvent Database Overview Excel
            (must contain a "Geographies" sheet).
        available_iso3: Set of ISO3 country codes present in the productivity
            baseline. Missing constituents are retained so lookup coverage remains honest.
        weight_provider: Shared SENSE country-weight provider used for every
            composite location.

    Returns:
        Mapping from ecoinvent shortname to normalized ``(ISO3, weight)``
        pairs. Unresolved locations are omitted and logged.
    """
    logger.info("Building ecoinvent → productivity baseline location mapping...")

    geo_df = _parse_geographies(ecoinvent_geographies_excel)
    result: dict[str, list[tuple[str, float]]] = {}
    unmapped: list[str] = []

    # Locations resolved at activity level, not here
    _DEFERRED_LOCATIONS = {"RoW", "RoE"}

    for _, row in geo_df.iterrows():
        shortname = str(row["Shortname"]).strip()
        iso2_string = row.get("simplified_ISO2")

        if shortname in _DEFERRED_LOCATIONS:
            continue

        if not iso2_string or pd.isna(iso2_string):
            unmapped.append(shortname)
            continue

        # Parse ISO2 codes (single or semicolon-separated)
        iso2_list = [c.strip() for c in str(iso2_string).split(";") if c.strip()]

        # Keep the full country universe, including countries absent from the baseline.
        iso3_candidates = []
        for iso2 in iso2_list:
            iso3 = iso2_to_iso3(iso2)
            if iso3 and iso3 not in iso3_candidates:
                iso3_candidates.append(iso3)

        if not iso3_candidates:
            unmapped.append(shortname)
            continue

        # Single country → weight 1.0
        if len(iso3_candidates) == 1:
            result[shortname] = [(iso3_candidates[0], 1.0)]
            continue

        # Multiple countries → use the same provider as social aggregation.
        weights = weight_provider.resolve_iso3(iso3_candidates).weights

        # Keep every constituent for completeness checks. A zero placeholder
        # records an unavailable weight; it does not assert zero economic size.
        weighted = [(iso3, weights.get(iso3, 0.0)) for iso3 in iso3_candidates]

        # Re-normalize after filtering
        total = sum(w for _, w in weighted)
        if total > 0:
            weighted = [(iso3, w / total) for iso3, w in weighted]

        if weighted:
            result[shortname] = weighted
        else:
            unmapped.append(shortname)

    # Summary logging
    n_single = sum(1 for v in result.values() if len(v) == 1)
    n_composite = sum(1 for v in result.values() if len(v) > 1)
    logger.info(
        f"  Mapped {len(result)} ecoinvent locations "
        f"({n_single} single-country, {n_composite} composite)"
    )
    if unmapped:
        logger.warning(
            f"  {len(unmapped)} locations could not be mapped to any baseline country: "
            f"{unmapped[:15]}{'...' if len(unmapped) > 15 else ''}"
        )

    return result
