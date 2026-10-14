"""Physical working-hour fallbacks and their auditable reference distribution."""

import math
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

import numpy as np


def canonical_unit(unit: str) -> str:
    """Normalize only established aliases, without combining different units."""
    value = " ".join(str(unit).casefold().split())
    return {"kg": "kilogram", "m3": "cubic meter", "m³": "cubic meter", "kwh": "kilowatt hour"}.get(
        value, value
    )


def reference_distribution(
    records: Iterable[dict[str, Any]],
    percentile: float,
) -> dict[str, dict[str, float | int]]:
    """Take one positive ordinary direct-hour observation per activity and unit.

    Risk factors and treatment placeholders are not observations. Linear
    interpolation is explicit, and sorting makes the calculation order invariant.
    """
    if not math.isfinite(percentile) or not 0 < percentile <= 100:
        raise ValueError("Treatment reference percentile must be greater than 0 and at most 100")
    values: dict[str, list[float]] = defaultdict(list)
    seen = set()
    for record in records:
        if record.get("Working_hours_source") == "treatment_fallback":
            continue
        price = record.get("Price")
        hours = record.get("Working_hours")
        if price is None or float(price) <= 0:
            continue
        if not math.isfinite(float(price)):
            raise ValueError(f"Invalid reference price for {record.get('Activity code')}")
        if hours is None:
            raise ValueError(f"Missing reference working hours for {record.get('Activity code')}")
        unit = canonical_unit(record["Unit"])
        identity = record["Activity code"]
        if identity in seen:
            raise ValueError(f"Duplicate reference activity: {identity}")
        seen.add(identity)
        hours = float(hours)
        if not math.isfinite(hours) or hours <= 0:
            raise ValueError(f"Invalid reference working hours for {identity}")
        values[unit].append(hours)
    return {
        unit: {
            "hours": float(np.quantile(sorted(hours), percentile / 100, method="linear")),
            "donor_count": len(hours),
        }
        for unit, hours in sorted(values.items())
    }


def fallback_overrides(
    activities: list[dict], reference: dict, *, profiles_only=False
) -> dict[str, float]:
    """Resolve hours for eligible treatments, failing before exchange writes."""
    result = {}
    for activity in activities:
        if activity.get("calculation route") != "treatment_fallback":
            continue
        unit = canonical_unit(activity["unit"])
        if profiles_only:
            value = 1.0  # Risk factors only, never a reference observation.
        else:
            if unit not in reference:
                raise ValueError(
                    f"No treatment working-hour reference for unit {unit!r}; supply a justified positive price or a reference database with ordinary hours for this unit"
                )
            value = float(reference[unit]["hours"])
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"Invalid treatment fallback for unit {unit!r}")
        result[str(activity["code"])] = value
    return result


def finalize_profiles(records: list[dict], reference: dict, rounding: int | None) -> list[dict]:
    """Scale prepared treatment risk profiles by physical hours exactly once."""
    for record in records:
        if record.get("Working_hours_source") == "treatment_fallback":
            unit = canonical_unit(record["Unit"])
            if unit not in reference:
                raise ValueError(f"No treatment working-hour reference for unit {unit!r}")
            hours = float(reference[unit]["hours"])
            if not math.isfinite(hours) or hours <= 0:
                raise ValueError(f"Invalid treatment fallback for unit {unit!r}")
            if record["Working_hours"] != 1.0 or record.get("Fallback_finalized"):
                raise ValueError("Treatment profile was already finalized")
            record["Working_hours"] = hours
            record["Fallback_unit"] = unit
            record["Fallback_donor_count"] = reference[unit]["donor_count"]
            record["Fallback_finalized"] = True
            for exchange in record["Averaged_Exchanges"]:
                exchange["meanValue"] *= hours
        if rounding is not None:
            for exchange in record["Averaged_Exchanges"]:
                value = exchange["meanValue"]
                exchange["meanValue"] = round(value, rounding) or value
    return records


def annotate_fallbacks(records: list[dict], reference: dict) -> None:
    """Attach the same provenance to directly calculated and prepared records."""
    for record in records:
        if record.get("Working_hours_source") != "treatment_fallback":
            continue
        unit = canonical_unit(record["Unit"])
        observation = reference[unit]
        if not math.isclose(
            float(record["Working_hours"]), float(observation["hours"]), rel_tol=1e-12
        ):
            raise ValueError(
                f"Treatment hours disagree with the reference for {record['Activity code']!r}"
            )
        record["Fallback_unit"] = unit
        record["Fallback_donor_count"] = observation["donor_count"]
        record["Fallback_finalized"] = True
