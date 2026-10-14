"""Direct working-hour calculations independent of social-sector matching."""

from __future__ import annotations

import math
from typing import Any

from sense.utils.helpers import norm_code


def build_productivity_working_hour_records(
    activities: list[dict[str, Any]],
    intensities: dict[str, float],
    price_factor: float,
    direct_hours_override: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Convert priced activities and productivity intensities to working hours.

    Args:
        activities: Activity metadata using the canonical SENSE output keys.
        intensities: Activity code to positive hours-per-monetary-unit values.
        price_factor: Multiplier aligning activity prices with the productivity
            baseline's monetary basis.

    Returns:
        Brightway-ready activity records with direct working-hour amounts.

    Raises:
        ValueError: If an activity code, price, intensity, or factor is invalid.
    """

    try:
        factor = float(price_factor)
    except (TypeError, ValueError) as exc:
        raise ValueError("productivity price factor must be positive and finite") from exc
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError("productivity price factor must be positive and finite")

    records: list[dict[str, Any]] = []
    direct_hours_override = direct_hours_override or {}
    seen: set[str] = set()
    for activity in activities:
        code = norm_code(activity.get("Activity code"))
        if not code:
            raise ValueError("Productivity-only activity is missing its activity code")
        if code in seen:
            raise ValueError(f"Duplicate productivity-only activity code: {code!r}")
        seen.add(code)

        if code in direct_hours_override:
            hours = float(direct_hours_override[code])
            if not math.isfinite(hours) or hours <= 0:
                raise ValueError(f"Invalid direct working hours for {code!r}")
            records.append(
                {
                    **activity,
                    "Activity code": code,
                    "Working_hours": hours,
                    "Working_hours_source": "treatment_fallback",
                    "Averaged_Exchanges": [],
                }
            )
            continue

        raw_price = activity.get("Price")
        if raw_price is None:
            raise ValueError(f"Activity {code!r} has no valid price")
        try:
            price = float(raw_price)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Activity {code!r} has no valid price") from exc
        if not math.isfinite(price) or price <= 0:
            raise ValueError(f"Activity {code!r} requires a positive finite price")

        try:
            intensity = float(intensities[code])
        except KeyError as exc:
            raise ValueError(f"Activity {code!r} has no productivity intensity") from exc
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Activity {code!r} has an invalid productivity intensity") from exc
        if not math.isfinite(intensity) or intensity <= 0:
            raise ValueError(f"Activity {code!r} requires a positive productivity intensity")

        converted_price = price * factor
        working_hours = intensity * converted_price
        if not math.isfinite(working_hours) or working_hours <= 0:
            raise ValueError(f"Activity {code!r} produced invalid working hours")

        records.append(
            {
                **activity,
                "Activity code": code,
                "Price": price,
                "Price_conversion_factor": factor,
                "Converted_price": converted_price,
                "Averaged_Exchanges": [],
                "Working_hours": working_hours,
                "Working_hours_source": "external_productivity",
            }
        )
    return records
