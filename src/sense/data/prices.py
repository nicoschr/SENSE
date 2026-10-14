"""Resolve and validate reference-product prices on an explicit monetary basis."""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from sense.utils.helpers import norm_code

logger = logging.getLogger(__name__)

# ---- Types ----
ProdKey = tuple[str, str]  # user-facing seed key: (unit, normalized product)


@dataclass(frozen=True)
class BrightwayPriceCatalog:
    """Validated ecoinvent prices retained by the Brightway importer."""

    source_database: str
    by_code: dict[str, tuple[ProdKey, float]]
    by_product: dict[ProdKey, float]


def reference_price_property(activity: Any) -> tuple[float, str] | None:
    """Return the reference exchange's price amount and monetary unit, if present."""

    productions = list(activity.production())
    activity_key = tuple(activity.key)
    references = [
        exchange for exchange in productions if tuple(exchange.get("input") or ()) == activity_key
    ]
    if not references and len(productions) == 1:
        references = productions
    if len(references) != 1:
        return None

    price = (references[0].get("properties") or {}).get("price")
    if price is None:
        return None
    if not isinstance(price, dict) or price.get("amount") is None or not price.get("unit"):
        raise ValueError(
            f"Activity {activity.key!r} has an incomplete reference-exchange price property"
        )
    try:
        amount = float(price["amount"])
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Activity {activity.key!r} has a non-numeric reference-exchange price"
        ) from error
    if not np.isfinite(amount) or amount < 0:
        raise ValueError(
            f"Activity {activity.key!r} has an invalid reference-exchange price: {amount!r}"
        )
    return amount, str(price["unit"]).strip()


def build_brightway_price_catalog(
    activities: Sequence[Any],
    *,
    source_database: str,
    price_currency: str,
    price_year: int,
    price_floor_percentile: float,
) -> BrightwayPriceCatalog:
    """Extract exact and product-average prices from an ecoinvent Brightway database."""

    if not 0 <= price_floor_percentile <= 100:
        raise ValueError("price_floor_percentile must be between 0 and 100")
    expected_unit = f"{price_currency}{price_year}".replace(" ", "").casefold()
    raw_by_code: dict[str, tuple[ProdKey, float]] = {}
    raw_by_product: dict[ProdKey, list[float]] = defaultdict(list)

    from tqdm import tqdm

    for activity in tqdm(
        activities,
        desc=f"Reading prices: {source_database}",
        unit="activity",
        dynamic_ncols=True,
    ):
        result = reference_price_property(activity)
        if result is None:
            continue
        price, monetary_unit = result
        if monetary_unit.replace(" ", "").casefold() != expected_unit:
            raise ValueError(
                f"Activity {activity.key!r} uses price unit {monetary_unit!r}; "
                f"expected {price_currency}{price_year}"
            )
        code = norm_code(activity.get("code")) or str(activity.key[1])
        product = prod_key(
            str(activity.get("unit") or ""),
            str(activity.get("reference product") or activity.get("name") or ""),
        )
        if code in raw_by_code:
            raise ValueError(f"Duplicate activity code in price source {source_database!r}: {code}")
        raw_by_code[code] = (product, price)
        raw_by_product[product].append(price)

    q = float(price_floor_percentile) / 100.0
    raw_product_means = {
        product: float(sum(values) / len(values)) for product, values in raw_by_product.items()
    }
    floor_by_unit: dict[str, float] = {}
    if q > 0:
        values_by_unit: dict[str, list[float]] = defaultdict(list)
        for (unit, _), value in raw_product_means.items():
            if value > 0:
                values_by_unit[unit].append(value)
        floor_by_unit = {
            unit: float(pd.Series(values).quantile(q)) for unit, values in values_by_unit.items()
        }

    def clean(product: ProdKey, price: float) -> float:
        return floor_by_unit.get(product[0], 0.0) if price == 0 else price

    by_code = {
        code: (product, clean(product, price)) for code, (product, price) in raw_by_code.items()
    }
    by_product = {
        product: float(sum(clean(product, price) for price in prices) / len(prices))
        for product, prices in raw_by_product.items()
    }
    return BrightwayPriceCatalog(
        source_database=source_database,
        by_code=by_code,
        by_product=by_product,
    )


def normalize_name(name: str) -> str:
    """Normalize a product name for product-level seed matching."""

    return re.sub(r"\s+", " ", (name or "").lower().strip().replace(",", "").replace("-", ""))


def prod_key(unit: str, reference_product: str) -> ProdKey:
    """Return the normalized unit-product key used by manual price seeds."""

    return (str(unit or "").lower().strip(), normalize_name(reference_product))
