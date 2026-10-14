"""Construct the activity-specific price graph and infer gross input costs."""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

from sense.utils.helpers import norm_code

from .prices import ProdKey, prod_key, reference_price_property

logger = logging.getLogger(__name__)

PRICE_INFERENCE_VERSION = 6

# ---- Types ----
ActivityKey = tuple[str, str]  # exact Brightway key: (database, activity code)


@dataclass
class ProductGraphCache:
    """In-memory product graph extracted from a Brightway database."""

    activity_to_id: dict[ActivityKey, int]
    id_to_product: list[ProdKey]
    id_to_meta: list[dict[str, Any]]

    recipe_out: list[int]
    recipe_inputs: list[list[tuple[int, float]]]
    recipe_act_db: list[str]
    recipe_act_code: list[str]
    recipe_act_name: list[str]
    recipe_act_location: list[str]
    producers_by_pid: list[list[int]]

    pid_activity_price_seed: dict[int, float]


def build_product_graph(
    target_database: str,
    biosphere_database: str = "biosphere3",
    *,
    activities: Iterable[Any] | None = None,
    should_expand_external: Callable[[Any], bool] | None = None,
    expected_price_unit: str | None = None,
    resolve_price: Callable[[Any], float | None] | None = None,
) -> ProductGraphCache:
    """Build a product graph from a Brightway database.

    Iterates the target database once. For each activity:
    - Creates one product node per exact Brightway activity key
    - Records a recipe using exact linked supplier activities
    - Captures activity-level or reference-exchange prices as seeds

    When ``should_expand_external`` is supplied, linked activities outside the
    target database are added recursively only when the callback returns true.
    This lets callers follow unpriced foreground dependencies without expanding
    already priced background databases.

    Args:
        target_database: Name of the Brightway database to analyze.
        biosphere_database: Name of the biosphere database (excluded from inputs).
        activities: Optional preloaded activity snapshot. Supplying it avoids a
            second complete database scan when metadata was already collected.
        should_expand_external: Optional predicate selecting linked activities
            whose recipes must also be included in the graph.
        expected_price_unit: Monetary unit required on reference-exchange price
            properties, such as ``"EUR2005"``. Activity-level numeric prices
            are assumed to use this configured basis.
        resolve_price: Shared price resolver used by the extraction pipeline.
            When provided, replaces the graph's standalone seed lookup.

    Returns:
        ProductGraphCache with the complete product graph.
    """
    from bw2data import Database
    from tqdm import tqdm

    activity_queue = list(Database(target_database) if activities is None else activities)
    activity_by_key = {tuple(activity.key): activity for activity in activity_queue}
    queued_keys = {tuple(activity.key) for activity in activity_queue}
    external_expansion: dict[ActivityKey, bool] = {}

    activity_to_id: dict[ActivityKey, int] = {}
    id_to_product: list[ProdKey] = []
    id_to_meta: list[dict[str, Any]] = []

    recipe_out: list[int] = []
    recipe_inputs: list[list[tuple[int, float]]] = []
    recipe_act_db: list[str] = []
    recipe_act_code: list[str] = []
    recipe_act_name: list[str] = []
    recipe_act_location: list[str] = []
    pid_activity_price_seed: dict[int, float] = {}

    def get_pid(
        activity_key: ActivityKey,
        product_key: ProdKey,
        meta: dict[str, Any] | None = None,
    ) -> int:
        pid = activity_to_id.get(activity_key)
        if pid is not None:
            return pid
        pid = len(id_to_product)
        activity_to_id[activity_key] = pid
        id_to_product.append(product_key)
        id_to_meta.append(
            meta
            or {
                "database": activity_key[0],
                "code": activity_key[1],
                "unit": product_key[0],
                "reference product": product_key[1],
                "location": "",
            }
        )
        return pid

    def capture_activity_price(activity: Any, pid: int) -> None:
        price: float | None
        if resolve_price is not None:
            price = resolve_price(activity)
            if price is not None:
                pid_activity_price_seed[pid] = price
            return
        raw_price = activity.get("price")
        try:
            price = float(raw_price)
        except (TypeError, ValueError):
            price = None
        if price is None:
            reference_price = reference_price_property(activity)
            if reference_price is None:
                price = None
            else:
                price, monetary_unit = reference_price
                if expected_price_unit is not None and (
                    monetary_unit.replace(" ", "").casefold()
                    != expected_price_unit.replace(" ", "").casefold()
                ):
                    raise ValueError(
                        f"Activity {activity.key!r} uses price unit {monetary_unit!r}; "
                        f"expected {expected_price_unit}"
                    )
        if price is None:
            return
        if np.isfinite(price) and price > 0 and pid not in pid_activity_price_seed:
            pid_activity_price_seed[pid] = price

    logger.info("Building product graph from '%s'...", target_database)
    progress = tqdm(
        total=len(activity_queue),
        desc=f"Building product graph: {target_database}",
        unit="activity",
        dynamic_ncols=True,
    )
    queue_index = 0
    try:
        while queue_index < len(activity_queue):
            act = activity_queue[queue_index]
            queue_index += 1
            code = norm_code(act.get("code")) or str(act.key[1])
            name = act.get("name", "")
            ref_product = str(act.get("reference product") or name or "").strip()
            unit = str(act.get("unit", "")).lower().strip()
            location = act.get("location", "")
            act_db = act.key[0]

            meta = {
                "database": act_db,
                "code": code,
                "name": name,
                "reference product": ref_product,
                "unit": unit,
                "location": location,
            }

            out_pk = prod_key(unit, ref_product)
            out_activity_key = (str(act_db), code)
            out_pid = get_pid(out_activity_key, out_pk, meta=meta)

            capture_activity_price(act, out_pid)

            # Collect purchased technosphere inputs. Negative exchanges represent
            # outputs or credits, not expenditure, and are excluded from this gross
            # production-cost proxy. Resolve suppliers from the snapshot first;
            # ``exc.input`` performs a database lookup per exchange.
            coeffs: dict[int, float] = defaultdict(float)
            for exc in act.technosphere():
                try:
                    amt = float(exc["amount"])
                except (KeyError, TypeError, ValueError) as exc_error:
                    raise ValueError(
                        f"Invalid technosphere amount on activity {act.key!r}"
                    ) from exc_error
                if not np.isfinite(amt):
                    raise ValueError(f"Non-finite technosphere amount on activity {act.key!r}")
                if amt <= 0:
                    continue

                try:
                    input_key = tuple(exc["input"])
                except (KeyError, TypeError):
                    inp = exc.input
                    input_key = tuple(inp.key)
                else:
                    inp = activity_by_key.get(input_key)
                    if inp is None:
                        inp = exc.input
                        activity_by_key[input_key] = inp

                inp_ref = str(inp.get("reference product") or inp.get("name") or "").strip()
                inp_unit = str(inp.get("unit", "")).lower().strip()
                inp_db = input_key[0]

                if inp_db == biosphere_database:
                    continue

                inp_code = norm_code(inp.get("code")) or str(inp.key[1])
                inp_pk = prod_key(inp_unit, inp_ref)
                inp_pid = get_pid(
                    (str(inp_db), inp_code),
                    inp_pk,
                    meta={
                        "database": inp_db,
                        "code": inp_code,
                        "name": inp.get("name", ""),
                        "reference product": inp_ref,
                        "unit": inp_unit,
                        "location": inp.get("location", ""),
                    },
                )
                capture_activity_price(inp, inp_pid)
                coeffs[inp_pid] += amt

                normalized_input_key = (str(inp_db), inp_code)
                if should_expand_external is not None and normalized_input_key not in queued_keys:
                    if normalized_input_key not in external_expansion:
                        external_expansion[normalized_input_key] = should_expand_external(inp)
                    if external_expansion[normalized_input_key]:
                        activity_queue.append(inp)
                        queued_keys.add(normalized_input_key)
                        progress.total += 1

            # Brightway usually normalizes the reference exchange to one unit, but
            # custom foreground databases need not do so. Ecoinvent waste-treatment
            # activities use a negative reference amount because the waste is
            # consumed. Prices are still expressed per unit, so normalize input
            # costs by the magnitude of either kind of reference exchange.
            from sense.brightway.repository import reference_production_amount

            output_amount = reference_production_amount(act.key, list(act.production()))

            normalized_coeffs = [
                (input_pid, amount / output_amount) for input_pid, amount in coeffs.items()
            ]

            rid = len(recipe_out)
            recipe_out.append(out_pid)
            recipe_inputs.append(normalized_coeffs)
            recipe_act_db.append(act_db)
            recipe_act_code.append(code)
            recipe_act_name.append(name)
            recipe_act_location.append(location)
            progress.update()

    finally:
        progress.close()

    producers_by_pid: list[list[int]] = [[] for _ in range(len(id_to_product))]
    for rid, out_pid in enumerate(recipe_out):
        producers_by_pid[out_pid].append(rid)

    logger.info(
        "  Product graph: %d products, %d recipes, %d with activity price seeds",
        len(id_to_product),
        len(recipe_out),
        len(pid_activity_price_seed),
    )

    return ProductGraphCache(
        activity_to_id=activity_to_id,
        id_to_product=id_to_product,
        id_to_meta=id_to_meta,
        recipe_out=recipe_out,
        recipe_inputs=recipe_inputs,
        recipe_act_db=recipe_act_db,
        recipe_act_code=recipe_act_code,
        recipe_act_name=recipe_act_name,
        recipe_act_location=recipe_act_location,
        producers_by_pid=producers_by_pid,
        pid_activity_price_seed=pid_activity_price_seed,
    )


def run_product_inference(
    cache: ProductGraphCache,
    initial_prices: dict[int, float],
    initial_sources: dict[int, str],
) -> tuple[dict[int, float], dict[int, str]]:
    """Run BFS price inference across the product graph.

    For each unpriced activity product, checks whether its recipe
    has all its inputs priced. If so, the product price = sum of
    (input_amount × input_price) across that recipe's inputs.

    Args:
        cache: Product graph from build_product_graph().
        initial_prices: {product_id: price} for seeded products.
        initial_sources: {product_id: source_label} for seeded products.

    Returns:
        (prices_by_pid, source_by_pid) — complete price dict after inference.
    """
    prices: dict[int, float] = dict(initial_prices)
    if any(not np.isfinite(value) or value < 0 for value in prices.values()):
        raise ValueError("Initial prices must be finite and non-negative")
    source: dict[int, str] = dict(initial_sources)
    n_recipes = len(cache.recipe_out)

    # Build dependency tracking
    dependents: dict[int, list[tuple[int, float]]] = defaultdict(list)
    missing_count = [0] * n_recipes
    partial_sum = [0.0] * n_recipes
    ready: deque[int] = deque()

    for rid in range(n_recipes):
        out_pid = cache.recipe_out[rid]
        if out_pid in prices:
            continue

        # An activity without technosphere inputs has no cost recipe. Pricing
        # it at zero would silently omit primary inputs and value added; leave
        # it unresolved so the seed workflow requests an explicit price.
        if not cache.recipe_inputs[rid]:
            continue

        miss = 0
        total = 0.0
        for inp_pid, coeff in cache.recipe_inputs[rid]:
            if not np.isfinite(coeff) or coeff < 0:
                raise ValueError("Price-inference coefficients must be finite and non-negative")
            p = prices.get(inp_pid)
            if p is not None:
                total += coeff * p
            else:
                miss += 1
                dependents[inp_pid].append((rid, coeff))

        missing_count[rid] = miss
        partial_sum[rid] = total
        if miss == 0 and total >= 0:
            ready.append(rid)

    resolved_new = 0
    while ready:
        rid = ready.popleft()
        out_pid = cache.recipe_out[rid]
        if out_pid in prices:
            continue

        val = partial_sum[rid]
        if not np.isfinite(val):
            raise ValueError("Price inference overflowed; check input prices and quantities")
        if val is None or val < 0:
            continue

        prices[out_pid] = float(val)
        source[out_pid] = "inferred_from_inputs"
        resolved_new += 1

        for dep_rid, coeff in dependents.get(out_pid, []):
            out2 = cache.recipe_out[dep_rid]
            if out2 in prices:
                continue
            partial_sum[dep_rid] += coeff * float(val)
            missing_count[dep_rid] -= 1
            if missing_count[dep_rid] == 0 and partial_sum[dep_rid] >= 0:
                ready.append(dep_rid)

    logger.info(
        "  Inference: %d products newly resolved, %d total priced", resolved_new, len(prices)
    )
    return prices, source
