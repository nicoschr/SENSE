"""Find minimal seed-price worklists and validate user-supplied prices."""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sense.artifacts import write_json_atomic

from .price_inference import ProductGraphCache
from .prices import ProdKey, normalize_name

logger = logging.getLogger(__name__)


def _tarjan_scc(graph: dict[int, set[int]]) -> list[list[int]]:
    """Find strongly connected components without a Python recursion-depth limit.

    Explicit DFS frames preserve the recursive algorithm's traversal and output
    order, including references to nodes without their own adjacency entry.
    """
    stack: list[int] = []
    lowlink: dict[int, int] = {}
    index: dict[int, int] = {}
    on_stack: set[int] = set()
    result: list[list[int]] = []

    def enter(v: int) -> tuple[int, Iterator[int]]:
        index[v] = lowlink[v] = len(index)
        stack.append(v)
        on_stack.add(v)
        return v, iter(graph.get(v, set()))

    for node in graph.keys():
        if node in index:
            continue
        frames = [enter(node)]
        while frames:
            v, neighbours = frames[-1]
            w = next(neighbours, None)
            if w is not None:
                if w not in index:
                    frames.append(enter(w))
                elif w in on_stack:
                    lowlink[v] = min(lowlink[v], index[w])
                continue

            frames.pop()
            if lowlink[v] == index[v]:
                comp = []
                while True:
                    member = stack.pop()
                    on_stack.remove(member)
                    comp.append(member)
                    if member == v:
                        break
                result.append(comp)
            if frames:
                parent = frames[-1][0]
                lowlink[parent] = min(lowlink[parent], lowlink[v])

    return result


def build_minimal_seed_worklist(
    cache: ProductGraphCache,
    priced: dict[int, float],
) -> pd.DataFrame:
    """Suggest source products that need manual prices.

    Analyzes the unresolved portion of the product graph to find:
    - External missing inputs (needed but no internal producer)
    - Products with no usable recipe
    - One representative per root SCC in the dependency graph

    This is a worklist heuristic, not a proof of a globally minimal seed set.
    Complex cycles can require another pass after the suggested seeds are filled.

    Args:
        cache: Product graph containing activities, products, and dependencies.
        priced: Prices already resolved, keyed by graph product ID.

    Returns:
        Required seeds with product identity, reason, and an example activity.
    """
    produced_pids: set[int] = set(cache.recipe_out)
    unresolved_outputs = [pid for pid in produced_pids if pid not in priced]
    if not unresolved_outputs:
        return pd.DataFrame()

    unresolved_set = set(unresolved_outputs)

    def example_producer(pid: int) -> dict[str, Any]:
        rids = cache.producers_by_pid[pid] if pid < len(cache.producers_by_pid) else []
        if not rids:
            return {
                "example_activity_name": None,
                "location": cache.id_to_meta[pid].get("location", ""),
            }
        rid = rids[0]
        return {
            "example_activity_name": cache.recipe_act_name[rid],
            "location": cache.recipe_act_location[rid],
        }

    rows: list[dict[str, Any]] = []

    # Select the recipe requiring the fewest unresolved inputs for each output.
    # Subsequent diagnostics must use the same recipe; otherwise an unused
    # alternative can create redundant seed requests.
    dep_graph: dict[int, set[int]] = defaultdict(set)
    best_recipe: dict[int, int] = {}
    external_dependencies: dict[int, set[int]] = defaultdict(set)

    for pid in unresolved_outputs:
        best: int | None = None
        best_score: tuple[int, int] | None = None
        for rid in cache.producers_by_pid[pid]:
            ins = cache.recipe_inputs[rid]
            if not ins:
                continue
            n_missing = sum(1 for input_pid, _ in ins if input_pid not in priced)
            score = (n_missing, len(ins))
            if best_score is None or score < best_score:
                best, best_score = rid, score
        if best is not None:
            best_recipe[pid] = best
            for input_pid, _ in cache.recipe_inputs[best]:
                if input_pid in unresolved_set:
                    dep_graph[pid].add(input_pid)
                elif input_pid not in priced and input_pid not in produced_pids:
                    external_dependencies[pid].add(input_pid)
        else:
            pk = cache.id_to_product[pid]
            rows.append(
                {
                    "unit": pk[0],
                    "reference_product": cache.id_to_meta[pid].get("reference product", pk[1]),
                    "seed_type": "no_recipe",
                    **example_producer(pid),
                }
            )

    # External missing inputs used by the selected recipes.
    seen_external: set[int] = set()
    for dependencies in external_dependencies.values():
        for input_pid in dependencies:
            if input_pid in seen_external:
                continue
            seen_external.add(input_pid)
            pk = cache.id_to_product[input_pid]
            rows.append(
                {
                    "unit": pk[0],
                    "reference_product": cache.id_to_meta[input_pid].get(
                        "reference product", pk[1]
                    ),
                    "seed_type": "external_missing",
                    **example_producer(input_pid),
                }
            )

    # Root SCCs
    for pid in unresolved_outputs:
        dep_graph.setdefault(pid, set())

    sccs = _tarjan_scc(dep_graph)
    scc_id = {}
    for i, comp in enumerate(sccs):
        for node in comp:
            scc_id[node] = i

    has_dep = [False] * len(sccs)
    for v in unresolved_outputs:
        vid = scc_id.get(v)
        if vid is None:
            continue
        if external_dependencies.get(v) or v not in best_recipe:
            has_dep[vid] = True
        for child in dep_graph.get(v, set()):
            cid = scc_id.get(child)
            if cid is not None and cid != vid:
                has_dep[vid] = True

    for i, comp in enumerate(sccs):
        if not comp or has_dep[i]:
            continue
        rep = comp[0]
        pk = cache.id_to_product[rep]
        rows.append(
            {
                "unit": pk[0],
                "reference_product": cache.id_to_meta[rep].get("reference product", pk[1]),
                "seed_type": "root_scc",
                **example_producer(rep),
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates(subset=["unit", "reference_product"]).copy()
    return df


def load_seed_template(
    seed_json_path: str,
    *,
    expected_currency: str | None = None,
    expected_year: int | None = None,
) -> dict[ProdKey, float]:
    """Load product prices entered in a SENSE-managed seed template.

    When expected monetary metadata is provided, conflicting file-level or
    entry-level declarations raise an error so differently based prices are not
    silently combined during recursive inference.

    Args:
        seed_json_path: SENSE-managed seed template to read.
        expected_currency: Required currency when validating declared metadata.
        expected_year: Required price year when validating declared metadata.

    Returns:
        Non-negative finite prices keyed by normalized unit and product name.
    """
    import json

    if not seed_json_path or not os.path.exists(seed_json_path):
        return {}

    with open(seed_json_path, encoding="utf-8") as f:
        data = json.load(f)

    expected_currency_norm = str(expected_currency).strip().upper() if expected_currency else None
    expected_year_norm = int(expected_year) if expected_year is not None else None
    metadata = data.get("metadata", {})
    declared_currency = metadata.get("expected_currency")
    declared_year = metadata.get("expected_price_year")

    if (
        expected_currency_norm
        and declared_currency
        and str(declared_currency).strip().upper() != expected_currency_norm
    ):
        raise ValueError(
            f"Seed file {seed_json_path} declares currency {declared_currency!r}, "
            f"but {expected_currency_norm!r} is configured. Convert the seed "
            "prices or use a matching ecoinvent_price_currency."
        )
    if (
        expected_year_norm is not None
        and declared_year is not None
        and int(declared_year) != expected_year_norm
    ):
        raise ValueError(
            f"Seed file {seed_json_path} declares price year {declared_year!r}, "
            f"but {expected_year_norm} is configured. Convert the seed prices "
            "or use a matching ecoinvent_price_year."
        )

    out: dict[ProdKey, float] = {}
    for s in data.get("seeds", []):
        price = s.get("price")
        if price is None:
            continue
        try:
            price = float(price)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Seed prices must be numeric: {seed_json_path}") from error
        if not np.isfinite(price):
            raise ValueError(f"Seed prices must be finite: {seed_json_path}")
        if price < 0:
            raise ValueError(f"Seed prices must be non-negative: {seed_json_path}")

        entry_currency = s.get("currency")
        entry_year = s.get("year")
        if (
            expected_currency_norm
            and entry_currency
            and str(entry_currency).strip().upper() != expected_currency_norm
        ):
            raise ValueError(
                f"Seed entry {s.get('reference_product')!r} in {seed_json_path} "
                f"declares currency {entry_currency!r}; expected "
                f"{expected_currency_norm!r}."
            )
        if (
            expected_year_norm is not None
            and entry_year is not None
            and int(entry_year) != expected_year_norm
        ):
            raise ValueError(
                f"Seed entry {s.get('reference_product')!r} in {seed_json_path} "
                f"declares year {entry_year!r}; expected {expected_year_norm}."
            )

        unit = str(s.get("unit", "")).lower().strip()
        ref = s.get("reference_product") or ""
        ref = str(ref).strip()
        if unit and ref:
            key = (unit, normalize_name(ref))
            if key in out and out[key] != price:
                raise ValueError(f"Conflicting seed prices for {key!r}")
            out[key] = price
        else:
            raise ValueError("Filled seed prices require a unit and reference_product")

    return out


def write_or_merge_seed_template(
    seed_json_path: str,
    worklist_df: pd.DataFrame,
    database_name: str,
    price_currency: str = "EUR",
    price_year: int = 2005,
) -> tuple[int, int, int]:
    """Write a seed price template or merge new missing products into an existing one.

    If the file does not exist, creates it with all products from worklist_df.
    If it exists, loads it, keeps all existing entries (preserving user-filled
    prices), and appends any new missing products not yet in the file.

    Args:
        seed_json_path: Path to the seed price JSON file.
        worklist_df: DataFrame from build_minimal_seed_worklist() with columns
            unit, reference_product, seed_type, example_activity_name.
        database_name: Name of the Brightway database (for metadata).
        price_currency: Expected currency for prices (informational).
        price_year: Expected price reference year (informational).

    Returns:
        (n_loaded, n_new, n_still_missing): count of seeds with valid prices,
        count of newly added template entries, count of entries still missing prices.
    """
    import json

    existing_seeds: list[dict] = []
    metadata: dict = {}

    if os.path.exists(seed_json_path):
        with open(seed_json_path, encoding="utf-8") as f:
            data = json.load(f)
        existing_seeds = data.get("seeds", [])
        metadata = data.get("metadata", {})

    required_keys: set[ProdKey] = set()
    for _, row in worklist_df.iterrows():
        unit = str(row.get("unit", "")).lower().strip()
        ref = str(row.get("reference_product", "")).strip()
        if unit and ref:
            required_keys.add((unit, normalize_name(ref)))

    # Index existing entries by (unit, normalized_product) for dedup
    existing_keys: dict[ProdKey, int] = {}
    for i, s in enumerate(existing_seeds):
        unit = str(s.get("unit", "")).lower().strip()
        ref = s.get("reference_product") or ""
        ref = str(ref).strip()
        if unit and ref:
            key = (unit, normalize_name(ref))
            existing_keys[key] = i
            s["required"] = key in required_keys

    # Merge new missing products
    n_new = 0
    for _, row in worklist_df.iterrows():
        unit = str(row.get("unit", "")).lower().strip()
        ref = str(row.get("reference_product", "")).strip()
        pk = (unit, normalize_name(ref))

        if pk in existing_keys:
            continue

        existing_seeds.append(
            {
                "unit": row.get("unit", ""),
                "reference_product": ref,
                "activity_name": row.get("example_activity_name", ""),
                "seed_type": row.get("seed_type", ""),
                "price": None,
                "currency": None,
                "year": None,
                "comment": None,
                "source": None,
                "required": True,
            }
        )
        existing_keys[pk] = len(existing_seeds) - 1
        n_new += 1

    # Count loaded vs still missing
    n_loaded = 0
    n_still_missing = 0
    for s in existing_seeds:
        if not s.get("required", False):
            continue
        price = s.get("price")
        if price is not None:
            try:
                if float(price) >= 0:
                    n_loaded += 1
                    continue
            except (TypeError, ValueError):
                pass
        n_still_missing += 1

    # Update metadata
    metadata.update(
        {
            "database": database_name,
            "expected_currency": price_currency,
            "expected_price_year": price_year,
            "note": (
                "Fill in 'price' for each product (in the expected currency and year). "
                "Every entry marked 'required' must have a non-negative price before "
                "the pipeline can continue. Historical entries are retained but do "
                "not block the current run. You can also fill in 'comment' and "
                "'source' for documentation."
            ),
        }
    )

    write_json_atomic(
        Path(seed_json_path),
        {"metadata": metadata, "seeds": existing_seeds},
    )

    return n_loaded, n_new, n_still_missing
