"""Checks on database boundaries and paired inventory construction."""

from __future__ import annotations

from collections import Counter


def technology_candidates(original_name: str, candidates: list) -> list:
    """Keep explicit energy technologies during semantic provider selection."""
    name = original_name.lower()
    markers = (
        "wind",
        "photovoltaic",
        "hydropower",
        "nuclear",
        "biomethane",
        "electrolysis",
        "hard coal",
        "lignite",
        "natural gas",
        "biogas",
        "geothermal",
        "solar thermal",
    )
    required = [marker for marker in markers if marker in name]
    return [
        candidate
        for candidate in candidates
        if all(marker in candidate.get("name", "").lower() for marker in required)
    ]


def check_climate_method(method: tuple) -> None:
    """Require ordinary and spatialized representations to use equivalent climate CFs."""
    import math

    import bw2data as bd

    standard = (method[0].removesuffix(" regionalized"), *method[1:])
    if standard not in bd.methods or method not in bd.methods:
        raise ValueError("Both standard and regionalized climate methods must exist")
    standard_cfs = {tuple(row[0]): float(row[1]) for row in bd.Method(standard).load()}
    selected_cfs = {tuple(row[0]): float(row[1]) for row in bd.Method(method).load()}
    lookup = {}
    for key, factor in standard_cfs.items():
        if not factor:
            continue
        flow = bd.get_activity(key)
        identity = (flow.get("name"), tuple(flow.get("categories", ())), flow.get("unit"))
        lookup[identity] = factor
        if not math.isclose(selected_cfs.get(key, 0), factor, rel_tol=1e-10):
            raise ValueError(f"Climate method omits/changes an ordinary climate flow: {key}")
    for flow in bd.Database("biosphere3_spatialized_flows"):
        name = flow.get("name", "").rsplit(", ", 1)[0]
        identity = (name, tuple(flow.get("categories", ())), flow.get("unit"))
        if identity in lookup and not math.isclose(
            selected_cfs.get(tuple(flow.key), 0), lookup[identity], rel_tol=1e-10
        ):
            raise ValueError(
                f"Climate method omits/changes an equivalent spatialized flow: {flow.key}"
            )


def check_backgrounds(base: str, regionalized: str, regioinvent: str) -> dict:
    """Check actual technosphere dependencies, including unprocessed inventory edits."""
    import bw2data as bd
    from bw2data.backends.peewee import ExchangeDataset

    for name in (base, regionalized, regioinvent):
        if name not in bd.databases:
            raise ValueError(f"Missing background database: {name}")
    dependencies = {}
    for name in (base, regionalized, regioinvent):
        if bd.databases[name].get("backend", "sqlite") != "sqlite":
            raise ValueError("The hydrogen study requires Brightway's SQLite backend")
        query = (
            ExchangeDataset.select(ExchangeDataset.input_database)
            .where(
                ExchangeDataset.output_database == name,
                ExchangeDataset.type == "technosphere",
            )
            .distinct()
        )
        dependencies[name] = {row.input_database for row in query} - {name}
    if dependencies[base] & {regionalized, regioinvent}:
        raise ValueError("Original ecoinvent depends on the regionalized system")
    if (
        regioinvent not in dependencies[regionalized]
        or regionalized not in dependencies[regioinvent]
    ):
        raise ValueError("The two regionalized backgrounds must be connected")
    if base in dependencies[regionalized] | dependencies[regioinvent]:
        raise ValueError("The regionalized background reaches original ecoinvent")
    return {k: sorted(v) for k, v in dependencies.items()}


def check_foreground_boundaries(names: list[str], allowed: set[str]) -> None:
    """Check actual exchanges of the small foregrounds, including non-FU dependencies."""
    import bw2data as bd

    failures = []
    for name in names:
        for act in bd.Database(name):
            for exchange in act.technosphere():
                if exchange.input.key[0] not in allowed:
                    failures.append((act.key, exchange.input.key))
    if failures:
        raise ValueError(
            f"Foreground crosses scenario boundary ({len(failures)} links): {failures[:5]}"
        )


def inventory_signature(activity) -> Counter:
    """Physical input recipe, independent of provider name/location and social flows."""
    return Counter(
        (e.input.get("reference product"), e.input.get("unit"), float(e["amount"]))
        for e in activity.technosphere()
    )


def environmental_signature(activity) -> Counter:
    """Compare direct emissions/resources without their spatial representation."""
    rows = []
    for exchange in activity.biosphere():
        flow = exchange.input
        if flow.get("sense_kind") or flow.get("sense_managed"):
            continue
        name = flow.get("name", "")
        if "spatialized" in flow.key[0]:
            name = name.rsplit(", ", 1)[0]
        rows.append(
            (name, tuple(flow.get("categories", ())), flow.get("unit"), float(exchange["amount"]))
        )
    return Counter(rows)


def check_pairs(baseline: list[str], regionalized: list[str], countries: list[str]):
    """Compare activities within corresponding database pairs, in list order.

    Each position identifies one foreground role (e.g. hydrogen or methanol).
    Different roles can legitimately contain the same activity identity.
    Identities must still be unique within each database.
    """
    import bw2data as bd

    from sense.brightway.repository import reference_production_amount

    if (
        not baseline
        or len(baseline) != len(regionalized)
        or len(set(baseline)) != len(baseline)
        or len(set(regionalized)) != len(regionalized)
    ):
        raise ValueError("Provide equally sized, nonempty lists of distinct paired databases")

    missing = [name for name in (*baseline, *regionalized) if name not in bd.databases]
    if missing:
        raise ValueError(
            f"Missing foreground databases: {', '.join(missing)}. "
            "Complete python -m case_studies.hydrogen.run foregrounds before "
            "running sense-foregrounds or calculate."
        )

    def index(names):
        result = {}
        for role, name in enumerate(names):
            for act in bd.Database(name):
                if act.get("location") not in countries:
                    continue
                key = (
                    role,
                    act.get("name"),
                    act.get("reference product"),
                    act.get("unit"),
                    act.get("location"),
                )
                if key in result:
                    raise ValueError(f"Ambiguous foreground identity in {name}: {key[1:]}")
                result[key] = act
        if not result:
            raise ValueError(
                f"No foreground activities for {countries} in {names}. "
                "Complete python -m case_studies.hydrogen.run foregrounds first."
            )
        return result

    left, right = index(baseline), index(regionalized)
    if set(left) != set(right):
        raise ValueError("Baseline and regionalized foreground activity sets differ")
    for key, act in left.items():
        other = right[key]
        if inventory_signature(act) != inventory_signature(other):
            raise ValueError(f"Paired foreground input quantities/products differ: {key}")
        if environmental_signature(act) != environmental_signature(other):
            raise ValueError(f"Paired foreground direct emissions/resources differ: {key}")
        left_production, right_production = list(act.production()), list(other.production())
        left_amount = reference_production_amount(act.key, left_production)
        right_amount = reference_production_amount(other.key, right_production)
        left_sign = float(left_production[0]["amount"]) if left_production else 1.0
        right_sign = float(right_production[0]["amount"]) if right_production else 1.0
        if left_amount != right_amount or (left_sign < 0) != (right_sign < 0):
            raise ValueError(f"Paired reference production differs: {key}")
    return [(left[k], right[k]) for k in sorted(left)]
