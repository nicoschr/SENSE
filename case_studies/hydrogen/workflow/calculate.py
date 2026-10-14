"""Paired hydrogen LCA and contribution export."""

from __future__ import annotations

import json
import warnings
from collections import defaultdict
from collections.abc import Sequence

import bw2data as bd
import pandas as pd
from bw2calc import LCA
from tqdm.auto import tqdm

from case_studies.hydrogen.audit_indicators import audit_study

from .carbon_methods import prepare as prepare_carbon_methods
from .case_config import ecoinvent_database_name, load
from .case_config import path as configured_path
from .paired_results import select_panels
from .result_archive import (
    ResultsWriter,
    activity_contributions,
    activity_metadata,
    capture_methods,
    characterization_rows,
)
from .study_checks import check_climate_method, check_foreground_boundaries, check_pairs
from .study_inputs import load_study_inputs
from .study_protocol import SCENARIOS, calculation_context, foregrounds, load_protocol
from .study_runs import require_run


def build_setup_exact(
    project_name: str,
    location_source_db: str,
    search_dbs: Sequence[str],
    activity_specs: Sequence[tuple[str, str, float]],
    target_groups: Sequence[str],
    setup_name: str,
    strict_unique: bool = True,
    verbose: bool = True,
    countries: Sequence[str] | None = None,
    selected_methods: Sequence[tuple] | None = None,
) -> tuple[str, list[dict], list[tuple], dict, list]:
    """Save an exact technology/location calculation setup in Brightway.

    ``activity_specs`` contains (name, reference product, demand amount) tuples.
    Read locations from ``location_source_db`` and search ``search_dbs`` in order.
    Record missing combinations without substitution. Duplicate matches raise
    ``ValueError`` when ``strict_unique`` is true; otherwise use the first match.

    Select ``target_groups``, scoping SENSE methods to ``sociosphere``; raise
    ``ValueError`` if none match. Set the active project and write ``setup_name``.
    Return (setup name, FU dictionaries, method keys, per-location diagnostics,
    duplicate records). No LCA is solved here.
    """
    bd.projects.set_current(project_name)

    # Locations
    src_db = bd.Database(location_source_db)
    locations = (
        list(countries)
        if countries is not None
        else sorted({act.get("location") for act in src_db})
    )
    if verbose:
        print(f"🌍 Locations from '{location_source_db}' (n={len(locations)}).")

    # Index DBs once
    def index_db(dbname: str):
        idx = defaultdict(list)
        db = bd.Database(dbname)
        for act in db:
            key = (act.get("name"), act.get("reference product"), act.get("location"))
            idx[key].append(act)
        return idx

    db_indices = {dbn: index_db(dbn) for dbn in search_dbs}
    if verbose:
        print(f"🗂️ Indexed DBs: {', '.join(search_dbs)}")

    def exact_hits(name: str, ref_prod: str, loc: str):
        hits = []
        for dbn in search_dbs:
            hits.extend([(dbn, a) for a in db_indices[dbn].get((name, ref_prod, loc), [])])
        return hits

    func_units: list[dict] = []
    duplicates = []
    diagnostics = {}

    for loc in locations:
        found_count = 0
        not_found_specs = []
        chosen_keys_for_loc = []

        for name, ref_prod, amount in activity_specs:
            hits = exact_hits(name, ref_prod, loc)
            if not hits:
                not_found_specs.append(f"{name} [{ref_prod}]")
                continue

            if len(hits) > 1:
                duplicate = (loc, name, ref_prod, len(hits), ",".join([h[0] for h in hits]))
                duplicates.append(duplicate)
                if strict_unique:
                    raise ValueError(
                        f"Expected one activity for {name!r}, {ref_prod!r}, {loc!r}; "
                        f"found {len(hits)} across {duplicate[-1]}."
                    )

            chosen = hits[0][1]  # first by db priority order
            if countries is not None and chosen.get("unit") not in {"kg", "kilogram"}:
                raise ValueError(
                    f"Hydrogen functional unit is not measured in kilograms: {chosen.key}"
                )
            func_units.append({chosen.key: amount})
            found_count += 1
            chosen_keys_for_loc.append(chosen.key)

        diagnostics[loc] = {
            "found": found_count,
            "not_found": len(not_found_specs),
            "missing_specs": not_found_specs,
            "found_keys": chosen_keys_for_loc,
        }

    # Impact methods
    impact_methods = [
        m
        for m in bd.methods
        if (
            (m[:2] == ("SENSE", "sociosphere") and m[2] in set(target_groups))
            or (m[0] != "SENSE" and m[0] in set(target_groups))
        )
    ]
    if selected_methods is not None:
        impact_methods = list(selected_methods)
        if any(method not in bd.methods for method in impact_methods):
            raise ValueError("Selected characterization methods are missing")
    if countries is not None and any(row["not_found"] for row in diagnostics.values()):
        raise ValueError(f"Incomplete functional-unit coverage: {diagnostics}")
    if not impact_methods:
        raise ValueError(f"No methods matched the top-level groups: {target_groups}")

    bd.calculation_setups[setup_name] = {"inv": func_units, "ia": impact_methods}

    if verbose:
        print(f"\n📦 Functional units created: {len(func_units)}")
        print(f"🧪 Methods selected: {len(impact_methods)}")
        print(f"✅ Saved calculation setup as '{setup_name}'")

        # compact diagnostics
        for loc, d in list(diagnostics.items())[:20]:
            print(f"  - {loc}: found={d['found']}, not_found={d['not_found']}")
        if len(diagnostics) > 20:
            print(f"  … (+{len(diagnostics) - 20} more locations)")

        if duplicates:
            print("\n⚠️ Duplicates (kept first by DB priority order):")
            for loc, n, r, cnt, dbs in duplicates[:10]:
                print(f"  - {loc} | {n} | {r} | {cnt} hits | {dbs}")
            if len(duplicates) > 10:
                print(f"  … and {len(duplicates) - 10} more")

    return setup_name, func_units, impact_methods, diagnostics, duplicates


def warmup_lci_across_fu_databases(func_units: Sequence[dict]) -> LCA:
    """Factorize an LCA spanning every database represented in the functional units.

    Use one tiny demand per database to establish matrix coverage. Subsequent
    ``redo_lci`` calls replace this initialization demand with the actual FU.
    """
    rep_by_db: dict[str, tuple] = {}
    for fu in func_units:
        k = next(iter(fu))
        rep_by_db.setdefault(k[0], k)

    warmup_demand = dict.fromkeys(rep_by_db.values(), 1e-12)
    lca = LCA(warmup_demand)
    lca.lci(factorize=True)
    return lca


def run():
    warnings.filterwarnings(
        "once",
        message=r"\(almost\) singular matrix!.*",
        module=r"scikits\.umfpack\.umfpack",
    )
    CFG = load()
    PROJECT = CFG["brightway"]["project"]
    STUDY_INPUTS = load_study_inputs(
        configured_path("hydrogen_inventory", CFG),
        ecoinvent_version=CFG["versions"]["ecoinvent"],
    )
    ACTIVITY_SPECS = [tuple(row) for row in STUDY_INPUTS["activity_specs"]]
    SPEC = load_protocol()
    indicator_coverage, indicator_evidence = audit_study(CFG)
    COUNTRIES = SPEC["countries"]
    TECHNOLOGIES = [row[0] for row in ACTIVITY_SPECS]

    if len(TECHNOLOGIES) != 5 or set(TECHNOLOGIES) != set(SPEC["technology_labels"]):
        raise ValueError(
            "The study workbook must specify the five fixed manuscript hydrogen technologies"
        )

    if any(float(row[2]) != 1.0 for row in ACTIVITY_SPECS):
        raise ValueError("The paired study requires a demand of 1 kg for each hydrogen route")

    bd.projects.set_current(PROJECT)

    carbon_accounting = prepare_carbon_methods(
        configured_path("data_dir", CFG), f"ecoinvent-{CFG['versions']['ecoinvent']}-biosphere"
    )
    climate_method = tuple(SPEC["panels"][0]["method"])
    check_climate_method(climate_method)
    method_table = capture_methods(
        bd.methods,
        SPEC["environmental_family"],
        SPEC["social_layers"]["geometric"],
        climate_method=climate_method,
    )
    captured_methods = [tuple(json.loads(key)) for key in method_table.method]
    print(
        f"Capturing {len(captured_methods)} methods: {method_table.kind.value_counts().to_dict()}"
    )

    check_pairs(
        list(foregrounds("baseline").values()),
        list(foregrounds("regionalized").values()),
        COUNTRIES,
    )

    run_directories = {}

    for database in [
        ecoinvent_database_name(CFG),
        ecoinvent_database_name(CFG, regionalized=True),
        CFG["brightway"]["regioinvent_database"],
        *foregrounds("baseline").values(),
        *foregrounds("regionalized").values(),
    ]:
        run_directories[database] = str(require_run(CFG, SPEC, database))

    func_units, fu_scenarios = [], {}

    for scenario in SCENARIOS:
        names = foregrounds(scenario)
        allowed = set(names.values()) | {
            ecoinvent_database_name(CFG, regionalized=scenario == "regionalized")
        }
        if scenario == "regionalized":
            allowed.add(CFG["brightway"]["regioinvent_database"])
        check_foreground_boundaries(list(names.values()), allowed)
        _, units, methods, diagnostics, duplicates = build_setup_exact(
            PROJECT,
            names["hydrogen"],
            [names["hydrogen"], names["subset"]],
            ACTIVITY_SPECS,
            [SPEC["environmental_family"]],
            f"hydrogen_{scenario}",
            countries=COUNTRIES,
            selected_methods=captured_methods,
        )
        func_units.extend(units)
        fu_scenarios.update({next(iter(fu)): scenario for fu in units})

    if len(func_units) != 30:
        raise ValueError(f"Expected 30 unique functional units; got {len(func_units)}")

    print("Initializing one factorized LCA across all hydrogen foreground databases...")
    lca = warmup_lci_across_fu_databases(func_units)
    characterization = characterization_rows(lca, method_table)

    # Inventory columns are activities, not product rows. Preserve exact identities
    # once for the entire archive; repeated names in different databases stay distinct.
    reverse_activities = lca.reverse_dict()[0]
    metadata_by_key = {}
    for database in sorted({key[0] for key in reverse_activities.values()}):
        for act in bd.Database(database):
            metadata_by_key[act.key] = activity_metadata(act.key, act)

    activities = pd.DataFrame(
        [
            {"activity_index": i, **metadata_by_key[reverse_activities[i]]}
            for i in range(lca.inventory.shape[1])
        ]
    )
    output = configured_path("data_dir", CFG) / "paired_hydrogen"
    metadata = {
        "context": calculation_context(CFG, SPEC),
        "carbon_accounting": carbon_accounting,
        "study_at_calculation": SPEC,
        "technologies": TECHNOLOGIES,
        "run_directories": run_directories,
        "indicator_evidence": indicator_evidence,
        "contribution_scope": "Every signed direct activity contribution across the "
        "full supply chain; exact zeros implicit. No top-N cutoff or aggregated residual. "
        "These are not overlapping recursive supply-chain subtree totals.",
    }
    with ResultsWriter(output / "results.zip", method_table, activities, metadata) as writer:
        for fu in tqdm(func_units, desc="Hydrogen systems, all methods and contributors"):
            lca.redo_lci(fu)
            key = next(iter(fu))
            activity = metadata_by_key[key]
            contributions, totals = activity_contributions(characterization, lca.inventory)
            writer.add(
                {
                    "technology": activity["name"],
                    "country": activity["location"],
                    "scenario": fu_scenarios[key],
                    "demand": float(fu[key]),
                    "activity_database": key[0],
                    "activity_code": key[1],
                },
                contributions,
                totals,
            )
        scores = writer.finish(indicator_coverage)

    scores.to_csv(output / "scores.csv", index=False)
    # A convenient small table of the current figure choice; the complete archive
    # remains the source for subsequent presentation changes.
    select_panels(scores, SPEC, TECHNOLOGIES).to_csv(output / "figure_scores.csv", index=False)
    print(f"Saved {len(scores)} scores; main-figure selection remains three panels.")
