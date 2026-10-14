"""Regionalization stage and calculation helpers."""

from pathlib import Path

import numpy as np
import pandas as pd
from brightway2 import Database, calculation_setups, get_activity, methods, projects
from bw2calc import LCA
from bw2calc.lca import factorized
from scipy import sparse
from tqdm import tqdm

from sense.artifacts import atomic_output_path, write_json_atomic
from sense.brightway.repository import empty_inventory_keys
from sense.config import SenseConfig
from sense.eligibility import exclusion_reason
from sense.utils.cache import file_hash, read_manifest, stable_hash

from .case_config import ecoinvent_database_name, load
from .case_config import path as configured_path
from .study_protocol import load_protocol
from .study_runs import require_run, run_references

EXCLUDE_TERMS_STARTING_WITH = SenseConfig().exclude_terms_starting_with
EXCLUDE_TERMS = SenseConfig().exclude_terms

SCORE_METADATA = ["name", "reference product", "location", "unit", "database"]


def _score_receipt_path(path: Path) -> Path:
    return path.with_suffix(".manifest.json")


def valid_score_file(path: Path, *, calculation: dict | None = None) -> bool:
    """Require checksum, schema, finite scores, unique identities and all rows."""
    receipt = read_manifest(_score_receipt_path(path))
    if (
        receipt is None
        or receipt.get("schema") != 1
        or not path.is_file()
        or receipt.get("sha256") != file_hash(path)
        or (calculation is not None and receipt.get("calculation") != calculation)
    ):
        return False
    try:
        table = pd.read_csv(path, keep_default_na=False, float_precision="round_trip")
        columns = receipt["columns"]
        if not isinstance(columns, list) or columns[: len(SCORE_METADATA)] != SCORE_METADATA:
            return False
        scores = columns[len(SCORE_METADATA) :]
        return (
            bool(scores)
            and table.columns.tolist() == columns
            and len(table) == receipt["rows"]
            and len(table) > 0
            and not table[SCORE_METADATA].duplicated().any()
            and np.isfinite(table[scores].to_numpy(dtype=float)).all()
            and stable_hash(table[SCORE_METADATA].to_dict("records")) == receipt["population"]
            and stable_hash(table[SCORE_METADATA[:-1]].to_dict("records"))
            == receipt["paired_population"]
        )
    except (OSError, ValueError, TypeError, KeyError, pd.errors.ParserError):
        return False


def valid_score_pair(directory: Path, runs: list[dict]) -> bool:
    """A pair is reusable only after both completed artifacts were published."""
    receipt = read_manifest(directory / "regionalization_runs.json")
    if receipt is None or receipt.get("schema") != 2 or receipt.get("runs") != runs:
        return False
    paths = [
        directory / f"my_batch_run_{label}_total_scores.csv" for label in ("base", "regionalized")
    ]
    if not all(valid_score_file(path) for path in paths):
        return False
    checksums = {path.name: file_hash(path) for path in paths}
    first = read_manifest(_score_receipt_path(paths[0]))
    second = read_manifest(_score_receipt_path(paths[1]))
    if first is None or second is None:
        return False
    return (
        receipt.get("artifacts") == checksums
        and first["rows"] == second["rows"]
        and first["columns"] == second["columns"]
        and first["paired_population"] == second["paired_population"]
    )


def set_project(project_name: str) -> None:
    """Set the active Brightway project."""
    projects.set_current(project_name)
    print(f"Active Brightway project: {projects.current}")


def build_by_triple(db: Database) -> dict:
    """Return activity-key lists indexed by (name, reference product, location).

    Retain duplicate keys for the caller's uniqueness check; do not choose a match.
    """
    by_triple: dict[tuple, list] = {}
    for act in db:
        triple = (act.get("name"), act.get("reference product"), act.get("location"))
        by_triple.setdefault(triple, []).append(act.key)
    return by_triple


def should_exclude_activity(activity, exclude_terms_starting_with=None, exclude_terms=None) -> bool:
    """Apply the package's shared activity eligibility rules.

    ``None`` selects the study's exclusion lists; an empty sequence disables
    that part of the filter.
    """
    if exclude_terms_starting_with is None:
        exclude_terms_starting_with = EXCLUDE_TERMS_STARTING_WITH
    if exclude_terms is None:
        exclude_terms = EXCLUDE_TERMS

    return (
        exclusion_reason(
            activity,
            prefixes=exclude_terms_starting_with,
            terms=exclude_terms,
            exact=SenseConfig().exclude_exact,
        )
        is not None
    )


def construct_functional_units(
    db_base_name: str,
    db_regionalized_name: str,
    global_amount: float = 1.0,
    max_activities: int | None = None,
) -> tuple[list[dict], list[dict]]:
    """Return aligned base and regionalized FU lists for common activity triples.

    Each demand uses ``global_amount`` reference-product units. Sort common
    (name, reference product, location) triples and apply ``max_activities`` before
    the study's exclusions. Require unique activity matches for retained triples;
    raise ``ValueError`` on ambiguity. Read the active project without writing it.
    """
    db_base = Database(db_base_name)
    db_reg = Database(db_regionalized_name)

    base_by_triple = build_by_triple(db_base)
    reg_by_triple = build_by_triple(db_reg)
    empty_inventory = empty_inventory_keys(list(db_base)) | empty_inventory_keys(list(db_reg))

    common_triples = sorted(set(base_by_triple.keys()).intersection(reg_by_triple.keys()))
    print(f"Found {len(common_triples)} common activities (name, reference product, location).")

    if max_activities is not None:
        common_triples = common_triples[:max_activities]
        print(f"Limiting to first {len(common_triples)} common activities for processing.")

    func_units_base: list[dict] = []
    func_units_reg: list[dict] = []

    for triple in common_triples:
        # ---- Base database activity (expect exactly 1 key per triple) ----
        keys_base = base_by_triple.get(triple, [])
        if len(keys_base) != 1:
            raise ValueError(
                f"Non-unique or missing triple in BASE DB for {triple}: "
                f"expected 1 activity, found {len(keys_base)}."
            )
        key_base = keys_base[0]
        activity_base = get_activity(key_base)
        if key_base in empty_inventory or should_exclude_activity(activity_base):
            continue

        # ---- Regionalized database activity (expect exactly 1 key per triple) ----
        keys_reg = reg_by_triple.get(triple, [])
        if len(keys_reg) != 1:
            raise ValueError(
                f"Non-unique or missing triple in REGIONALIZED DB for {triple}: "
                f"expected 1 activity, found {len(keys_reg)}."
            )
        key_reg = keys_reg[0]
        activity_reg = get_activity(key_reg)
        if key_reg in empty_inventory or should_exclude_activity(activity_reg):
            continue

        # ---- Add matched functional units ----
        func_units_base.append({key_base: global_amount})
        func_units_reg.append({key_reg: global_amount})

    print(f"Functional units (base):         {len(func_units_base)}")
    print(f"Functional units (regionalized): {len(func_units_reg)}")

    # sanity check: should remain aligned one-to-one
    if len(func_units_base) != len(func_units_reg):
        raise RuntimeError(
            "Base and regionalized functional unit lists differ in length. "
            "This should not happen under one-to-one matching."
        )

    return func_units_base, func_units_reg


def select_impact_methods(target_groups: list[str]) -> list:
    """Return selected method groups, restricting SENSE keys to ``sociosphere``."""
    selected = [
        m
        for m in methods
        if (
            (m[:2] == ("SENSE", "sociosphere") and m[2] in target_groups)
            or (m[0] != "SENSE" and m[0] in target_groups)
        )
    ]
    print(f"Selected {len(selected)} impact methods.")
    return selected


def save_calculation_setups(
    setup_name_base: str,
    setup_name_reg: str,
    func_units_base: list[dict],
    func_units_reg: list[dict],
    impact_methods: list,
) -> None:
    """Save Brightway calculation setups for base and regionalized FUs."""
    calculation_setups[setup_name_base] = {
        "inv": func_units_base,
        "ia": impact_methods,
    }
    calculation_setups[setup_name_reg] = {
        "inv": func_units_reg,
        "ia": impact_methods,
    }

    print(f"\nSetup saved as '{setup_name_base}' (base database).")
    print(f"Functional units (base):         {len(func_units_base)}")
    print(f"\nSetup saved as '{setup_name_reg}' (regionalized database).")
    print(f"Functional units (regionalized): {len(func_units_reg)}")


def process_func_units_and_lca(
    cs: dict,
    setup_name: str,
    output_dir: Path,
    num_activities: int | None = None,
    overwrite_existing: bool = False,
) -> Path:
    """Write total LCA scores for a calculation setup and return the CSV path.

    ``cs`` supplies ``inv`` FU dictionaries and ``ia`` method keys. Process the first
    ``num_activities`` entries, or all when it is ``None``. A new calculation requires
    a nonempty FU list and solves the transposed system once per method, reusing
    one factorization. This gives the same totals without an inventory per demand.

    Write one activity per row, metadata and score columns atomically. Reuse only
    when content and the complete demand/method identity match its manifest.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    func_units = cs["inv"]
    methods_list = cs["ia"]
    method_labels = [
        " | ".join(m[2:] if m[:2] == ("SENSE", "sociosphere") else m) for m in methods_list
    ]

    if num_activities is not None:
        func_units = func_units[:num_activities]
    if not func_units or not methods_list or len(set(method_labels)) != len(method_labels):
        raise ValueError("Regionalization requires nonempty demands and distinct method labels")
    calculation = {
        "demands": stable_hash(
            [sorted((list(key), amount) for key, amount in demand.items()) for demand in func_units]
        ),
        "methods": [list(method) for method in methods_list],
    }

    output_csv_path = output_dir / f"{setup_name}_total_scores.csv"

    if not overwrite_existing and valid_score_file(output_csv_path, calculation=calculation):
        print(f"File for {setup_name} already exists at '{output_csv_path}'. Skipping LCA.")
        return output_csv_path

    # Include every requested database, including disconnected foregrounds.
    print(f"\nInitializing LCA for setup '{setup_name}' ...")
    lca = LCA({key: 1.0 for demand in func_units for key in demand})
    lca.load_lci_data()
    print("Factorizing the transposed technosphere matrix once ...")
    solve = factorized(lca.technosphere_matrix.T.tocsc())
    rows, columns, amounts = [], [], []
    for column, demand in enumerate(func_units):
        for key, amount in demand.items():
            rows.append(lca.product_dict[key])
            columns.append(column)
            amounts.append(amount)
    demand_matrix = sparse.coo_matrix(
        (amounts, (rows, columns)),
        shape=(lca.technosphere_matrix.shape[0], len(func_units)),
    ).tocsr()
    scores = np.empty((len(func_units), len(methods_list)))
    for column, method in enumerate(tqdm(methods_list, desc="Solving impact methods")):
        lca.switch_method(method)
        # c B A^-1 f = (A^-T B^T c^T)^T f, including signed flows and demands.
        factors = np.asarray(lca.characterization_matrix.sum(axis=0)).ravel()
        direct = np.asarray(lca.biosphere_matrix.T @ factors).ravel()
        scores[:, column] = demand_matrix.T @ solve(direct)

    results = []
    for index, fu in enumerate(tqdm(func_units, desc=f"Writing FUs in {setup_name}")):
        row_score = {}
        fu_key = list(fu.keys())[0]

        # Activity metadata
        activity = get_activity(fu_key)
        row_score["name"] = activity["name"]
        row_score["reference product"] = activity["reference product"]
        row_score["location"] = activity["location"]
        row_score["unit"] = activity.get("unit", "")
        row_score["database"] = activity["database"]

        # LCA scores for each method
        for j, label in enumerate(method_labels):
            row_score[label] = scores[index, j]

        results.append(row_score)

    df_results = pd.DataFrame(results)
    if not np.isfinite(df_results[method_labels].to_numpy(dtype=float)).all():
        raise ValueError(f"Non-finite regionalization scores in {setup_name}; no CSV was replaced")
    with atomic_output_path(output_csv_path) as temporary:
        df_results.to_csv(temporary, index=False)
    # Hash the serialized/round-tripped metadata, preserving country code NA.
    metadata = pd.read_csv(output_csv_path, keep_default_na=False)[SCORE_METADATA]
    write_json_atomic(
        _score_receipt_path(output_csv_path),
        {
            "schema": 1,
            "sha256": file_hash(output_csv_path),
            "rows": len(df_results),
            "columns": df_results.columns.tolist(),
            "population": stable_hash(metadata.to_dict("records")),
            "paired_population": stable_hash(metadata.drop(columns="database").to_dict("records")),
            "calculation": calculation,
        },
    )

    print(
        f"\nTotal scores for {len(results)} functional units from '{setup_name}' "
        f"written to '{output_csv_path}'."
    )
    return output_csv_path


def run(*, rebuild=False):
    """Execute this stage using the configured study inputs."""
    CFG = load()

    PROJECT_NAME = CFG["brightway"]["project"]
    DB_BASE = ecoinvent_database_name(CFG)
    DB_REGIONALIZED = ecoinvent_database_name(CFG, regionalized=True)
    SETUP_NAME_BASE = "my_batch_run_base"
    SETUP_NAME_REG = "my_batch_run_regionalized"
    GLOBAL_AMOUNT = 1.0
    NUM_ACTIVITIES_TO_PROCESS = None
    OUTPUT_DIR = configured_path("data_dir", CFG) / "case_study_results/hydrogen_case_study_results"
    TARGET_GROUPS = [
        "Local Community",
        "Society",
        "Value Chain Actors",
        "Workers",
        "Children",
        "Consumer",
    ]
    REBUILD_LCA_RESULTS = rebuild  # True explicitly recalculates both score CSVs
    SPEC = load_protocol()

    base_csv = OUTPUT_DIR / f"{SETUP_NAME_BASE}_total_scores.csv"

    reg_csv = OUTPUT_DIR / f"{SETUP_NAME_REG}_total_scores.csv"

    requests = [
        (DB_BASE, "geometric"),
        (DB_REGIONALIZED, "geometric"),
        (CFG["brightway"]["regioinvent_database"], "geometric"),
    ]

    run_sources = run_references(CFG, SPEC, requests)
    for database, mean in requests:
        require_run(CFG, SPEC, database, mean)

    receipt_path = OUTPUT_DIR / "regionalization_runs.json"

    if not REBUILD_LCA_RESULTS and valid_score_pair(OUTPUT_DIR, run_sources):
        print("Skipping regionalization LCA setup, database scans and solves: saved results exist.")
    else:
        set_project(PROJECT_NAME)

        # 1) Build functional units from base and regionalized databases
        func_units_base, func_units_reg = construct_functional_units(
            db_base_name=DB_BASE,
            db_regionalized_name=DB_REGIONALIZED,
            global_amount=GLOBAL_AMOUNT,
            max_activities=NUM_ACTIVITIES_TO_PROCESS,
        )

        # 2) Select impact methods
        impact_methods = select_impact_methods(TARGET_GROUPS)

        # 3) Save calculation setups
        save_calculation_setups(
            setup_name_base=SETUP_NAME_BASE,
            setup_name_reg=SETUP_NAME_REG,
            func_units_base=func_units_base,
            func_units_reg=func_units_reg,
            impact_methods=impact_methods,
        )

        cs_base = calculation_setups[SETUP_NAME_BASE]
        cs_reg = calculation_setups[SETUP_NAME_REG]

        # 4) Run LCAs (skips if CSVs already exist, unless overwrite_existing=True)
        base_csv = process_func_units_and_lca(
            cs=cs_base,
            setup_name=SETUP_NAME_BASE,
            output_dir=OUTPUT_DIR,
            num_activities=NUM_ACTIVITIES_TO_PROCESS,
            overwrite_existing=True,
        )
        reg_csv = process_func_units_and_lca(
            cs=cs_reg,
            setup_name=SETUP_NAME_REG,
            output_dir=OUTPUT_DIR,
            num_activities=NUM_ACTIVITIES_TO_PROCESS,
            overwrite_existing=True,
        )
        write_json_atomic(
            receipt_path,
            {
                "schema": 2,
                "runs": run_sources,
                "artifacts": {path.name: file_hash(path) for path in (base_csv, reg_csv)},
            },
        )
