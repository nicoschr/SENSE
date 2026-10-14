"""Build or validate the two foreground scenarios."""

from __future__ import annotations

import bw2data as bd

from sense.data.embeddings import load_encoder

from .carbon_methods import prepare as prepare_carbon_methods
from .case_config import ecoinvent_database_name, load
from .case_config import path as configured_path
from .foreground_construction import (
    _DATABASE_CANDIDATE_INDEXES,
    SemanticMatcher,
    localize_ecoinvent_subset,
    localize_foreground_database,
)
from .inventory_operations import (
    _ensure_project,
    import_foreground_from_excel,
    remove_self_referencing_technosphere_exchanges,
    replace_technosphere_exchange_in_activity,
)
from .study_checks import check_foreground_boundaries, check_pairs
from .study_inputs import load_study_inputs
from .study_protocol import SCENARIOS, foregrounds, load_protocol


def run():
    """Execute this stage using the configured study inputs."""
    CFG = load()
    DATA_DIR = configured_path("data_dir", CFG)
    STUDY_INPUTS = load_study_inputs(
        configured_path("hydrogen_inventory", CFG), ecoinvent_version=CFG["versions"]["ecoinvent"]
    )
    METHANOL_EXCEL = configured_path("methanol_inventory", CFG)
    HYDROGEN_EXCEL = configured_path("hydrogen_inventory", CFG)
    TROUBLESHOOTING_DIR = DATA_DIR / "troubleshooting"

    TROUBLESHOOTING_DIR.mkdir(parents=True, exist_ok=True)

    SPEC = load_protocol()
    PROJECT = CFG["brightway"]["project"]
    ECOINVENT_VERSION = CFG["versions"]["ecoinvent"]

    ECOINVENT_BIOSPHERE_DB = f"ecoinvent-{ECOINVENT_VERSION}-biosphere"
    REGIOINVENT_DB = CFG["brightway"]["regioinvent_database"]
    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    SIMILARITY_THRESHOLD = 0.75
    ECOINVENT_SUBSET_TARGET_ACTS = {tuple(row) for row in STUDY_INPUTS["subset_targets"]}

    _ensure_project(PROJECT)

    _DATABASE_CANDIDATE_INDEXES.clear()

    matcher = None

    for scenario in SCENARIOS:
        outputs = foregrounds(scenario)
        existing = [name in bd.databases for name in outputs.values()]
        if any(existing) and not all(existing):
            raise ValueError(
                f"Incomplete {scenario} foreground set; inspect before rebuilding: {outputs}"
            )
        if all(existing):
            print(f"Reusing {scenario} foregrounds")
            continue
        if matcher is None:
            matcher = SemanticMatcher(load_encoder(MODEL_NAME))
        background = ecoinvent_database_name(CFG, regionalized=scenario == "regionalized")
        backgrounds = [background] + ([REGIOINVENT_DB] if scenario == "regionalized" else [])
        staging_methanol = f"02_methanol_inventories_source_{scenario}"
        staging_hydrogen = f"01_hydrogen_inventories_source_{scenario}"
        if staging_methanol in bd.databases or staging_hydrogen in bd.databases:
            raise ValueError(
                f"Staging databases already exist for {scenario}; inspect the interrupted import"
            )
        # Workbooks contain ordinary elemental flow names; localize spatial flows below.
        import_foreground_from_excel(
            METHANOL_EXCEL,
            staging_methanol,
            ECOINVENT_BIOSPHERE_DB,
            background,
            unlinked_csv=TROUBLESHOOTING_DIR / f"{scenario}_methanol_unlinked.csv",
        )
        remove_self_referencing_technosphere_exchanges(staging_methanol)
        replace_technosphere_exchange_in_activity(
            staging_methanol, **STUDY_INPUTS["methanol_replacement"]
        )
        localize_foreground_database(
            staging_methanol,
            outputs["methanol"],
            backgrounds,
            SPEC["countries"],
            SIMILARITY_THRESHOLD,
            matcher,
            TROUBLESHOOTING_DIR / f"{scenario}_methanol_mapping.xlsx",
        )
        import_foreground_from_excel(
            HYDROGEN_EXCEL,
            staging_hydrogen,
            ECOINVENT_BIOSPHERE_DB,
            background,
            [staging_methanol],
            TROUBLESHOOTING_DIR / f"{scenario}_hydrogen_unlinked.csv",
        )
        localize_foreground_database(
            staging_hydrogen,
            outputs["hydrogen"],
            [outputs["methanol"], *backgrounds],
            SPEC["countries"],
            SIMILARITY_THRESHOLD,
            matcher,
            TROUBLESHOOTING_DIR / f"{scenario}_hydrogen_mapping.xlsx",
            foreground_sources={staging_methanol: outputs["methanol"]},
        )
        localize_ecoinvent_subset(
            background,
            outputs["subset"],
            backgrounds,
            SPEC["countries"],
            ECOINVENT_SUBSET_TARGET_ACTS,
            matcher,
            SIMILARITY_THRESHOLD,
            TROUBLESHOOTING_DIR / f"{scenario}_subset_mapping.xlsx",
        )

    for scenario in SCENARIOS:
        names = list(foregrounds(scenario).values())
        allowed_backgrounds = {
            ecoinvent_database_name(CFG, regionalized=scenario == "regionalized")
        }
        if scenario == "regionalized":
            allowed_backgrounds.add(REGIOINVENT_DB)
        check_foreground_boundaries(names, set(names) | allowed_backgrounds)

    pairs = check_pairs(
        list(foregrounds("baseline").values()),
        list(foregrounds("regionalized").values()),
        SPEC["countries"],
    )

    print(f"Checked {len(pairs)} paired foreground activities")

    prepare_carbon_methods(DATA_DIR, ECOINVENT_BIOSPHERE_DB)
