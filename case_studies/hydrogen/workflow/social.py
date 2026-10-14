"""SENSE mapping stages."""

from .case_config import ecoinvent_database_name, enable_sense_progress, load
from .paired_prices import synchronize_prices
from .study_protocol import SCENARIOS, load_protocol
from .study_protocol import foregrounds as scenario_foregrounds
from .study_runs import run_layer


def backgrounds():
    """Execute this stage using the configured study inputs."""
    CFG = load()
    SPEC = load_protocol()

    enable_sense_progress()

    backgrounds = [
        ecoinvent_database_name(CFG),
        ecoinvent_database_name(CFG, regionalized=True),
        CFG["brightway"]["regioinvent_database"],
    ]

    for database in backgrounds:
        run_layer(CFG, SPEC, database, "geometric")


def arithmetic():
    """Execute this stage using the configured study inputs."""
    CFG = load()
    SPEC = load_protocol()

    enable_sense_progress()

    run_layer(CFG, SPEC, ecoinvent_database_name(CFG), "arithmetic")


def foregrounds():
    """Execute this stage using the configured study inputs."""
    CFG = load()
    SPEC = load_protocol()

    enable_sense_progress()

    synchronize_prices(CFG, SPEC)

    for scenario in SCENARIOS:
        for database in scenario_foregrounds(scenario).values():
            run_layer(CFG, SPEC, database)
