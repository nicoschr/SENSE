"""Keep the calibrated package default and manuscript protocol aligned."""

from dataclasses import asdict
from inspect import signature

import pytest

from case_studies.hydrogen import run
from case_studies.hydrogen.workflow.study_protocol import (
    SCENARIOS,
    foregrounds,
    load_protocol,
    sense_config,
)
from sense import SenseConfig
from sense.core.matching import match_from_stores


def test_calibrated_threshold_is_shared_by_package_and_every_study_run():
    spec = load_protocol()
    assert spec["sense"]["similarity_delta"] == SenseConfig().similarity_delta == 0.028
    assert signature(match_from_stores).parameters["threshold"].default == 0.028
    for mean in ("geometric", "arithmetic"):
        for database in [
            None,
            *(name for scenario in SCENARIOS for name in foregrounds(scenario).values()),
        ]:
            assert asdict(sense_config(spec, mean, database))["similarity_delta"] == 0.028


def test_recalculation_reuses_inventories_and_rejects_retired_sensitivity():
    assert run.GROUPS["recalculate-hydrogen"] == (
        "check-backgrounds",
        "sense-backgrounds",
        "sense-foregrounds",
        "calculate",
        "figure-hydrogen",
        "figure-contributions",
    )
    with pytest.raises(SystemExit):
        run.main(["threshold-sensitivity", "--dry-run"])
