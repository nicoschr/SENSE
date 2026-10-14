"""Supplementary reconstruction boundaries and portable output checks."""

import json

import numpy as np
import pandas as pd
import pytest

from case_studies.hydrogen import run as runner
from case_studies.hydrogen.workflow import case_config, si_common, si_matching, si_tables
from sense.utils.cache import file_hash


def config():
    return {
        "similarity_delta": 0.028,
        "social_flow_mean": "geometric",
        "productivity_baseline_path": None,
        "rounding_digits": None,
        "psilca_process_overrides": {},
        "zero_price_policy": "treatment_working_hours",
    }


def archived_inputs(tmp_path, monkeypatch, *, treatment=False):
    from zipfile import ZipFile

    cfg = {
        "paths": {"data_dir": str(tmp_path)},
        "brightway": {"project": "study"},
        "versions": {"ecoinvent": "3.12", "psilca": "4.0"},
    }
    monkeypatch.setattr(case_config, "load", lambda: cfg)
    root = tmp_path / "Output"
    source = root / "PSILCA/psilca_assessments_v4.0.parquet"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"source used only for hash verification")
    source_manifest = {"artifact_hashes": {"assessments": file_hash(source)}}
    source.with_suffix(".manifest.json").write_text(json.dumps(source_manifest))
    directory = root / "Runs/run-1"
    snapshots = directory / "InputSnapshots"
    snapshots.mkdir(parents=True)
    activity = {"code": "a", "calculation route": "treatment_fallback" if treatment else "monetary"}
    values = {"priced_activities": [{"Activities": [activity]}], "psilca_manifest": source_manifest}
    inputs = {}
    for name, value in values.items():
        path = snapshots / (name + ".json")
        path.write_text(json.dumps(value))
        inputs[name] = {
            "path": "C:\\old\\run-1\\InputSnapshots\\" + path.name,
            "sha256": file_hash(path),
        }
    identity = {
        "project": "study",
        "database": "inventory",
        "versions": cfg["versions"],
        "sociosphere": "sociosphere",
        "config": config(),
        "inputs": inputs,
    }
    (directory / "inputs.json").write_text(json.dumps(identity))
    (directory / "diagnostics.json").write_text(json.dumps({"processing_issues": []}))
    complete = {
        **identity,
        "artifacts": {
            p.relative_to(directory).as_posix().replace("/", "\\"): file_hash(p)
            for p in directory.rglob("*")
            if p.is_file()
        },
    }
    (directory / "complete.json").write_text(json.dumps(complete))
    archive = tmp_path / "paired_hydrogen/results.zip"
    archive.parent.mkdir()
    with ZipFile(archive, "w") as zipped:
        zipped.writestr(
            "manifest.json",
            json.dumps({"metadata": {"run_directories": {"inventory": "C:\\old\\Runs\\run-1"}}}),
        )
    return directory


def test_archived_runs_resolve_from_new_root_and_allow_no_treatment_recipients(
    tmp_path, monkeypatch
):
    directory = archived_inputs(tmp_path, monkeypatch)
    inputs = si_common.load_inputs()
    assert inputs.runs == {"inventory": str(directory)}
    inputs.check_unchanged()
    inputs.archive.write_bytes(b"changed")
    with pytest.raises(ValueError, match="archive changed"):
        inputs.check_unchanged()


def test_archived_treatment_recipients_require_hour_reference(tmp_path, monkeypatch):
    archived_inputs(tmp_path, monkeypatch, treatment=True)
    with pytest.raises(ValueError, match="lack an archived working-hour reference"):
        si_common.load_inputs()


def test_changed_run_snapshot_is_rejected(tmp_path, monkeypatch):
    directory = archived_inputs(tmp_path, monkeypatch)
    (directory / "InputSnapshots/priced_activities.json").write_text("[]")
    with pytest.raises(ValueError, match="Changed run artifact"):
        si_common.load_inputs()


def test_run_artifact_cannot_escape_the_archive_directory(tmp_path, monkeypatch):
    directory = archived_inputs(tmp_path, monkeypatch)
    path = directory / "complete.json"
    complete = json.loads(path.read_text())
    complete["artifacts"] = {"..\\outside.json": "ignored"}
    path.write_text(json.dumps(complete))
    with pytest.raises(ValueError, match="Unsafe run artifact path"):
        si_common.load_inputs()


def test_coefficient_cache_requires_content_and_input_identity(tmp_path):
    path = tmp_path / "cache.parquet"
    frame = pd.DataFrame({"value": [1.0, 2.0]})
    si_common.save_coefficients(frame, path, "inputs-1")
    pd.testing.assert_frame_equal(si_common.cached_coefficients(path, "inputs-1"), frame)
    assert si_common.cached_coefficients(path, "inputs-2") is None
    path.write_bytes(b"corrupted")
    assert si_common.cached_coefficients(path, "inputs-1") is None


def test_risk_method_selection_does_not_select_same_named_opportunity():
    stem = [
        "SENSE",
        "sociosphere",
        "Workers",
        "Health and safety",
        "Rate of fatal accidents at workplace",
    ]
    methods = pd.DataFrame(
        {
            "method": [json.dumps(stem + [kind]) for kind in ("risk", "opportunity")],
            "method_index": [3, 4],
        }
    )
    assert si_common.method_index(methods, stem[-1]) == 3
    with pytest.raises(ValueError, match="exactly one method"):
        si_common.method_index(pd.concat([methods, methods]), stem[-1])


def test_si_group_runs_diagnostics_before_exports_and_expensive_comparison():
    stages = runner.GROUPS["all"]
    for analysis, output in [
        ("si-productivity", "figure-si-productivity"),
        ("si-missing-assessments", "table-si-missing-assessments"),
        ("si-carbon", "table-si-carbon"),
        ("benchmark", "table-si-coverage"),
    ]:
        assert stages.index(analysis) < stages.index(output) < stages.index("regionalization")
    assert "build-backgrounds" not in stages
    assert not {"regionalization", "benchmark", "sense-backgrounds"} & set(runner.GROUPS["si"])


def test_matching_figures_require_only_aggregate_inputs_and_configured_output(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(case_config, "load", lambda: {"paths": {"data_dir": str(tmp_path)}})
    si_matching.run()
    for name in (
        "Fig1_semantic_vs_baseline_TEST",
        "Fig2_semantic_model_choice_DEV_TEST",
        "Fig3_delta_selection_utility_high_res",
    ):
        assert (tmp_path / "figures" / (name + ".pdf")).is_file()
        assert (tmp_path / "figures" / (name + ".svg")).is_file()
        assert (tmp_path / "figures" / (name + ".png")).is_file()
    evidence = json.loads((tmp_path / "si/matching/rendered.json").read_text())
    assert evidence["threshold"] == 0.028
    assert "workbook" not in evidence["inputs"]
    metrics = pd.read_csv(si_matching.EVIDENCE / "encoder_metrics.csv")
    assert (
        metrics.loc[
            metrics.model.eq("BAAI/bge-base-en-v1.5") & metrics.split.eq("test"), "top1"
        ].item()
        == 0.81
    )


def test_carbon_table_requires_complete_country_scenario_pairs(tmp_path, monkeypatch):
    monkeypatch.setattr(case_config, "load", lambda: {"paths": {"data_dir": str(tmp_path)}})
    rows = [
        {
            "technology": technology,
            "country": country,
            "scenario": scenario,
            "conventional_climate_score": value,
            "explicit_climate_score": value / 2,
        }
        for technology, _ in si_tables.ROUTES
        for country in ("DE", "CN", "US")
        for value, scenario in ((2.0, "baseline"), (4.0, "regionalized"))
    ]
    path = tmp_path / "si/carbon/carbon_convention_comparison.csv"
    path.parent.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    si_tables.carbon()
    assert (
        "2.00--4.00 & 1.00--2.00" in (tmp_path / "si/carbon_convention_comparison.tex").read_text()
    )
    pd.DataFrame(rows[:-1]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="all six"):
        si_tables.carbon()


def test_soca_coverage_separates_positive_pairs_from_exchange_coverage(tmp_path):
    from case_studies.hydrogen.workflow.si_coverage import write_comparison_coverage

    cache = tmp_path / "cache/soca"
    cache.mkdir(parents=True)
    pd.DataFrame({"dataset_id": [1, 2, 3, 4, 5]}).to_parquet(cache / "mapping_df.parquet")
    pd.DataFrame({"dataset_id": [1, 2, 3]}).to_parquet(cache / "xml_df.parquet")
    for mean in ("geometric", "arithmetic"):
        with pd.ExcelWriter(tmp_path / f"sense_soca_activity_coverage_{mean}.xlsx") as writer:
            pd.DataFrame({"dataset_id": [4]}).to_excel(writer, sheet_name="Only_SENSE", index=False)
            pd.DataFrame({"dataset_id": [3]}).to_excel(writer, sheet_name="Only_SOCA", index=False)
    tables = {
        "geometric": pd.DataFrame({"dataset_id": [1, 1, 2]}),
        "arithmetic": pd.DataFrame({"dataset_id": [1, 2]}),
    }
    counts = write_comparison_coverage(tables, tmp_path, tmp_path / "si")
    assert counts["geometric"]["positive_pairs"] == 3
    assert counts["geometric"]["both_exchange_coverage"] == 2
    assert counts["arithmetic"]["neither_exchange_coverage"] == 1
    output = tmp_path / "si/comparison_coverage.tex"
    assert "Population & Geometric & Arithmetic" in output.read_text()
    tables["arithmetic"] = tables["geometric"].copy()
    write_comparison_coverage(tables, tmp_path, tmp_path / "si")
    rendered = output.read_text()
    assert "Population & Count" in rendered
    assert "Counts are identical for both aggregation variants." in rendered
    assert "Positive matched activity--indicator pairs & 3" in rendered
    tables["geometric"] = pd.DataFrame({"dataset_id": [3]})
    with pytest.raises(ValueError, match="lack shared coverage"):
        write_comparison_coverage(tables, tmp_path, tmp_path / "si")


def test_missing_data_geometry_distinguishes_exposure_share_from_risk_sensitivity():
    # Half the source candidates lack assessments, but they account for 3/4
    # of labour. Geometric risk sensitivity uses 1/2, not the labour share.
    hours = np.array([1.0, 3.0])
    no_data = np.array([False, True])
    assert np.sum(hours[no_data]) / np.sum(hours) == 0.75
    from case_studies.hydrogen.workflow.si_missing import coefficients
    from case_studies.hydrogen.workflow.si_productivity import extract
    from sense.core.averaging import average_sector_mappings_w_prices

    def records(factor):
        enriched = pd.DataFrame(
            [
                {
                    "Activity code": "a",
                    "Price": 1.0,
                    "PSILCA_country": "US",
                    "Matched_sector": "Sector",
                    "PSILCA_process": str(i),
                    "Country_weight": 1.0,
                    "PSILCA_Exchanges": [
                        {
                            "category": "Workers",
                            "subcategory": "Health and safety",
                            "indicator": "Rate of fatal accidents at workplace",
                            "type": "risk",
                            "unit": "h",
                            "meanValue": h,
                            "risk level": "no data" if absent else "medium",
                        }
                    ],
                }
                for i, (h, absent) in enumerate(zip(hours, no_data, strict=True))
            ]
        )
        return average_sector_mappings_w_prices(
            enriched, {"medium": 1.0, "no data": factor}, mean_method="geometric"
        )

    base, changed = extract(records(0.1)), extract(records(10.0))
    assert changed[0][2][0]["meanValue"] / base[0][2][0]["meanValue"] == pytest.approx(10.0)
    assert callable(coefficients)
