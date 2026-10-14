"""Boundary tests for package loading, caching, and pipeline orchestration."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

import sense.core.matching as matching
import sense.pipeline as pipeline
import sense.preparation as preparation
from sense.artifacts import write_json_atomic
from sense.config import SenseConfig
from sense.geography import read_ecoinvent_geographies


def test_publication_metadata_and_source_distribution_are_consistent():
    root = Path(__file__).resolve().parents[1]
    metadata = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    citation = yaml.safe_load((root / "CITATION.cff").read_text(encoding="utf-8"))
    project = metadata["project"]
    assert citation["version"] == project["version"]
    assert [
        {"name": f"{author['given-names']} {author['family-names']}"}
        for author in citation["authors"]
    ] == project["authors"]
    assert citation["repository-code"] == project["urls"]["Repository"]
    targets = metadata["tool"]["hatch"]["build"]["targets"]
    included = set(targets["sdist"]["only-include"])
    assert {"README.md", "LICENSE", "CITATION.cff", "environment.yml", "pyproject.toml"} <= included
    assert "environment-macos.yml" in included
    workflow = yaml.safe_load((root / ".github/workflows/ci.yml").read_text(encoding="utf-8"))
    # Keep expensive suites opt-in without locking the runner labels or expressions.
    events = yaml.load(
        (root / ".github/workflows/ci.yml").read_text(encoding="utf-8"), Loader=yaml.BaseLoader
    )["on"]
    assert events["workflow_dispatch"]["inputs"]["full_suite"]["default"] == "false"
    assert workflow["concurrency"]["cancel-in-progress"] is True
    assert all((root / path).is_file() for path in included)
    for directory in ("src", "tests", "case_studies", "tools"):
        python_files = {
            path.relative_to(root).as_posix()
            for path in (root / directory).rglob("*.py")
            if "Data" not in path.parts
        }
        assert python_files <= included, python_files - included
    assert set(targets["wheel"]["only-include"]) == {
        path for path in included if path.startswith("src/sense/")
    }
    readme = (root / "README.md").read_text(encoding="utf-8")
    case_study_readme = (root / "case_studies/hydrogen/README.md").read_text(encoding="utf-8")
    # The case study is documented separately; the package README only links to it.
    assert "case_studies/hydrogen/README.md" in readme
    for text in (readme, case_study_readme):
        assert "/Users/" not in text
    assert (root / "src/sense/resources/THIRD_PARTY_NOTICES.txt").is_file()


def test_package_import_is_lazy_about_brightway():
    environment = dict(os.environ)
    environment["PYTHONPATH"] = "src"
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import inspect, sys, sense; "
                "assert 'bw2data' not in sys.modules; "
                "from sense import run_sense; "
                "params = inspect.signature(run_sense).parameters; "
                "assert 'project' in params; "
                "assert 'ecoinvent_dir' not in params; "
                "assert 'biosphere_db' not in params; "
                "print(sense.__version__)"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert completed.returncode == 0, completed.stderr


def test_atomic_json_failure_preserves_previous_file(tmp_path):
    destination = tmp_path / "artifact.json"
    destination.write_text('{"state": "valid"}', encoding="utf-8")

    with pytest.raises(TypeError):
        write_json_atomic(destination, {"invalid": object()})

    assert destination.read_text(encoding="utf-8") == '{"state": "valid"}'


def test_brightway_cache_token_uses_metadata_without_database_scan(monkeypatch):
    monkeypatch.setattr(
        preparation,
        "databases",
        {"technosphere": {"backend": "sqlite", "modified": "2026-01-01"}},
    )
    monkeypatch.setattr(
        preparation,
        "Database",
        lambda _: (_ for _ in ()).throw(AssertionError("database was traversed")),
    )

    first = preparation._brightway_database_token("technosphere")
    second = preparation._brightway_database_token("technosphere")

    assert first == second


def test_activity_price_cache_tracks_linked_database_tokens(monkeypatch):
    tokens = {"target": "target-v1", "linked": "linked-v1"}
    monkeypatch.setattr(preparation, "databases", {"target": {}, "linked": {}})
    monkeypatch.setattr(
        preparation,
        "_brightway_database_token",
        lambda database_name: tokens[database_name],
    )
    expected = {"schema": 2, "price_inference_version": 4}
    manifest = {
        **expected,
        "pricing_dependency_tokens": {
            "target": "target-v1",
            "linked": "linked-v1",
        },
    }

    assert preparation._activity_cache_matches(manifest, expected)
    tokens["linked"] = "linked-v2"
    assert not preparation._activity_cache_matches(manifest, expected)


@pytest.mark.parametrize("source", ["target", "linked"])
def test_activity_cache_refresh_preserves_source_and_dependency_invalidation(
    tmp_path, monkeypatch, source
):
    tokens = {"target": "target-v1", "linked": "linked-v1"}
    monkeypatch.setattr(preparation, "databases", {name: {} for name in tokens})
    monkeypatch.setattr(preparation, "_brightway_database_token", tokens.__getitem__)
    manifest_path = tmp_path / "activity.manifest.json"
    preparation.write_manifest(
        manifest_path,
        {
            "brightway_database_token": tokens["target"],
            "price_source_database": source,
            "price_source_database_token": tokens[source],
            "pricing_dependency_tokens": dict(tokens),
        },
    )

    tokens["target"] = "target-v2"
    preparation._refresh_activity_cache_token(manifest_path, "target")
    manifest = preparation.read_manifest(manifest_path)
    expected = {
        "brightway_database_token": tokens["target"],
        "price_source_database": source,
        "price_source_database_token": tokens[source],
    }
    assert preparation._activity_cache_matches(manifest, expected)

    # Refreshing our own writes must not hide changes to a linked database.
    tokens["linked"] = "linked-v2"
    preparation._refresh_activity_cache_token(manifest_path, "target")
    assert not preparation._activity_cache_matches(
        preparation.read_manifest(manifest_path), expected
    )


def test_matching_accepts_preloaded_activity_entry(monkeypatch):
    entry = {
        "ISO2": "AA",
        "PSILCA_mapping": "AA",
        "Activities": [
            {
                "code": "activity",
                "name": "Activity",
                "reference product": "Product",
                "unit": "kg",
                "price": 2.0,
            }
        ],
    }

    def fake_load(_directory, name):
        if name.startswith("ecoinvent_"):
            return np.array([[1.0, 0.0]]), pd.Series(["activity"]), {}
        return np.array([[1.0, 0.0]]), pd.Series(["AA::Sector"]), {}

    monkeypatch.setattr(matching, "load_embeddings", fake_load)
    result = matching.match_from_stores(
        "ecoinvent",
        "psilca",
        "AA",
        activity_entry=entry,
    )

    assert result[["Activity code", "Matched_sector", "Price"]].to_dict("records") == [
        {"Activity code": "activity", "Matched_sector": "Sector", "Price": 2.0}
    ]


def test_shared_geography_normalization(monkeypatch, tmp_path):
    source = pd.DataFrame(
        [
            {
                "Shortname": "US-CO",
                "Name": "Colorado",
                "Geography Classification": "Province or State",
                "Contained and Overlapping Geographies": "",
            },
            {
                "Shortname": "RER",
                "Name": "Europe",
                "Geography Classification": "ecoinvent Region",
                "Contained and Overlapping Geographies": "DE; FR; DE-AT",
            },
        ]
    )
    monkeypatch.setattr(pd, "read_excel", lambda *_args, **_kwargs: source)

    result = read_ecoinvent_geographies(tmp_path / "geographies.xlsx")

    assert result["simplified_ISO2"].tolist() == ["US", "DE; FR"]


def test_run_sense_wires_prepared_data_into_one_region(monkeypatch, tmp_path):
    monkeypatch.setattr("sense.brightway.state.capture_database_state", lambda *a, **kw: {})
    psilca_dir = tmp_path / "psilca"
    psilca_dir.mkdir()
    (psilca_dir / "synthetic.xml").write_text("preparation is stubbed in this test")
    workbook = tmp_path / "Database-Overview-for-ecoinvent-test.xlsx"
    workbook.touch()

    class Activity(dict):
        key = ("technosphere", "activity")

    activity = Activity(code="activity", location="AA")
    monkeypatch.setattr(pipeline, "_load_technosphere_activities", lambda _: [activity])
    selected_projects: list[str] = []
    monkeypatch.setattr(
        pipeline,
        "projects",
        type(
            "Projects",
            (),
            {
                "__contains__": lambda self, name: True,
                "set_current": lambda self, name: selected_projects.append(name),
            },
        )(),
    )
    monkeypatch.setattr(
        pipeline,
        "Database",
        lambda name: [activity] if name == "technosphere" else [],
    )
    monkeypatch.setattr(
        pipeline,
        "databases",
        {"technosphere": {"modified": "1"}, "sociosphere": {}},
    )
    monkeypatch.setattr(pipeline, "_prepare_psilca_data", lambda **_: None)
    monkeypatch.setattr(
        pipeline,
        "_prepare_location_mapping",
        lambda **_: pd.DataFrame(
            [{"Shortname": "AA", "PSILCA_mapping": "AA", "simplified_ISO2": "AA"}]
        ),
    )
    monkeypatch.setattr(pipeline, "_prepare_embedding_stores", lambda **_: None)
    monkeypatch.setattr(
        pipeline,
        "_prepare_activity_list",
        lambda **_: [
            {
                "ISO2": "AA",
                "PSILCA_mapping": "AA",
                "Activities": [{"code": "activity", "location": "AA", "price": 1.0}],
            }
        ],
    )
    social_cleanup_calls: list[str] = []
    monkeypatch.setattr(
        pipeline,
        "_clear_managed_exchanges",
        lambda **kwargs: social_cleanup_calls.append(kwargs["technosphere_db"]) or 0,
    )
    received: list[str] = []

    def run_region(**kwargs):
        received.append(kwargs["target_iso2"])
        assert kwargs["iso2_data"]["AA"]["Activities"][0]["code"] == "activity"
        assert kwargs["techno_lookup"] == {"activity": activity}
        return {
            "ISO2": "AA",
            "num_activities": 1,
            "num_matches": 1,
            "social_added": 0,
            "wh_added": 0,
            "issues": [],
        }

    monkeypatch.setattr(pipeline, "_run_social_for_iso2", run_region)

    result = pipeline.run_sense(
        project="project",
        technosphere_db="technosphere",
        sociosphere_db="sociosphere",
        psilca_dir=psilca_dir,
        ecoinvent_version="test",
        psilca_version="4.0",
        config=SenseConfig(
            iso2_selection=["AA"],
            enable_social_flows=False,
            enable_working_hours_flows=False,
        ),
        data_dir=tmp_path,
        ecoinvent_geographies_excel=workbook,
    )

    assert selected_projects == ["project"]
    assert received == ["AA"]
    assert social_cleanup_calls == ["technosphere", "technosphere"]
    assert result[0]["num_matches"] == 1
    assert list((tmp_path / "Output" / "Runs").glob("*/Monetary/*.json"))
    diagnostic_path = next((tmp_path / "Output" / "Diagnostics").glob("*.json"))
    diagnostics = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    assert diagnostics["regions_processed"] == 1
    assert diagnostics["processing_issues"] == []


def test_release_allowlists_are_files_and_notebook_metadata_is_empty():
    import tomllib

    root = Path(__file__).resolve().parents[1]
    targets = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["tool"]["hatch"][
        "build"
    ]["targets"]
    for target in ("sdist", "wheel"):
        for name in targets[target]["only-include"]:
            assert (root / name).is_file() and not (root / name).is_symlink()
            assert not any(token in name for token in ("..", "*", ".local.", "REVISION_NOTES"))
            if name.endswith(".ipynb"):
                notebook = json.loads((root / name).read_text(encoding="utf-8"))
                for cell in notebook["cells"]:
                    assert not cell.get("outputs") and not cell.get("attachments")
                    assert cell.get("execution_count") is None


def test_fallback_price_cache_tracks_reference_price_and_output_amount(monkeypatch):
    import sense.preparation as preparation

    class Record(dict):
        key = ("synthetic", "a")

        def technosphere(self):
            return []

        def production(self):
            return [production]

    activity = Record({"production amount": 1})
    production = Record({"amount": 1, "properties": {"price": {"amount": 2, "unit": "USD"}}})
    production.input = activity
    monkeypatch.setattr(preparation, "Database", lambda _: [activity])
    monkeypatch.setattr(preparation, "databases", {"synthetic": {}})
    first = preparation._brightway_database_token("synthetic")
    production["properties"]["price"]["amount"] = 3
    second = preparation._brightway_database_token("synthetic")
    assert second != first
    activity["production amount"] = 2
    assert preparation._brightway_database_token("synthetic") != second
