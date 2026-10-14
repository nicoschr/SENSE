"""Failure-path regressions for the full repository cleanup."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from filelock import FileLock

from case_studies.hydrogen.workflow import soca, study_protocol, study_runs
from sense import SenseConfig
from sense.artifacts import write_json_atomic
from sense.data.embeddings import load_embeddings, save_embeddings, store_is_complete
from sense.data.price_inference import ProductGraphCache, run_product_inference
from sense.data.seed_prices import load_seed_template
from sense.pipeline import _select_iso2_regions
from sense.run_context import RunEvidence
from sense.utils.locking import exclusive_run


@pytest.mark.parametrize(
    "options",
    [
        {"iso2_selection": []},
        {"iso2_selection": "DE"},
        {"iso2_selection": ["DE", "DE"]},
        {"iso2_selection": [""]},
        {"use_psilca": "false"},
        {"enable_social_flows": "false"},
        {"minimum_similarity": "0.5"},
        {"similarity_delta": True},
        {"embedding_model": ""},
        {"ecoinvent_to_psilca_price_factor": -1},
        {"productivity_eur_per_usd": float("inf")},
        {"level_weights": {}},
        {"level_weights": {"low": True}},
        {"psilca_process_overrides": {1: "x.xml"}},
        {"productivity_baseline_path": "x.parquet", "productivity_location_proxies": {"X": "DEU"}},
    ],
)
def test_malformed_configuration_is_rejected(options):
    with pytest.raises(ValueError):
        SenseConfig(**options)


def test_explicit_selection_cannot_silently_ignore_misspelled_regions():
    with pytest.raises(ValueError, match="absent"):
        _select_iso2_regions({"DE": {}}, SenseConfig(iso2_selection=["DE", "ZZ"]))
    assert _select_iso2_regions({"CN": {}, "DE": {}}, SenseConfig(iso2_selection=["DE"])) == ["DE"]


@pytest.mark.parametrize("corruption", ["array", "manifest", "path"])
def test_embedding_cache_detects_torn_writes_and_rejects_paths(tmp_path, corruption):
    save_embeddings(str(tmp_path), "test", np.eye(2), pd.Series(["a", "b"]), {})
    if corruption == "array":
        np.save(tmp_path / "test.npy", np.ones((2, 2), dtype=np.float32))
    else:
        path = tmp_path / "test.manifest.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        if corruption == "path":
            data["emb_path"] = "../other.npy"
        else:
            data = []
        path.write_text(json.dumps(data))
    assert not store_is_complete(str(tmp_path), "test")
    with pytest.raises(ValueError):
        load_embeddings(str(tmp_path), "test")


def test_completed_layers_survive_mutable_cache_refresh_but_detect_tampering(tmp_path):
    import bw2data as bd

    from sense.brightway.state import capture_database_state

    bd.projects.set_current("test")
    bd.Database("ei").write(
        {
            ("ei", "a"): {
                "name": "Invented production",
                "unit": "kg",
                "location": "DE",
                "exchanges": [],
            }
        }
    )
    cfg = {
        "brightway": {"project": "test"},
        "versions": {"ecoinvent": "3.12", "psilca": "4.0"},
        "paths": {"data_dir": str(tmp_path)},
    }
    spec = study_protocol.load_protocol()
    cache = tmp_path / "activity.manifest.json"
    source = tmp_path / "geography.xlsx"
    source.write_bytes(b"synthetic workbook")
    runs = {}
    for mean in ("geometric", "arithmetic"):
        cache.write_text(json.dumps({"token": mean}))
        evidence = RunEvidence.start(
            tmp_path / "Output",
            "test",
            "ei",
            spec["social_layers"][mean],
            study_protocol.sense_config(spec, mean, "ei"),
            {"activity_manifest": cache, "geographies": source},
            snapshot_inputs={"activity_manifest"},
            versions=cfg["versions"],
        )
        cache.write_text('{"token":"post-write"}')
        write_json_atomic(evidence.directory / "diagnostics.json", {"processing_issues": []})
        evidence.complete(database_state=capture_database_state("ei", spec["social_layers"][mean]))
        study_runs.record_run(cfg, spec, "ei", mean, [{"run_directory": str(evidence.directory)}])
        runs[mean] = evidence.directory
    for mean, directory in runs.items():
        assert study_runs.require_run(cfg, spec, "ei", mean) == directory
    (runs["geometric"] / "diagnostics.json").write_text('{"processing_issues":[],"changed":true}')
    with pytest.raises(ValueError, match="artifact changed"):
        study_runs.require_run(cfg, spec, "ei")
    source.write_bytes(b"changed source")
    with pytest.raises(ValueError, match="input changed"):
        study_runs.require_run(cfg, spec, "ei", "arithmetic")


@pytest.mark.parametrize("prices", [[-1], ["oops"], [1, 2]])
def test_bad_or_conflicting_seed_prices_are_not_silently_discarded(tmp_path, prices):
    path = tmp_path / "seeds.json"
    path.write_text(
        json.dumps(
            {"seeds": [{"unit": "kg", "reference_product": "P", "price": p} for p in prices]}
        )
    )
    with pytest.raises(ValueError):
        load_seed_template(str(path))


def test_recursive_price_overflow_stops_instead_of_publishing_infinity():
    cache = ProductGraphCache(
        activity_to_id={},
        id_to_product=[("kg", "P"), ("kg", "I")],
        id_to_meta=[{}, {}],
        recipe_out=[0],
        recipe_inputs=[[(1, 1e200)]],
        recipe_act_db=["d"],
        recipe_act_code=["a"],
        recipe_act_name=["a"],
        recipe_act_location=["DE"],
        producers_by_pid=[[0], []],
        pid_activity_price_seed={},
    )
    with pytest.raises(ValueError, match="overflowed"):
        run_product_inference(cache, {1: 1e200}, {1: "test"})


def test_writer_locks_fail_fast_and_release_after_exceptions(tmp_path):
    @exclusive_run
    def run(**kwargs):
        if kwargs.get("fail"):
            raise ValueError("simulated failure")
        return "done"

    with FileLock(tmp_path / ".sense-run.lock"):
        with pytest.raises(RuntimeError, match="Another SENSE run"):
            run(data_dir=tmp_path)
    with pytest.raises(ValueError, match="simulated failure"):
        run(data_dir=tmp_path, fail=True)
    assert run(data_dir=tmp_path) == "done"


def test_soca_parquet_cache_verifies_both_tables(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "process.xml").write_text("synthetic")
    ctx = SimpleNamespace(
        spold_folder=source,
        spold_pattern="*.xml",
        use_cache=True,
        force_rebuild_cache=False,
        cache_dir=tmp_path,
        cache_mapping=tmp_path / "mapping.parquet",
        cache_xml=tmp_path / "xml.parquet",
        cache_meta=tmp_path / "cache.json",
    )
    mapping = pd.DataFrame({"file": ["source/process.xml"], "dataset_id": [1]})
    exchanges = pd.DataFrame({"dataset_id": [1], "amount": [2.0]})
    calls = []
    monkeypatch.setattr(
        soca, "process_spold_folder", lambda ctx: calls.append(True) or (mapping, exchanges)
    )
    soca.load_or_parse_spold(ctx)
    cached, _ = soca.load_or_parse_spold(ctx)
    pd.testing.assert_frame_equal(cached, mapping)
    assert len(calls) == 1
    exchanges.assign(amount=99).to_parquet(ctx.cache_xml, index=False)
    _, repaired = soca.load_or_parse_spold(ctx)
    assert len(calls) == 2 and repaired.amount.tolist() == [2.0]


@pytest.fixture
def temporary_project():
    import shutil
    import uuid

    import bw2data as bd

    previous = bd.projects.current
    root = bd.projects._use_temp_directory()
    name = "cleanup-" + uuid.uuid4().hex
    bd.projects.set_current(name)
    try:
        yield bd
    finally:
        bd.projects.set_current("default")
        bd.projects.delete_project(name, delete_dir=False)
        bd.projects._restore_orig_directory()
        bd.projects.set_current(previous)
        shutil.rmtree(root, ignore_errors=True)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -1, "oops"])
def test_invalid_exchange_never_deletes_previous_inventory(temporary_project, invalid):
    from sense.brightway.flows import add_working_hours_flows_and_exchanges

    bd = temporary_project
    key = ("tech", "a")
    bd.Database("tech").write(
        {
            key: {
                "name": "A",
                "unit": "kg",
                "location": "DE",
                "exchanges": [{"input": key, "type": "production", "amount": 1}],
            }
        }
    )
    bd.Database("social").write({})
    data = [{"Activity code": "a", "ISO2": "DE", "Working_hours": 2}]
    add_working_hours_flows_and_exchanges(data, "tech", "social")
    data[0]["Working_hours"] = invalid
    with pytest.raises(ValueError, match="finite and non-negative"):
        add_working_hours_flows_and_exchanges(data, "tech", "social")
    assert [e["amount"] for e in bd.get_activity(key).biosphere()] == [2]
    # A corrupt existing amount is repaired, not accepted as equal through NaN arithmetic.
    exchange = next(iter(bd.get_activity(key).biosphere()))
    exchange["amount"] = float("nan")
    exchange.save()
    data[0]["Working_hours"] = 2
    add_working_hours_flows_and_exchanges(data, "tech", "social")
    assert [e["amount"] for e in bd.get_activity(key).biosphere()] == [2]


def test_subset_construction_uses_the_stage_call_signature(temporary_project, tmp_path):
    from case_studies.hydrogen.workflow.foreground_construction import localize_ecoinvent_subset

    bd = temporary_project
    key = ("base", "a")
    bd.Database("base").write(
        {
            key: {
                "name": "Hydrogen",
                "reference product": "hydrogen",
                "unit": "kg",
                "location": "GLO",
                "exchanges": [{"input": key, "type": "production", "amount": 2}],
            }
        }
    )
    localize_ecoinvent_subset(
        "base",
        "subset",
        ["base"],
        ["DE", "CN", "US"],
        {("Hydrogen", "hydrogen", "GLO")},
        SimpleNamespace(),
        0.75,
        tmp_path / "mapping.xlsx",
    )
    assert {a["location"] for a in bd.Database("subset")} == {"DE", "CN", "US"}
    assert all(next(iter(a.production()))["amount"] == 2 for a in bd.Database("subset"))


def test_batched_matching_preserves_dense_selection_and_order():
    from sense.core.matching import _retained_matches

    rng = np.random.default_rng(42)
    sectors, activities = rng.normal(size=(7, 5)), rng.normal(size=(13, 5))
    columns = np.array([12, 2, 5, 0, 7, 8, 1])
    scores = sectors @ activities[columns].T
    best = scores.max(axis=0)
    s, a = (scores >= best[None, :] - 0.8).nonzero()
    expected = [(int(i), int(j), scores[i, j], best[j]) for i, j in zip(s, a, strict=True)]
    np.testing.assert_allclose(
        _retained_matches(sectors, activities, columns, 0.8, batch_size=2), expected
    )


@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf"), "bad"])
def test_invalid_productivity_override_is_never_silently_ignored(bad):
    from sense.core.averaging import average_sector_mappings_w_prices

    with pytest.raises(ValueError, match="hours_override"):
        average_sector_mappings_w_prices(pd.DataFrame(), {}, hours_override={"a": bad})


def test_conflicting_prices_within_one_activity_are_rejected():
    from sense.core.averaging import average_sector_mappings_w_prices

    rows = pd.DataFrame({"Activity code": ["a", "a"], "Price": [1, 2]})
    with pytest.raises(ValueError, match="Conflicting prices"):
        average_sector_mappings_w_prices(rows, {})


def _activity(database, code, *, amount=1, exchanges=()):
    key = (database, code)
    return {
        "name": code,
        "reference product": code,
        "location": "DE",
        "unit": "kg",
        "exchanges": [{"input": key, "type": "production", "amount": amount}, *exchanges],
    }


def test_price_inference_uses_real_production_exchange(temporary_project):
    from sense.data.price_inference import build_product_graph

    bd = temporary_project
    product = _activity(
        "graph",
        "product",
        amount=2,
        exchanges=[{"input": ("graph", "input"), "type": "technosphere", "amount": 6}],
    )
    product["production amount"] = 99  # Deliberately stale metadata.
    bd.Database("graph").write(
        {("graph", "product"): product, ("graph", "input"): _activity("graph", "input")}
    )
    graph = build_product_graph("graph")
    out_pid = graph.activity_to_id[("graph", "product")]
    recipe = graph.recipe_out.index(out_pid)
    assert graph.recipe_inputs[recipe] == [(graph.activity_to_id[("graph", "input")], 3)]


def test_background_check_reads_unprocessed_links(temporary_project):
    from case_studies.hydrogen.workflow.study_checks import check_backgrounds

    bd = temporary_project
    for name in ("base", "regionalized", "regioinvent"):
        bd.Database(name).write({(name, "a"): _activity(name, "a")})
    for name, provider in (("regionalized", "regioinvent"), ("regioinvent", "regionalized")):
        bd.get_activity((name, "a")).new_exchange(
            input=(provider, "a"), type="technosphere", amount=1
        ).save()
    # Metadata still reflects the previously processed, empty technosphere.
    assert check_backgrounds("base", "regionalized", "regioinvent")["regionalized"] == [
        "regioinvent"
    ]
    bd.get_activity(("base", "a")).new_exchange(
        input=("regioinvent", "a"), type="technosphere", amount=1
    ).save()
    with pytest.raises(ValueError, match="Original ecoinvent depends"):
        check_backgrounds("base", "regionalized", "regioinvent")


def test_paired_foregrounds_report_missing_prerequisite(temporary_project):
    from case_studies.hydrogen.workflow.study_checks import check_pairs

    with pytest.raises(ValueError, match="Missing foreground databases: baseline, regionalized"):
        check_pairs(["baseline"], ["regionalized"], ["DE"])
    assert "baseline" not in temporary_project.databases


def test_paired_foregrounds_report_empty_selection(temporary_project):
    from case_studies.hydrogen.workflow.study_checks import check_pairs

    for name in ("baseline", "regionalized"):
        temporary_project.Database(name).write({(name, "a"): _activity(name, "a")})
    with pytest.raises(ValueError, match="No foreground activities"):
        check_pairs(["baseline"], ["regionalized"], ["CN"])


def test_paired_foregrounds_keep_same_identity_in_separate_database_roles(temporary_project):
    from case_studies.hydrogen.workflow.study_checks import check_pairs

    bd = temporary_project
    baseline = ["hydrogen_base", "methanol_base"]
    regionalized = ["hydrogen_regional", "methanol_regional"]
    for names in (baseline, regionalized):
        for amount, name in enumerate(names, 1):
            bd.Database(name).write({(name, "shared"): _activity(name, "shared", amount=amount)})
    pairs = check_pairs(baseline, regionalized, ["DE"])
    assert [(left.key[0], right.key[0]) for left, right in pairs] == list(
        zip(baseline, regionalized, strict=True)
    )
    assert len(pairs) == 2
    # Different role-specific production amounts expose accidental cross-pairing.
    with pytest.raises(ValueError, match="production"):
        check_pairs(baseline, list(reversed(regionalized)), ["DE"])


def test_paired_foregrounds_still_reject_duplicates_within_one_database(temporary_project):
    from case_studies.hydrogen.workflow.study_checks import check_pairs

    bd = temporary_project
    duplicate = _activity("baseline", "a")
    duplicate["exchanges"] = [{"input": ("baseline", "b"), "type": "production", "amount": 1}]
    bd.Database("baseline").write(
        {("baseline", "a"): _activity("baseline", "a"), ("baseline", "b"): duplicate}
    )
    bd.Database("regionalized").write({("regionalized", "a"): _activity("regionalized", "a")})
    with pytest.raises(ValueError, match="Ambiguous foreground identity in baseline"):
        check_pairs(["baseline"], ["regionalized"], ["DE"])


@pytest.mark.parametrize(
    "baseline,regionalized",
    [([], []), (["a"], ["b", "c"]), (["a", "a"], ["b", "c"]), (["a", "b"], ["c", "c"])],
)
def test_paired_foregrounds_require_one_database_per_role(
    temporary_project, baseline, regionalized
):
    from case_studies.hydrogen.workflow.study_checks import check_pairs

    with pytest.raises(ValueError, match="distinct paired databases"):
        check_pairs(baseline, regionalized, ["DE"])


def test_paired_foregrounds_reject_production_sign_difference(temporary_project):
    from case_studies.hydrogen.workflow.study_checks import check_pairs

    bd = temporary_project
    for name, amount in (("baseline", 1), ("regionalized", -1)):
        bd.Database(name).write({(name, "a"): _activity(name, "a", amount=amount)})
    with pytest.raises(ValueError, match="production"):
        check_pairs(["baseline"], ["regionalized"], ["DE"])


def test_input_replacement_preserves_uncertainty_and_checks_missing_provider(temporary_project):
    from case_studies.hydrogen.workflow.inventory_operations import (
        replace_technosphere_exchange_in_activity,
    )

    bd = temporary_project
    target = _activity(
        "fg",
        "target",
        exchanges=[
            {
                "input": ("fg", "old"),
                "type": "technosphere",
                "amount": 2,
                "uncertainty type": 2,
                "loc": 0.5,
                "scale": 0.1,
            }
        ],
    )
    bd.Database("fg").write(
        {
            ("fg", "target"): target,
            **{("fg", name): _activity("fg", name) for name in ("old", "new")},
        }
    )
    args = ("fg", "target", "target", ["old"], "old", "new", "new")
    assert replace_technosphere_exchange_in_activity(*args) == (1, 0)
    exchange = next(iter(bd.get_activity(("fg", "target")).technosphere()))
    assert exchange.input.key == ("fg", "new")
    assert (
        exchange["amount"],
        exchange["uncertainty type"],
        exchange["loc"],
        exchange["scale"],
    ) == (2, 2, 0.5, 0.1)
    assert replace_technosphere_exchange_in_activity(*args) == (0, 1)
    with pytest.raises(ValueError, match="provider is missing"):
        replace_technosphere_exchange_in_activity(*args[:-2], "missing", "missing")


def test_psilca_cache_rebuilds_when_only_one_artifact_changes(tmp_path, monkeypatch):
    from sense import preparation

    source = tmp_path / "source"
    source.mkdir()
    (source / "a.xml").write_text("synthetic")
    calls = []

    def extract(**kwargs):
        from pathlib import Path

        calls.append(True)
        Path(kwargs["combined_output_path"]).write_bytes(b"assessments")
        Path(kwargs["css_list_output_path"]).write_text("{}")

    monkeypatch.setattr(preparation, "combine_psilca_css_and_extract_social_exchanges", extract)
    args = (
        source,
        SenseConfig(),
        tmp_path / "cache.parquet",
        tmp_path / "sectors.json",
        tmp_path / "zero.csv",
    )
    preparation._prepare_psilca_data(*args)
    preparation._prepare_psilca_data(*args)
    assert len(calls) == 1
    args[3].write_text('{"modified":true}')
    preparation._prepare_psilca_data(*args)
    assert len(calls) == 2


def test_failed_download_preserves_existing_input_and_removes_temporary_file(tmp_path, monkeypatch):
    import io

    from case_studies.hydrogen import fetch_open_data

    destination = tmp_path / "trade.db"
    destination.write_bytes(b"old")
    monkeypatch.setattr(
        fetch_open_data.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(b"wrong")
    )
    with pytest.raises(RuntimeError, match="SHA-256"):
        fetch_open_data._fetch_verified(
            "https://example.org/data", "expected", destination, label="test", size_hint="small"
        )
    assert destination.read_bytes() == b"old"
    assert list(tmp_path.iterdir()) == [destination]


def test_model_loader_requires_safe_weights_without_remote_code(monkeypatch):
    import sys

    from sense.data.embeddings import load_encoder

    calls = []
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=lambda *a, **kw: calls.append((a, kw))),
    )
    load_encoder("test-model", revision="a" * 40)
    assert calls == [
        (
            ("test-model",),
            {
                "revision": "a" * 40,
                "trust_remote_code": False,
                "model_kwargs": {"use_safetensors": True},
            },
        )
    ]


def test_version_labels_cannot_escape_output_directory(tmp_path):
    from sense.run_context import RunPaths

    with pytest.raises(ValueError, match="path separators"):
        RunPaths.create(
            data_dir=tmp_path,
            technosphere_database="db",
            ecoinvent_version="../../private",
            psilca_version="4",
            use_psilca=True,
        )
