"""Scientific and persistence invariants for the single-project paired study."""

from __future__ import annotations

import ast
import itertools
import json
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from case_studies.hydrogen.workflow import paired_results, study_protocol
from case_studies.hydrogen.workflow.study_checks import technology_candidates


def test_fresh_foregrounds_reach_import_with_pinned_encoder(monkeypatch, tmp_path):
    import sentence_transformers

    from case_studies.hydrogen.workflow import foregrounds
    from sense.data.embeddings import PINNED_MODEL_REVISIONS

    cfg = {
        "brightway": {"project": "invented-fresh", "regioinvent_database": "InventedRegio"},
        "versions": {"ecoinvent": "3.12", "psilca": "4.0"},
    }
    monkeypatch.setattr(foregrounds, "load", lambda: cfg)
    monkeypatch.setattr(foregrounds, "configured_path", lambda key, config: tmp_path / key)
    monkeypatch.setattr(
        foregrounds, "load_study_inputs", lambda *args, **kwargs: {"subset_targets": []}
    )
    monkeypatch.setattr(foregrounds, "_ensure_project", lambda project: None)
    monkeypatch.setattr(foregrounds.bd, "databases", {})
    model_calls = []

    def constructor(name, **kwargs):
        model_calls.append((name, kwargs))
        return object()

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", constructor)

    class ReachedImport(Exception):
        pass

    def first_import(workbook, database, biosphere, background, **kwargs):
        assert workbook == tmp_path / "methanol_inventory"
        assert database == "02_methanol_inventories_source_baseline"
        assert biosphere == "ecoinvent-3.12-biosphere"
        assert background == "ecoinvent-3.12-cutoff"
        raise ReachedImport

    monkeypatch.setattr(foregrounds, "import_foreground_from_excel", first_import)
    with pytest.raises(ReachedImport):
        foregrounds.run()
    name = "sentence-transformers/all-MiniLM-L6-v2"
    assert model_calls == [
        (
            name,
            {
                "revision": PINNED_MODEL_REVISIONS[name],
                "trust_remote_code": False,
                "model_kwargs": {"use_safetensors": True},
            },
        )
    ]


def synthetic_scores():
    spec = study_protocol.load_protocol()
    technologies = [f"Technology {i}" for i in range(5)]
    rows = []
    for ti, country, scenario, panel in itertools.product(
        range(5), spec["countries"], ("baseline", "regionalized"), spec["panels"]
    ):
        rows.append(
            {
                "technology": technologies[ti],
                "country": country,
                "scenario": scenario,
                "panel": panel["id"],
                "score": (ti + 1) * (1.2 if scenario == "regionalized" else 1),
                "method": json.dumps(panel["method"]),
                "unit": "synthetic units",
                "activity_database": scenario,
                "activity_code": f"{ti}-{country}",
            }
        )
    return spec, technologies, pd.DataFrame(rows)


def test_paired_results_reject_missing_duplicate_and_mixed_method_rows():
    spec, technologies, scores = synthetic_scores()
    paired_results.validate_scores(scores, spec, technologies)
    for bad in (scores.iloc[1:], pd.concat([scores, scores.iloc[:1]])):
        with pytest.raises(ValueError, match="Incomplete/duplicate"):
            paired_results.validate_scores(bad, spec, technologies)
    scores.loc[0, "method"] = json.dumps(["another method"])
    with pytest.raises(ValueError, match="Incorrect characterization"):
        paired_results.validate_scores(scores, spec, technologies)


def test_paired_figure_retains_zero_negative_and_positive_native_values():
    matplotlib = pytest.importorskip("matplotlib")

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    spec, technologies, scores = synthetic_scores()
    scores.loc[0, "score"] = 0
    scores.loc[1, "score"] = -2
    fig = paired_results.plot_paired(scores, spec, technologies)
    try:
        assert len(fig.axes) == 3
        assert all(ax.get_xscale() == "linear" for ax in fig.axes)
        assert sum(len(ax.lines) for ax in fig.axes) == 3 * (5 * 3 * 3 + 4)
    finally:
        plt.close(fig)


def test_fixed_study_and_independent_copies_preserve_existing_mapping_settings():
    spec = study_protocol.load_protocol()
    assert spec["countries"] == ["DE", "CN", "US"]
    assert len(spec["technology_labels"]) == 5
    assert [p["method"][-2] for p in spec["panels"][1:]] == [
        "Children in employment, total",
        "Rate of fatal accidents at workplace",
    ]
    assert study_protocol.sense_config(spec).country_weighting == "gdp"
    modified = study_protocol.load_protocol()
    modified["panels"][1]["method"][-2] = "changed"
    modified["countries"].append("CH")
    assert study_protocol.load_protocol() == spec
    assert study_protocol.sense_config(spec).level_weights["no"] == 0.001
    foreground = study_protocol.foregrounds("baseline")["hydrogen"]
    assert study_protocol.sense_config(spec, database=foreground).iso2_selection == [
        "DE",
        "CN",
        "US",
    ]


def test_localization_preserves_explicit_energy_technology():
    candidates = [
        {"name": "electricity production, wind, offshore"},
        {"name": "electricity production, natural gas"},
    ]
    assert (
        technology_candidates("electricity production, wind, onshore", candidates) == candidates[:1]
    )


@pytest.mark.parametrize("mean", ["geometric", "arithmetic"])
def test_biogas_bookkeeping_exclusions_are_exact_and_foreground_only(mean):
    from sense.eligibility import exclusion_reason

    spec = study_protocol.load_protocol()
    names = spec["foreground_exclude_exact"]
    assert len(names) == 3
    for scenario in study_protocol.SCENARIOS:
        databases = study_protocol.foregrounds(scenario)
        config = study_protocol.sense_config(spec, mean, databases["methanol"])
        assert config.exclude_exact == names
        for name in names:
            assert exclusion_reason({"name": name}, exact=config.exclude_exact) is not None
            # A shared prefix must not exclude operational treatments or variants.
            for other in (name.split(", cut-off")[0], name + ", operational variant"):
                assert exclusion_reason({"name": other}, exact=config.exclude_exact) is None
        for role in ("hydrogen", "subset"):
            assert study_protocol.sense_config(spec, mean, databases[role]).exclude_exact == []
    for database in (
        None,
        "ecoinvent-3.12-cutoff",
        "ecoinvent-3.12-cutoff - regionalized",
        "Regioinvent",
        "custom",
    ):
        config = study_protocol.sense_config(spec, mean, database)
        assert config.exclude_exact == []
        assert all(
            exclusion_reason({"name": name}, exact=config.exclude_exact) is None for name in names
        )


def test_foreground_exclusion_changes_calculation_identity_only_for_affected_mapping():
    spec = study_protocol.load_protocol()
    changed = study_protocol.load_protocol()
    changed["foreground_exclude_exact"] = []
    cfg = {"versions": {}, "brightway": {"project": "test"}}
    assert study_protocol.calculation_context(cfg, spec) != study_protocol.calculation_context(
        cfg, changed
    )
    assert study_protocol.sense_config(spec, database="Regioinvent") == study_protocol.sense_config(
        changed, database="Regioinvent"
    )
    database = study_protocol.foregrounds("baseline")["methanol"]
    assert study_protocol.sense_config(spec, database=database) != study_protocol.sense_config(
        changed, database=database
    )


def test_geometric_and_arithmetic_social_layers_coexist_and_rerun_independently():
    from bw2calc import LCA
    from bw2data import Database, projects

    from sense.brightway.flows import add_social_biosphere_flows_and_exchanges
    from sense.brightway.methods import add_social_impact_categories
    from sense.core.averaging import average_sector_mappings_w_prices

    previous = projects.current
    temporary_root = projects._use_temp_directory()
    project = "paired-test-" + uuid.uuid4().hex
    projects.set_current(project)
    try:
        key = ("technosphere", "activity")
        Database(key[0]).write(
            {
                key: {
                    "name": "Synthetic",
                    "reference product": "product",
                    "unit": "kg",
                    "location": "AA",
                    "type": "process",
                    "exchanges": [{"input": key, "type": "production", "amount": 1}],
                }
            }
        )
        rows = pd.DataFrame(
            [
                {
                    "Activity code": "activity",
                    "Activity name": "Synthetic",
                    "Reference product": "product",
                    "Unit": "kg",
                    "ISO2": "AA",
                    "Price": 1,
                    "PSILCA_country": "AA",
                    "Matched_sector": sector,
                    "PSILCA_Exchanges": [
                        {
                            "indicator": "Risk",
                            "risk level": "medium",
                            "type": "risk",
                            "unit": "h",
                            "category": "Workers",
                            "subcategory": "Test",
                            "meanValue": hours,
                        }
                    ],
                }
                for sector, hours in [("one", 1), ("two", 9)]
            ]
        )
        for mean in ("geometric", "arithmetic", "arithmetic"):
            layer = f"sociosphere_{mean}"
            from bw2data import databases

            if layer not in databases:
                Database(layer).write({})
            records = average_sector_mappings_w_prices(
                rows, level_weights={"medium": 1}, mean_method=mean
            )
            add_social_biosphere_flows_and_exchanges(records, key[0], layer)
            add_social_impact_categories(layer)
        assert len(list(Database(key[0]).get(key[1]).biosphere())) == 2
        for mean, expected in [("geometric", 3), ("arithmetic", 5)]:
            lca = LCA({key: 1}, ("SENSE", f"sociosphere_{mean}", "Workers", "Test", "Risk", "risk"))
            lca.lci()
            lca.lcia()
            assert lca.score == pytest.approx(expected)
    finally:
        import shutil

        projects.set_current("default")
        projects.delete_project(project, delete_dir=False)
        projects._restore_orig_directory()
        projects.set_current(previous)
        shutil.rmtree(temporary_root, ignore_errors=True)


def test_validation_extractor_filters_social_layer_before_aggregation():
    # A synthetic activity with both layers must not yield their sum.
    from types import SimpleNamespace

    module = Path(__file__).parents[1] / "case_studies/hydrogen/workflow/benchmark_comparison.py"
    source = module.read_text(encoding="utf-8")
    node = next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef) and n.name == "extract_bw_social_flows"
    )
    activity_key = ("ei", "a")
    metadata = {
        "sense_kind": "social",
        "categories": ("Workers", "Test"),
        "name": "Risk (risk), AA",
        "unit": "h",
    }
    loaded = {
        activity_key: {
            "name": "A",
            "location": "AA",
            "reference product": "P",
            "exchanges": [
                {"type": "biosphere", "input": ("geo", "f"), "amount": 3},
                {"type": "biosphere", "input": ("arith", "f"), "amount": 5},
            ],
        }
    }
    scope = {
        "pd": pd,
        "np": np,
        "tqdm": lambda items, **kwargs: items,
        "Database": lambda name: SimpleNamespace(load=lambda: {(name, "f"): metadata}),
        "get_stakeholder_from_categories": lambda categories: categories,
        "parse_sense_flow_name": lambda name: ("Risk", "risk"),
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), "extractor", "exec"), scope)
    mapping = pd.DataFrame({"bw_key_meta": [activity_key], "dataset_id": [1]})
    result = scope["extract_bw_social_flows"](SimpleNamespace(load=lambda: loaded), mapping, "geo")
    assert result["amount"].tolist() == [3]


def test_paired_price_sync_changes_metadata_only_and_rejects_conflicting_seeds(
    tmp_path, monkeypatch
):
    import shutil

    import bw2data as bd

    from case_studies.hydrogen.workflow import paired_prices
    from sense.data import ecoinvent
    from sense.utils.helpers import slug

    spec = study_protocol.load_protocol()
    cfg = {
        "brightway": {"project": "prices-test"},
        "paths": {"data_dir": str(tmp_path)},
        "versions": {"ecoinvent": "3.12"},
    }
    temporary_root = bd.projects._use_temp_directory()
    bd.projects.set_current(cfg["brightway"]["project"])
    try:
        for scenario in ("baseline", "regionalized"):
            for role, name in study_protocol.foregrounds(scenario).items():
                key = (name, "a")
                bd.Database(name).write(
                    {
                        key: {
                            "name": role,
                            "reference product": role,
                            "unit": "kg",
                            "location": "DE",
                            "price": 99.0,
                            "exchanges": [{"input": key, "type": "production", "amount": 2}],
                        }
                    }
                )

        def prices(_mapping, _output, database, **kwargs):
            assert (
                kwargs["exclude_exact"]
                == study_protocol.sense_config(spec, database=database).exclude_exact
            )
            return pd.DataFrame({"code": ["a"], "price": [7.0]})

        monkeypatch.setattr(ecoinvent, "fetch_ecoinvent_activities", prices)
        audit = paired_prices.synchronize_prices(cfg, spec)
        assert len(audit) == 3
        for scenario in ("baseline", "regionalized"):
            for name in study_protocol.foregrounds(scenario).values():
                activity = bd.Database(name).get("a")
                assert activity["price"] == 7
                assert list(activity.production())[0]["amount"] == 2
                assert not list(activity.technosphere())
        regional = study_protocol.foregrounds("regionalized")["hydrogen"]
        seed = tmp_path / "Output/SeedPrices" / f"seed_prices_{slug(regional)}.json"
        seed.write_text(
            json.dumps(
                {
                    "metadata": {"expected_currency": "EUR", "expected_price_year": 2005},
                    "seeds": [{"unit": "kg", "reference_product": "hydrogen", "price": 9}],
                }
            )
        )
        with pytest.raises(ValueError, match="conflicts with paired price"):
            paired_prices.synchronize_prices(cfg, spec)
        assert bd.Database(regional).get("a")["price"] == 7
    finally:
        bd.projects.set_current("default")
        bd.projects.delete_project(cfg["brightway"]["project"], delete_dir=False)
        bd.projects._restore_orig_directory()
        shutil.rmtree(temporary_root, ignore_errors=True)


def test_run_references_keep_both_layers_in_one_project(tmp_path):
    from case_studies.hydrogen.workflow import study_runs

    spec = study_protocol.load_protocol()
    cfg = {"brightway": {"project": "shared"}, "paths": {"data_dir": str(tmp_path)}}
    registry = {
        json.dumps(["shared", "ei", spec["social_layers"][mean]]): str(tmp_path / mean)
        for mean in ("geometric", "arithmetic")
    }
    (tmp_path / "study_runs.json").write_text(json.dumps(registry))
    refs = study_runs.run_references(cfg, spec, [("ei", "geometric"), ("ei", "arithmetic")])
    assert len({r["run_directory"] for r in refs}) == 2
    with pytest.raises(ValueError, match="Missing study run"):
        study_runs.run_references(cfg, spec, [("Regioinvent", "arithmetic")])


def test_worker_coverage_distinguishes_missing_processes_from_duplicate_sector_labels():
    from case_studies.hydrogen.audit_indicators import summarize

    def record(label, source, indicator, risk):
        return {
            "location": "DE",
            "reference_name": label,
            "source_file": source,
            "subcategory": "Working time",
            "indicator": indicator,
            "assessment_type": "risk",
            "risk_level": risk,
        }

    records = pd.DataFrame(
        [
            record("A", "industry.xml", "hours", "no"),
            record("A", "commodity.xml", "hours", "no data"),
            record("B", "other.xml", "another indicator", "low"),
            record("C", "valid.xml", "hours", "low"),
            record("C", "valid.xml", "hours", "high"),
        ]
    )
    hours = summarize(records).set_index("indicator").loc["hours"]
    assert hours.country_sector_labels == 3
    assert hours.assessed_sector_labels == 1
    assert hours.country_processes == 4
    assert hours.assessed_processes == 2  # Explicit 'no' is an assessment.
    assert hours.absent_processes == 1
    assert hours.unassessed_processes == 2
    assert hours.assessment_coverage_fraction == 0.5


def test_fixed_indicator_audit_reports_gaps_and_rejects_absent_indicators(tmp_path, capsys):
    from case_studies.hydrogen.audit_indicators import audit_study

    spec = study_protocol.load_protocol()
    rows = [
        {
            "location": country,
            "reference_name": "Sector",
            "source_file": country + ".xml",
            "category": "Workers",
            "subcategory": panel["method"][-3],
            "indicator": panel["method"][-2],
            "assessment_type": "risk",
            "risk_level": "medium",
        }
        for country in spec["countries"]
        for panel in spec["panels"][1:]
    ]
    # Excluded stakeholders must not enter the worker assessment universe.
    rows.append({**rows[0], "category": "Local community", "source_file": "community.xml"})
    source = tmp_path / "Output/PSILCA/psilca_assessments_v4.0.parquet"
    source.parent.mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(source)
    cfg = {"paths": {"data_dir": str(tmp_path)}, "versions": {"psilca": "4.0"}}
    coverage, evidence = audit_study(cfg)
    assert len(coverage) == 6
    assert coverage.assessed_processes.eq(1).all()
    assert len(evidence["sha256"]) == 64
    assert evidence["missing_data_factor"] == 0.1
    saved = json.loads((tmp_path / "paired_hydrogen/indicator_audit/evidence.json").read_text())
    assert saved == evidence
    rows[0]["risk_level"] = "no data"
    pd.DataFrame(rows).to_parquet(source)
    coverage, evidence = audit_study(cfg)
    assert evidence["domestic_assessment_gaps"] == [
        {
            "location": "DE",
            "indicator": "Children in employment, total",
            "country_sector_labels": 1,
            "unassessed_sector_labels": 1,
        }
    ]
    assert "[DATA GAP] DE" in capsys.readouterr().out
    pd.DataFrame(rows[1:]).to_parquet(source)
    with pytest.raises(ValueError, match="Missing or ambiguous indicator coverage record"):
        audit_study(cfg)


def test_raw_worker_audit_normalizes_levels_and_filters_stakeholders(tmp_path):
    from case_studies.hydrogen.audit_indicators import audit

    source = tmp_path / "export"
    source.mkdir()
    (source / "process_1.xml").write_text("""
    <ecoSpold xmlns="http://www.EcoInvent.org/EcoSpold01"><dataset>
      <referenceFunction name="Sector"/><geography location="DE"/>
      <exchange category="Workers" subCategory="Working time" name="Hours; no risk"/>
      <exchange category="Local community" subCategory="Test" name="Another; high risk"/>
    </dataset></ecoSpold>""")
    result = audit(source, tmp_path / "output")
    assert len(result) == 1
    assert result.iloc[0].indicator == "Hours"
    assert result.iloc[0].assessed_sector_labels == 1


def test_capture_methods_includes_all_stakeholders_opportunities_and_hours_in_one_layer():
    from case_studies.hydrogen.workflow.result_archive import capture_methods

    family = "Example midpoint"
    methods = {
        (family, "climate"): {"unit": "kg"},
        (family, "water"): {"unit": "m3"},
        ("SENSE", "sociosphere", "Workers", "Child labour", "Children", "risk"): {"unit": "h"},
        ("SENSE", "sociosphere", "Society", "Health", "Health", "risk"): {"unit": "h"},
        ("SENSE", "sociosphere", "Workers", "Benefits", "Leave", "opportunity"): {"unit": "h"},
        ("SENSE", "sociosphere", "Social", "Working hours", "working hours"): {"unit": "h"},
        ("SENSE", "sociosphere_arithmetic", "Workers", "Child labour", "Children", "risk"): {
            "unit": "h"
        },
        ("Other environmental family", "climate"): {"unit": "kg"},
    }
    captured = capture_methods(methods, family, "sociosphere")
    assert len(captured) == 6
    assert captured.kind.value_counts().to_dict() == {
        "environmental": 2,
        "social_risk": 2,
        "social_opportunity": 1,
        "working_hours": 1,
    }
    assert captured.method_id.is_unique
    assert not captured.method.str.contains("arithmetic").any()
    methods.pop(("SENSE", "sociosphere", "Social", "Working hours", "working hours"))
    with pytest.raises(ValueError, match="total working-hours"):
        capture_methods(methods, family, "sociosphere")


def test_complete_archive_preserves_tiny_signed_contributions_and_all_activity_identities(tmp_path):
    from scipy import sparse

    from case_studies.hydrogen.workflow import result_archive

    spec = study_protocol.load_protocol()
    cfg = {"versions": {"ecoinvent": "3.12", "psilca": "4.0"}, "brightway": {"project": "test"}}
    context = study_protocol.calculation_context(cfg, spec)
    registered = {tuple(p["method"]): {"unit": "h"} for p in spec["panels"]}
    registered[
        (spec["environmental_family"], "climate change", "global warming potential (GWP100)")
    ] = {"unit": "h"}
    hours = ("SENSE", "sociosphere", "Social", "Working hours", "working hours")
    registered[hours] = {"unit": "h"}
    methods = result_archive.capture_methods(
        registered,
        spec["environmental_family"],
        "sociosphere",
        climate_method=tuple(spec["panels"][0]["method"]),
    )
    activities = pd.DataFrame(
        [
            {
                "activity_index": i,
                "database": db,
                "code": "0001",
                "name": "Same label",
                "isic_code": ["0111", "0111", "Unclassified", "3510"][i],
                "isic_division": ["01", "01", "Unclassified", "35"][i],
                "reference_product": "Same product",
                "location": "NA",
                "unit": "kg",
            }
            for i, db in enumerate(["first", "second", "third", "zero"])
        ]
    )
    values = sparse.csr_matrix(np.tile([2.0, -0.5, 1e-18, 0.0], (len(methods), 1)))
    totals = np.asarray(values.sum(axis=1)).ravel()
    metadata = {
        "context": context,
        "technologies": list(spec["technology_labels"]),
        "study_at_calculation": spec,
    }
    target = tmp_path / "results.zip"
    with result_archive.ResultsWriter(target, methods, activities, metadata) as writer:
        for technology, country, scenario in itertools.product(
            spec["technology_labels"], spec["countries"], ("baseline", "regionalized")
        ):
            writer.add(
                {
                    "technology": technology,
                    "country": country,
                    "scenario": scenario,
                    "demand": 1.0,
                    "activity_database": scenario,
                    "activity_code": technology + country,
                },
                values,
                totals,
            )
        scores = writer.finish(pd.DataFrame({"coverage": [0.5]}))
    assert len(scores) == 30 * len(methods)
    saved, saved_methods, saved_activities, unit = result_archive.load_contributions(target, 0)
    np.testing.assert_array_equal(saved.toarray(), values.toarray())
    assert saved.nnz == 3 * len(methods)  # Tiny and negative contributions survive.
    assert saved_activities.location.tolist() == ["NA"] * 4
    assert saved_activities.database.tolist() == ["first", "second", "third", "zero"]
    assert saved_activities.code.tolist() == ["0001"] * 4
    assert saved_activities.isic_division.tolist() == ["01", "01", "Unclassified", "35"]
    contributors = result_archive.contributor_table(target, 0, hours)
    grouped = contributors.groupby("isic_division").contribution.sum()
    assert grouped["01"] == pytest.approx(1.5)
    assert grouped["Unclassified"] == pytest.approx(1e-18, abs=1e-25)
    assert len(contributors) == 4  # Zero contributions remain filterable too.
    assert grouped.sum() == pytest.approx(totals[0])
    assert contributors.contribution_unit.eq("h").all()
    with pytest.raises(ValueError, match="Method absent"):
        result_archive.contributor_table(target, 0, ("missing",))
    assert saved_methods.method_id.tolist() == methods.method_id.tolist()
    assert unit["demand"] == 1.0

    # Change the plot to hours after calculation, without touching or opening Brightway.
    updated = study_protocol.load_protocol()
    updated["panels"][1] = {"id": "hours", "label": "Worker hours", "method": list(hours)}
    assert study_protocol.calculation_context(cfg, updated) == context
    loaded, _ = result_archive.load_results(target, context=context)
    selected = paired_results.select_panels(loaded, updated, list(spec["technology_labels"]))
    assert len(selected) == 90
    assert set(selected.panel) == {"climate", "hours", "fatal_accidents"}
    updated["panels"][1]["method"] = ["Not captured"]
    with pytest.raises(ValueError, match="Method was not captured"):
        paired_results.select_panels(loaded, updated, list(spec["technology_labels"]))
    with pytest.raises(ValueError, match="another study calculation"):
        result_archive.load_results(target, context={})

    original = target.read_bytes()
    with pytest.raises(ValueError, match="Incomplete"):
        with result_archive.ResultsWriter(target, methods, activities, metadata) as writer:
            writer.finish(pd.DataFrame())
    assert target.read_bytes() == original
    with pytest.raises(ValueError, match="reconcile"):
        with result_archive.ResultsWriter(target, methods, activities, metadata) as writer:
            writer.add({}, values, totals + 1)
    assert target.read_bytes() == original
    assert list(tmp_path.iterdir()) == [target]


def test_batched_characterization_matches_brightway_scores_and_contributors():
    import shutil

    import bw2data as bd
    from bw2calc import LCA

    from case_studies.hydrogen.workflow.result_archive import (
        activity_contributions,
        characterization_rows,
    )

    root = bd.projects._use_temp_directory()
    project = "all-methods-" + uuid.uuid4().hex
    bd.projects.set_current(project)
    try:
        flow = ("biosphere", "f")
        hours = ("biosphere", "hours")
        bd.Database("biosphere").write(
            {
                flow: {"name": "Emission", "unit": "kg", "type": "emission"},
                hours: {"name": "Worker hours", "unit": "h", "type": "emission"},
            }
        )
        a, b = ("tech", "a"), ("tech", "b")
        bd.Database("tech").write(
            {
                a: {
                    "name": "A",
                    "unit": "kg",
                    "exchanges": [
                        {"input": a, "type": "production", "amount": 1},
                        {"input": b, "type": "technosphere", "amount": 2},
                        {"input": flow, "type": "biosphere", "amount": -1},
                        {"input": hours, "type": "biosphere", "amount": 1e-12},
                    ],
                },
                b: {
                    "name": "B",
                    "unit": "kg",
                    "exchanges": [
                        {"input": b, "type": "production", "amount": 1},
                        {"input": flow, "type": "biosphere", "amount": 3},
                        {"input": hours, "type": "biosphere", "amount": 5},
                    ],
                },
            }
        )
        methods = [("test", "environment"), ("test", "hours")]
        for key, cfs in zip(methods, [[(flow, 2)], [(hours, 1)]], strict=True):
            method = bd.Method(key)
            method.register(unit="test")
            method.write(cfs)
        lca = LCA({a: 1})
        lca.lci(factorize=True)
        table = pd.DataFrame({"method": [json.dumps(m) for m in methods]})
        cf_rows = characterization_rows(lca, table)
        for demand in ({a: 1}, {b: 3}):
            lca.redo_lci(demand)
            contributions, scores = activity_contributions(cf_rows, lca.inventory)
            for i, method in enumerate(methods):
                lca.switch_method(method)
                lca.lcia()
                assert scores[i] == pytest.approx(lca.score)
                expected = np.asarray(lca.characterized_inventory.sum(axis=0)).ravel()
                np.testing.assert_allclose(contributions.getrow(i).toarray().ravel(), expected)
        # An explicit negative direct contribution remains visible in the full inventory.
        lca.redo_lci({a: 1})
        contributions, _ = activity_contributions(cf_rows, lca.inventory)
        assert contributions[0, lca.activity_dict[a]] == -2
        assert contributions[1, lca.activity_dict[a]] == pytest.approx(1e-12, abs=1e-20)
    finally:
        bd.projects.set_current("default")
        bd.projects.delete_project(project, delete_dir=False)
        bd.projects._restore_orig_directory()
        shutil.rmtree(root, ignore_errors=True)


def test_activity_sectors_preserve_classifications_and_do_not_invent_missing_labels():
    from case_studies.hydrogen.workflow.result_archive import activity_metadata

    activity = {
        "name": "Same name",
        "reference product": "Product",
        "location": "NA",
        "unit": "kg",
        "classifications": [("ISIC rev.4 ecoinvent", "0111:Growing cereals"), ("CPC", "01111")],
    }
    first = activity_metadata(("original", "a"), activity)
    second = activity_metadata(("regionalized", "b"), activity)
    assert (first["database"], first["code"]) != (second["database"], second["code"])
    assert first["isic_code"] == "0111"
    assert first["isic_division"] == "01"
    assert first["isic_sector"] == "0111: Growing cereals"
    assert first["isic_status"] == "classified"
    assert json.loads(first["classifications"])[1] == ["CPC", "01111"]
    for values, status in [
        ([], "missing"),
        ([("ISIC rev.3", "0111:Other revision")], "missing"),
        ([("ISIC rev.4 ecoinvent", "not a code")], "unparseable"),
        (
            [("ISIC rev.4 ecoinvent", "0111:Cereals"), ("ISIC rev.4", "3510:Electricity")],
            "ambiguous",
        ),
    ]:
        result = activity_metadata(("foreground", "h"), {**activity, "classifications": values})
        assert result["isic_status"] == status
        assert result["isic_division"] == "Unclassified"
        assert result["isic_code"] == "Unclassified"
