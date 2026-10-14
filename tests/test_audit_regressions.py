"""Numerical and state-transition regressions found in the repository audit."""

from __future__ import annotations

import ast
import json
import shutil
import uuid
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sense import SenseConfig, run_sense
from sense.brightway.flows import add_social_biosphere_flows_and_exchanges
from sense.core.averaging import average_sector_mappings_w_prices
from sense.data.ecoinvent import detect_unmapped_locations, fetch_ecoinvent_activities
from sense.exceptions import SenseInputRequiredError
from sense.preparation import _check_location_concordance
from sense.productivity.geography import build_location_mapping
from sense.productivity.loader import ProductivityBaseline
from sense.regional import _preflight_productivity_coverage
from sense.run_context import latest_run_directory
from sense.weighting import CountryWeightProvider

WORKFLOW = Path(__file__).resolve().parents[1] / "case_studies/hydrogen/workflow"


@pytest.fixture
def brightway():
    """Isolate real database writes and release connections before Windows cleanup."""
    import bw2data as bd

    root = bd.projects._use_temp_directory()
    name = f"sense-audit-{uuid.uuid4().hex}"
    bd.projects.set_current(name)
    try:
        yield bd
    finally:
        bd.projects.set_current("default")
        bd.projects.delete_project(name, delete_dir=False)
        bd.projects._restore_orig_directory()
        shutil.rmtree(root, ignore_errors=True)


def _database(bd, name, activities):
    bd.Database(name).write(
        {
            (name, code): {
                "name": code,
                "reference product": "product",
                "unit": "kg",
                "location": location,
                "price": 10.0,
                "classifications": [("ISIC rev.4 ecoinvent", "20")],
                "exchanges": [{"input": (name, code), "amount": amount, "type": "production"}],
            }
            for code, location, amount in activities
        }
    )


def _inputs(tmp_path):
    geo = tmp_path / "geographies.xlsx"
    pd.DataFrame(
        [
            {
                "Shortname": code,
                "Name": name,
                "Geography Classification": kind,
                "Contained and Overlapping Geographies": countries,
            }
            for code, name, kind, countries in [
                ("CH", "Switzerland", "Country", "CH"),
                ("DE", "Germany", "Country", "DE"),
                ("GLO", "World", "World", "CH; DE"),
                ("RER", "Europe", "ecoinvent Region", "CH; DE"),
            ]
        ]
    ).to_excel(geo, sheet_name="Geographies", index=False)
    baseline = tmp_path / "productivity.parquet"
    pd.DataFrame(
        [
            {"country": c, "year": 2020, "isic4_code": "20", "productivity_hrs_per_usd": 2.0}
            for c in ["CHE", "DEU"]
        ]
    ).to_parquet(baseline)
    return geo, baseline


def test_reference_production_rerun_scope_and_shared_methods(brightway, tmp_path):
    """Fresh and repeated runs yield the same per-unit results without cross-DB damage."""
    bd = brightway
    LCA = pytest.importorskip("bw2calc").LCA
    geo, baseline = _inputs(tmp_path)
    _database(bd, "first", [("a", "CH", 2), ("b", "CH", 1), ("c", "DE", 1)])
    _database(bd, "second", [("d", "DE", 1)])

    def run(database, **options):
        return run_sense(
            project=bd.projects.current,
            technosphere_db=database,
            sociosphere_db="social",
            ecoinvent_version="3.12",
            data_dir=tmp_path,
            ecoinvent_geographies_excel=geo,
            config=SenseConfig(
                exclude_empty_inventories=False,
                use_psilca=False,
                productivity_baseline_path=baseline,
                productivity_eur_per_usd=1,
                country_weighting="equal",
                **options,
            ),
        )

    run("first", enable_isic_working_hours_methods=True)
    original_run = latest_run_directory(tmp_path / "Output", bd.projects.current, "first")
    original_manifest = (original_run / "complete.json").read_bytes()
    method = ("SENSE", "social", "Social", "Working hours", "working hours")
    lca = LCA({("first", "a"): 1}, method)
    lca.lci()
    lca.lcia()
    assert lca.score == pytest.approx(20)
    assert {exc["amount"] for exc in bd.get_activity(("first", "a")).biosphere()} == {40}

    run("second")
    run("first", exclude_exact=["b"])
    assert not list(bd.get_activity(("first", "b")).biosphere())
    assert all(
        exc.input.get("sense_kind") != "working_hours_isic"
        for act in bd.Database("first")
        for exc in act.biosphere()
    )
    run("first", iso2_selection=["CH"], enable_working_hours_flows=False)
    assert not list(bd.get_activity(("first", "a")).biosphere())
    assert list(bd.get_activity(("first", "c")).biosphere())
    assert list(bd.get_activity(("second", "d")).biosphere())
    assert method in bd.methods
    assert (original_run / "complete.json").read_bytes() == original_manifest
    assert latest_run_directory(tmp_path / "Output", bd.projects.current, "first") != original_run

    run("second", exclude_exact=["d"])
    assert not list(bd.get_activity(("second", "d")).biosphere())

    # The same production basis applies to risk and opportunity exchanges.
    add_social_biosphere_flows_and_exchanges(
        [
            {
                "Activity code": "a",
                "ISO2": "CH",
                "Averaged_Exchanges": [
                    {
                        "indicator": "I",
                        "category": "Workers",
                        "subcategory": "A",
                        "type": kind,
                        "unit": "h",
                        "meanValue": value,
                    }
                    for kind, value in [("risk", 3), ("opportunity", 5)]
                ],
            }
        ],
        "first",
        "social",
    )
    assert sorted(exc["amount"] for exc in bd.get_activity(("first", "a")).biosphere()) == [6, 10]


def test_incomplete_productivity_region_is_reported_and_rejected(tmp_path):
    geo, baseline = _inputs(tmp_path)
    frame = pd.read_parquet(baseline)
    frame[frame["country"] == "CHE"].to_parquet(baseline)
    provider = CountryWeightProvider.equal(reference_year=2020)
    mapping = build_location_mapping(geo, {"CHE"}, provider)
    source = ProductivityBaseline(baseline, mapping, weight_provider=provider)
    for result in [
        source.lookup_with_status("RER", "20"),
        source.lookup_row_with_status(["CHE", "DEU"], "20"),
    ]:
        assert result.status == "override_partial_aggregate"
        assert (result.n_countries_attempted, result.n_countries_used) == (2, 1)
    with pytest.raises(ValueError, match="complete productivity coverage"):
        _preflight_productivity_coverage(
            regions=["RER"],
            iso2_data={"RER": {"Activities": [{"code": "a", "ISIC rev.4 ecoinvent": "20"}]}},
            config=SenseConfig(),
            baseline=source,
        )


def test_unusable_geography_never_silently_omits_an_activity(brightway, tmp_path):
    mapping = pd.DataFrame({"Shortname": ["CH", "CUSTOM"], "PSILCA_mapping": ["CH", None]})
    assert detect_unmapped_locations({"CUSTOM"}, mapping) == ["CUSTOM"]
    (tmp_path / "location_concordance_test.json").write_text(
        json.dumps(
            {
                "locations": [{"location_code": "CUSTOM", "countries": "ZZ"}],
            }
        ),
        encoding="utf-8",
    )
    sectors = tmp_path / "sectors.json"
    sectors.write_text(json.dumps({"CH": ["sector"]}), encoding="utf-8")
    with pytest.raises(SenseInputRequiredError, match="no countries available"):
        _check_location_concordance("test", mapping, tmp_path, "test", sectors, {"CUSTOM"})
    _database(brightway, "test", [("lost", "CUSTOM", 1)])
    with pytest.raises(ValueError, match="no usable country mapping"):
        fetch_ecoinvent_activities(
            mapping, str(tmp_path / "activities.json"), "test", exclude_empty_inventories=False
        )
    assert not (tmp_path / "activities.json").exists()


def test_rest_of_world_subtracts_constituents_of_existing_regions(brightway, tmp_path):
    bd = brightway
    _database(bd, "tech", [("rest", "RoW", 1), ("europe", "RER", 1)])
    for activity in bd.Database("tech"):
        activity["name"] = "Same production activity"
        activity.save()
    mapping = pd.DataFrame(
        {
            "Shortname": ["GLO", "RER"],
            "simplified_ISO2": ["CH; DE; FR", "CH; DE"],
            "PSILCA_mapping": ["CH; DE; FR", "CH; DE"],
        }
    )
    output = tmp_path / "activities.json"
    fetch_ecoinvent_activities(mapping, str(output), "tech", exclude_empty_inventories=False)
    regions = {row["ISO2"]: row for row in json.loads(output.read_text(encoding="utf-8"))}
    assert regions["RoW"]["Activities"][0]["PSILCA_mapping"] == "FR"


@pytest.mark.parametrize("hours", [-2, float("nan"), float("inf")])
def test_invalid_matched_psilca_hours_stop_before_empty_output(hours):
    frame = pd.DataFrame(
        [
            {
                "Activity code": "a",
                "Price": 1,
                "PSILCA_country": "CH",
                "Matched_sector": "s",
                "PSILCA_Exchanges": [
                    {
                        "indicator": "I",
                        "meanValue": hours,
                        "risk level": "medium",
                        "type": "risk",
                        "unit": "h",
                    }
                ],
            }
        ]
    )
    with pytest.raises(ValueError, match="Invalid PSILCA worker-hour"):
        average_sector_mappings_w_prices(frame, SenseConfig().level_weights)


def _workflow_functions(filename, names, namespace):
    definitions = [
        node
        for path in (
            [WORKFLOW / "benchmark_comparison.py", WORKFLOW / "soca.py"]
            if filename == "benchmark.py"
            else [WORKFLOW / filename]
        )
        for node in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    exec(compile(ast.Module(body=definitions, type_ignores=[]), filename, "exec"), namespace)
    return namespace


def test_validation_preserves_stakeholder_and_subgroup_identity():
    functions = _workflow_functions(
        "benchmark.py",
        {"build_compare", "get_stakeholder_from_categories"},
        {"pd": pd, "np": np, "STAKEHOLDER_GROUPS": {"Workers", "Society"}},
    )
    data = pd.DataFrame(
        {
            "dataset_id": [1, 1],
            "stakeholder_group": ["Workers", "Society"],
            "subgroup": ["A", "B"],
            "indicator": ["Same", "Same"],
            "type": ["risk", "risk"],
            "amount": [1, 9],
        }
    )
    soca = data.rename(columns={"amount": "medium_risk_opportunity_hours"}).copy()
    soca["medium_risk_opportunity_hours"] = [9, 1]
    result = functions["build_compare"](data, soca)
    assert len(result) == 2
    assert np.allclose(result["log_diff"].abs(), np.log10(9))
    assert functions["get_stakeholder_from_categories"](("Workers", "A")) == ("Workers", "A")


def test_validation_checks_source_version_and_normalized_product_units():
    pytest.importorskip("bw2io")
    overview = pd.DataFrame(
        {
            "ecoQuery URL": ["https://ecoquery.ecoinvent.org/3.12/cutoff/dataset/1/"],
            "Activity UUID": ["uuid"],
            "Activity Name": ["activity"],
            "Geography": ["CH"],
            "Reference Product Name": ["product"],
            "Unit": ["kg"],
        }
    )
    functions = _workflow_functions(
        "benchmark.py",
        {"build_ao_and_bw_match"},
        {
            "pd": pd,
            "defaultdict": defaultdict,
            "CFG": {"versions": {"ecoinvent": "3.12"}},
            "OVERVIEW_XLSX": "unused",
            "read_excel_sheet": lambda *a, **kw: overview.copy(),
            "norm": lambda x: str(x or "").strip().lower(),
        },
    )

    class Activity(dict):
        key = ("tech", "a")

    activity = Activity(
        name="activity", **{"reference product": "product", "location": "CH", "unit": "kilogram"}
    )
    source = pd.DataFrame({"dataset_id": [1], "eco_version": ["3.12"], "system_model": ["cutoff"]})
    from functools import partial
    from types import SimpleNamespace

    compare = partial(
        functions["build_ao_and_bw_match"],
        SimpleNamespace(cfg={"versions": {"ecoinvent": "3.12"}}, overview_xlsx="unused"),
    )
    assert compare(source, [activity])["bw_key_meta"].tolist() == [("tech", "a")]
    activity["unit"] = "kilowatt hour"
    assert compare(source, [activity])["bw_key_meta"].isna().all()
    with pytest.raises(ValueError, match="source version"):
        compare(source.assign(eco_version="3.11"), [activity])


def test_soca_cache_detects_content_changes_with_unchanged_metadata(tmp_path):
    import os

    function = _workflow_functions("benchmark.py", {"spold_signature"}, {"Path": Path})[
        "spold_signature"
    ]
    path = tmp_path / "source.spold"
    path.write_text("first", encoding="utf-8")
    stamp = path.stat()
    first = function(tmp_path, "*.spold")
    path.write_text("other", encoding="utf-8")
    os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    assert function(tmp_path, "*.spold") != first


def test_figure_inputs_require_recalculation_after_analysis_changes(tmp_path, monkeypatch):
    from case_studies.hydrogen.workflow import figure_io

    path = tmp_path / "inputs.zip"
    figure_io.save_inputs(path, {"values": pd.DataFrame({"x": [1]})}, {"context": {}})
    monkeypatch.setattr(figure_io, "_analysis_signature", lambda: "changed-analysis")
    assert not figure_io.reuse_inputs(path, context={})
    with pytest.raises(ValueError, match="analysis code changes"):
        figure_io.load_inputs(path, context={})


@pytest.mark.parametrize("amount", [0, float("nan"), float("inf"), -float("inf")])
def test_unsupported_reference_production_cannot_change_existing_exchanges(brightway, amount):
    bd = brightway
    _database(bd, "tech", [("a", "CH", 1)])
    bd.Database("social").write({})
    data = [
        {
            "Activity code": "a",
            "ISO2": "CH",
            "Averaged_Exchanges": [
                {
                    "indicator": "I",
                    "category": "Workers",
                    "subcategory": "A",
                    "type": "risk",
                    "unit": "h",
                    "meanValue": 3,
                },
            ],
        }
    ]
    add_social_biosphere_flows_and_exchanges(data, "tech", "social")
    activity = bd.get_activity(("tech", "a"))
    production = next(iter(activity.production()))
    production["amount"] = amount
    production.save()
    data[0]["Averaged_Exchanges"][0]["meanValue"] = 9
    with pytest.raises(ValueError, match="nonzero finite reference production"):
        add_social_biosphere_flows_and_exchanges(data, "tech", "social")
    assert next(iter(activity.biosphere()))["amount"] == 3


def test_localized_inventory_preserves_production_metadata_and_flow_geography(brightway):
    from case_studies.hydrogen.workflow.inventory_localization import (
        BiosphereLocalizer,
        clone_activity,
    )

    bd = brightway
    bd.Database("biosphere3_spatialized_flows").write(
        {
            ("biosphere3_spatialized_flows", loc): {
                "name": f"Water, {loc}",
                "unit": "kg",
                "categories": ("water",),
                "type": "emission",
            }
            for loc in ["CH", "DE"]
        }
    )
    _database(bd, "source", [("a", "CH", 2)])
    activity = bd.get_activity(("source", "a"))
    activity.new_exchange(
        input=("biosphere3_spatialized_flows", "CH"),
        amount=4,
        type="biosphere",
        **{"uncertainty type": 3, "loc": 4, "scale": 0.2},
    ).save()
    clone = clone_activity(activity, ("target", "a"), "DE")
    clone["exchanges"].append(BiosphereLocalizer().exchange(next(iter(activity.biosphere())), "DE"))
    bd.Database("target").write({("target", "a"): clone})
    result = bd.get_activity(("target", "a"))
    assert next(iter(result.production()))["amount"] == 2
    assert result["classifications"] == activity["classifications"]
    exchange = next(iter(result.biosphere()))
    assert exchange.input.key == ("biosphere3_spatialized_flows", "DE")
    assert (exchange["amount"], exchange["scale"]) == (4, 0.2)
    with pytest.raises(ValueError, match="spatial counterpart"):
        BiosphereLocalizer().exchange(next(iter(activity.biosphere())), "FR")


@pytest.mark.parametrize("production", [-1.0, -2.0])
def test_negative_reference_treatment_preserves_labour_and_credit_signs(
    brightway, tmp_path, production
):
    bd = brightway
    LCA = pytest.importorskip("bw2calc").LCA
    geo, baseline = _inputs(tmp_path)
    _database(bd, "waste", [("treatment", "CH", production)])
    # Three units of waste leave the generator and require treatment.
    bd.Database("generator").write(
        {
            ("generator", "source"): {
                "name": "waste generator",
                "reference product": "product",
                "unit": "kg",
                "location": "CH",
                "exchanges": [
                    {"input": ("generator", "source"), "amount": 1.0, "type": "production"},
                    {"input": ("waste", "treatment"), "amount": -3.0, "type": "technosphere"},
                ],
            }
        }
    )
    config = SenseConfig(
        exclude_empty_inventories=False,
        use_psilca=False,
        productivity_baseline_path=baseline,
        productivity_eur_per_usd=1,
        country_weighting="equal",
    )

    def run():
        return run_sense(
            project=bd.projects.current,
            technosphere_db="waste",
            sociosphere_db="social",
            ecoinvent_version="3.12",
            data_dir=tmp_path,
            ecoinvent_geographies_excel=geo,
            config=config,
        )

    run()
    run()  # Idempotent; do not duplicate labour exchanges on retry.
    treatment = bd.get_activity(("waste", "treatment"))
    assert next(iter(treatment.production()))["amount"] == production
    assert [exc["amount"] for exc in treatment.biosphere()] == [20 * abs(production)]
    method = ("SENSE", "social", "Social", "Working hours", "working hours")
    for demand, expected in [
        ({("generator", "source"): 1}, 60),
        ({("waste", "treatment"): -1}, 20),
        ({("waste", "treatment"): 1}, -20),
    ]:
        lca = LCA(demand, method)
        lca.lci()
        lca.lcia()
        assert lca.score == pytest.approx(expected)
    add_social_biosphere_flows_and_exchanges(
        [
            {
                "Activity code": "treatment",
                "ISO2": "CH",
                "Averaged_Exchanges": [
                    {
                        "indicator": "I",
                        "category": "Workers",
                        "subcategory": "A",
                        "type": "risk",
                        "unit": "h",
                        "meanValue": 3,
                    }
                ],
            }
        ],
        "waste",
        "social",
    )
    risk = [exc for exc in treatment.biosphere() if exc.input.get("sense_kind") != "working_hours"]
    assert len(risk) == 1 and risk[0]["amount"] == 3 * abs(production)


def test_production_market_exclusion_removes_stale_hours_and_keeps_suppliers(brightway, tmp_path):
    """Regioinvent's supplier mix must not add a second production workforce."""
    bd = brightway
    LCA = pytest.importorskip("bw2calc").LCA
    geo, baseline = _inputs(tmp_path)
    _database(bd, "trade", [("factory", "DE", 1), ("mix", "CH", 1)])
    market = bd.get_activity(("trade", "mix"))
    market["name"] = "production market for product"
    market.save()
    market.new_exchange(input=("trade", "factory"), amount=1.0, type="technosphere").save()
    physical = [(e.input.key, e["amount"]) for e in market.technosphere()]

    def run(**kwargs):
        run_sense(
            project=bd.projects.current,
            technosphere_db="trade",
            sociosphere_db="social",
            ecoinvent_version="3.12",
            data_dir=tmp_path,
            ecoinvent_geographies_excel=geo,
            config=SenseConfig(
                exclude_empty_inventories=False,
                use_psilca=False,
                productivity_baseline_path=baseline,
                productivity_eur_per_usd=1.0,
                country_weighting="equal",
                enable_isic_working_hours_methods=True,
                **kwargs,
            ),
        )

    # Reproduce the previous default and populate its stale direct labour.
    run(exclude_terms_starting_with=["market for", "market group for", "generic market for"])
    method = ("SENSE", "social", "Social", "Working hours", "working hours")
    old = LCA({market.key: 1.0}, method)
    old.lci()
    old.lcia()
    assert old.score == pytest.approx(40.0)
    run()
    assert not list(market.biosphere())
    assert [(e.input.key, e["amount"]) for e in market.technosphere()] == physical
    assert len(list(bd.get_activity(("trade", "factory")).biosphere())) == 2
    corrected = LCA({market.key: 1.0}, method)
    corrected.lci()
    corrected.lcia()
    assert corrected.score == pytest.approx(20.0)


@pytest.mark.parametrize("rest,base", [("RoW", "GLO"), ("RoE", "RER")])
@pytest.mark.parametrize(
    "other,geography,proxy,expected",
    [
        ("CX", "CX", "AU; DE; FR", "AU; DE; FR"),
        ("ISLAND", "CX", "AU; DE", "AU; DE; FR"),
        ("AU-NSW", "AU", "AU", "DE; FR"),
        ("REGION", "AU; DE; CX", "AU; DE", "FR"),
    ],
)
def test_rest_excludes_actual_geography_not_social_proxy(
    brightway, tmp_path, rest, base, other, geography, proxy, expected
):
    _database(brightway, "tech", [("rest", rest, 1), ("other", other, 1)])
    for activity in brightway.Database("tech"):
        activity["name"] = "Same production activity"
        activity.save()
    mapping = pd.DataFrame(
        [
            {
                "Shortname": base,
                "simplified_ISO2": "AU; DE; FR; CX",
                "PSILCA_mapping": "AU; DE; FR",
            },
            {"Shortname": other, "simplified_ISO2": geography, "PSILCA_mapping": proxy},
        ]
    )
    output = tmp_path / "activities.json"
    fetch_ecoinvent_activities(mapping, str(output), "tech", exclude_empty_inventories=False)
    regions = {row["ISO2"]: row for row in json.loads(output.read_text(encoding="utf-8"))}
    assert regions[rest]["Activities"][0]["PSILCA_mapping"] == expected
    assert regions[other]["PSILCA_mapping"] == proxy


@pytest.mark.parametrize("rest,base", [("RoW", "GLO"), ("RoE", "RER")])
@pytest.mark.parametrize("geography", [None, "", "AU; DE; FR"])
def test_rest_still_rejects_missing_geography_or_genuinely_empty_residual(
    brightway, tmp_path, rest, base, geography
):
    _database(brightway, "tech", [("rest", rest, 1), ("other", "REGION", 1)])
    for activity in brightway.Database("tech"):
        activity["name"] = "Same production activity"
        activity.save()
    mapping = pd.DataFrame(
        [
            {"Shortname": base, "simplified_ISO2": "AU; DE; FR", "PSILCA_mapping": "AU; DE; FR"},
            {"Shortname": "REGION", "simplified_ISO2": geography, "PSILCA_mapping": "AU; DE; FR"},
        ]
    )
    output = tmp_path / "activities.json"
    expected = "empty .* country mapping" if geography else "no actual country membership"
    with pytest.raises(ValueError, match=expected):
        fetch_ecoinvent_activities(mapping, str(output), "tech", exclude_empty_inventories=False)
    assert not output.exists()


def test_rest_exclusions_keep_reference_products_separate(brightway, tmp_path):
    _database(brightway, "tech", [("rest", "RoW", 1), ("other", "GLO", 1)])
    for activity in brightway.Database("tech"):
        activity["name"] = "Same production activity"
        activity["reference product"] = activity["code"]
        activity.save()
    mapping = pd.DataFrame(
        [{"Shortname": "GLO", "simplified_ISO2": "AU; DE", "PSILCA_mapping": "AU; DE"}]
    )
    output = tmp_path / "activities.json"
    fetch_ecoinvent_activities(mapping, str(output), "tech", exclude_empty_inventories=False)
    regions = {row["ISO2"]: row for row in json.loads(output.read_text(encoding="utf-8"))}
    assert regions["RoW"]["Activities"][0]["PSILCA_mapping"] == "AU; DE"


def test_rest_labels_do_not_require_a_fixed_country_concordance(brightway, tmp_path):
    _database(brightway, "tech", [("world", "RoW", 1), ("europe", "RoE", 1), ("local", "DE", 1)])
    for activity in brightway.Database("tech"):
        activity["name"] = "Same production activity"
        activity.save()
    mapping = pd.DataFrame(
        [
            {"Shortname": "GLO", "simplified_ISO2": "AU; DE; FR", "PSILCA_mapping": "AU; DE; FR"},
            {"Shortname": "RER", "simplified_ISO2": "DE; FR", "PSILCA_mapping": "DE; FR"},
            {"Shortname": "DE", "simplified_ISO2": "DE", "PSILCA_mapping": "DE"},
            {"Shortname": "RoW", "simplified_ISO2": "", "PSILCA_mapping": ""},
            {"Shortname": "RoE", "simplified_ISO2": "", "PSILCA_mapping": ""},
        ]
    )
    output = tmp_path / "activities.json"
    fetch_ecoinvent_activities(mapping, str(output), "tech", exclude_empty_inventories=False)
    regions = {row["ISO2"]: row for row in json.loads(output.read_text(encoding="utf-8"))}
    assert regions["RoW"]["Activities"][0]["PSILCA_mapping"] == "AU; FR"
    assert regions["RoE"]["Activities"][0]["PSILCA_mapping"] == "FR"
