"""Treatment estimates remain distinct from prices and monetary productivity."""

import json
import shutil
import uuid
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from sense import SenseConfig, run_sense
from sense.brightway.repository import empty_inventory_keys
from sense.core.averaging import average_sector_mappings_w_prices
from sense.core.treatment import fallback_overrides, finalize_profiles, reference_distribution
from sense.eligibility import exclusion_reason, is_treatment
from sense.labour_reference import active_regions, load_or_build_reference
from sense.run_context import latest_run_directory


def eligible(activity):
    config = SenseConfig()
    return (
        exclusion_reason(
            activity,
            prefixes=config.exclude_terms_starting_with,
            terms=config.exclude_terms,
            exact=config.exclude_exact,
        )
        is None
    )


@pytest.mark.parametrize(
    "name",
    [
        "market for waste",
        "market group for waste",
        "generic market for copper",
        "production market for electricity",
        "electricity production mix",
        "electricity consumption mix",
        "electricity technology mix",
        "natural gas, import from RU",
        "electricity, high voltage, import from FR",
    ],
)
def test_bookkeeping_exclusions_include_regioinvent(name):
    assert not eligible({"name": name, "reference product": name.split(", import from ")[0]})


def test_import_operations_and_real_collection_remain_eligible():
    assert eligible(
        {"name": "evaporation of natural gas, import from RU", "reference product": "natural gas"}
    )
    assert eligible({"name": "treatment of concrete, collection for final disposal"})
    assert not eligible(
        {
            "name": "waste routing",
            "comment": "This is a renaming activity. Nothing is treated here.",
        }
    )
    assert is_treatment({}, -1)
    assert not is_treatment({"name": "waste product"}, 1)
    assert is_treatment({"sense_activity_role": "treatment"}, 1)


def donor(code, hours, unit="kg", **extra):
    return {
        "Activity code": code,
        "Unit": unit,
        "Price": 1.0,
        "Working_hours": hours,
        "Working_hours_source": "psilca",
        **extra,
    }


def test_percentile_is_linear_per_activity_and_order_independent():
    records = [
        donor("a", 4),
        donor("b", 20, "kilogram"),
        donor("ignored", 10000, Working_hours_source="treatment_fallback"),
        donor("zero", 0, Price=0),
        donor("electricity", 100, "kWh"),
    ]
    expected = {
        "kilogram": {"hours": 7.2, "donor_count": 2},
        "kilowatt hour": {"hours": 100, "donor_count": 1},
    }
    assert reference_distribution(records, 20) == expected
    assert reference_distribution(reversed(records), 20) == expected
    with pytest.raises(ValueError, match="Duplicate"):
        reference_distribution([records[0], records[0]], 20)
    with pytest.raises(ValueError, match="Missing reference"):
        reference_distribution([donor("missing", None)], 20)


def test_zero_stays_in_input_snapshot_but_is_not_matched():
    entries = [
        {
            "ISO2": "DE",
            "Activities": [
                {"code": "zero", "price": 0, "calculation route": "zero"},
                {"code": "treatment", "price": 0, "calculation route": "treatment_fallback"},
            ],
        }
    ]
    assert [a["code"] for a in active_regions(entries)["DE"]["Activities"]] == ["treatment"]
    assert len(entries[0]["Activities"]) == 2


def test_zero_is_a_disable_signal_not_a_reference_quantile():
    with pytest.raises(ValueError, match="greater than 0"):
        reference_distribution([donor("a", 4)], 0)


@pytest.mark.parametrize("mean,expected_factor", [("arithmetic", 5.5), ("geometric", np.sqrt(10))])
def test_physical_hours_bypass_currency_and_labour_intensities(mean, expected_factor):
    rows = [
        {
            "Activity code": "treatment",
            "Activity name": "Treatment",
            "Reference product": "waste",
            "Unit": "kg",
            "ISO2": "DE",
            "Price": None,
            "PSILCA_country": "DE",
            "Matched_sector": sector,
            "PSILCA_Exchanges": [
                {
                    "indicator": "Accidents",
                    "risk level": risk,
                    "type": "risk",
                    "unit": "h",
                    "category": "Workers",
                    "subcategory": "Safety",
                    "meanValue": intensity,
                }
            ],
        }
        for sector, risk, intensity in [("first", "medium", 200), ("second", "high", 5000)]
    ]
    records = average_sector_mappings_w_prices(
        pd.DataFrame(rows),
        level_weights={"medium": 1, "high": 10, "no data": 0.1},
        mean_method=mean,
        direct_hours_override={"treatment": 7.2},
        psilca_price_factor=999,
        productivity_price_factor=888,
    )
    assert records[0]["Working_hours"] == 7.2
    assert records[0]["Averaged_Exchanges"][0]["meanValue"] == pytest.approx(7.2 * expected_factor)
    assert records[0]["Working_hours_source"] == "treatment_fallback"
    with pytest.raises(ValueError, match="cannot overlap"):
        average_sector_mappings_w_prices(
            pd.DataFrame(rows),
            direct_hours_override={"treatment": 1},
            hours_override={"treatment": 2},
            level_weights={"medium": 1},
        )


def test_missing_unit_fails_and_profiles_cannot_be_finalized_twice():
    activities = [{"code": "treatment", "unit": "kg", "calculation route": "treatment_fallback"}]
    with pytest.raises(ValueError, match="No treatment working-hour reference"):
        fallback_overrides(activities, {})
    profile = donor(
        "treatment",
        1,
        Working_hours_source="treatment_fallback",
        Averaged_Exchanges=[{"meanValue": 10}],
    )
    finalize_profiles([profile], {"kilogram": {"hours": 1, "donor_count": 2}}, None)
    with pytest.raises(ValueError, match="already finalized"):
        finalize_profiles([profile], {"kilogram": {"hours": 1, "donor_count": 2}}, None)


def test_reference_cache_reuses_valid_result_and_rejects_corruption(tmp_path):
    calls = []

    def prepare(region):
        calls.append(region)
        return {"records": [donor(region, 4 if region == "DE" else 20)], "summary": {}}

    arguments = {
        "cache_directory": tmp_path,
        "identity": {"source": "background"},
        "regions": ["US", "DE"],
        "prepare_region": prepare,
        "percentile": 20,
    }
    reference, path, _ = load_or_build_reference(**arguments)
    assert calls == ["DE", "US"]
    assert reference["kilogram"]["hours"] == 7.2
    load_or_build_reference(**arguments)
    assert len(calls) == 2
    corrupt = json.loads(path.read_text())
    corrupt["reference"]["kilogram"]["hours"] = 1000
    path.write_text(json.dumps(corrupt))
    load_or_build_reference(**arguments)
    assert len(calls) == 4


@pytest.fixture
def brightway():
    import bw2data as bd

    root = bd.projects._use_temp_directory()
    project = f"sense-treatment-{uuid.uuid4().hex}"
    bd.projects.set_current(project)
    try:
        yield bd
    finally:
        bd.projects.set_current("default")
        bd.projects.delete_project(project, delete_dir=False)
        bd.projects._restore_orig_directory()
        shutil.rmtree(root, ignore_errors=True)


def inputs(tmp_path):
    geography = tmp_path / "geographies.xlsx"
    pd.DataFrame(
        [
            {
                "Shortname": code,
                "Name": name,
                "Geography Classification": "Country",
                "Contained and Overlapping Geographies": code,
            }
            for code, name in [("DE", "Germany"), ("US", "United States")]
        ]
    ).to_excel(geography, sheet_name="Geographies", index=False)
    productivity = tmp_path / "productivity.parquet"
    pd.DataFrame(
        [
            {"country": code, "year": 2020, "isic4_code": "20", "productivity_hrs_per_usd": 2.0}
            for code in ["DEU", "USA"]
        ]
    ).to_parquet(productivity)
    return geography, productivity


def write_database(bd, name, rows):
    bd.Database(name).write(
        {
            (name, code): {
                "name": title,
                "reference product": product,
                "unit": "kg",
                "location": country,
                "classifications": [("ISIC rev.4 ecoinvent", "20")],
                "exchanges": [
                    {
                        "input": (name, code),
                        "amount": reference,
                        "type": "production",
                        "properties": {"price": {"amount": price, "unit": "EUR2005"}},
                    }
                ],
            }
            for code, title, product, country, reference, price in rows
        }
    )


@pytest.mark.parametrize("invalid", ["false", 0, None])
def test_empty_inventory_setting_requires_a_boolean(invalid):
    assert SenseConfig().exclude_empty_inventories is True
    with pytest.raises(ValueError, match="exclude_empty_inventories"):
        SenseConfig(exclude_empty_inventories=invalid)


def test_empty_inventory_detection_is_independent_of_database_origin(brightway):
    bd = brightway
    name = "ecoinvent-3.12-cutoff"
    write_database(
        bd,
        name,
        [
            ("empty", "Recovered heat", "heat", "DE", 1, 10),
            ("treatment", "Clinker production", "waste", "DE", -1, 0),
            ("operation", "Physical operation", "product", "DE", 1, 2),
            ("zero-input", "Zero exchange", "product", "DE", 1, 3),
        ],
    )
    bd.Database("environment").write(
        {("environment", "emission"): {"name": "Emission", "unit": "kg", "type": "emission"}}
    )
    bd.Database("social").write(
        {
            ("social", "managed"): {
                "name": "Working hours",
                "unit": "h",
                "type": "emission",
                "sense_managed": True,
            }
        }
    )
    empty = bd.get_activity((name, "empty"))
    empty.new_exchange(input=("social", "managed"), amount=999, type="biosphere").save()
    operation = bd.get_activity((name, "operation"))
    operation.new_exchange(input=("environment", "emission"), amount=1, type="biosphere").save()
    zero_input = bd.get_activity((name, "zero-input"))
    zero_input.new_exchange(input=operation.key, amount=0, type="technosphere").save()
    assert empty_inventory_keys(list(bd.Database(name))) == {
        (name, "empty"),
        (name, "treatment"),
        (name, "zero-input"),
    }
    write_database(bd, "custom", [("custom", "Labour-only foreground", "service", "DE", 1, 5)])
    assert empty_inventory_keys(list(bd.Database("custom"))) == {("custom", "custom")}
    write_database(bd, "Regioinvent", [("derived", "Recovered heat", "heat", "US", 1, 10)])
    derived = bd.get_activity(("Regioinvent", "derived"))
    assert empty_inventory_keys([derived]) == {derived.key}
    # Technosphere-only inventories remain eligible even without emissions.
    zero_input.new_exchange(input=operation.key, amount=1, type="technosphere").save()
    assert zero_input.key not in empty_inventory_keys(list(bd.Database(name)))
    # Non-SENSE environmental information must survive even in a mixed namespace.
    unowned = bd.Database("social").new_activity("unowned")
    unowned.update(name="Unowned emission", unit="kg", type="emission")
    unowned.save()
    empty.new_exchange(input=unowned.key, amount=1, type="biosphere").save()
    assert (name, "empty") not in empty_inventory_keys(list(bd.Database(name)))


def test_empty_exclusion_rebuilds_reference_and_removes_stale_hours(brightway, tmp_path):
    bd = brightway
    geo, productivity = inputs(tmp_path)
    name = "custom-foreground"
    write_database(
        bd,
        name,
        [
            ("real", "Factory", "product", "DE", 1, 2),
            ("empty", "Recovered coproduct", "coproduct", "DE", 1, 100),
            ("t", "Waste treatment", "waste", "DE", -1, 0),
        ],
    )
    bd.Database("environment").write(
        {("environment", "flow"): {"name": "Emission", "unit": "kg", "type": "emission"}}
    )
    for code in ("real", "t"):
        bd.get_activity((name, code)).new_exchange(
            input=("environment", "flow"), amount=1, type="biosphere"
        ).save()

    def run(exclude_empty):
        return run_sense(
            project=bd.projects.current,
            technosphere_db=name,
            sociosphere_db="social",
            price_source_db=name,
            ecoinvent_version="3.12",
            data_dir=tmp_path,
            ecoinvent_geographies_excel=geo,
            config=SenseConfig(
                use_psilca=False,
                productivity_baseline_path=productivity,
                productivity_eur_per_usd=1,
                country_weighting="equal",
                exclude_empty_inventories=exclude_empty,
            ),
        )

    run(False)
    treatment = bd.get_activity((name, "t"))
    before = [exc["amount"] for exc in treatment.biosphere() if exc.input.get("sense_managed")]
    assert before == pytest.approx([43.2])
    run(True)
    assert not [
        exc
        for exc in bd.get_activity((name, "empty")).biosphere()
        if exc.input.get("sense_managed")
    ]
    after = [exc["amount"] for exc in treatment.biosphere() if exc.input.get("sense_managed")]
    assert after == pytest.approx([4.0])
    run(True)
    assert [
        exc["amount"] for exc in treatment.biosphere() if exc.input.get("sense_managed")
    ] == pytest.approx(after)
    run(False)
    assert [
        exc["amount"] for exc in treatment.biosphere() if exc.input.get("sense_managed")
    ] == pytest.approx(before)
    assert [
        exc["amount"]
        for exc in bd.get_activity((name, "empty")).biosphere()
        if exc.input.get("sense_managed")
    ] == pytest.approx([200.0])


def test_full_pipeline_reference_scope_zero_cleanup_and_regionalized_reuse(brightway, tmp_path):
    bd = brightway
    geo, productivity = inputs(tmp_path)
    rows = [
        ("a", "Factory A", "product A", "DE", 1, 2),
        ("b", "Factory B", "product B", "US", 1, 10),
        ("t", "Treatment", "waste", "DE", -2, 0),
        ("zero", "Land transformation", "land", "DE", 1, 0),
        ("market", "production market for product", "product", "DE", 1, 5),
    ]
    write_database(bd, "base", rows)
    zero = bd.get_activity(("base", "zero"))
    zero.new_exchange(input=("base", "a"), amount=3, type="technosphere").save()
    physical = {
        act.key: [(exc["type"], exc["amount"]) for exc in act.exchanges()]
        for act in bd.Database("base")
    }

    def run(database="base", **kwargs):
        return run_sense(
            project=bd.projects.current,
            technosphere_db=database,
            sociosphere_db="social",
            price_source_db="base",
            ecoinvent_version="3.12",
            data_dir=tmp_path,
            ecoinvent_geographies_excel=geo,
            config=SenseConfig(
                exclude_empty_inventories=False,
                use_psilca=False,
                productivity_baseline_path=productivity,
                productivity_eur_per_usd=1,
                country_weighting="equal",
                save_intermediate=True,
                **kwargs,
            ),
        )

    run(iso2_selection=["DE"])
    treatment = bd.get_activity(("base", "t"))
    assert [exc["amount"] for exc in treatment.biosphere()] == pytest.approx([14.4])
    assert not list(bd.get_activity(("base", "b")).biosphere())
    for code in ("zero", "market"):
        assert not list(bd.get_activity(("base", code)).biosphere())
    LCA = pytest.importorskip("bw2calc").LCA
    method = ("SENSE", "social", "Social", "Working hours", "working hours")
    calculation = LCA({zero.key: 1}, method)
    calculation.lci()
    calculation.lcia()
    assert calculation.score == pytest.approx(12)  # Upstream hours survive a zero direct price.
    calculation = LCA({treatment.key: -1}, method)
    calculation.lci()
    calculation.lcia()
    assert calculation.score == pytest.approx(7.2)
    run()
    assert [exc["amount"] for exc in treatment.biosphere()] == pytest.approx([14.4])
    for act in bd.Database("base"):
        assert [
            (exc["type"], exc["amount"]) for exc in act.exchanges() if exc["type"] != "biosphere"
        ] == physical[act.key]
    # A much larger transformed donor population must not alter the reference.
    regionalized = deepcopy(rows) + [("extra", "Additional factory", "extra", "US", 1, 1000)]
    write_database(bd, "regio", regionalized)
    run("regio")
    assert [exc["amount"] for exc in bd.get_activity(("regio", "t")).biosphere()] == pytest.approx(
        [14.4]
    )
    folder = latest_run_directory(tmp_path / "Output", bd.projects.current, "regio")
    manifest = json.loads((folder / "inputs.json").read_text())
    assert "treatment_working_hours_reference" in manifest["inputs"]
    assert "pricing_audit" in manifest["inputs"]
    # Previously mapped activities that acquire explicit zero prices lose only
    # their direct managed inventory on rerun.
    activity = bd.get_activity(("base", "a"))
    activity["price"] = 0
    activity.save()
    run(iso2_selection=["DE"])
    assert not list(activity.biosphere())
    assert [exc["amount"] for exc in treatment.biosphere()] == pytest.approx([40])


@pytest.mark.parametrize("percentile", [0, 20])
def test_unknown_treatment_unit_fails_before_inventory_mutation(brightway, tmp_path, percentile):
    bd = brightway
    geo, productivity = inputs(tmp_path)
    write_database(
        bd,
        "base",
        [
            ("factory", "Factory", "product", "DE", 1, 2),
            ("treatment", "Treatment", "waste", "DE", -1, 0),
        ],
    )
    treatment = bd.get_activity(("base", "treatment"))
    treatment["unit"] = "meter"
    treatment.save()

    def run():
        run_sense(
            project=bd.projects.current,
            technosphere_db="base",
            sociosphere_db="social",
            ecoinvent_version="3.12",
            data_dir=tmp_path,
            ecoinvent_geographies_excel=geo,
            config=SenseConfig(
                exclude_empty_inventories=False,
                use_psilca=False,
                productivity_baseline_path=productivity,
                productivity_eur_per_usd=1,
                country_weighting="equal",
                treatment_working_hours_percentile=percentile,
            ),
        )

    if percentile:
        with pytest.raises(
            ValueError, match="No treatment working-hour reference for unit 'meter'"
        ):
            run()
        assert "social" not in bd.databases
        assert not list(bd.get_activity(("base", "factory")).biosphere())
    else:
        run()
        assert list(bd.get_activity(("base", "factory")).biosphere())
    assert not list(treatment.biosphere())


@pytest.mark.parametrize("percentile", [0, 20])
def test_missing_treatment_price_does_not_enter_price_inference(
    brightway, tmp_path, monkeypatch, percentile
):
    from sense.data import ecoinvent

    bd = brightway
    write_database(
        bd,
        "tech",
        [
            ("factory", "Factory", "product", "DE", 1, 2),
            ("treatment", "Treatment", "waste", "DE", -1, 0),
            ("zero", "Land transformation", "land", "DE", 1, 0),
            ("priced_treatment", "Priced treatment", "other waste", "DE", -1, 3),
        ],
    )
    treatment = bd.get_activity(("tech", "treatment"))
    for exchange in treatment.production():
        exchange["properties"] = {}
        exchange.save()

    def no_inference(**kwargs):
        raise AssertionError(
            "Zero prices and treatment fallbacks must not enter input-cost inference"
        )

    monkeypatch.setattr(ecoinvent, "build_product_graph", no_inference)
    result = ecoinvent.fetch_ecoinvent_activities(
        pd.DataFrame([{"Shortname": "DE", "PSILCA_mapping": "DE"}]),
        str(tmp_path / "priced.json"),
        "tech",
        exclude_empty_inventories=False,
        treatment_working_hours_percentile=percentile,
    )
    routes = result.set_index("code")["calculation route"].to_dict()
    assert routes == {
        "factory": "monetary",
        "treatment": "treatment_fallback" if percentile else "zero",
        "zero": "zero",
        "priced_treatment": "monetary",
    }
    snapshot = json.loads((tmp_path / "priced.json").read_text())
    unpriced = next(a for a in snapshot[0]["Activities"] if a["code"] == "treatment")
    assert unpriced["price"] is None
    assert unpriced["reference amount"] == -1
    assert result.set_index("code").loc["priced_treatment", "price"] == 3
    audit = json.loads((tmp_path / "priced.pricing.json").read_text())
    missing = next(row for row in audit["activities"] if row["code"] == "treatment")
    assert missing["treatment_fallback_disabled"] is (percentile == 0)


@pytest.mark.parametrize(
    "policy,expected", [("treatment_working_hours", 0), ("legacy_price_floor", 2)]
)
def test_price_floor_requires_explicit_legacy_policy(brightway, tmp_path, policy, expected):
    from sense.data.ecoinvent import fetch_ecoinvent_activities

    bd = brightway
    write_database(
        bd,
        "tech",
        [
            ("factory", "Factory", "product", "DE", 1, 2),
            ("treatment", "Treatment", "waste", "DE", -1, 0),
        ],
    )
    result = fetch_ecoinvent_activities(
        pd.DataFrame([{"Shortname": "DE", "PSILCA_mapping": "DE"}]),
        str(tmp_path / "priced.json"),
        "tech",
        exclude_empty_inventories=False,
        zero_price_policy=policy,
    )
    assert result.set_index("code").loc["treatment", "price"] == expected


@pytest.mark.parametrize("price", [-1, float("nan"), "invalid"])
def test_invalid_activity_prices_stop_before_fallback(brightway, tmp_path, price):
    from sense.data.ecoinvent import fetch_ecoinvent_activities

    bd = brightway
    write_database(bd, "tech", [("treatment", "Treatment", "waste", "DE", -1, 0)])
    activity = bd.get_activity(("tech", "treatment"))
    activity["price"] = price
    activity.save()
    with pytest.raises(ValueError, match="activity price"):
        fetch_ecoinvent_activities(
            pd.DataFrame([{"Shortname": "DE", "PSILCA_mapping": "DE"}]),
            str(tmp_path / "priced.json"),
            "tech",
            exclude_empty_inventories=False,
        )


@pytest.mark.parametrize("exclude_empty", [False, True])
def test_psilca_pipeline_prepares_once_and_reuses_reference_for_risk_changes(
    brightway, tmp_path, monkeypatch, exclude_empty
):
    import sense.pipeline as pipeline
    import sense.preparation as preparation
    import sense.regional as regional
    from sense.data.embeddings import save_embeddings

    bd = brightway
    geo, _ = inputs(tmp_path)
    source = tmp_path / "synthetic_psilca"
    source.mkdir()
    for country, hours in [("DE", 2), ("US", 6)]:
        (source / f"{country}.xml").write_text(
            f"""<ecoSpold><dataset>
<metaInformation><processInformation><referenceFunction name="Manufacturing"/>
<geography location="{country}"/></processInformation></metaInformation>
<flowData><exchange name="Accidents; high risk" unit="h" meanValue="{hours}"
category="Workers" subCategory="Safety"><outputGroup>4</outputGroup></exchange>
<exchange name="Training; medium opportunity" unit="h" meanValue="{hours}"
category="Workers" subCategory="Safety"><outputGroup>4</outputGroup></exchange>
</flowData></dataset></ecoSpold>""",
            encoding="utf-8",
        )
    write_database(
        bd,
        "base",
        [
            ("a", "Factory A", "product A", "DE", 1, 2),
            ("b", "Factory B", "product B", "US", 1, 10),
            ("t", "Treatment", "waste", "DE", -2, 0),
        ],
    )

    if exclude_empty:
        # A fresh default run excludes a priced bookkeeping activity while retaining
        # actual production and treatment with environmental exchanges.
        bd.Database("environment").write(
            {("environment", "flow"): {"name": "Invented emission", "type": "emission"}}
        )
        for activity in bd.Database("base"):
            activity.new_exchange(input=("environment", "flow"), amount=1, type="biosphere").save()
        empty = bd.Database("base").new_activity("empty")
        empty.update(
            name="Invented bookkeeping", location="DE", unit="kg", **{"reference product": "empty"}
        )
        empty.save()
        empty.new_exchange(
            input=empty.key,
            amount=1,
            type="production",
            properties={"price": {"amount": 999, "unit": "EUR2005"}},
        ).save()

    def synthetic_embeddings(**kwargs):
        # Use real matching with deterministic vectors, avoiding model downloads.
        for country in ("DE", "US"):
            save_embeddings(
                str(kwargs["psilca_emb_dir"]),
                f"psilca_{country}",
                np.array([[1, 0]], dtype="float32"),
                pd.Series([f"{country}::Manufacturing"], name="id"),
                {},
            )
        for country, entry in kwargs["iso2_data"].items():
            codes = [activity["code"] for activity in entry["Activities"]]
            if codes:
                save_embeddings(
                    str(kwargs["ecoinvent_emb_dir"]),
                    f"ecoinvent_{country}",
                    np.tile(np.array([[1, 0]], dtype="float32"), (len(codes), 1)),
                    pd.Series(codes, name="id"),
                    {},
                )

    monkeypatch.setattr(pipeline, "_prepare_embedding_stores", synthetic_embeddings)
    monkeypatch.setattr(preparation, "_prepare_embedding_stores", synthetic_embeddings)
    original_match = regional.match_from_stores
    calls = []

    def counted_matching(**kwargs):
        calls.append(kwargs["target_iso2"])
        return original_match(**kwargs)

    monkeypatch.setattr(regional, "match_from_stores", counted_matching)

    def run(factor, percentile=20):
        weights = SenseConfig().level_weights
        weights["high"] = factor
        run_sense(
            project=bd.projects.current,
            technosphere_db="base",
            sociosphere_db="social",
            ecoinvent_version="3.12",
            psilca_version="4.0",
            psilca_dir=source,
            data_dir=tmp_path,
            ecoinvent_geographies_excel=geo,
            config=SenseConfig(
                exclude_empty_inventories=exclude_empty,
                iso2_selection=["DE"],
                country_weighting="equal",
                ecoinvent_to_psilca_price_factor=3,
                save_intermediate=True,
                level_weights=weights,
                treatment_working_hours_percentile=percentile,
            ),
        )

    run(10)
    assert calls == ["DE", "US"]  # No repeated matching to write baseline profiles.
    if exclude_empty:
        assert not list(bd.get_activity(("base", "empty")).biosphere())
    treatment = bd.get_activity(("base", "t"))
    hours = next(
        exc for exc in treatment.biosphere() if exc.input.get("sense_kind") == "working_hours"
    )
    assert hours["amount"] == pytest.approx(2 * 45.6)
    risk = next(exc for exc in treatment.biosphere() if "(risk)" in exc.input["name"])
    assert risk["amount"] == pytest.approx(2 * 45.6 * 10)
    opportunity = next(exc for exc in treatment.biosphere() if "(opportunity)" in exc.input["name"])
    assert opportunity["amount"] == pytest.approx(2 * 45.6)
    folder = latest_run_directory(tmp_path / "Output", bd.projects.current, "base")
    assert not (folder / "PreparedReference").exists()
    averaged = list((folder / "Sector mappings/Averaged/3.12").glob("*.json"))
    assert [path.name for path in averaged] == ["DE_averaged.json"]
    profile = next(
        record for record in json.loads(averaged[0].read_text()) if record["Activity code"] == "t"
    )
    assert profile["Working_hours"] == pytest.approx(45.6)
    assert profile["Fallback_donor_count"] == 2

    run(100)
    assert calls == ["DE", "US", "DE"]  # Reference reused when only risk factors change.
    assert len(list((tmp_path / "Output/WorkingHoursReferences").glob("*.json"))) == 1
    risk = next(exc for exc in treatment.biosphere() if "(risk)" in exc.input["name"])
    assert risk["amount"] == pytest.approx(2 * 45.6 * 100)

    # Changing only the percentile must invalidate cached routing, clean every
    # managed treatment indicator, and skip the now-unneeded donor reference.
    ordinary = bd.get_activity(("base", "a"))
    ordinary_before = [(exc.input.key, exc["amount"]) for exc in ordinary.biosphere()]
    physical_before = [
        (exc.input.key, exc["type"], exc["amount"])
        for exc in treatment.exchanges()
        if exc["type"] != "biosphere"
    ]

    def no_reference(**kwargs):
        raise AssertionError("A zero percentile must not build a treatment reference")

    with monkeypatch.context() as disabled:
        disabled.setattr(pipeline, "prepare_labour_reference", no_reference)
        run(100, 0)
    assert not [
        exchange for exchange in treatment.biosphere() if exchange.input.get("sense_managed")
    ]
    assert [(exc.input.key, exc["amount"]) for exc in ordinary.biosphere()] == ordinary_before
    assert [
        (exc.input.key, exc["type"], exc["amount"])
        for exc in treatment.exchanges()
        if exc["type"] != "biosphere"
    ] == physical_before
    folder = latest_run_directory(tmp_path / "Output", bd.projects.current, "base")
    assert (
        "treatment_working_hours_reference"
        not in json.loads((folder / "inputs.json").read_text())["inputs"]
    )
    # The same cached study can re-enable a different percentile and restore
    # the original default without manual cache deletion.
    run(100, 50)
    hours = next(
        exc for exc in treatment.biosphere() if exc.input.get("sense_kind") == "working_hours"
    )
    assert hours["amount"] == pytest.approx(2 * 96)
    run(100, 20)
    hours = next(
        exc for exc in treatment.biosphere() if exc.input.get("sense_kind") == "working_hours"
    )
    assert hours["amount"] == pytest.approx(2 * 45.6)


def test_paired_missing_treatment_prices_remain_missing(brightway, tmp_path, monkeypatch):
    from case_studies.hydrogen.workflow import paired_prices

    bd = brightway
    write_database(bd, "source", [("factory", "Factory", "product", "DE", 1, 2)])
    for database in ("base", "regio"):
        write_database(bd, database, [("t", "Treatment", "waste", "DE", -1, 0)])
        for exchange in bd.get_activity((database, "t")).production():
            exchange["properties"] = {}
            exchange.save()
    spec = paired_prices.study_protocol.load_protocol()
    spec["countries"] = ["DE"]
    spec["sense"]["exclude_empty_inventories"] = False
    monkeypatch.setattr(
        paired_prices.study_protocol,
        "foregrounds",
        lambda scenario: {"hydrogen": "base" if scenario == "baseline" else "regio"},
    )
    monkeypatch.setattr(paired_prices.case_config, "path", lambda *args: tmp_path)
    monkeypatch.setattr(
        paired_prices.case_config, "ecoinvent_database_name", lambda *args: "source"
    )
    audit = paired_prices.synchronize_prices({"brightway": {"project": bd.projects.current}}, spec)
    assert len(audit) == 1 and audit[0]["price"] is None
    assert all(
        bd.get_activity((database, "t")).get("price") is None for database in ("base", "regio")
    )
    seed = tmp_path / "Output/SeedPrices/seed_prices_regio.json"
    seed.write_text(
        json.dumps(
            {
                "metadata": {"expected_currency": "EUR", "expected_price_year": 2005},
                "seeds": [{"unit": "kg", "reference_product": "waste", "price": 9}],
            }
        )
    )
    with pytest.raises(ValueError, match="conflicts with paired price"):
        paired_prices.synchronize_prices({"brightway": {"project": bd.projects.current}}, spec)


def test_empty_eligibility_cache_tracks_physical_and_flow_changes(brightway, tmp_path, monkeypatch):
    from sense import preparation
    from sense.brightway import repository

    bd = brightway
    name = "custom-cache"
    write_database(bd, name, [("a", "Labour service", "service", "DE", 1, 5)])
    bd.Database("social-cache").write(
        {
            ("social-cache", "flow"): {
                "name": "Owned hours",
                "unit": "h",
                "type": "emission",
                "sense_managed": True,
            }
        }
    )
    act = bd.get_activity((name, "a"))
    exchange = act.new_exchange(input=("social-cache", "flow"), amount=1, type="biosphere")
    exchange.save()
    original = repository.empty_inventory_keys
    scans = []

    def counted(*args, **kwargs):
        scans.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(repository, "empty_inventory_keys", counted)
    path = tmp_path / "activities.json"

    def prepare():
        return preparation._prepare_activity_list(
            name,
            name,
            pd.DataFrame([{"Shortname": "DE", "PSILCA_mapping": "DE"}]),
            SenseConfig(),
            path,
            tmp_path / "seeds",
            name,
        )

    assert prepare() == []
    assert prepare() == []
    assert len(scans) == 1  # An unchanged call does not inspect exchanges.
    flow = bd.get_activity(("social-cache", "flow"))
    flow["name"] = "Renamed owned hours"
    flow.save()
    with monkeypatch.context() as context:

        def unexpected_repricing(**kwargs):
            raise AssertionError("Unchanged eligibility must not rebuild prices")

        context.setattr(preparation, "fetch_ecoinvent_activities", unexpected_repricing)
        assert prepare() == []
    assert len(scans) == 2
    flow["sense_managed"] = False
    flow.save()
    assert any(entry["Activities"] for entry in prepare())
    assert len(scans) == 3  # Ownership changes invalidate even without target changes.
    prepare()
    assert len(scans) == 3
    exchange["amount"] = 0
    exchange.save()
    assert prepare() == []
    assert len(scans) == 4  # Biosphere-only amount changes invalidate correctly.
    metadata = dict(bd.databases[name])
    metadata.pop("modified", None)
    bd.databases[name] = metadata
    prepare()
    prepare()
    assert len(scans) == 6  # Untracked backends never reuse eligibility blindly.


def test_empty_inventory_scan_decodes_only_unresolved_activity_amounts(brightway, monkeypatch):
    import pickle

    bd = brightway
    name = "many-inputs"
    write_database(bd, name, [("a", "Factory", "product", "DE", 1, 5)])
    act = bd.get_activity((name, "a"))
    for _ in range(25):
        act.new_exchange(input=act.key, amount=1, type="technosphere").save()
    original = pickle.loads
    decoded = []

    def counted(payload, *args, **kwargs):
        decoded.append(True)
        return original(payload, *args, **kwargs)

    monkeypatch.setattr(pickle, "loads", counted)
    assert not empty_inventory_keys([act])
    assert len(decoded) == 1  # Repeated inputs do not require repeated decoding.
