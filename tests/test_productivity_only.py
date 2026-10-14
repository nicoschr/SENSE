"""Contracts for running SENSE without the licensed PSILCA layer."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import sense.cli as cli
import sense.data.ecoinvent as ecoinvent
import sense.economics as economics
import sense.pipeline as pipeline
import sense.regional as regional
from sense.config import SenseConfig
from sense.core.working_hours import build_productivity_working_hour_records
from sense.productivity.loader import (
    LOOKUP_STATUS_NO_ISIC,
    LOOKUP_STATUS_OVERRIDE_EXACT,
    LookupResult,
    ProductivityBaseline,
)
from sense.run_context import RunPaths
from sense.weighting import CountryWeightProvider


def _productivity_config(tmp_path: Path, **overrides) -> SenseConfig:
    values = {
        "use_psilca": False,
        "productivity_baseline_path": tmp_path / "productivity.parquet",
    }
    values.update(overrides)
    return SenseConfig(**values)


def test_configuration_rejects_incoherent_values(tmp_path):
    assert _productivity_config(tmp_path).enable_social_flows is False
    with pytest.raises(ValueError, match="enable_social_flows requires"):
        SenseConfig(
            use_psilca=False,
            enable_social_flows=True,
            productivity_baseline_path=tmp_path / "productivity.parquet",
        )
    with pytest.raises(ValueError, match="require productivity_baseline_path"):
        SenseConfig(use_psilca=False, enable_social_flows=False)
    with pytest.raises(ValueError, match="proxies requires"):
        SenseConfig(productivity_location_proxies={"XX": ["AAA"]})
    with pytest.raises(ValueError, match="requires enable_working_hours_flows"):
        SenseConfig(
            enable_working_hours_flows=False,
            enable_isic_working_hours_methods=True,
        )
    with pytest.raises(ValueError, match="iso2_selection"):
        SenseConfig(iso2_selection=["DE"], iso2_start_at="FR")
    with pytest.raises(ValueError, match="minimum_similarity"):
        SenseConfig(minimum_similarity=2.0)


@pytest.mark.parametrize("region", ["RoW", "RoE"])
def test_productivity_overrides_and_audits_keep_activity_identity(region):
    def lookup(countries, isic, year):
        assert countries == ["DEU", "FRA"]
        assert year == 2020
        return LookupResult(
            value=2.0 if isic == "20" else None,
            status=LOOKUP_STATUS_OVERRIDE_EXACT if isic == "20" else LOOKUP_STATUS_NO_ISIC,
        )

    overrides, audit = regional._build_productivity_overrides(
        activities=[
            {"code": "priced-activity", "PSILCA_mapping": " DE;FR; ", "ISIC rev.4 ecoinvent": "20"},
            {"code": "missing-activity", "PSILCA_mapping": " DE;FR; "},
        ],
        target_iso2=region,
        config=SenseConfig(),
        baseline=SimpleNamespace(lookup_row_with_status=lookup),
    )
    assert overrides == {"priced-activity": 2.0}
    assert len(audit) == 1
    assert audit[0]["activity_code"] == "missing-activity"


def test_direct_productivity_calculation_is_strict_and_dimensionally_explicit():
    activities = [
        {
            "Activity code": "activity",
            "Activity name": "Example",
            "Reference product": "product",
            "ISIC": "20",
            "Unit": "kg",
            "ISO2": "AA",
            "Price": 4.0,
        }
    ]

    records = build_productivity_working_hour_records(
        activities,
        intensities={"activity": 0.5},
        price_factor=0.25,
    )

    assert records[0]["Converted_price"] == pytest.approx(1.0)
    assert records[0]["Working_hours"] == pytest.approx(0.5)
    assert records[0]["Averaged_Exchanges"] == []
    with pytest.raises(ValueError, match="no productivity intensity"):
        build_productivity_working_hour_records(activities, {}, 0.25)


def test_productivity_location_proxies_use_shared_weights_and_require_all_donors():
    baseline = ProductivityBaseline.__new__(ProductivityBaseline)
    baseline._location_mapping = {}
    baseline._available_iso3 = {"AAA", "BBB"}
    baseline._weight_provider = CountryWeightProvider.equal(reference_year=2020)

    applied = baseline.apply_location_proxies({"XX": ["AAA", "BBB"]})

    assert applied == {"XX": ["AAA", "BBB"]}
    assert baseline._location_mapping["XX"] == [("AAA", 0.5), ("BBB", 0.5)]
    with pytest.raises(ValueError, match="absent from the baseline"):
        baseline.apply_location_proxies({"YY": ["AAA", "CCC"]})


def test_productivity_only_economics_never_resolves_psilca(monkeypatch, tmp_path):
    years: list[int] = []
    provider = CountryWeightProvider.equal(reference_year=2024)
    monkeypatch.setattr(
        economics,
        "resolve_psilca_price_conversion",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("PSILCA conversion")),
    )
    monkeypatch.setattr(
        economics,
        "_country_weight_provider",
        lambda _config, year: years.append(year) or provider,
    )
    monkeypatch.setattr(economics, "_load_productivity", lambda **_: (None, 1.0, None))
    config = SenseConfig(
        use_psilca=False,
        enable_social_flows=False,
        enable_working_hours_flows=False,
        productivity_year=2024,
    )

    context = economics.resolve_economic_context(
        config=config,
        psilca_version=None,
        ecoinvent_geographies_excel=tmp_path / "unused.xlsx",
    )

    assert context.psilca_conversion is None
    assert years == [2024]


@pytest.mark.parametrize("use_psilca", [False, True])
def test_productivity_requires_explicit_monetary_conversion(monkeypatch, tmp_path, use_psilca):
    baseline_path = tmp_path / "productivity.parquet"
    baseline_path.touch()

    class Baseline:
        def apply_location_proxies(self, _proxies):
            return {}

    monkeypatch.setattr(
        economics.pd,
        "read_parquet",
        lambda *_args, **_kwargs: pd.DataFrame({"country": ["AAA"]}),
    )
    monkeypatch.setattr(economics, "build_location_mapping", lambda **_: {})
    monkeypatch.setattr(economics, "ProductivityBaseline", lambda **_: Baseline())

    with pytest.raises(ValueError, match="explicit productivity monetary conversion"):
        economics._load_productivity(
            config=_productivity_config(tmp_path, use_psilca=use_psilca),
            workbook=tmp_path / "geographies.xlsx",
            weight_provider=CountryWeightProvider.equal(reference_year=2020),
        )


def test_productivity_only_paths_do_not_collide_with_psilca_paths(tmp_path):
    integrated = RunPaths.create(
        data_dir=tmp_path,
        technosphere_database="ecoinvent",
        ecoinvent_version="3.12",
        psilca_version="4.0",
    )
    productivity = RunPaths.create(
        data_dir=tmp_path,
        technosphere_database="ecoinvent",
        ecoinvent_version="3.12",
        psilca_version=None,
        use_psilca=False,
    )

    assert integrated.ecoinvent_activity_list != productivity.ecoinvent_activity_list
    assert integrated.psilca_assessments.suffix == ".parquet"
    assert integrated.output_dir == tmp_path / "Output"
    assert productivity.psilca_tag == "disabled"


def test_location_mapping_without_psilca_retains_all_countries(monkeypatch, tmp_path):
    source = pd.DataFrame(
        [
            {"Shortname": "AA", "simplified_ISO2": "AA"},
            {"Shortname": "REG", "simplified_ISO2": "AA; BB"},
        ]
    )
    monkeypatch.setattr(ecoinvent, "read_ecoinvent_geographies", lambda _: source)

    mapped = ecoinvent.map_ecoinvent_locations(str(tmp_path / "unused.xlsx"))

    assert mapped["PSILCA_mapping"].tolist() == ["AA", "AA; BB"]


@pytest.mark.parametrize("price_source_db", [None, "custom-apos-background"])
def test_run_without_psilca_skips_parsing_embeddings_and_version(
    monkeypatch, tmp_path, price_source_db
):
    monkeypatch.setattr("sense.brightway.state.capture_database_state", lambda *a, **kw: {})
    workbook = tmp_path / "geographies.xlsx"
    workbook.touch()

    class Activity(dict):
        key = ("technosphere", "activity")

    activity = Activity(code="activity", location="AA")
    monkeypatch.setattr(pipeline, "_load_technosphere_activities", lambda _: [activity])
    events: list[str] = []

    class Baseline:
        def lookup_with_status(self, location, raw_isic, year):
            events.append("productivity preflight")
            assert (location, raw_isic, year) == ("AA", "20", 2020)
            return LookupResult(value=0.5, status=LOOKUP_STATUS_OVERRIDE_EXACT)

    class Sociosphere(list):
        def write(self, _data):
            events.append("sociosphere creation")

    def forbidden(**_):
        raise AssertionError("PSILCA path invoked")

    monkeypatch.setattr(pipeline, "_prepare_psilca_data", forbidden)
    monkeypatch.setattr(pipeline, "_prepare_embedding_stores", forbidden)
    monkeypatch.setattr(
        pipeline,
        "projects",
        type(
            "Projects",
            (),
            {"__contains__": lambda self, name: True, "set_current": lambda self, name: None},
        )(),
    )
    monkeypatch.setattr(
        pipeline,
        "Database",
        lambda name: [activity] if name == "technosphere" else Sociosphere(),
    )
    monkeypatch.setattr(
        pipeline,
        "databases",
        {
            "technosphere": {"modified": "1"},
            "ecoinvent-3.12-cutoff": {},
            "custom-apos-background": {},
        },
    )
    monkeypatch.setattr(
        pipeline,
        "resolve_economic_context",
        lambda **_: SimpleNamespace(
            psilca_conversion=None,
            country_weights=CountryWeightProvider.equal(reference_year=2020),
            productivity=Baseline(),
            productivity_price_factor=0.25,
        ),
    )
    monkeypatch.setattr(pipeline, "write_economic_assumptions", lambda **_: None)
    monkeypatch.setattr(pipeline, "write_productivity_audit", lambda **_: None)
    monkeypatch.setattr(pipeline, "_refresh_activity_cache_token", lambda *_: None)
    monkeypatch.setattr(pipeline, "_create_impact_methods", lambda *_: None)
    monkeypatch.setattr(pipeline, "_clear_managed_exchanges", lambda **_: 0)
    monkeypatch.setattr(
        pipeline,
        "_prepare_location_mapping",
        lambda **kwargs: (
            pytest.fail("PSILCA CSS supplied")
            if kwargs["css_list_path"] is not None
            else pd.DataFrame(
                [{"Shortname": "AA", "simplified_ISO2": "AA", "PSILCA_mapping": "AA"}]
            )
        ),
    )
    monkeypatch.setattr(pipeline, "_check_location_concordance", lambda **kwargs: kwargs["loc_df"])
    price_sources = []
    monkeypatch.setattr(
        pipeline,
        "_prepare_activity_list",
        lambda **kwargs: price_sources.append(kwargs["price_source_database"])
        or [
            {
                "ISO2": "AA",
                "PSILCA_mapping": "AA",
                "Activities": [
                    {
                        "code": "activity",
                        "name": "Example",
                        "reference product": "product",
                        "unit": "kg",
                        "price": 4.0,
                        "ISIC rev.4 ecoinvent": "20",
                    }
                ],
            }
        ],
    )
    written: list[dict] = []
    monkeypatch.setattr(
        regional,
        "add_working_hours_flows_and_exchanges",
        lambda **kwargs: written.extend(kwargs["data"]) or {"added": 1, "updated": 0, "removed": 0},
    )
    monkeypatch.setattr(regional, "summarize_and_optionally_save", lambda **_: [])

    result = pipeline.run_sense(
        project="project",
        technosphere_db="technosphere",
        price_source_db=price_source_db,
        sociosphere_db="sociosphere",
        ecoinvent_version="3.12",
        config=_productivity_config(tmp_path, iso2_selection=["AA"]),
        data_dir=tmp_path,
        ecoinvent_geographies_excel=workbook,
    )

    assert written[0]["Working_hours"] == pytest.approx(0.5)
    assert result[0]["productivity_override_applied"] == 1
    assert events == ["productivity preflight", "sociosphere creation"]
    assert price_sources == [price_source_db or "technosphere"]


def test_productivity_preflight_rejects_an_unresolved_activity(tmp_path):
    class Baseline:
        def lookup_with_status(self, _location, _raw_isic, _year):
            return LookupResult(value=None, status=LOOKUP_STATUS_NO_ISIC)

    with pytest.raises(ValueError, match="complete productivity coverage"):
        regional._preflight_productivity_coverage(
            regions=["AA"],
            iso2_data={
                "AA": {
                    "Activities": [
                        {
                            "code": "activity",
                            "name": "Example",
                            "ISIC rev.4 ecoinvent": "",
                        }
                    ]
                }
            },
            config=_productivity_config(tmp_path),
            baseline=Baseline(),
        )


def test_cli_productivity_mode_passes_no_psilca_inputs(monkeypatch, tmp_path):
    received = {}

    def execute(**kwargs):
        received.update(kwargs)
        return [{"ISO2": "AA", "issues": []}]

    monkeypatch.setattr(cli, "run_sense", execute)
    result = cli.main(
        [
            "run",
            "--project",
            "project",
            "--technosphere-db",
            "technosphere",
            "--no-psilca",
            "--productivity-baseline",
            str(tmp_path / "productivity.parquet"),
            "--data-dir",
            str(tmp_path / "artifacts"),
        ]
    )

    assert result == 0
    assert received["psilca_dir"] is None
    assert received["psilca_version"] is None
    assert received["data_dir"] == tmp_path / "artifacts"
    assert received["config"].use_psilca is False


@pytest.mark.parametrize("kind", ["infinite", "duplicate", "fractional_year", "class_code"])
def test_invalid_productivity_is_rejected_before_lookup(tmp_path, kind):
    row = {"country": "AAA", "year": 2020, "isic4_code": "10", "productivity_hrs_per_usd": 2.0}
    if kind == "infinite":
        row["productivity_hrs_per_usd"] = float("inf")
    elif kind == "fractional_year":
        row["year"] = 2020.5
    elif kind == "class_code":
        row["isic4_code"] = "1000"
    data = pd.DataFrame([row, row] if kind == "duplicate" else [row])
    path = tmp_path / "baseline.parquet"
    data.to_parquet(path)
    with pytest.raises(ValueError):
        ProductivityBaseline(
            path, {}, weight_provider=CountryWeightProvider.equal(reference_year=2020)
        )


def test_productivity_numeric_text_is_normalized_before_lookup(tmp_path):
    path = tmp_path / "baseline.parquet"
    pd.DataFrame(
        {
            "country": ["AAA"],
            "year": ["2020"],
            "isic4_code": ["10"],
            "productivity_hrs_per_usd": ["2.5"],
        }
    ).to_parquet(path)
    baseline = ProductivityBaseline(
        path, {}, weight_provider=CountryWeightProvider.equal(reference_year=2020)
    )
    assert baseline._lookup_single_with_year("AAA", "10", 2020) == (2.5, 2020)
    assert baseline._available_years == {2020}
