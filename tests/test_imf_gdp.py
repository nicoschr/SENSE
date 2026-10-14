"""Source priority, units, reproducibility and shared GDP weighting regressions."""

import hashlib
import importlib.util
import json
from importlib import resources
from pathlib import Path

import pandas as pd
import pytest

from sense.config import SenseConfig
from sense.economics import _country_weight_provider
from sense.productivity.loader import ProductivityBaseline
from sense.weighting import (
    BUNDLED_GDP_FILE,
    BUNDLED_IMF_GDP_FILE,
    BUNDLED_IMF_METADATA_FILE,
    CountryWeightProvider,
    attach_country_weights,
)


def test_bundled_fallback_values_years_and_sources():
    provider = CountryWeightProvider.bundled_gdp(reference_year=2023)
    codes = ["DEU", "TWN", "SSD", "ERI", "PRK"]
    result = provider.resolve_iso3(codes)
    assert result.missing_countries == ("PRK",)
    assert result.data_years == {"DEU": 2023, "TWN": 2023, "SSD": 2023, "ERI": 2019}
    assert result.data_sources == {
        "DEU": "world_bank",
        "TWN": "imf_weo",
        "SSD": "imf_weo",
        "ERI": "imf_weo",
    }
    assert provider._observation("TWN") == (757_387_000_000.0, 2023)
    assert provider._observation("SSD") == (7_401_000_000.0, 2023)
    assert provider._observation("ERI") == (1_982_000_000.0, 2019)
    assert sum(result.weights.values()) == pytest.approx(1.0)
    assert result.weights["TWN"] / result.weights["SSD"] == pytest.approx(757.387 / 7.401)
    assert provider.resolve_iso3(codes) == result  # Cache retains provenance.
    audit = provider.audit_dict()
    assert audit["country_observations"]["ERI"]["year"] == 2019
    assert audit["data_source_counts"] == {"world_bank": 2, "imf_weo": 6}
    assert audit["sources"][1]["source"] == "World Economic Outlook (April 2026)"
    assert audit["license"] is None


def test_all_existing_world_bank_lookups_are_preserved():
    source = resources.files("sense.resources").joinpath(BUNDLED_GDP_FILE)
    with source.open() as handle:
        data = pd.read_csv(handle)
    # Check every reference year represented by the bundled datasets.
    for year in range(2000, 2025):
        primary = CountryWeightProvider.from_dataframe(data, reference_year=year)
        combined = CountryWeightProvider.bundled_gdp(reference_year=year)
        for country in primary._values:
            value, observed_year = primary._observation(country)
            if value is not None:
                assert combined._observation_with_source(country) == (
                    value,
                    observed_year,
                    "world_bank",
                )


@pytest.mark.parametrize(
    "primary_year,fallback_year,expected",
    [
        (2018, 2023, (10.0, 2018, "user_provided")),
        (2017, 2023, (99.0, 2023, "imf_weo")),
        (2017, 2018, (99.0, 2018, "imf_weo")),
        (2017, 2017, (None, None, None)),
    ],
)
def test_source_priority_precedes_year_proximity(primary_year, fallback_year, expected):
    provider = CountryWeightProvider(
        method="gdp", reference_year=2023, values={"AAA": {primary_year: 10.0}}
    )
    provider._fallback_values = {"AAA": {fallback_year: 99.0}}
    assert provider._observation_with_source("AAA") == expected


def test_custom_csv_is_not_silently_supplemented(tmp_path):
    path = tmp_path / "custom.csv"
    pd.DataFrame([{"country_code": "DEU", "year": 2023, "gdp_current_usd": 100.0}]).to_csv(
        path, index=False
    )
    cfg = SenseConfig(country_gdp_path=path)
    provider = _country_weight_provider(cfg, 2023)
    result = provider.resolve_iso3(["DEU", "TWN"])
    assert result.weights == {"DEU": 1.0}
    assert result.missing_countries == ("TWN",)
    assert provider.audit_dict()["sources"] == []
    assert CountryWeightProvider.equal(reference_year=2023).resolve_iso3(
        ["DEU", "TWN"]
    ).weights == {"DEU": 0.5, "TWN": 0.5}


def test_productivity_and_social_rows_share_default_source_weights(tmp_path):
    provider = _country_weight_provider(SenseConfig(), 2023)
    path = tmp_path / "productivity.parquet"
    pd.DataFrame(
        [
            {"country": c, "year": 2023, "isic4_code": "20", "productivity_hrs_per_usd": v}
            for c, v in [("DEU", 2.0), ("TWN", 8.0)]
        ]
    ).to_parquet(path)
    baseline = ProductivityBaseline(path, {}, weight_provider=provider, reference_year=2023)
    productivity = baseline.lookup_row_with_status(["DEU", "TWN"], "20")
    rows = attach_country_weights(
        pd.DataFrame(
            [
                {"Activity code": "a", "PSILCA_country": "DE"},
                {"Activity code": "a", "PSILCA_country": "TW"},
            ]
        ),
        provider,
    ).set_index("PSILCA_country")
    assert productivity.status == "override_full_aggregate"
    assert productivity.value == pytest.approx(
        rows.loc["DE", "Country_weight"] * 2 + rows.loc["TW", "Country_weight"] * 8
    )
    assert rows.loc["TW", "Country_weight_data_source"] == "imf_weo"
    assert rows.loc["DE", "Country_weight_data_source"] == "world_bank"


def test_snapshot_units_scope_and_integrity(monkeypatch, tmp_path):
    source = resources.files("sense.resources")
    for name in [BUNDLED_GDP_FILE, BUNDLED_IMF_GDP_FILE, BUNDLED_IMF_METADATA_FILE]:
        (tmp_path / name).write_bytes(source.joinpath(name).read_bytes())
    frame = pd.read_csv(tmp_path / BUNDLED_IMF_GDP_FILE)
    assert frame.year.min() == 2000 and frame.year.max() == 2024
    assert not frame.country_code.isin(["WEOWORLD", "ADVEC", "UVK", "WBG"]).any()
    assert {"XKX", "PSE", "TWN", "SSD", "ERI"} <= set(frame.country_code)
    assert not frame.duplicated(["country_code", "year"]).any()
    with (tmp_path / BUNDLED_IMF_GDP_FILE).open("a") as handle:
        handle.write("\n")
    monkeypatch.setattr(resources, "files", lambda _: tmp_path)
    with pytest.raises(ValueError, match="provenance hash"):
        CountryWeightProvider.bundled_gdp(reference_year=2023)


def test_snapshot_builder_filters_totals_and_preserves_decimal_units(tmp_path):
    path = Path(__file__).resolve().parents[1] / "tools/build_imf_gdp_snapshot.py"
    spec = importlib.util.spec_from_file_location("build_imf_gdp_snapshot", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data = tmp_path / "data.json"
    data.write_text(
        json.dumps(
            {
                "values": {
                    "NGDPD": {
                        "TWN": {"1999": 1, "2023": 757.387, "2025": 2},
                        "UVK": {"2023": 10.1},
                        "WBG": {"2023": 3.4},
                        "WEOWORLD": {"2023": 100000},
                    }
                }
            }
        )
    )
    indicators = tmp_path / "indicators.json"
    metadata = {
        "indicators": {
            "NGDPD": {
                "unit": "Billions of U.S. dollars",
                "dataset": "WEO",
                "source": "Test release",
            }
        }
    }
    indicators.write_text(json.dumps(metadata))
    countries = tmp_path / "countries.json"
    countries.write_text(
        json.dumps({"countries": {c: {"label": c} for c in ["TWN", "UVK", "WBG"]}})
    )
    result = module.build(data, indicators, countries, tmp_path / "out")
    out = tmp_path / "out" / BUNDLED_IMF_GDP_FILE
    frame = pd.read_csv(out).set_index("country_code")
    assert frame.loc["TWN", "gdp_current_usd"] == 757_387_000_000
    assert set(frame.index) == {"TWN", "XKX", "PSE"}
    assert result["sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()
    assert (
        module.build(data, indicators, countries, tmp_path / "again")["sha256"] == result["sha256"]
    )
    metadata["indicators"]["NGDPD"]["unit"] = "Billions of international dollars"
    indicators.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="current US dollars"):
        module.build(data, indicators, countries, tmp_path / "wrong")
