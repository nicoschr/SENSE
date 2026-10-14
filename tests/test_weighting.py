import math
from pathlib import Path

import pandas as pd
import pytest

import sense.core.averaging as averaging
import sense.productivity.geography as geography
import sense.productivity.loader as loader
import sense.weighting as weighting
from sense.weighting import CountryWeightProvider


def _gdp_data():
    return pd.DataFrame(
        [
            {"country_code": "AAA", "year": 2019, "gdp_current_usd": 75.0},
            {"country_code": "BBB", "year": 2018, "gdp_current_usd": 25.0},
        ]
    )


def _social_row(country: str, value: float):
    return {
        "Activity code": "activity-1",
        "Activity name": "Example",
        "Reference product": "product",
        "ISIC": "10",
        "Unit": "kg",
        "ISO2": "RER",
        "Price": 1.0,
        "PSILCA_country": country,
        "Matched_sector": "Example sector",
        "PSILCA_Exchanges": [
            {
                "indicator": "Example risk",
                "risk level": "medium",
                "type": "risk",
                "unit": "h",
                "category": "Workers",
                "subcategory": "Example",
                "meanValue": value,
            }
        ],
    }


def test_bundled_world_bank_snapshot_loads_without_external_files():
    provider = weighting.CountryWeightProvider.bundled_gdp(reference_year=2019)
    resolution = provider.resolve_iso3(["DEU", "FRA"])

    assert resolution.method == "gdp"
    assert resolution.data_years == {"DEU": 2019, "FRA": 2019}
    assert math.isclose(sum(resolution.weights.values()), 1.0)
    assert resolution.weights["DEU"] > resolution.weights["FRA"]


def test_provider_uses_nearest_year_and_reports_partial_coverage():
    provider = weighting.CountryWeightProvider.from_dataframe(_gdp_data(), reference_year=2019)

    resolution = provider.resolve_iso3(["AAA", "BBB", "CCC"])

    assert resolution.method == "gdp_available_countries"
    assert resolution.data_years == {"AAA": 2019, "BBB": 2018}
    assert resolution.missing_countries == ("CCC",)
    assert resolution.weights == pytest.approx({"AAA": 0.75, "BBB": 0.25})
    assert provider.audit_dict()["missing_country_counts"] == {"CCC": 1}


def test_country_weights_change_only_cross_country_aggregation(monkeypatch):
    monkeypatch.setattr(
        weighting,
        "iso2_to_iso3",
        lambda code: {"AA": "AAA", "BB": "BBB"}.get(code),
    )
    provider = weighting.CountryWeightProvider.from_dataframe(_gdp_data(), reference_year=2019)

    rows = [
        _social_row("AA", 2.0),
        _social_row("AA", 6.0),  # within-country arithmetic mean = 4
        _social_row("BB", 12.0),
    ]
    enriched = weighting.attach_country_weights(pd.DataFrame(rows), provider)
    result = averaging.average_sector_mappings_w_prices(
        enriched,
        level_weights={"medium": 1.0, "no data": 0.1},
        mean_method="arithmetic",
    )[0]

    # Cross-country result: 4 * 0.75 + 12 * 0.25 = 6.
    assert result["Working_hours"] == pytest.approx(6.0)
    assert result["Averaged_Exchanges"][0]["meanValue"] == pytest.approx(6.0)
    assert result["PSILCA_country_weights"] == pytest.approx({"AA": 0.75, "BB": 0.25})


def test_country_weights_are_renormalized_for_each_available_indicator(monkeypatch):
    monkeypatch.setattr(
        weighting,
        "iso2_to_iso3",
        lambda code: {"AA": "AAA", "BB": "BBB"}.get(code),
    )
    provider = weighting.CountryWeightProvider.from_dataframe(_gdp_data(), reference_year=2019)
    rows = [_social_row("AA", 2.0), _social_row("BB", 8.0)]
    rows[1]["PSILCA_Exchanges"][0]["indicator"] = "Other risk"

    result = averaging.average_sector_mappings_w_prices(
        weighting.attach_country_weights(pd.DataFrame(rows), provider),
        level_weights={"medium": 1.0, "no data": 0.1},
        mean_method="arithmetic",
    )[0]
    values = {
        exchange["indicator"]: exchange["meanValue"] for exchange in result["Averaged_Exchanges"]
    }

    assert values == pytest.approx({"Example risk": 2.0, "Other risk": 8.0})


def test_weighted_geometric_mean_uses_weights_as_exponents():
    rows = [_social_row("AA", 1.0), _social_row("BB", 100.0)]
    for row, weight in zip(rows, (0.9, 0.1), strict=True):
        row["Country_weight"] = weight
    result = averaging.average_sector_mappings_w_prices(
        pd.DataFrame(rows), level_weights={"medium": 1.0}, mean_method="geometric"
    )[0]

    assert result["Averaged_Exchanges"][0]["meanValue"] == pytest.approx(100.0**0.1)
    # Working hours use arithmetic averaging even when social scores are geometric.
    assert result["Working_hours"] == pytest.approx(10.9)


@pytest.mark.parametrize("method", ["geometric", "arithmetic"])
def test_aggregation_rejects_negative_worker_hour_intensities(method):
    rows = [_social_row("AA", -1.0), _social_row("BB", 1.0)]
    with pytest.raises(ValueError, match="Invalid PSILCA worker-hour intensity"):
        averaging.average_sector_mappings_w_prices(
            pd.DataFrame(rows), level_weights={"medium": 1.0}, mean_method=method
        )


def test_productivity_geography_uses_the_shared_provider(monkeypatch):
    provider = weighting.CountryWeightProvider.from_dataframe(_gdp_data(), reference_year=2019)
    monkeypatch.setattr(
        geography,
        "iso2_to_iso3",
        lambda code: {"AA": "AAA", "BB": "BBB"}.get(code),
    )
    monkeypatch.setattr(
        geography,
        "_parse_geographies",
        lambda _: pd.DataFrame(
            [
                {
                    "Shortname": "RER",
                    "simplified_ISO2": "AA; BB",
                }
            ]
        ),
    )

    mapping = geography.build_location_mapping(
        Path("not-read.xlsx"),
        available_iso3={"AAA", "BBB"},
        weight_provider=provider,
    )

    assert dict(mapping["RER"]) == pytest.approx({"AAA": 0.75, "BBB": 0.25})


def test_productivity_row_lookup_reuses_shared_provider(tmp_path):
    provider = weighting.CountryWeightProvider.from_dataframe(_gdp_data(), reference_year=2019)
    baseline_path = tmp_path / "productivity.parquet"
    pd.DataFrame(
        [
            {
                "country": "AAA",
                "year": 2020,
                "isic4_code": "10",
                "productivity_hrs_per_usd": 2.0,
            },
            {
                "country": "BBB",
                "year": 2020,
                "isic4_code": "10",
                "productivity_hrs_per_usd": 10.0,
            },
        ]
    ).to_parquet(baseline_path, index=False)

    baseline = loader.ProductivityBaseline(
        parquet_path=baseline_path,
        location_mapping={},
        reference_year=2020,
        weight_provider=provider,
    )
    result = baseline.lookup_row_with_status(["AAA", "BBB"], "10", 2020)

    assert result.value == pytest.approx(4.0)
    assert result.n_countries_used == 2


def test_custom_gdp_does_not_claim_world_bank_rights_or_allow_infinity():
    data = pd.DataFrame({"country_code": ["AAA"], "year": [2020], "gdp_current_usd": [1.0]})
    custom = CountryWeightProvider.from_dataframe(data, reference_year=2020).audit_dict()
    assert custom["license"] is None and custom["source_url"] is None
    bundled = CountryWeightProvider.bundled_gdp(reference_year=2020).audit_dict()
    assert bundled["license"] is None  # Mixed source rights are recorded separately.
    assert bundled["sources"][0]["license"] == "CC BY 4.0"
    assert bundled["sources"][1]["source_id"] == "imf_weo"
    data.loc[0, "gdp_current_usd"] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        CountryWeightProvider.from_dataframe(data, reference_year=2020)
