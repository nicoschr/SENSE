"""Regional completeness must include constituents with missing GDP weights."""

from pathlib import Path

import pandas as pd
import pytest

from sense.productivity import geography
from sense.productivity.loader import ProductivityBaseline
from sense.weighting import CountryWeightProvider


@pytest.mark.parametrize(
    "available_gdp,expected_value,expected_status,expected_used",
    [
        (["DEU", "FRA"], 3.0, "override_full_aggregate", 2),
        (["DEU"], 2.0, "override_partial_aggregate", 1),
        (["USA"], 4.0, "override_partial_aggregate", 2),
    ],
)
@pytest.mark.parametrize("kind", ["named_region", "rest_of_region"])
def test_gdp_gaps_are_visible_in_both_region_paths(
    tmp_path,
    monkeypatch,
    available_gdp,
    expected_value,
    expected_status,
    expected_used,
    kind,
):
    provider = CountryWeightProvider.from_dataframe(
        pd.DataFrame(
            [
                {
                    "country_code": code,
                    "year": 2023,
                    "gdp_current_usd": 3.0 if code == "DEU" else 1.0,
                }
                for code in available_gdp
            ]
        ),
        reference_year=2023,
    )
    monkeypatch.setattr(
        geography,
        "_parse_geographies",
        lambda _: pd.DataFrame(
            [
                {"Shortname": "RER", "simplified_ISO2": "DE; FR"},
                {"Shortname": "FR", "simplified_ISO2": "FR"},
            ]
        ),
    )
    mapping = geography.build_location_mapping(Path("not-read.xlsx"), {"DEU", "FRA"}, provider)
    path = tmp_path / "baseline.parquet"
    pd.DataFrame(
        [
            {"country": c, "year": 2023, "isic4_code": "20", "productivity_hrs_per_usd": v}
            for c, v in [("DEU", 2.0), ("FRA", 6.0)]
        ]
    ).to_parquet(path, index=False)
    baseline = ProductivityBaseline(path, mapping, weight_provider=provider, reference_year=2023)
    if kind == "named_region":
        result = baseline.lookup_with_status("RER", "20")
    else:
        result = baseline.lookup_row_with_status(["DEU", "FRA"], "20")
    assert result.value == pytest.approx(expected_value)
    assert result.status == expected_status
    assert result.n_countries_attempted == 2
    assert result.n_countries_used == expected_used
    # A single-country lookup needs no relative GDP weight.
    assert baseline.lookup_with_status("FR", "20").status == "override_exact"
