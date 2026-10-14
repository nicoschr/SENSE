"""Tests for explicit monetary-basis conversion."""

import json
import math

import pandas as pd
import pytest

import sense.core.averaging as averaging
import sense.data.seed_prices as seed_prices
import sense.monetary as monetary
from sense.data.seed_prices import load_seed_template


def test_known_psilca_versions_resolve_documented_defaults():
    v3 = monetary.resolve_psilca_price_conversion("v3.1.1")
    assert v3.source.label == "EUR2005"
    assert v3.target.label == "USD2011"
    assert v3.target.approximate is True
    assert math.isclose(v3.factor, 1.5731857218119631, rel_tol=1e-12)

    v4 = monetary.resolve_psilca_price_conversion("4")
    assert v4.target.label == "USD2015"
    assert v4.target.approximate is True
    assert math.isclose(v4.factor, 1.3108611656722589, rel_tol=1e-12)


def test_explicit_year_and_factor_override_known_or_unknown_version():
    conversion = monetary.resolve_psilca_price_conversion(
        "5.0",
        psilca_year=2024,
        factor_override=1.23,
    )
    assert conversion.target.label == "USD2024"
    assert conversion.factor == 1.23
    assert conversion.user_supplied is True


def test_ambiguous_or_invalid_conversion_is_rejected_instead_of_guessed():
    with pytest.raises(ValueError, match="No representative monetary year"):
        monetary.resolve_psilca_price_conversion("5.0")

    with pytest.raises(ValueError, match="No built-in conversion"):
        monetary.resolve_psilca_price_conversion("4.0", psilca_year=2020)

    with pytest.raises(ValueError, match="positive and finite"):
        monetary.resolve_psilca_price_conversion("4.0", factor_override=0)


def test_psilca_and_external_productivity_use_their_own_factor_once():
    enriched = pd.DataFrame(
        [
            {
                "Activity code": "activity-1",
                "Activity name": "Example",
                "Reference product": "product",
                "ISIC": "20",
                "Unit": "kg",
                "ISO2": "CO",
                "Price": 10.0,
                "PSILCA_country": "CO",
                "Matched_sector": "Land transport",
                "PSILCA_Exchanges": [
                    {
                        "indicator": "Example risk",
                        "risk level": "medium",
                        "type": "risk",
                        "unit": "h",
                        "category": "Workers",
                        "subcategory": "Example",
                        "meanValue": 2.0,
                    }
                ],
            }
        ]
    )

    psilca_result = averaging.average_sector_mappings_w_prices(
        enriched,
        level_weights={"medium": 1.0, "no data": 0.1},
        psilca_price_factor=3.0,
        productivity_price_factor=4.0,
    )[0]
    assert psilca_result["Working_hours"] == 60.0
    assert psilca_result["Converted_price"] == 30.0
    assert psilca_result["Averaged_Exchanges"][0]["meanValue"] == pytest.approx(60.0)

    productivity_result = averaging.average_sector_mappings_w_prices(
        enriched,
        level_weights={"medium": 1.0, "no data": 0.1},
        hours_override={"activity-1": 5.0},
        psilca_price_factor=3.0,
        productivity_price_factor=4.0,
    )[0]
    assert productivity_result["Working_hours"] == 200.0
    assert productivity_result["Converted_price"] == 40.0
    assert productivity_result["Averaged_Exchanges"][0]["meanValue"] == pytest.approx(200.0)

    second_activity = enriched.copy()
    second_activity["Activity code"] = "activity-2"
    mixed_results = averaging.average_sector_mappings_w_prices(
        pd.concat([enriched, second_activity], ignore_index=True),
        level_weights={"medium": 1.0, "no data": 0.1},
        hours_override={"activity-2": 5.0},
        psilca_price_factor=3.0,
        productivity_price_factor=4.0,
    )
    mixed_by_code = {result["Activity code"]: result for result in mixed_results}
    assert mixed_by_code["activity-1"]["Averaged_Exchanges"][0]["meanValue"] == 60.0
    assert mixed_by_code["activity-2"]["Averaged_Exchanges"][0]["meanValue"] == 200.0


def test_seed_template_basis_mismatch_is_rejected(tmp_path):
    seed_path = tmp_path / "seeds.json"
    seed_path.write_text(
        json.dumps(
            {
                "metadata": {
                    "expected_currency": "EUR",
                    "expected_price_year": 2005,
                },
                "seeds": [
                    {
                        "unit": "kg",
                        "reference_product": "example",
                        "price": 2.5,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    loaded = seed_prices.load_seed_template(
        str(seed_path), expected_currency="EUR", expected_year=2005
    )
    assert loaded[("kg", "example")] == 2.5

    with pytest.raises(ValueError, match="declares currency"):
        seed_prices.load_seed_template(str(seed_path), expected_currency="USD", expected_year=2005)


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
def test_nonfinite_seed_never_enters_inference(tmp_path, value):
    path = tmp_path / "seed.json"
    path.write_text(
        json.dumps({"seeds": [{"unit": "kg", "reference_product": "x", "price": value}]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="finite"):
        load_seed_template(str(path))
