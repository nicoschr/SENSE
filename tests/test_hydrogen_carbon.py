"""Physical carbon cycles must not receive an uptake-only climate credit."""

import pytest

from case_studies.hydrogen.workflow.carbon_methods import explicit_factors


def test_closed_co2_cycle_and_methane_oxidation_are_counted_consistently():
    names = [
        "Carbon dioxide, in air",
        "Carbon dioxide, non-fossil",
        "Carbon dioxide, non-fossil, resource correction",
        "Methane, fossil",
        "Methane, non-fossil",
        "Carbon dioxide, to soil or biomass stock",
    ]
    flows = {
        ("bio", str(i)): {"name": name, "categories": ("air",), "unit": "kilogram"}
        for i, name in enumerate(names)
    }
    source = {("bio", "0"): -1.0, ("bio", "3"): 36.0, ("bio", "4"): 34.0, ("bio", "5"): -1.0}
    result = explicit_factors(source, flows)
    assert 32.8 * (result["bio", "0"] + result["bio", "1"]) == 0
    assert result["bio", "2"] == -1
    assert result["bio", "4"] == result["bio", "3"] == 36
    assert result["bio", "5"] == -1
    assert ("bio", "1") not in source  # no changes to imported factors
    assert explicit_factors(result, flows) == result
    del source["bio", "3"]
    with pytest.raises(ValueError, match="fossil-methane"):
        explicit_factors(source, flows)


def test_spatialized_carbon_flows_get_the_same_factors():
    names = [
        "Carbon dioxide, in air",
        "Carbon dioxide, non-fossil",
        "Carbon dioxide, non-fossil, resource correction",
    ]
    flows = {
        ("biosphere3_spatialized_flows", str(i)): {
            "name": name + ", US",
            "categories": ("air",),
            "unit": "kilogram",
        }
        for i, name in enumerate(names)
    }
    assert list(explicit_factors({}, flows).values()) == [-1, 1, -1]
