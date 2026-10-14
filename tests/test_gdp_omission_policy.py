"""GDP omissions may be accepted; missing retained coefficients must not be."""

from pathlib import Path

import pandas as pd
import pytest

from sense.config import SenseConfig
from sense.economics import write_productivity_audit
from sense.productivity.loader import ProductivityBaseline
from sense.regional import _preflight_productivity_coverage
from sense.weighting import CountryWeightProvider


def setup_case(
    tmp_path,
    available_gdp,
    available_productivity,
    policy,
    *,
    region="RER",
    isic="20",
    use_psilca=False,
):
    provider = CountryWeightProvider.from_dataframe(
        pd.DataFrame(
            [
                {"country_code": c, "year": 2023, "gdp_current_usd": v}
                for c, v in available_gdp.items()
            ]
        ),
        reference_year=2023,
    )
    countries = ["DEU", "FRA"]
    weights = provider.resolve_iso3(countries).weights
    path = tmp_path / "productivity.parquet"
    pd.DataFrame(
        [
            {"country": c, "year": 2023, "isic4_code": "20", "productivity_hrs_per_usd": v}
            for c, v in available_productivity.items()
        ]
    ).to_parquet(path, index=False)
    baseline = ProductivityBaseline(
        path,
        {"RER": [(c, weights.get(c, 0.0)) for c in countries], "FR": [("FRA", 1.0)]},
        weight_provider=provider,
        reference_year=2023,
    )
    config = SenseConfig(
        use_psilca=use_psilca,
        productivity_baseline_path=path,
        productivity_year=2023,
        productivity_eur_per_usd=1.0,
        productivity_gdp_policy=policy,
    )
    activity = {
        "code": "a",
        "name": "Example",
        "ISIC rev.4 ecoinvent": isic,
        "PSILCA_mapping": "DE; FR",
    }
    return {
        "regions": [region],
        "iso2_data": {region: {"Activities": [activity]}},
        "config": config,
        "baseline": baseline,
    }


@pytest.mark.parametrize("region", ["RER", "RoW", "RoE"])
@pytest.mark.parametrize("use_psilca", [False, True])
def test_accept_only_gdp_omissions_and_preserve_audit(tmp_path, region, use_psilca):
    kwargs = setup_case(
        tmp_path,
        {"DEU": 3},
        {"DEU": 2},
        "available_countries",
        region=region,
        use_psilca=use_psilca,
    )
    # FRA has neither GDP nor productivity: its GDP omission is explicit.
    values, rows = _preflight_productivity_coverage(**kwargs)[region]
    assert values == {"a": 2.0}
    assert len(rows) == 1
    row = rows[0]
    assert row["status"] == "override_partial_aggregate"
    assert row["accepted_gdp_omissions"] is True
    assert row["omitted_gdp_countries"] == ["FRA"]
    assert row["missing_productivity_countries"] == []
    assert row["n_countries_attempted"] == 2
    assert row["n_countries_used"] == 1
    write_productivity_audit(
        output_dir=tmp_path,
        technosphere_database="db",
        results=[
            {
                "num_activities": 1,
                "productivity_override_applied": 1,
                "productivity_audit_misses": [],
                "productivity_audit_gdp_omissions": rows,
            }
        ],
    )
    saved = pd.read_csv(tmp_path / "ProductivityAudit/productivity_audit_db.csv")
    assert len(saved) == 1
    assert saved.iloc[0]["accepted_gdp_omissions"]
    assert saved.iloc[0]["omitted_gdp_countries"] == "['FRA']"


@pytest.mark.parametrize("region", ["RER", "RoW"])
@pytest.mark.parametrize("failure", ["strict", "no_gdp", "missing_retained", "unmapped", "no_isic"])
def test_errors_still_stop(tmp_path, region, failure):
    gdp = {"DEU": 3}
    productivity = {"DEU": 2, "FRA": 6}
    policy = "available_countries"
    isic = "20"
    if failure == "strict":
        policy = "strict"
    elif failure == "no_gdp":
        gdp = {"USA": 1}
    elif failure == "missing_retained":
        gdp["FRA"] = 1
        productivity.pop("FRA")
    elif failure == "unmapped":
        region = "UNKNOWN"
    elif failure == "no_isic":
        isic = ""
    kwargs = setup_case(tmp_path, gdp, productivity, policy, region=region, isic=isic)
    with pytest.raises(ValueError, match="complete productivity coverage"):
        _preflight_productivity_coverage(**kwargs)


def test_no_gdp_needed_for_direct_country(tmp_path):
    kwargs = setup_case(
        tmp_path, {"DEU": 3}, {"DEU": 2, "FRA": 6}, "available_countries", region="FR"
    )
    assert _preflight_productivity_coverage(**kwargs)["FR"] == ({"a": 6.0}, [])


def test_complete_weighted_mean_unchanged(tmp_path):
    kwargs = setup_case(tmp_path, {"DEU": 3, "FRA": 1}, {"DEU": 2, "FRA": 6}, "available_countries")
    assert _preflight_productivity_coverage(**kwargs)["RER"] == ({"a": 3.0}, [])


def test_missing_productivity_not_hidden_by_simultaneous_gdp_gap(tmp_path):
    kwargs = setup_case(tmp_path, {"DEU": 3}, {"FRA": 6}, "available_countries")
    with pytest.raises(ValueError, match="complete productivity coverage"):
        _preflight_productivity_coverage(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"productivity_gdp_policy": "typo"},
        {"productivity_gdp_policy": "available_countries"},
        {
            "productivity_gdp_policy": "available_countries",
            "productivity_baseline_path": Path("x"),
            "country_weighting": "equal",
        },
    ],
)
def test_invalid_policy_configuration(kwargs):
    with pytest.raises(ValueError):
        SenseConfig(**kwargs)


def test_default_policy_remains_strict():
    assert SenseConfig().productivity_gdp_policy == "strict"
