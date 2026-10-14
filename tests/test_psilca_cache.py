"""Tests for the compact PSILCA assessment cache."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from sense.core.averaging import average_sector_mappings_w_prices
from sense.core.matching import PSILCALookupStore, add_social_exchanges, match_from_stores
from sense.data.embeddings import save_embeddings
from sense.data.psilca import (
    PSILCA_CACHE_SCHEMA_VERSION,
    combine_psilca_css_and_extract_social_exchanges,
)


def _process_xml(*, sector: str, exchange: str, value: float) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<ecoSpold>
  <dataset>
    <metaInformation>
      <processInformation>
        <referenceFunction name="{sector}" />
        <geography location="AA" />
      </processInformation>
    </metaInformation>
    <flowData>
      <exchange name="{exchange}" unit="h" meanValue="{value}"
                category="Workers" subCategory="Conditions">
        <outputGroup>4</outputGroup>
      </exchange>
    </flowData>
  </dataset>
</ecoSpold>
"""


def test_compact_cache_preserves_conflicting_processes_and_optional_selection(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "a.xml").write_text(
        _process_xml(sector="Shared sector", exchange="Indicator; high risk", value=2),
        encoding="utf-8",
    )
    (source / "b.xml").write_text(
        _process_xml(
            sector="Shared sector",
            exchange="Indicator; very high opportunity",
            value=3,
        ),
        encoding="utf-8",
    )
    cache = tmp_path / "assessments.parquet"
    sectors = tmp_path / "sectors.json"

    combine_psilca_css_and_extract_social_exchanges(
        str(source),
        str(cache),
        str(sectors),
    )

    metadata = pq.read_metadata(cache).metadata
    assert metadata[b"sense_schema"] == str(PSILCA_CACHE_SCHEMA_VERSION).encode()
    assert json.loads(sectors.read_text(encoding="utf-8")) == {"AA": ["Shared sector"]}
    expected = {
        ("aa", "shared sector", "b.xml"): [
            {
                "indicator": "Indicator",
                "type": "opportunity",
                "risk level": "very high",
                "meanValue": 3.0,
                "unit": "h",
                "category": "Workers",
                "subcategory": "Conditions",
            }
        ]
    }
    all_processes = PSILCALookupStore(cache).for_countries(["AA"])
    assert set(all_processes) == {
        ("aa", "shared sector", "a.xml"),
        ("aa", "shared sector", "b.xml"),
    }
    assert all_processes[("aa", "shared sector", "a.xml")][0]["meanValue"] == 2
    assert (
        all_processes[("aa", "shared sector", "b.xml")]
        == expected[("aa", "shared sector", "b.xml")]
    )
    store = PSILCALookupStore(
        cache,
        max_cached_countries=1,
        process_overrides={"AA | Shared sector": "b.xml"},
    )
    assert store.for_countries(["AA"]) == expected
    assert store.for_countries(["AA"]) == expected


@pytest.mark.parametrize("failure", ["empty", "invalid_xml", "write_failure"])
def test_failed_psilca_parse_preserves_previous_files_and_removes_temporary(
    tmp_path, monkeypatch, failure
):
    import sense.data.psilca as psilca

    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "assessments.parquet"
    sectors = tmp_path / "sectors.json"
    output.write_bytes(b"previous cache")
    sectors.write_text("previous sectors", encoding="utf-8")
    if failure == "invalid_xml":
        (source / "test.xml").write_text("<unfinished>", encoding="utf-8")
    elif failure == "write_failure":
        (source / "test.xml").write_text("<ecoSpold/>", encoding="utf-8")

        def fail(*_):
            raise OSError("synthetic disk failure")

        monkeypatch.setattr(psilca, "_write_parquet_batch", fail)
    with pytest.raises((FileNotFoundError, RuntimeError, OSError)):
        combine_psilca_css_and_extract_social_exchanges(str(source), str(output), str(sectors))
    assert output.read_bytes() == b"previous cache"
    assert sectors.read_text(encoding="utf-8") == "previous sectors"
    assert {p.name for p in tmp_path.iterdir()} == {"source", output.name, sectors.name}


@pytest.mark.parametrize("mean_method", ["arithmetic", "geometric"])
@pytest.mark.parametrize("external_hours", [None, 7.0])
@pytest.mark.parametrize("reverse", [False, True])
def test_semantic_ties_aggregate_every_source_without_multiplying_hours(
    tmp_path, mean_method, external_hours, reverse
):
    # Two same-name sources plus a nearby different label all qualify. A fourth
    # label is outside tolerance. The duplicate must count as a candidate, not
    # overwrite another source or split a special name-level weight.
    source = tmp_path / "source"
    source.mkdir()
    recipes = [
        ("a.xml", "Shared sector", "low", 2.0),
        ("b.xml", "Shared sector", "high", 4.0),
        ("c.xml", "Near sector", "medium", 6.0),
        ("d.xml", "Distant sector", "high", 100.0),
    ]
    for filename, label, risk, hours in recipes:
        (source / filename).write_text(
            _process_xml(sector=label, exchange=f"Indicator; {risk} risk", value=hours),
            encoding="utf-8",
        )
    cache = tmp_path / "assessments.parquet"
    combine_psilca_css_and_extract_social_exchanges(
        str(source), str(cache), str(tmp_path / "sectors.json")
    )
    eco = tmp_path / "eco"
    psi = tmp_path / "psi"
    save_embeddings(
        str(eco),
        "ecoinvent_AA",
        np.array([[1.0, 0.0]], dtype="float32"),
        pd.Series(["activity-1"], name="id"),
        {},
    )
    scores = np.array([0.9, 0.897, 0.8], dtype="float32")
    vectors = np.column_stack([scores, np.sqrt(1 - scores**2)])
    save_embeddings(
        str(psi),
        "psilca_AA",
        vectors,
        pd.Series(["AA::Shared sector", "AA::Near sector", "AA::Distant sector"], name="id"),
        {},
    )
    mapped = match_from_stores(
        str(eco),
        str(psi),
        "AA",
        threshold=0.005,
        activity_entry={
            "PSILCA_mapping": "AA",
            "Activities": [
                {
                    "code": "activity-1",
                    "name": "Example",
                    "reference product": "Product",
                    "ISIC rev.4 ecoinvent": "20",
                    "unit": "kg",
                    "price": 1.0,
                }
            ],
        },
    )
    lookup = PSILCALookupStore(cache).for_countries(["AA"])
    if reverse:
        lookup = dict(reversed(list(lookup.items())))
        mapped = mapped.iloc[::-1]
    enriched = add_social_exchanges(mapped, lookup)
    assert set(enriched["PSILCA_process"]) == {"a.xml", "b.xml", "c.xml"}
    assert enriched.loc[enriched.Matched_sector == "Shared sector", "Semantic_score"].nunique() == 1
    kwargs = {} if external_hours is None else {"hours_override": {"activity-1": external_hours}}
    result = average_sector_mappings_w_prices(
        enriched,
        level_weights={"low": 0.1, "medium": 1.0, "high": 10.0},
        mean_method=mean_method,
        rounding=9,
        **kwargs,
    )[0]
    expected_hours = 4.0 if external_hours is None else external_hours
    assert result["Working_hours"] == pytest.approx(expected_hours)
    observations = (
        np.array([0.2, 40.0, 6.0])
        if external_hours is None
        else np.array([0.1, 10.0, 1.0]) * external_hours
    )
    expected_risk = (
        observations.mean() if mean_method == "arithmetic" else np.exp(np.log(observations).mean())
    )
    assert result["Averaged_Exchanges"][0]["meanValue"] == pytest.approx(expected_risk)
    assert result["PSILCA_processes_used"] == {"AA": ["a.xml", "b.xml", "c.xml"]}


def test_identical_assessments_are_separate_candidates_and_legacy_lookup_works():
    exchange = {
        "indicator": "I",
        "type": "risk",
        "risk level": "low",
        "meanValue": 1.0,
        "unit": "h",
    }
    mapped = pd.DataFrame(
        [{"PSILCA_country": "AA", "Matched_sector": "Shared", "Semantic_score": 0.9}]
    )
    lookup = {("aa", "shared", name): [dict(exchange)] for name in ["a.xml", "b.xml"]}
    result = add_social_exchanges(mapped, lookup)
    assert result["PSILCA_process"].tolist() == ["a.xml", "b.xml"]
    assert len(result) == 2
    legacy = add_social_exchanges(mapped, {("aa", "shared"): [exchange]})
    assert len(legacy) == 1
    assert legacy.iloc[0]["PSILCA_Exchanges"] == [exchange]
