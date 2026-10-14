"""Tests for cache validation, configuration, and failure signaling."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import sense.cli as cli
import sense.core.matching as matching
import sense.data.ecoinvent as ecoinvent
import sense.pipeline as pipeline
import sense.preparation as preparation
from sense.config import SenseConfig
from sense.data.ecoinvent import write_or_merge_location_concordance
from sense.data.embeddings import (
    BGE_QUERY_PROMPT,
    _sha,
    prep_query_bge,
    save_embeddings,
    store_is_complete,
)
from sense.data.seed_prices import write_or_merge_seed_template
from sense.exceptions import SenseInputRequiredError


def test_technosphere_loader_reports_nearby_database_name(monkeypatch):
    monkeypatch.setattr(
        preparation,
        "databases",
        {"ecoinvent-3.10.1-cutoff - regionalized": {}},
    )

    with pytest.raises(ValueError, match="Did you mean 'ecoinvent-3.10.1-cutoff - regionalized'"):
        preparation._load_technosphere_activities("ecoinvent-3.10.1-cutoff regionalized")


def test_unknown_resume_region_fails_instead_of_running_nothing():
    with pytest.raises(ValueError, match="was not found"):
        pipeline._select_iso2_regions({"AA": {}}, SenseConfig(iso2_start_at="ZZ"))


def test_embedding_cache_rejects_stale_ids_and_metadata(tmp_path):
    save_embeddings(
        str(tmp_path),
        "example",
        np.ones((2, 3), dtype=np.float32),
        pd.Series(["a", "b"]),
        {"model": "model-a", "input_hash": "hash-a"},
    )

    assert store_is_complete(
        str(tmp_path),
        "example",
        expected_ids=["a", "b"],
        expected_meta={"model": "model-a", "input_hash": "hash-a"},
    )
    assert not store_is_complete(str(tmp_path), "example", expected_ids=["a", "changed"])
    assert not store_is_complete(str(tmp_path), "example", expected_meta={"model": "model-b"})


def test_ecoinvent_embedding_cache_accepts_harmless_activity_reordering(tmp_path):
    stored_text = {"b": "text b", "a": "text a"}
    stored_ids = ["b", "a"]
    base_meta = {
        "model": "model-a",
        "prompt": BGE_QUERY_PROMPT,
        "normalized": True,
        "source": "ecoinvent_activities",
        "iso2": "AA",
    }
    stored_meta = {
        **base_meta,
        "input_hash": _sha(json.dumps([prep_query_bge(stored_text[code]) for code in stored_ids])),
    }
    save_embeddings(
        str(tmp_path),
        "ecoinvent_AA",
        np.ones((2, 3), dtype=np.float32),
        pd.Series(stored_ids),
        stored_meta,
    )

    reordered = pd.DataFrame([("a", "text a"), ("b", "text b")], columns=["code", "text"])
    expected_meta = {
        **base_meta,
        "input_hash": _sha(json.dumps(reordered["text"].map(prep_query_bge).tolist())),
    }
    assert ecoinvent._embedding_store_matches_content(
        str(tmp_path), "ecoinvent_AA", reordered, expected_meta
    )

    reordered.loc[reordered["code"] == "a", "text"] = "changed"
    assert not ecoinvent._embedding_store_matches_content(
        str(tmp_path), "ecoinvent_AA", reordered, expected_meta
    )


def test_only_current_location_concordances_block(tmp_path):
    path = tmp_path / "locations.json"
    path.write_text(
        json.dumps(
            {
                "locations": [
                    {"location_code": "OLD", "countries": None},
                    {"location_code": "CURRENT", "countries": "DE"},
                ]
            }
        ),
        encoding="utf-8",
    )

    loaded, new, missing = write_or_merge_location_concordance(str(path), ["CURRENT"], "database")

    assert (loaded, new, missing) == (1, 0, 0)
    entries = json.loads(path.read_text(encoding="utf-8"))["locations"]
    assert {entry["location_code"]: entry["required"] for entry in entries} == {
        "OLD": False,
        "CURRENT": True,
    }


def test_only_current_seed_worklist_entries_block(tmp_path):
    path = tmp_path / "seeds.json"
    path.write_text(
        json.dumps(
            {
                "seeds": [
                    {"unit": "kg", "reference_product": "old", "price": None},
                    {"unit": "kg", "reference_product": "current", "price": 2.0},
                ]
            }
        ),
        encoding="utf-8",
    )
    worklist = pd.DataFrame(
        [
            {
                "unit": "kg",
                "reference_product": "current",
                "seed_type": "external_missing",
                "example_activity_name": "Example",
            }
        ]
    )

    loaded, new, missing = write_or_merge_seed_template(str(path), worklist, "database")

    assert (loaded, new, missing) == (1, 0, 0)
    entries = json.loads(path.read_text(encoding="utf-8"))["seeds"]
    assert {entry["reference_product"]: entry["required"] for entry in entries} == {
        "old": False,
        "current": True,
    }


@pytest.mark.parametrize("include_empty", [False, True])
def test_cli_returns_nonzero_for_expected_pipeline_stop(monkeypatch, capsys, include_empty):
    def stop(**_):
        assert _["config"].exclude_empty_inventories is (not include_empty)
        raise SenseInputRequiredError("Seed prices required", "/tmp/seeds.json")

    monkeypatch.setattr(cli, "run_sense", stop)
    return_code = cli.main(
        [
            "run",
            "--project",
            "project",
            "--technosphere-db",
            "technosphere",
            "--psilca-dir",
            "/tmp/psilca",
        ]
        + (["--include-empty-inventories"] if include_empty else [])
    )

    assert return_code == 2
    assert "SENSE stopped" in capsys.readouterr().err


def test_absolute_match_threshold_retains_and_flags_bad_match(monkeypatch):
    activity_entry = {
        "ISO2": "AA",
        "PSILCA_mapping": "AA",
        "Activities": [
            {
                "code": "activity",
                "name": "Activity",
                "reference product": "Product",
                "unit": "kg",
                "price": 1.0,
            }
        ],
    }

    def fake_load(_directory, name):
        if name.startswith("ecoinvent_"):
            return np.array([[1.0, 0.0]]), pd.Series(["activity"]), {}
        return np.array([[0.0, 1.0]]), pd.Series(["AA::Unrelated sector"]), {}

    monkeypatch.setattr(matching, "load_embeddings", fake_load)
    flagged = matching.match_from_stores(
        "ecoinvent",
        "psilca",
        "AA",
        activity_entry=activity_entry,
        minimum_similarity=0.5,
    )
    unflagged = matching.match_from_stores(
        "ecoinvent",
        "psilca",
        "AA",
        activity_entry=activity_entry,
        minimum_similarity=None,
    )

    assert len(flagged) == 1
    assert bool(flagged.iloc[0]["Below minimum similarity"])
    assert len(unflagged) == 1
    assert not bool(unflagged.iloc[0]["Below minimum similarity"])


def test_missing_mapped_embedding_store_is_fatal(monkeypatch):
    activity_entry = {
        "ISO2": "AA",
        "PSILCA_mapping": "AA",
        "Activities": [
            {
                "code": "activity",
                "name": "Activity",
                "reference product": "Product",
                "unit": "kg",
                "price": 1.0,
            }
        ],
    }

    def fake_load(_directory, name):
        if name.startswith("ecoinvent_"):
            return np.array([[1.0, 0.0]]), pd.Series(["activity"]), {}
        raise FileNotFoundError(name)

    monkeypatch.setattr(matching, "load_embeddings", fake_load)
    with pytest.raises(FileNotFoundError, match="mapped country 'AA'"):
        matching.match_from_stores(
            "ecoinvent",
            "psilca",
            "AA",
            activity_entry=activity_entry,
        )


def test_missing_matched_psilca_exchanges_are_fatal():
    mapped = pd.DataFrame(
        [
            {
                "ISO2": "AA",
                "Activity code": "activity",
                "PSILCA_country": "AA",
                "Matched_sector": "Missing sector",
            }
        ]
    )

    with pytest.raises(ValueError, match="missing 1 matched country-sector"):
        matching.add_social_exchanges(mapped, {})


@pytest.mark.parametrize(
    "kwargs",
    [
        {"update_tol": float("nan")},
        {"update_tol": float("inf")},
        {"embedding_batch_size": 1.5},
        {"rounding_digits": 1.5},
        {"productivity_year": 2020.5},
        {"psilca_price_year": True},
    ],
)
def test_invalid_configuration_stops_at_construction(kwargs):
    with pytest.raises(ValueError):
        SenseConfig(**kwargs)


def test_large_diagnostic_export_serializes_in_bounded_chunks(tmp_path, monkeypatch):
    import gzip
    import json

    import pandas as pd

    from sense.diagnostics import save_intermediate_artifacts

    frame = pd.DataFrame(
        {
            "code": list(range(150)),
            "text": ['ü; quoted "text"'] * 150,
            "PSILCA_Exchanges": [[{"meanValue": 0.123456789123, "risk": "high"}]] * 150,
            "missing": [float("nan")] * 150,
        }
    )
    original = pd.DataFrame.to_json
    expected = json.loads(original(frame, orient="records", force_ascii=False))
    chunk_sizes = []

    def checked(self, *args, **kwargs):
        chunk_sizes.append(len(self))
        assert len(self) <= 64
        return original(self, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_json", checked)
    save_intermediate_artifacts("RoW", frame, [], str(tmp_path), "test")
    path = tmp_path / "Sector mappings/Raw/test/matched_RoW_prices_w_social_exchanges.json.gz"
    assert json.loads(gzip.decompress(path.read_bytes())) == expected
    assert chunk_sizes == [64, 64, 22]
    save_intermediate_artifacts("Empty", frame.iloc[:0], [], str(tmp_path), "test")
    assert (
        json.loads(
            gzip.decompress(
                (path.parent / "matched_Empty_prices_w_social_exchanges.json.gz").read_bytes()
            )
        )
        == []
    )
