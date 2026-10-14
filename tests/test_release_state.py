"""Release-boundary regressions with entirely invented inventories and records."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from sense import SenseConfig
from sense.data.embeddings import model_revision


@pytest.fixture
def inventory():
    import bw2data as bd

    previous = bd.projects.current
    bd.projects.set_current("release-state-regressions")
    for name in ("inventory", "background", "social", "other-social"):
        if name in bd.databases:
            del bd.databases[name]
    for name in ("social", "other-social"):
        bd.Database(name).write(
            {
                (name, "hours"): {
                    "name": "Invented working hours",
                    "unit": "hour",
                    "type": "emission",
                    "sense_managed": True,
                    "sense_kind": "working_hours",
                    "exchanges": [],
                }
            }
        )
    bd.Database("background").write(
        {
            ("background", "b"): {
                "name": "Invented background production",
                "reference product": "Invented B",
                "unit": "kg",
                "location": "DE",
                "exchanges": [{"input": ("background", "b"), "amount": 1, "type": "production"}],
            }
        }
    )
    bd.Database("inventory").write(
        {
            ("inventory", "a"): {
                "name": "Invented production",
                "reference product": "Invented A",
                "unit": "kg",
                "location": "DE",
                "exchanges": [
                    {
                        "input": ("inventory", "a"),
                        "amount": 1,
                        "type": "production",
                        "properties": {"price": {"amount": 4, "unit": "EUR2005"}},
                    },
                    {"input": ("background", "b"), "amount": 2, "type": "technosphere"},
                    {"input": ("social", "hours"), "amount": 3, "type": "biosphere"},
                ],
            }
        }
    )
    method = bd.Method(("SENSE", "social", "Workers", "Invented", "risk"))
    method.register(unit="med risk hours")
    method.write([(("social", "hours"), 1)])
    yield bd
    bd.projects.set_current(previous)
    # Remove its registry entry now and let the session fixture clean the files.
    bd.projects.delete_project("release-state-regressions", delete_dir=False)


@pytest.mark.parametrize(
    "change", ["price", "technosphere", "social", "flow", "method", "dependency"]
)
def test_completed_state_rejects_changed_calculation_inputs(inventory, change):
    from sense.brightway.state import capture_database_state, validate_database_state

    bd = inventory
    recorded = capture_database_state("inventory", "social")
    validate_database_state(recorded)
    activity = bd.get_activity(("inventory", "a"))
    if change == "price":
        exchange = next(iter(activity.production()))
        exchange["properties"] = {"price": {"amount": 40, "unit": "EUR2005"}}
        exchange.save()
    elif change in {"technosphere", "social"}:
        exchange = next(
            iter(activity.technosphere() if change == "technosphere" else activity.biosphere())
        )
        exchange["amount"] *= 2
        exchange.save()
    elif change == "flow":
        flow = bd.get_activity(("social", "hours"))
        flow["unit"] = "day"
        flow.save()
    elif change == "method":
        bd.Method(("SENSE", "social", "Workers", "Invented", "risk")).write(
            [(("social", "hours"), 10)]
        )
    else:
        background = bd.get_activity(("background", "b"))
        background["price"] = 7
        background.save()
    with pytest.raises(ValueError, match="changed after"):
        validate_database_state(recorded)


def test_other_social_layer_does_not_invalidate_completed_state(inventory):
    from sense.brightway.state import capture_database_state, validate_database_state

    bd = inventory
    recorded = capture_database_state("inventory", "social")
    bd.get_activity(("inventory", "a")).new_exchange(
        input=("other-social", "hours"), amount=15, type="biosphere"
    ).save()
    # Adding unused shared-layer flows/methods for a different mapped database is
    # also legitimate and must not invalidate this database's receipt.
    flow = bd.Database("social").new_activity(
        code="unused",
        name="Invented unrelated flow",
        unit="hour",
        type="emission",
        sense_managed=True,
        sense_kind="working_hours",
    )
    flow.save()
    method = bd.Method(("SENSE", "social", "Workers", "Invented", "risk"))
    method.write([(("social", "hours"), 1), (("social", "unused"), 1)])
    validate_database_state(recorded)
    assert capture_database_state("inventory", "social") == recorded


@pytest.mark.parametrize("location", [None, "", "  ", 17])
def test_unusable_locations_fail_before_parsing_or_exchange_changes(
    inventory, tmp_path, monkeypatch, location
):
    import sense.pipeline as pipeline

    bd = inventory
    bd.databases["inventory"]["searchable"] = False
    bd.databases.flush()
    activity = bd.get_activity(("inventory", "a"))
    activity["location"] = location
    activity.save()
    source = tmp_path / "xml"
    source.mkdir()
    (source / "invented.xml").write_text("<ecoSpold/>")
    workbook = tmp_path / "geographies.xlsx"
    workbook.write_bytes(b"not parsed before rejection")

    def forbidden(*args, **kwargs):
        raise AssertionError("PSILCA parsing preceded location validation")

    monkeypatch.setattr(pipeline, "_prepare_psilca_data", forbidden)
    with pytest.raises(ValueError, match="non-empty location"):
        pipeline.run_sense(
            project=bd.projects.current,
            technosphere_db="inventory",
            sociosphere_db="social",
            config=SenseConfig(country_weighting="equal", ecoinvent_to_psilca_price_factor=1),
            psilca_path=source,
            ecoinvent_version="3.12",
            psilca_version="4",
            data_dir=tmp_path,
            ecoinvent_geographies_excel=workbook,
        )
    assert [exchange["amount"] for exchange in activity.biosphere()] == [3]


def test_missing_location_key_is_rejected(inventory, tmp_path, monkeypatch):
    import sense.pipeline as pipeline

    activity = inventory.get_activity(("inventory", "a"))
    data = dict(activity)
    data.pop("location")

    # The early validator works on the fetched inventory, not the price cache.
    class MissingLocation(dict):
        key = activity.key

    monkeypatch.setattr(
        pipeline, "_load_technosphere_activities", lambda _: [MissingLocation(data)]
    )
    source = tmp_path / "xml"
    source.mkdir()
    (source / "invented.xml").write_text("<ecoSpold/>")
    workbook = tmp_path / "geographies.xlsx"
    workbook.write_bytes(b"not parsed")
    with pytest.raises(ValueError, match="non-empty location"):
        pipeline.run_sense(
            project=inventory.projects.current,
            technosphere_db="inventory",
            sociosphere_db="social",
            config=SenseConfig(country_weighting="equal", ecoinvent_to_psilca_price_factor=1),
            psilca_path=source,
            ecoinvent_version="3.12",
            psilca_version="4",
            data_dir=tmp_path,
            ecoinvent_geographies_excel=workbook,
        )
    assert next(iter(activity.biosphere()))["amount"] == 3


@pytest.mark.parametrize(
    "name",
    [
        "../model",
        "/tmp/model",
        "C:\\model",
        "~/model",
        "BAAI/../model",
        "https://example.org/model",
    ],
)
def test_model_paths_rejected_before_dependency_import(monkeypatch, name):
    from sense.data.embeddings import load_encoder

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(
            SentenceTransformer=lambda *args, **kwargs: pytest.fail(
                "unsafe model passed to dependency"
            )
        ),
    )
    with pytest.raises(ValueError, match="identifier"):
        load_encoder(name, revision="a" * 40)


def test_local_directory_cannot_shadow_hub_identifier(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "BAAI/bge-base-en-v1.5").mkdir(parents=True)
    with pytest.raises(ValueError, match="local model path"):
        model_revision("BAAI/bge-base-en-v1.5")


def test_custom_encoder_requires_immutable_revision():
    with pytest.raises(ValueError, match="immutable"):
        SenseConfig(embedding_model="invented/model")
    with pytest.raises(ValueError, match="immutable"):
        SenseConfig(embedding_revision="main")
    assert model_revision("invented/model", "a" * 40) == "a" * 40


@pytest.mark.parametrize("kind", ["psilca", "ecoinvent"])
def test_embedding_revision_change_and_legacy_manifest_rebuild_stores(tmp_path, monkeypatch, kind):
    from sense.data.ecoinvent import embed_ecoinvent_to_store
    from sense.data.psilca import embed_psilca_to_store

    calls = []

    class Encoder:
        def __init__(self, *args, **kwargs):
            calls.append(kwargs["revision"])

        def encode(self, texts, **kwargs):
            return np.ones((len(texts), 2), dtype=np.float32)

    monkeypatch.setitem(
        sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=Encoder)
    )
    source = tmp_path / "invented.json"
    store = tmp_path / "store"
    if kind == "psilca":
        source.write_text(json.dumps({"DE": ["Invented sector"]}))

        def embed(revision):
            embed_psilca_to_store(str(source), str(store), revision=revision)

    else:
        source.write_text(
            json.dumps(
                [
                    {
                        "ISO2": "DE",
                        "Activities": [
                            {
                                "code": "a",
                                "name": "Invented production",
                                "reference product": "Invented A",
                            }
                        ],
                    }
                ]
            )
        )

        def embed(revision):
            embed_ecoinvent_to_store(
                str(source), str(store), target_iso2_list=["DE"], revision=revision
            )

    embed("a" * 40)
    embed("a" * 40)
    assert calls == ["a" * 40]
    embed("b" * 40)
    assert calls == ["a" * 40, "b" * 40]
    manifest = next(store.glob("*.manifest.json"))
    data = json.loads(manifest.read_text())
    assert data["meta"].pop("revision") == "b" * 40
    manifest.write_text(json.dumps(data))
    embed("b" * 40)
    assert calls == ["a" * 40, "b" * 40, "b" * 40]


def test_overflow_is_not_returned_as_missing_data():
    from sense.core.averaging import average_sector_mappings_w_prices

    frame = pd.DataFrame(
        [
            {
                "Activity code": "invented",
                "Price": 1e308,
                "PSILCA_country": "DE",
                "Matched_sector": "Invented",
                "PSILCA_Exchanges": [
                    {"indicator": "Invented risk", "meanValue": 10, "risk level": "medium"}
                ],
            }
        ]
    )
    with pytest.raises(ValueError, match="Non-finite computed"):
        average_sector_mappings_w_prices(frame, level_weights={"medium": 1})


def test_score_reuse_rejects_corruption_and_changed_demands(inventory, tmp_path, monkeypatch):
    from case_studies.hydrogen.workflow import regionalization as module

    bd = inventory
    setup = {
        "inv": [{("inventory", "a"): 1}],
        "ia": [("SENSE", "social", "Workers", "Invented", "risk")],
    }
    path = module.process_func_units_and_lca(setup, "test", tmp_path)
    assert module.valid_score_file(path)
    assert pd.read_csv(path).iloc[0, -1] == pytest.approx(3)
    original = path.read_bytes()

    def forbidden(*args, **kwargs):
        raise AssertionError("valid cached scores recalculated")

    with monkeypatch.context() as patch:
        patch.setattr(module, "LCA", forbidden)
        assert module.process_func_units_and_lca(setup, "test", tmp_path) == path
        changed = {**setup, "inv": [{("inventory", "a"): 2}]}
        with pytest.raises(AssertionError, match="recalculated"):
            module.process_func_units_and_lca(changed, "test", tmp_path)
    path.write_text("truncated\n")
    assert not module.valid_score_file(path)
    module.process_func_units_and_lca(setup, "test", tmp_path)
    assert path.read_bytes() == original
    table = pd.read_csv(path)
    table.iloc[0, -1] = 999
    table.to_csv(path, index=False)
    assert not module.valid_score_file(path)
    assert bd.projects.current == "release-state-regressions"


def test_adjoint_scores_match_forward_lca(inventory, tmp_path):
    from bw2calc import LCA

    from case_studies.hydrogen.workflow import regionalization as module

    bd = inventory
    bd.Database("adjoint-test").write(
        {
            ("adjoint-test", str(i)): {
                "name": f"Invented chain {i}",
                "reference product": f"Product {i}",
                "location": "DE",
                "unit": "kg",
                "exchanges": [
                    {"input": ("adjoint-test", str(i)), "type": "production", "amount": 2},
                    {"input": ("social", "hours"), "type": "biosphere", "amount": i - 3},
                    {
                        "input": ("adjoint-test", str((i + 1) % 30)),
                        "type": "technosphere",
                        "amount": 0.1,
                    },
                ],
            }
            for i in range(30)
        }
    )
    negative = ("Invented", "negative factors")
    bd.Method(negative).register(unit="h")
    bd.Method(negative).write([(("social", "hours"), -0.25)])
    methods = [("SENSE", "social", "Workers", "Invented", "risk"), negative]
    demands = [{("adjoint-test", str(i)): i + 0.5} for i in range(30)]
    demands += [{("inventory", "a"): -2, ("adjoint-test", "0"): 3}]
    expected = []
    lca = LCA({key: 1 for demand in demands for key in demand})
    lca.lci(factorize=True)
    for demand in demands:
        lca.redo_lci(demand)
        row = []
        for method in methods:
            lca.switch_method(method)
            lca.lcia_calculation()
            row.append(lca.score)
        expected.append(row)
    try:
        path = module.process_func_units_and_lca(
            {"inv": demands, "ia": methods}, "adjoint", tmp_path
        )
        actual = pd.read_csv(path).iloc[:, len(module.SCORE_METADATA) :].to_numpy()
        np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-12)
    finally:
        del bd.databases["adjoint-test"]
        bd.Method(negative).deregister()


def test_unchanged_state_does_not_scan_database_tables(inventory, monkeypatch):
    from bw2data.backends.peewee import ExchangeDataset

    from sense.brightway.state import capture_database_state

    recorded = capture_database_state("inventory", "social")
    db = ExchangeDataset._meta.database
    original = db.execute_sql

    def checked(sql, *args, **kwargs):
        if "FROM activitydataset" in sql or "FROM exchangedataset" in sql:
            raise AssertionError("Unchanged state unexpectedly scanned database tables")
        return original(sql, *args, **kwargs)

    monkeypatch.setattr(db, "execute_sql", checked)
    assert capture_database_state("inventory", "social") == recorded


def test_state_rejects_external_sql_changes_without_modified_metadata(inventory):
    import pickle
    import sqlite3
    from contextlib import closing

    from bw2data.backends.peewee import ExchangeDataset

    from sense.brightway.state import capture_database_state, validate_database_state

    bd = inventory
    recorded = capture_database_state("inventory", "social")
    modified = bd.databases["inventory"].get("modified")
    with closing(sqlite3.connect(ExchangeDataset._meta.database.database)) as connection:
        identifier, payload = connection.execute(
            "SELECT id, data FROM exchangedataset WHERE output_database = ? AND type = ?",
            ("inventory", "technosphere"),
        ).fetchone()
        data = pickle.loads(payload)
        data["amount"] *= 2
        connection.execute(
            "UPDATE exchangedataset SET data = ? WHERE id = ?", (pickle.dumps(data), identifier)
        )
        connection.commit()
    assert bd.databases["inventory"].get("modified") == modified
    with pytest.raises(ValueError, match="changed after"):
        validate_database_state(recorded)


def test_new_foreground_reuses_verified_upstream_content(inventory, monkeypatch):
    import sense.brightway.state as state

    bd = inventory
    state.capture_database_state("inventory", "social")
    bd.Database("additional").write(
        {
            ("additional", "a"): {
                "name": "Additional foreground",
                "reference product": "product",
                "unit": "kg",
                "location": "DE",
                "exchanges": [
                    {"input": ("additional", "a"), "amount": 1, "type": "production"},
                    {"input": ("background", "b"), "amount": 2, "type": "technosphere"},
                ],
            }
        }
    )
    original = state.loads

    def checked(payload):
        value = original(payload)
        if value.get("name") == "Invented background production":
            raise AssertionError("Unchanged upstream content decoded again")
        return value

    monkeypatch.setattr(state, "loads", checked)
    captured = state.capture_database_state("additional", "social")
    assert captured["inventories"]["background"]["activities"] == 1
    del bd.databases["additional"]


def test_unchanged_working_hours_method_is_not_rewritten(inventory, monkeypatch):
    from sense.brightway.methods import add_working_hours_impact_category

    bd = inventory
    add_working_hours_impact_category("social")
    original = bd.Method.write
    writes = []

    def counted(self, data, *args, **kwargs):
        writes.append(self.name)
        return original(self, data, *args, **kwargs)

    monkeypatch.setattr(bd.Method, "write", counted)
    add_working_hours_impact_category("social")
    assert not writes
    flow = bd.Database("social").new_activity(
        "second",
        name="Second hours",
        unit="hour",
        type="emission",
        sense_managed=True,
        sense_kind="working_hours",
    )
    flow.save()
    add_working_hours_impact_category("social")
    assert len(writes) == 1
    # Brightway iterates flows in random order. CF ordering must not trigger writes.
    for _ in range(5):
        add_working_hours_impact_category("social")
    assert len(writes) == 1
