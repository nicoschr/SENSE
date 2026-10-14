"""Regression tests for release-critical SENSE invariants."""

from __future__ import annotations

import shutil
import uuid
from types import SimpleNamespace

import pandas as pd
import pytest

import sense.brightway.methods as methods_module
from sense.brightway import repository
from sense.brightway.flows import (
    _attach_exchanges,
    add_social_biosphere_flows_and_exchanges,
    add_working_hours_flows_and_exchanges,
)
from sense.brightway.methods import (
    add_social_impact_categories,
    add_working_hours_impact_categories_by_isic,
    add_working_hours_impact_category,
)
from sense.core.averaging import average_sector_mappings_w_prices
from sense.data.ecoinvent import fetch_ecoinvent_activities
from sense.data.price_inference import ProductGraphCache, build_product_graph, run_product_inference
from sense.data.prices import build_brightway_price_catalog
from sense.data.seed_prices import build_minimal_seed_worklist


def _exchange(
    indicator: str,
    hours: float,
    *,
    exchange_type: str = "risk",
) -> dict:
    return {
        "indicator": indicator,
        "risk level": "medium",
        "type": exchange_type,
        "unit": "h",
        "category": "Workers",
        "subcategory": "Example",
        "meanValue": hours,
    }


def _row(sector: str, exchanges: list[dict], *, price: float = 1.0) -> dict:
    return {
        "Activity code": "activity-1",
        "Activity name": "Example activity",
        "Reference product": "example product",
        "ISIC": "10",
        "Unit": "kg",
        "ISO2": "AA",
        "Price": price,
        "PSILCA_country": "AA",
        "Matched_sector": sector,
        "PSILCA_Exchanges": exchanges,
    }


def test_working_hours_are_averaged_once_per_sector_not_per_indicator():
    enriched = pd.DataFrame(
        [
            _row("sector-1", [_exchange("indicator-1", 1), _exchange("indicator-2", 1)]),
            _row("sector-2", [_exchange("indicator-1", 9)]),
        ]
    )

    result = average_sector_mappings_w_prices(
        enriched,
        level_weights={"medium": 1.0, "no data": 0.1},
        mean_method="arithmetic",
    )[0]

    assert result["Working_hours"] == pytest.approx(5.0)


def test_missing_risk_level_is_retained_as_structured_input_diagnostic():
    exchange = _exchange("indicator", 1)
    exchange["risk level"] = ""
    result = average_sector_mappings_w_prices(
        pd.DataFrame([_row("sector-1", [exchange])]),
        level_weights={"medium": 1.0, "no data": 0.1},
    )[0]

    assert result["Input_diagnostics"] == {"risk_level_missing_or_unknown": 1}
    assert result["Averaged_Exchanges"][0]["meanValue"] == pytest.approx(0.1)


def test_inconsistent_sector_worker_hour_intensities_fail_safely():
    enriched = pd.DataFrame(
        [_row("sector-1", [_exchange("indicator-1", 1), _exchange("indicator-2", 2)])]
    )

    with pytest.raises(ValueError, match="Inconsistent PSILCA worker-hour intensities"):
        average_sector_mappings_w_prices(
            enriched,
            level_weights={"medium": 1.0, "no data": 0.1},
        )


def test_risk_and_opportunity_types_remain_distinct():
    enriched = pd.DataFrame(
        [
            _row(
                "sector-1",
                [
                    _exchange("same indicator", 1, exchange_type="risk"),
                    _exchange("same indicator", 1, exchange_type="opportunity"),
                ],
            )
        ]
    )

    result = average_sector_mappings_w_prices(
        enriched,
        level_weights={"medium": 1.0, "no data": 0.1},
        mean_method="arithmetic",
    )[0]

    assert {exchange["type"] for exchange in result["Averaged_Exchanges"]} == {
        "risk",
        "opportunity",
    }


@pytest.mark.parametrize("price", [None, 0.0, -1.0, float("nan")])
def test_missing_or_non_positive_prices_fail_before_aggregation(price):
    enriched = pd.DataFrame([_row("sector-1", [_exchange("indicator", 1)], price=price)])

    with pytest.raises(ValueError, match="positive finite price"):
        average_sector_mappings_w_prices(
            enriched,
            level_weights={"medium": 1.0, "no data": 0.1},
        )


def _price_cache() -> ProductGraphCache:
    """Two location-specific activities producing the same named product."""

    return ProductGraphCache(
        activity_to_id={
            ("db", "producer-a"): 0,
            ("db", "producer-b"): 1,
            ("db", "input-a"): 2,
            ("db", "input-b"): 3,
            ("db", "primary"): 4,
        },
        id_to_product=[
            ("kg", "same product"),
            ("kg", "same product"),
            ("kg", "input a"),
            ("kg", "input b"),
            ("kg", "primary product"),
        ],
        id_to_meta=[
            {"location": "AA"},
            {"location": "BB"},
            {"location": "AA"},
            {"location": "BB"},
            {"location": "AA"},
        ],
        recipe_out=[0, 1, 4],
        recipe_inputs=[[(2, 1.0)], [(3, 1.0)], []],
        recipe_act_db=["db", "db", "db"],
        recipe_act_code=["producer-a", "producer-b", "primary"],
        recipe_act_name=["Producer A", "Producer B", "Primary"],
        recipe_act_location=["AA", "BB", "AA"],
        producers_by_pid=[[0], [1], [], [], [2]],
        pid_activity_price_seed={},
    )


def test_price_inference_preserves_activity_specific_products_without_zero_defaults():
    prices, _ = run_product_inference(
        _price_cache(),
        initial_prices={2: 2.0, 3: 5.0},
        initial_sources={2: "test", 3: "test"},
    )

    assert prices[0] == pytest.approx(2.0)
    assert prices[1] == pytest.approx(5.0)
    assert 4 not in prices


class _GraphExchange(dict):
    def __init__(self, input_activity, amount, *, properties=None):
        super().__init__(amount=amount, properties=properties or {})
        self.input = input_activity


class _GraphActivity(dict):
    def __init__(
        self,
        code,
        product,
        location,
        *,
        database="db",
        production=1.0,
        price=None,
        reference_price=None,
        price_unit="EUR2005",
    ):
        super().__init__(
            code=code,
            name=f"Activity {code}",
            price=price,
            **{
                "reference product": product,
                "unit": "kg",
                "location": location,
            },
        )
        self.key = (database, code)
        self._technosphere = []
        self._production_amount = production
        self._reference_price = reference_price
        self._price_unit = price_unit

    def technosphere(self):
        return list(self._technosphere)

    def production(self):
        properties = (
            {"price": {"amount": self._reference_price, "unit": self._price_unit}}
            if self._reference_price is not None
            else None
        )
        return [_GraphExchange(self, self._production_amount, properties=properties)]


def test_brightway_reference_prices_build_exact_and_product_catalogs():
    priced = _GraphActivity("priced", "Priced product", "AA", reference_price=2.0)
    second = _GraphActivity("second", "Second product", "AA", reference_price=4.0)
    zero = _GraphActivity("zero", "Waste product", "AA", reference_price=0.0)

    catalog = build_brightway_price_catalog(
        [priced, second, zero],
        source_database="db",
        price_currency="EUR",
        price_year=2005,
        price_floor_percentile=50.0,
    )

    assert catalog.by_code["priced"] == (("kg", "priced product"), 2.0)
    assert catalog.by_product[("kg", "second product")] == 4.0
    assert catalog.by_code["zero"] == (("kg", "waste product"), 3.0)


def test_brightway_reference_price_unit_must_match_configured_basis():
    activity = _GraphActivity(
        "priced",
        "Product",
        "AA",
        reference_price=2.0,
        price_unit="USD2015",
    )

    with pytest.raises(ValueError, match="expected EUR2005"):
        build_brightway_price_catalog(
            [activity],
            source_database="db",
            price_currency="EUR",
            price_year=2005,
            price_floor_percentile=20.0,
        )


def test_graph_reads_price_from_brightway_reference_exchange(monkeypatch):
    activity = _GraphActivity("priced", "Product", "AA", reference_price=2.5)
    monkeypatch.setattr("bw2data.Database", lambda _: [activity])

    graph = build_product_graph("db")

    assert graph.pid_activity_price_seed == {0: 2.5}


def test_graph_rejects_reference_price_in_another_monetary_basis(monkeypatch):
    activity = _GraphActivity(
        "priced",
        "Product",
        "AA",
        reference_price=2.5,
        price_unit="USD2015",
    )
    monkeypatch.setattr("bw2data.Database", lambda _: [activity])

    with pytest.raises(ValueError, match="expected EUR2005"):
        build_product_graph("db", expected_price_unit="EUR2005")


def test_regionalized_activity_reuses_base_brightway_price(monkeypatch, tmp_path):
    source = _GraphActivity(
        "shared-code",
        "Product",
        "AA",
        database="ecoinvent-3.12-cutoff",
        reference_price=2.5,
    )
    regionalized = _GraphActivity(
        "shared-code",
        "Product",
        "AA",
        database="ecoinvent-3.12-cutoff - regionalized",
    )
    databases = {
        "ecoinvent-3.12-cutoff": [source],
        "ecoinvent-3.12-cutoff - regionalized": [regionalized],
    }
    monkeypatch.setattr("bw2data.Database", lambda name: databases[name])

    result = fetch_ecoinvent_activities(
        exclude_empty_inventories=False,
        mapping_df=pd.DataFrame([{"Shortname": "AA", "PSILCA_mapping": "AA"}]),
        output_path=str(tmp_path / "activities.json"),
        technosphere="ecoinvent-3.12-cutoff - regionalized",
        price_source_database="ecoinvent-3.12-cutoff",
        price_floor_percentile=0.0,
        activities=[regionalized],
    )

    assert result[["code", "price"]].to_dict("records") == [{"code": "shared-code", "price": 2.5}]


def test_excluded_unpriced_activity_does_not_create_a_seed_request(monkeypatch, tmp_path):
    source = _GraphActivity(
        "shared-code",
        "Product",
        "AA",
        database="ecoinvent-3.12-cutoff",
        reference_price=2.5,
    )
    included = _GraphActivity(
        "shared-code",
        "Product",
        "AA",
        database="regionalized",
    )
    excluded = _GraphActivity(
        "excluded",
        "Unpriced custom product",
        "AA",
        database="regionalized",
    )
    excluded["name"] = "consumption market for unpriced custom product"
    databases = {
        "ecoinvent-3.12-cutoff": [source],
        "regionalized": [included, excluded],
    }
    monkeypatch.setattr("bw2data.Database", lambda name: databases[name])
    monkeypatch.setattr(
        "sense.data.ecoinvent.build_product_graph",
        lambda **_: pytest.fail("excluded activity was incorrectly used as a graph root"),
    )

    result = fetch_ecoinvent_activities(
        exclude_empty_inventories=False,
        mapping_df=pd.DataFrame([{"Shortname": "AA", "PSILCA_mapping": "AA"}]),
        output_path=str(tmp_path / "activities.json"),
        technosphere="regionalized",
        price_source_database="ecoinvent-3.12-cutoff",
        exclude_terms=["consumption market for"],
        auto_seed_prices_path=str(tmp_path / "seeds.json"),
        activities=[included, excluded],
    )

    assert result["code"].tolist() == ["shared-code"]
    assert not (tmp_path / "seeds.json").exists()


def test_graph_uses_exact_supplier_keys_and_normalizes_reference_output(monkeypatch):
    input_a = _GraphActivity("input-a", "input", "AA")
    input_b = _GraphActivity("input-b", "input", "BB")
    producer_a = _GraphActivity("producer-a", "same product", "AA", production=2.0)
    producer_b = _GraphActivity("producer-b", "same product", "BB")
    producer_a._technosphere = [_GraphExchange(input_a, 4.0)]
    producer_b._technosphere = [_GraphExchange(input_b, 3.0)]
    activities = [producer_a, producer_b, input_a, input_b]

    monkeypatch.setattr("bw2data.Database", lambda _: activities)
    graph = build_product_graph("db")

    producer_a_pid = graph.activity_to_id[("db", "producer-a")]
    producer_b_pid = graph.activity_to_id[("db", "producer-b")]
    input_a_pid = graph.activity_to_id[("db", "input-a")]
    input_b_pid = graph.activity_to_id[("db", "input-b")]
    assert producer_a_pid != producer_b_pid
    assert graph.id_to_product[producer_a_pid] == graph.id_to_product[producer_b_pid]

    recipe_a = graph.producers_by_pid[producer_a_pid][0]
    recipe_b = graph.producers_by_pid[producer_b_pid][0]
    assert graph.recipe_inputs[recipe_a] == [(input_a_pid, 2.0)]
    assert graph.recipe_inputs[recipe_b] == [(input_b_pid, 3.0)]


def test_graph_normalizes_negative_waste_reference_output(monkeypatch):
    input_activity = _GraphActivity("input", "input", "AA")
    waste_treatment = _GraphActivity("treatment", "waste", "AA", production=-2.0)
    waste_treatment._technosphere = [_GraphExchange(input_activity, 4.0)]

    monkeypatch.setattr("bw2data.Database", lambda _: [waste_treatment, input_activity])
    graph = build_product_graph("db")

    treatment_pid = graph.activity_to_id[("db", "treatment")]
    input_pid = graph.activity_to_id[("db", "input")]
    recipe = graph.producers_by_pid[treatment_pid][0]
    assert graph.recipe_inputs[recipe] == [(input_pid, 2.0)]


def test_graph_excludes_negative_outputs_from_gross_cost_inference(monkeypatch):
    purchased_input = _GraphActivity("input", "input", "AA")
    waste_output = _GraphActivity("waste", "waste", "AA")
    producer = _GraphActivity("producer", "product", "AA")
    producer._technosphere = [
        _GraphExchange(purchased_input, 3.0),
        _GraphExchange(waste_output, -10.0),
    ]

    monkeypatch.setattr("bw2data.Database", lambda _: [producer])
    graph = build_product_graph("db")

    producer_pid = graph.activity_to_id[("db", "producer")]
    input_pid = graph.activity_to_id[("db", "input")]
    recipe = graph.producers_by_pid[producer_pid][0]
    assert graph.recipe_inputs[recipe] == [(input_pid, 3.0)]


def test_graph_expands_only_unpriced_linked_database_activities(monkeypatch):
    priced_background = _GraphActivity(
        "background", "background product", "AA", database="background", price=2.0
    )
    linked_foreground = _GraphActivity(
        "linked", "linked product", "AA", database="linked-foreground"
    )
    target = _GraphActivity("target", "target product", "AA", database="target")
    linked_foreground._technosphere = [_GraphExchange(priced_background, 3.0)]
    target._technosphere = [_GraphExchange(linked_foreground, 4.0)]

    monkeypatch.setattr("bw2data.Database", lambda _: [target])
    graph = build_product_graph(
        "target",
        should_expand_external=lambda activity: activity.get("price") is None,
    )
    prices, _ = run_product_inference(
        graph,
        initial_prices=graph.pid_activity_price_seed,
        initial_sources=dict.fromkeys(graph.pid_activity_price_seed, "activity_price"),
    )

    linked_pid = graph.activity_to_id[("linked-foreground", "linked")]
    target_pid = graph.activity_to_id[("target", "target")]
    assert len(graph.recipe_out) == 2
    assert prices[linked_pid] == pytest.approx(6.0)
    assert prices[target_pid] == pytest.approx(24.0)


def test_seed_worklist_does_not_request_a_product_downstream_of_missing_input():
    cache = ProductGraphCache(
        activity_to_id={
            ("foreground", "output"): 0,
            ("external", "missing"): 1,
        },
        id_to_product=[("kg", "output"), ("kg", "missing input")],
        id_to_meta=[
            {"reference product": "output", "location": "AA"},
            {"reference product": "missing input", "location": "AA"},
        ],
        recipe_out=[0],
        recipe_inputs=[[(1, 1.0)]],
        recipe_act_db=["foreground"],
        recipe_act_code=["output"],
        recipe_act_name=["Output"],
        recipe_act_location=["AA"],
        producers_by_pid=[[0], []],
        pid_activity_price_seed={},
    )

    worklist = build_minimal_seed_worklist(cache, priced={})

    assert worklist[["reference_product", "seed_type"]].to_dict("records") == [
        {"reference_product": "missing input", "seed_type": "external_missing"}
    ]


class _Flow(dict):
    def __init__(self, key, *, managed):
        super().__init__(managed=managed)
        self.key = key


class _Exchange(dict):
    def __init__(self, flow, amount):
        super().__init__(type="biosphere", amount=amount)
        self.input = flow
        self.saved = False
        self.deleted = False

    def save(self):
        self.saved = True

    def delete(self):
        self.deleted = True


class _Activity:
    def __init__(self, exchanges):
        self._exchanges = exchanges
        self.created = []

    def exchanges(self):
        return list(self._exchanges)

    def new_exchange(self, *, input, amount, type):
        flow = _Flow(input, managed=True)
        exchange = _Exchange(flow, amount)
        exchange["type"] = type
        self.created.append(exchange)
        return exchange


def test_exchange_sync_updates_adds_and_removes_only_managed_flows():
    current_flow = _Flow(("bio", "current"), managed=True)
    stale_flow = _Flow(("bio", "stale"), managed=True)
    user_flow = _Flow(("bio", "user"), managed=False)
    current = _Exchange(current_flow, 1.0)
    stale = _Exchange(stale_flow, 7.0)
    user = _Exchange(user_flow, 9.0)
    activity = _Activity([current, stale, user])
    new_flow = _Flow(("bio", "new"), managed=True)

    added, updated, removed = _attach_exchanges(
        {
            "activity": [
                {"flow": current_flow, "amount": 2.0},
                {"flow": new_flow, "amount": 3.0},
            ]
        },
        {"activity": activity},
        resolve_flow=lambda row: (row["flow"], row["amount"]),
        managed_flow=lambda flow: bool(flow.get("managed")),
    )

    assert (added, updated, removed) == (1, 1, 1)
    assert current["amount"] == pytest.approx(2.0)
    assert current.saved is True
    assert stale.deleted is True
    assert user.deleted is False
    assert activity.created[0].saved is True


def _discard_temporary_project(projects, project_name: str, temporary_root) -> None:
    """Remove a throwaway Brightway project created under ``_use_temp_directory``.

    Switch to another project first so bw2data releases the SQLite connections of the
    one being deleted: POSIX allows unlinking open files, Windows does not. Delete the
    registration only, leaving the directory to the ``ignore_errors`` cleanup below,
    because ``delete_project(delete_dir=True)`` calls ``shutil.rmtree`` unguarded and
    raises ``PermissionError: [WinError 32]`` if a handle has not been released yet.
    ``temporary_root`` is a throwaway directory, so a residual file is harmless.
    """

    projects.set_current("default")
    projects.delete_project(project_name, delete_dir=False)
    projects._restore_orig_directory()
    shutil.rmtree(temporary_root, ignore_errors=True)


def test_exchange_sync_is_transactional_and_idempotent_in_brightway():
    """Exercise the supported Brightway 3 SQLite adapter, not test doubles."""

    from bw2data import Database, projects

    temporary_root = projects._use_temp_directory()
    project_name = f"sense-test-{uuid.uuid4().hex}"
    projects.set_current(project_name)
    try:
        biosphere = Database("sense-test-biosphere")
        biosphere.write(
            {
                (biosphere.name, "current"): {
                    "name": "current",
                    "unit": "h",
                    "categories": ("Social", "Test"),
                    "type": "emission",
                },
                (biosphere.name, "new"): {
                    "name": "new",
                    "unit": "h",
                    "categories": ("Social", "Test"),
                    "type": "emission",
                },
                (biosphere.name, "stale"): {
                    "name": "stale",
                    "unit": "h",
                    "categories": ("Social", "Test"),
                    "type": "emission",
                },
                (biosphere.name, "user"): {
                    "name": "user",
                    "unit": "kg",
                    "categories": ("Natural resource",),
                    "type": "natural resource",
                },
            }
        )
        technosphere = Database("sense-test-technosphere")
        technosphere.write(
            {
                (technosphere.name, "activity"): {
                    "name": "Activity",
                    "reference product": "Product",
                    "unit": "kg",
                    "location": "GLO",
                    "type": "process",
                    "exchanges": [
                        {
                            "input": (technosphere.name, "activity"),
                            "amount": 1.0,
                            "type": "production",
                        },
                        {
                            "input": (biosphere.name, "current"),
                            "amount": 1.0,
                            "type": "biosphere",
                        },
                        {
                            "input": (biosphere.name, "stale"),
                            "amount": 8.0,
                            "type": "biosphere",
                        },
                        {
                            "input": (biosphere.name, "user"),
                            "amount": 9.0,
                            "type": "biosphere",
                        },
                    ],
                }
            }
        )
        activity = technosphere.get("activity")
        current = biosphere.get("current")
        new = biosphere.get("new")
        managed = {current.key, new.key, biosphere.get("stale").key}
        arguments = {
            "grouped": {
                "activity": [
                    {"flow": current, "amount": 2.0},
                    {"flow": new, "amount": 3.0},
                ]
            },
            "techno_lookup": {"activity": activity},
            "resolve_flow": lambda row: (row["flow"], row["amount"]),
            "managed_flow": lambda flow: flow.key in managed,
            "managed_flow_keys": managed,
        }

        assert _attach_exchanges(**arguments) == (1, 1, 1)
        amounts = {
            exchange.input.key: exchange["amount"]
            for exchange in technosphere.get("activity").biosphere()
        }
        assert amounts == {
            current.key: pytest.approx(2.0),
            new.key: pytest.approx(3.0),
            biosphere.get("user").key: pytest.approx(9.0),
        }
        assert _attach_exchanges(**arguments) == (0, 0, 0)
    finally:
        _discard_temporary_project(projects, project_name, temporary_root)


def test_social_flow_creation_and_type_separation_in_real_brightway():
    """Cover flow creation plus transactional exchange writing end to end."""

    from bw2data import Database, projects

    LCA = pytest.importorskip("bw2calc").LCA
    temporary_root = projects._use_temp_directory()
    project_name = f"sense-test-{uuid.uuid4().hex}"
    projects.set_current(project_name)
    try:
        technosphere = Database("technosphere")
        technosphere.write(
            {
                (technosphere.name, "activity"): {
                    "name": "Activity",
                    "reference product": "Product",
                    "unit": "kg",
                    "location": "AA",
                    "classifications": [("ISIC rev.4 ecoinvent", "20: Chemicals")],
                    "type": "process",
                    "exchanges": [
                        {
                            "input": (technosphere.name, "activity"),
                            "amount": 1.0,
                            "type": "production",
                        }
                    ],
                }
            }
        )
        sociosphere = Database("sociosphere")
        sociosphere.write({})
        records = [
            {
                "Activity code": "activity",
                "ISO2": "AA",
                "Working_hours": 4.0,
                "Averaged_Exchanges": [
                    {
                        "indicator": "Shared",
                        "type": "risk",
                        "unit": "med risk hours",
                        "category": "Workers",
                        "subcategory": "Conditions",
                        "meanValue": 2.0,
                    },
                    {
                        "indicator": "Shared",
                        "type": "opportunity",
                        "unit": "med opportunity hours",
                        "category": "Workers",
                        "subcategory": "Conditions",
                        "meanValue": 3.0,
                    },
                ],
            }
        ]

        first = add_social_biosphere_flows_and_exchanges(
            records,
            technosphere.name,
            sociosphere.name,
        )
        assert first["added"] == 2
        assert {flow.get("sense_kind") for flow in sociosphere} == {"social"}
        exchanges = list(technosphere.get("activity").biosphere())
        assert {exchange.input.get("name") for exchange in exchanges} == {
            "Shared (risk), AA",
            "Shared (opportunity), AA",
        }

        working = add_working_hours_flows_and_exchanges(
            records,
            technosphere.name,
            sociosphere.name,
            add_isic_flows=True,
        )
        assert working["added"] == 1
        assert working["isic_added"] == 1
        assert {flow.get("sense_kind") for flow in sociosphere} == {
            "social",
            "working_hours",
            "working_hours_isic",
        }

        add_social_impact_categories(sociosphere.name)
        add_working_hours_impact_category(sociosphere.name)
        add_working_hours_impact_categories_by_isic(sociosphere.name)
        lca = LCA(
            {technosphere.get("activity"): 1},
            ("SENSE", "sociosphere", "Workers", "Conditions", "Shared", "risk"),
        )
        lca.lci()
        lca.lcia()
        assert lca.score == pytest.approx(2.0)
        assert (
            "SENSE",
            "sociosphere",
            "Workers",
            "Conditions",
            "Shared",
            "risk",
        ) in methods_module.methods
        assert (
            "SENSE",
            "sociosphere",
            "Workers",
            "Conditions",
            "Shared",
            "opportunity",
        ) in methods_module.methods
        assert (
            "SENSE",
            "sociosphere",
            "Social",
            "Working hours",
            "working hours",
        ) in methods_module.methods
        assert (
            "SENSE",
            "sociosphere",
            "Social",
            "Working hours by ISIC",
            "ISIC 20",
        ) in methods_module.methods

        add_social_impact_categories(sociosphere.name)
        add_working_hours_impact_category(sociosphere.name)
        add_working_hours_impact_categories_by_isic(sociosphere.name)
        other = Database("other-sociosphere")
        other.write({})
        add_social_biosphere_flows_and_exchanges(records, technosphere.name, other.name)
        add_social_impact_categories(other.name)
        assert methods_module.remove_impact_categories(sociosphere.name, "social") == 2
        assert (
            "SENSE",
            other.name,
            "Workers",
            "Conditions",
            "Shared",
            "risk",
        ) in methods_module.methods
        assert not any(
            k[:2] == ("SENSE", sociosphere.name) and k[-1] == "risk" for k in methods_module.methods
        )

        second = add_social_biosphere_flows_and_exchanges(
            records,
            technosphere.name,
            sociosphere.name,
        )
        assert (second["added"], second["updated"], second["removed"]) == (0, 0, 0)
    finally:
        _discard_temporary_project(projects, project_name, temporary_root)


def test_social_methods_keep_category_and_assessment_type_separate(monkeypatch):
    marker = "Imported from PSILCA sector matching (country-resolved, typed)"
    flows = [
        _Flow(("bio", "workers-risk"), managed=True),
        _Flow(("bio", "workers-opportunity"), managed=True),
        _Flow(("bio", "community-risk"), managed=True),
    ]
    flows[0].update(
        name="Shared indicator (risk), AA",
        unit="med risk hours",
        categories=("Workers", "Conditions"),
        comment=marker,
        sense_kind="social",
    )
    flows[1].update(
        name="Shared indicator (opportunity), AA",
        unit="med opportunity hours",
        categories=("Workers", "Conditions"),
        comment=marker,
        sense_kind="social",
    )
    flows[2].update(
        name="Shared indicator (risk), AA",
        unit="med risk hours",
        categories=("Local community", "Conditions"),
        comment=marker,
        sense_kind="social",
    )

    method_metadata = {
        ("Old", "SENSE", "method"): {
            "description": "Auto-generated from SENSE social flows in sociosphere"
        }
    }
    written = {}

    class FakeMethod:
        def __init__(self, key):
            self.key = key

        def deregister(self):
            method_metadata.pop(self.key, None)
            written.pop(self.key, None)

        def register(self, **metadata):
            method_metadata[self.key] = metadata

        def write(self, cfs):
            written[self.key] = cfs

    monkeypatch.setattr(methods_module, "Database", lambda _: flows)
    monkeypatch.setattr(methods_module, "Method", FakeMethod)
    monkeypatch.setattr(methods_module, "methods", method_metadata)
    methods_module.add_social_impact_categories("sociosphere")

    workers_risk = ("SENSE", "sociosphere", "Workers", "Conditions", "Shared indicator", "risk")
    workers_opportunity = (
        "SENSE",
        "sociosphere",
        "Workers",
        "Conditions",
        "Shared indicator",
        "opportunity",
    )
    community_risk = (
        "SENSE",
        "sociosphere",
        "Local community",
        "Conditions",
        "Shared indicator",
        "risk",
    )
    assert set(written) == {workers_risk, workers_opportunity, community_risk}
    assert written[workers_risk] == [(flows[0].key, 1.0)]
    assert written[workers_opportunity] == [(flows[1].key, 1.0)]
    assert written[community_risk] == [(flows[2].key, 1.0)]
    assert ("Old", "SENSE", "method") in method_metadata

    assert methods_module.remove_impact_categories("sociosphere", "social") == 3
    assert set(method_metadata) == {("Old", "SENSE", "method")}
    assert written == {}


@pytest.mark.parametrize("price_unit", ["EUR2005", "USD2015"])
def test_direct_and_recursive_prices_share_precedence_and_unit_validation(
    monkeypatch, tmp_path, price_unit
):
    source = _GraphActivity("source", "input", "AA", database="source", reference_price=2)
    child = _GraphActivity(
        "child", "input", "AA", database="custom", reference_price=10, price_unit=price_unit
    )
    parent = _GraphActivity("parent", "output", "AA", database="custom")
    parent._technosphere = [_GraphExchange(child, 1)]
    monkeypatch.setattr("bw2data.Database", lambda _: [source])
    kwargs = {
        "mapping_df": pd.DataFrame([{"Shortname": "AA", "PSILCA_mapping": "AA"}]),
        "output_path": str(tmp_path / "activities.json"),
        "technosphere": "custom",
        "price_source_database": "source",
        "price_floor_percentile": 0,
        "activities": [child, parent],
    }
    if price_unit != "EUR2005":
        with pytest.raises(ValueError, match="expected EUR2005"):
            fetch_ecoinvent_activities(exclude_empty_inventories=False, **kwargs)
    else:
        result = fetch_ecoinvent_activities(exclude_empty_inventories=False, **kwargs)
        assert dict(zip(result["code"], result["price"], strict=False)) == {
            "child": 10,
            "parent": 10,
        }


def test_activity_prices_work_without_reference_price_catalog(tmp_path):
    activity = _GraphActivity("custom", "product", "AA", price=7)
    result = fetch_ecoinvent_activities(
        exclude_empty_inventories=False,
        mapping_df=pd.DataFrame([{"Shortname": "AA", "PSILCA_mapping": "AA"}]),
        output_path=str(tmp_path / "activities.json"),
        technosphere="db",
        activities=[activity],
    )
    assert result["price"].tolist() == [7]


def test_unpriced_foreground_produces_actionable_seed_request(tmp_path):
    from sense.exceptions import SenseInputRequiredError

    activity = _GraphActivity("custom", "product", "AA")
    seed_path = tmp_path / "seeds.json"
    with pytest.raises(SenseInputRequiredError) as error:
        fetch_ecoinvent_activities(
            exclude_empty_inventories=False,
            mapping_df=pd.DataFrame([{"Shortname": "AA", "PSILCA_mapping": "AA"}]),
            output_path=str(tmp_path / "activities.json"),
            technosphere="db",
            activities=[activity],
            auto_seed_prices_path=str(seed_path),
        )
    assert error.value.path == seed_path
    assert seed_path.is_file()


def test_method_collision_fails_without_touching_unowned_method(monkeypatch):
    key = ("SENSE", "db", "Social", "Working hours", "working hours")
    metadata = {key: {"description": "User-owned"}}
    flow = _Flow(("db", "hours"), managed=True)
    flow["sense_kind"] = "working_hours"
    monkeypatch.setattr(methods_module, "Database", lambda _: [flow])
    monkeypatch.setattr(methods_module, "methods", metadata)
    monkeypatch.setattr(
        methods_module, "Method", lambda _: pytest.fail("Mutation before ownership validation")
    )
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        add_working_hours_impact_category("db")
    assert metadata[key] == {"description": "User-owned"}


def test_precreated_unowned_flow_is_rejected_before_persistence(monkeypatch):
    flow = SimpleNamespace(key=("bio", "user-flow"))
    activity = SimpleNamespace(key=("tech", "example"))
    monkeypatch.setattr(
        repository, "_supports_sqlite_bulk", lambda _: pytest.fail("Persistence was reached")
    )
    with pytest.raises(ValueError, match="not owned"):
        repository.synchronize_exchanges(
            database_name="tech",
            grouped={"example": [{}]},
            techno_lookup={"example": activity},
            resolve_flow=lambda _: (flow, 1.0),
            managed_flow=lambda _: False,
            managed_flow_keys=set(),
            update_tol=0,
        )


def test_price_worklist_handles_deep_dependencies_and_cycles():
    from sense.data.seed_prices import _tarjan_scc

    graph = {node: {node + 1} for node in range(5000)}
    assert _tarjan_scc(graph) == [[node] for node in range(5000, -1, -1)]
    graph[5000] = {0}
    assert _tarjan_scc(graph) == [list(range(5000, -1, -1))]
    assert _tarjan_scc({0: {1}, 1: {0, 2}, 2: {3}, 3: {2}}) == [[3, 2], [1, 0]]
