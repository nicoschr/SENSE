"""Scientific accounting contracts for the contribution breakdown."""

import json

import numpy as np
import pandas as pd
from scipy import sparse

from case_studies.hydrogen.workflow.contribution_data import aggregate_locations
from case_studies.hydrogen.workflow.contribution_plot import build_display_table


def test_activity_grouping_preserves_credits_products_units_and_regional_labels():
    activities = pd.DataFrame(
        {
            "name": ["production"] * 6,
            "reference_product": ["gas", "gas", "gas", "oil", "gas", "gas"],
            "unit": ["m3", "m3", "m3", "kg", "kg", "m3"],
            "location": ["DE", "DE", "DE", "CN", "CN", "RoW"],
        }
    )
    grouped = aggregate_locations(activities, sparse.csr_matrix([[6, 4, -8, 5, 2, 1]]))
    assert len(grouped) == 5  # Copies combine; opposing signs and products do not.
    assert grouped.activity.nunique() == 3
    assert grouped.contribution.sum() == 10
    np.testing.assert_allclose(grouped.percent.abs().sum(), 100)
    np.testing.assert_allclose(grouped.percent.sum(), 1000 / 26)
    assert grouped.absolute_total.eq(26).all()
    assert set(grouped.location) == {"DE", "CN", "RoW"}
    assert grouped[grouped.sign.eq("negative")].contribution.item() == -8
    zero = aggregate_locations(activities, sparse.csr_matrix((1, len(activities))))
    assert zero.empty
    assert "percent" in zero


def test_display_pooling_preserves_signs_and_uses_stable_country_shades():
    rows = []
    for fu in (0, 1):
        for activity in range(8):
            for location, value in [
                ("CN-NM", 3),
                ("CN-SD", 2),
                ("DE", 4),
                ("RoW", 1),
                ("US", -0.5),
                ("NA", 0.25),
            ]:
                rows.append(
                    {
                        "fu_index": fu,
                        "panel": "climate",
                        "technology": "hydrogen",
                        "country": "US",
                        "scenario": "baseline" if fu == 0 else "regionalized",
                        "activity": json.dumps([f"activity {activity}", "gas", "m3"]),
                        "location": location,
                        "sign": "negative" if value < 0 else "positive",
                        "percent": value / (activity + 1) * (1 if fu == 0 else 2),
                        "contribution": value / (activity + 1),
                        "contribution_unit": "kg CO2-Eq",
                    }
                )
    data = pd.DataFrame(rows)
    # Non-US production must not affect selection, shades or totals.
    source = pd.concat([data, data.assign(country="CN", percent=9999)])
    displayed, legend = build_display_table(source)
    assert len(legend) == 3  # Two activities reach 10% in a bar, plus the remainder.
    for column in ("contribution", "percent"):
        before = data.groupby(["fu_index", "sign"])[column].sum()
        after = displayed.groupby(["fu_index", "sign"])[column].sum()
        np.testing.assert_allclose(before, after)
    assert "CN" in displayed.location_group.values
    assert "CN-NM" not in displayed.location_group.values
    assert "CN-SD" not in displayed.location_group.values
    assert "RoW" in displayed.location_group.values
    assert set(displayed.location_group) == {"CN", "DE", "RoW", "US", "NA"}
    assert set(displayed.country) == {"US"}
    # All five locations survive in every retained activity and the remainder.
    assert displayed.groupby("activity_group").location_group.nunique().eq(5).all()
    # A country has the same shade within its activity, independent of FU/scenario.
    assert displayed.groupby(["activity_group", "location_group"]).color.nunique().eq(1).all()
    retained = displayed[displayed.activity_number.eq(1)]
    np.testing.assert_allclose(retained[retained.location_group.eq("CN")].percent, [5, 10])


def test_fatal_accidents_has_paired_activity_and_country_panels():
    import matplotlib.pyplot as plt

    from case_studies.hydrogen.workflow.contribution_plot import plot_contributions

    rows = []
    technologies = [f"Route {i}" for i in range(5)]
    panels = [{"id": "fatal_accidents", "label": "Fatal accidents"}]
    for panel in panels:
        for ti, technology in enumerate(technologies):
            for si, scenario in enumerate(("baseline", "regionalized")):
                for location, value in (("DE", 90), ("CN", 9.9), ("US", 0.1)):
                    rows.append(
                        {
                            "panel": panel["id"],
                            "fu_index": ti * 2 + si,
                            "technology": technology,
                            "country": "US",
                            "scenario": scenario,
                            "activity": json.dumps(["Activity", "product", "kg"]),
                            "location": location,
                            "sign": "positive",
                            "percent": value,
                            "contribution": value,
                            "contribution_unit": "units",
                        }
                    )
    table, legend = build_display_table(pd.DataFrame(rows))
    spec = {"panels": panels, "technology_labels": {}}
    fig = plot_contributions(table, legend, spec, technologies)
    try:
        from matplotlib.text import Text

        np.testing.assert_allclose(fig.get_size_inches() * 25.4, [180, 180])
        fig.canvas.draw()
        visible_text = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text()]
        assert {t.get_fontsize() for t in visible_text} == {7, 8}
        activity_ax, country_ax = fig.axes[:2]
        assert len(activity_ax.patches) == 10
        np.testing.assert_allclose([patch.get_width() for patch in activity_ax.patches], 100)
        assert len({patch.get_facecolor() for patch in activity_ax.patches}) == 1
        assert len(country_ax.patches) == 30  # DE, CN and the pooled Other segment.
        np.testing.assert_allclose(sum(patch.get_width() for patch in country_ax.patches), 1000)
        for ax in (activity_ax, country_ax):
            assert len({round(patch.get_y(), 5) for patch in ax.patches}) == 10
            assert sum(text.get_text() == "O" for text in ax.texts) == 5
            assert sum(text.get_text() == "R" for text in ax.texts) == 5
            assert not any(text.get_text() == "US" for text in ax.texts)
            assert len(ax.get_yticklabels()) == 5
            assert all(spine.get_visible() for spine in ax.spines.values())
    finally:
        plt.close(fig)


def test_activity_selection_is_panel_specific_with_shared_colours_and_numbers():
    rows = []
    for panel, values in (
        ("climate", [30, 25, 20, 15, 10, 5, 2, 1]),
        ("fatal", [30, 1, 2, 5, 10, 15, 20, 25]),
    ):
        for index, value in enumerate(values):
            rows.append(
                {
                    "panel": panel,
                    "fu_index": 0,
                    "technology": "hydrogen",
                    "country": "US",
                    "scenario": "baseline",
                    "activity": json.dumps([f"Activity {index}", "product", "kg"]),
                    "location": "CN",
                    "sign": "positive",
                    "percent": value,
                    "contribution": value,
                    "contribution_unit": "units",
                }
            )
    source = pd.DataFrame(rows)
    table, _ = build_display_table(source)
    for panel, expected in (("climate", {0, 1, 2, 3, 4}), ("fatal", {0, 4, 5, 6, 7})):
        subset = table[table.panel.eq(panel)]
        named = subset[subset.activity_number.ne(0)]
        assert set(named.activity_group) == {
            json.dumps([f"Activity {i}", "product", "kg"]) for i in expected
        }
        assert subset[subset.activity_number.eq(0)].contribution.item() == 8
    np.testing.assert_allclose(
        source.groupby("panel").contribution.sum(), table.groupby("panel").contribution.sum()
    )
    climate_numbers = table.loc[
        table.panel.eq("climate") & table.activity_number.ne(0), "activity_number"
    ]
    assert sorted(climate_numbers.unique()) == [1, 2, 3, 4, 5]
    fatal_numbers = table.loc[
        table.panel.eq("fatal") & table.activity_number.ne(0), "activity_number"
    ]
    assert {6, 7, 8}.issubset(set(fatal_numbers))
    shared = table[table.activity_number.ne(0)].groupby("activity_group")
    assert shared.color.nunique().eq(1).all()
    assert shared.activity_number.nunique().eq(1).all()


def test_location_sums_preserve_opposing_signs():
    from case_studies.hydrogen.workflow.contribution_plot import sum_activity_locations

    table = pd.DataFrame(
        {
            "fu_index": [0, 0, 0],
            "technology": ["route"] * 3,
            "country": ["US"] * 3,
            "scenario": ["baseline"] * 3,
            "panel": ["climate"] * 3,
            "sign": ["positive", "positive", "negative"],
            "contribution_unit": ["kg"] * 3,
            "activity_group": ["activity"] * 3,
            "activity_number": [1] * 3,
            "location_group": ["DE", "CN", "US"],
            "contribution": [6, 4, -2],
            "percent": [50, 100 / 3, -100 / 6],
        }
    )
    result = sum_activity_locations(table, [{"activity": "activity", "color": "#0072B2"}])
    assert len(result) == 2
    assert result.set_index("sign").contribution.to_dict() == {"negative": -2, "positive": 10}
    np.testing.assert_allclose(result.percent.abs().sum(), 100)
    assert result.color.eq("#0072B2").all()


def test_country_breakdown_pools_small_locations_and_preserves_opposing_signs():
    from case_studies.hydrogen.workflow.contribution_plot import sum_country_activities

    data = pd.DataFrame(
        {
            "fu_index": [0] * 5,
            "technology": ["route"] * 5,
            "country": ["US"] * 5,
            "scenario": ["baseline"] * 5,
            "panel": ["fatal_accidents"] * 5,
            "contribution_unit": ["h"] * 5,
            "location_group": ["DE", "DE", "DE", "RoW", "NA"],
            "sign": ["positive", "positive", "negative", "positive", "positive"],
            "contribution": [50, 30, -10, 9.9, 0.1],
            "percent": [50, 30, -10, 9.9, 0.1],
        }
    )
    result, legend, order = sum_country_activities(data)
    assert set(result.location_group) == {"DE", "RoW", "Other countries/regions"}
    assert len(result) == 4
    assert order == ["DE", "RoW", "Other countries/regions"]
    assert (
        result[result.location_group.eq("Other countries/regions")].inline_label.item() == "Other"
    )
    assert any("Rest of world" in entry["label"] for entry in legend)
    np.testing.assert_allclose(result.percent.abs().sum(), 100)
    pd.testing.assert_series_equal(
        result.groupby("sign").contribution.sum(), data.groupby("sign").contribution.sum()
    )
