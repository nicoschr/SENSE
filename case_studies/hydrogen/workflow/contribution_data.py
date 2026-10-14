"""Group archived direct contributions without changing the LCA calculation."""

from __future__ import annotations

import io
import json
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy import sparse

from .paired_results import select_panels
from .result_archive import _read_table, load_results

ACTIVITY_KEY = ["name", "reference_product", "unit"]


def aggregate_locations(activities: pd.DataFrame, row) -> pd.DataFrame:
    """Combine database copies by activity identity and location, retaining both signs.

    A shared name alone is insufficient: different reference products or units
    remain different activities. Positive and negative terms never cancel during
    grouping. The denominator is the sum of absolute direct contributions, before
    any grouping, so credits cannot disappear or reverse sign in the display.
    """
    row = row.tocsr(copy=True)
    row.sum_duplicates()
    row.eliminate_zeros()
    if row.shape != (1, len(activities)) or not np.isfinite(row.data).all():
        raise ValueError("Invalid activity contribution row")
    data = activities.iloc[row.indices][[*ACTIVITY_KEY, "location"]].fillna("").copy()
    data["location"] = data.location.replace("", "Unspecified")
    data["contribution"] = row.data
    data["sign"] = np.where(row.data < 0, "negative", "positive")
    data = data.groupby([*ACTIVITY_KEY, "location", "sign"], as_index=False).contribution.sum()
    denominator = float(np.abs(row.data).sum())
    data["absolute_total"] = denominator
    data["percent"] = data.contribution / denominator * 100 if denominator else 0.0
    data["activity"] = [
        json.dumps(list(key), ensure_ascii=False)
        for key in data[ACTIVITY_KEY].itertuples(index=False, name=None)
    ]
    return data


def prepare_contributions(path, *, context, spec, production_country=None):
    """Read the axes once, then stream one FU matrix at a time; check all score sums."""
    scores, metadata = load_results(path, context=context)
    selected = select_panels(scores, spec, metadata["technologies"])
    frames = []
    with ZipFile(path) as archive:
        activities = _read_table(archive, "activities")
        methods = _read_table(archive, "methods")
        units = _read_table(archive, "functional_units")
        if activities.activity_index.tolist() != list(range(len(activities))):
            raise ValueError("Activity metadata is not in matrix column order")
        if methods.method_index.tolist() != list(range(len(methods))):
            raise ValueError("Method metadata is not in matrix row order")
        method_indices = {}
        for panel in spec["panels"]:
            matches = methods[
                methods.method.map(lambda key, method=panel["method"]: json.loads(key) == method)
            ]
            if len(matches) != 1:
                raise ValueError(f"Absent or ambiguous method: {panel['id']}")
            method_indices[panel["id"]] = int(matches.iloc[0].method_index)
        if production_country is not None:
            units = units[units.country.eq(production_country)]
            if units.empty:
                raise ValueError(f"No functional units for {production_country}")
        for position, unit in enumerate(units.itertuples(index=False), start=1):
            values = sparse.load_npz(io.BytesIO(archive.read(unit.contribution_file))).tocsr()
            if values.shape != (len(methods), len(activities)):
                raise ValueError("Contribution dimensions do not match archived axes")
            for panel in spec["panels"]:
                data = aggregate_locations(activities, values.getrow(method_indices[panel["id"]]))
                expected = selected[
                    selected.fu_index.eq(unit.fu_index) & selected.panel.eq(panel["id"])
                ]
                if len(expected) != 1 or not np.isclose(
                    data.contribution.sum(), expected.iloc[0].score, rtol=1e-10, atol=1e-12
                ):
                    raise ValueError(
                        f"Contributions do not reconcile: FU {unit.fu_index}, {panel['id']}"
                    )
                frames.append(
                    data.assign(
                        fu_index=unit.fu_index,
                        technology=unit.technology,
                        country=unit.country,
                        scenario=unit.scenario,
                        panel=panel["id"],
                        contribution_unit=expected.iloc[0].unit,
                    )
                )
            print(f"Read contributions: {position}/{len(units)}", flush=True)
    return pd.concat(frames, ignore_index=True), metadata


def select_activity_groups(data: pd.DataFrame, *, threshold_percent: float = 10):
    """Keep activities reaching the threshold in any system within each panel.

    Only each panel's own selection is shown there. The union defines shared
    colour/number identities across the figure; all other activities in a panel
    remain in Other activities. Magnitudes retain important negative credits.
    """
    magnitude = data.assign(magnitude=data.percent.abs())
    per_bar = magnitude.groupby(["panel", "fu_index", "activity"]).magnitude.sum()
    maxima = per_bar.groupby(["panel", "activity"]).max()
    keep = set()
    panel_selections = {}
    for panel in data.panel.unique():
        panel_maxima = maxima.loc[panel]
        selected = panel_maxima[panel_maxima.ge(threshold_percent)].index
        panel_selections[panel] = set(selected)
        keep.update(selected)
    totals = (
        magnitude.groupby("activity").magnitude.sum().sort_values(ascending=False, kind="stable")
    )
    retained = [activity for activity in totals.index if activity in keep]
    result = data.copy()
    result["activity_group"] = "Other activities"
    for panel, selected in panel_selections.items():
        mask = result.panel.eq(panel) & result.activity.isin(selected)
        result.loc[mask, "activity_group"] = result.loc[mask, "activity"]
    return result, retained
