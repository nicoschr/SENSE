"""Validate and plot paired results without importing or modifying Brightway."""

from __future__ import annotations

import itertools
import json

import numpy as np
import pandas as pd

from .figure_io import COLORS, COUNTRY_COLORS


def validate_scores(scores: pd.DataFrame, spec: dict, technologies: list[str]) -> None:
    keys = ["technology", "country", "scenario", "panel"]
    required = {*keys, "score", "unit", "method", "activity_database", "activity_code"}
    if not required.issubset(scores.columns):
        raise ValueError(f"Missing result columns: {required - set(scores.columns)}")
    if len(technologies) != 5 or len(set(technologies)) != 5:
        raise ValueError("Expected five distinct hydrogen technologies")
    expected = set(
        itertools.product(
            technologies,
            spec["countries"],
            ("baseline", "regionalized"),
            [p["id"] for p in spec["panels"]],
        )
    )
    actual = set(scores[keys].itertuples(index=False, name=None))
    if scores.duplicated(keys).any() or actual != expected:
        raise ValueError(
            f"Incomplete/duplicate paired results: missing {len(expected - actual)}, extra {len(actual - expected)}"
        )
    if not np.isfinite(pd.to_numeric(scores["score"], errors="coerce")).all():
        raise ValueError("Non-finite paired results")
    for panel in spec["panels"]:
        data = scores[scores.panel == panel["id"]]
        if data.unit.nunique() != 1 or data.unit.isna().any() or (data.unit == "").any():
            raise ValueError(f"Missing or inconsistent units for {panel['id']}")
        if any(tuple(json.loads(key)) != tuple(panel["method"]) for key in data.method):
            raise ValueError(f"Incorrect characterization method for {panel['id']}")


def select_panels(scores: pd.DataFrame, spec: dict, technologies: list[str]) -> pd.DataFrame:
    """Select presentation panels from the complete, method-keyed calculation table."""
    keys = scores.method.map(lambda key: tuple(json.loads(key)))
    frames = []
    for panel in spec["panels"]:
        frame = scores[keys.map(lambda key, method=tuple(panel["method"]): key == method)].copy()
        if frame.empty:
            raise ValueError(
                f"Method was not captured: {panel['method']}; rerun calculate only "
                "if the required method is outside the saved result set"
            )
        frame["panel"] = panel["id"]
        frames.append(frame)
    selected = pd.concat(frames, ignore_index=True)
    validate_scores(selected, spec, technologies)
    return selected


def plot_paired(scores: pd.DataFrame, spec: dict, technologies: list[str]):
    """Fixed manuscript panels with paired scenarios and native indicator units."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    validate_scores(scores, spec, technologies)
    fig, axes = plt.subplots(
        len(spec["panels"]), 1, figsize=(180 / 25.4, 180 / 25.4), squeeze=False
    )
    colors = [COUNTRY_COLORS[country] for country in spec["countries"]]
    markers = ["o", "s", "^"]  # Redundant country encoding for colour-deficient/grayscale viewing.
    country_names = {"DE": "Germany", "CN": "China", "US": "United States"}
    for ax in list(axes.flat)[len(spec["panels"]) :]:
        ax.set_visible(False)
    for letter, ax, panel in zip("abcd", axes.flat, spec["panels"], strict=False):
        data = scores[scores.panel == panel["id"]]
        for ti, technology in enumerate(technologies):
            for ci, country in enumerate(spec["countries"]):
                pair = data[(data.technology == technology) & (data.country == country)].set_index(
                    "scenario"
                )
                a, b = pair.loc["baseline", "score"], pair.loc["regionalized", "score"]
                y = ti + (ci - 1) * 0.21
                ax.plot([a, b], [y, y], color=colors[ci], linewidth=0.8)
                ax.plot(
                    a,
                    y,
                    marker=markers[ci],
                    color=colors[ci],
                    markerfacecolor="white",
                    markersize=3.5,
                )
                ax.plot(b, y, marker=markers[ci], color=colors[ci], markersize=3.5)
        labels = [spec.get("technology_labels", {}).get(name, name) for name in technologies]
        ax.set_yticks(range(len(technologies)), labels, fontsize=7)
        ax.set_ylim(len(technologies) - 0.5, -0.5)
        for boundary in np.arange(0.5, len(technologies) - 0.5):
            ax.axhline(boundary, color="#A6A6A6", linewidth=0.5, linestyle=(0, (3, 3)), zorder=0)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(0.65)
        ax.text(
            0,
            1.035,
            f"({letter})  {panel['label']}",
            transform=ax.transAxes,
            fontsize=8,
            fontweight="bold",
            ha="left",
            va="baseline",
        )
        unit_label = {
            "kg CO2-Eq": r"kg CO$_2$-eq",
            "med risk hours": "medium-risk-equivalent hours",
        }.get(data.unit.iloc[0], data.unit.iloc[0])
        ax.set_xlabel(f"{unit_label} per kg hydrogen", fontsize=7)
        ax.tick_params(axis="x", labelsize=7)
        ax.xaxis.get_offset_text().set_fontsize(7)
        ax.grid(axis="x", color=COLORS["grid"], linewidth=0.5)
        ax.set_axisbelow(True)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(-3, 4), useMathText=True)
    handles = [
        Line2D(
            [],
            [],
            color=color,
            marker=marker,
            markersize=3.5,
            label=country_names.get(country, country),
        )
        for country, color, marker in zip(spec["countries"], colors, markers, strict=True)
    ]
    handles += [
        Line2D(
            [],
            [],
            color=COLORS["ink"],
            marker="o",
            linestyle="none",
            markerfacecolor=face,
            label=label,
            markersize=4,
        )
        for face, label in [
            ("white", "Original supply chains"),
            (COLORS["ink"], "Regionalized supply chains"),
        ]
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=7,
        columnspacing=3.0,
        handlelength=2.0,
    )
    fig.tight_layout(rect=(0, 0.1, 1, 1), w_pad=1.0, h_pad=1.5)
    return fig
