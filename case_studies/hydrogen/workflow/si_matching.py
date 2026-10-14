"""Render the audited SI calibration figures in the SENSE style.

Uses bundled aggregate evidence from the original author-labelled audit.
Does not fit models, recreate manual labels, or access Brightway.
"""

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter

from sense.artifacts import write_json_atomic

from . import case_config
from .figure_io import COLORS, export_figure, figure_style
from .si_common import require

EVIDENCE = Path(__file__).resolve().parents[1] / "resources/matching_audit"
MODELS = [
    "baseline/tfidf_char_wb_3_5",
    "sentence-transformers/all-MiniLM-L6-v2",
    "BAAI/bge-base-en-v1.5",
    "sentence-transformers/all-mpnet-base-v2",
]
MODEL_LABELS = ["TF-IDF char (3–5)", "MiniLM-L6", "BGE-base", "MPNet-base"]
MODEL_COLORS = [COLORS["grey"], COLORS["indigo"], COLORS["burgundy"], COLORS["green"]]


def save(fig, name, output):
    target = output / name
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target.with_suffix(".png"), dpi=300)
    export_figure(fig, target)
    return {
        ext: hashlib.sha256(target.with_suffix("." + ext).read_bytes()).hexdigest()
        for ext in ("pdf", "svg")
    }


def axes_style(ax, title):
    ax.set_title(title, loc="left", fontsize=10, fontweight="bold", pad=8)
    ax.tick_params(labelsize=9, width=0.65)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.65)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.4)


def encoder_figures(metrics, output):
    outputs = {}
    fields = [
        ("top1", "Top-1 plausibility", "Rate"),
        ("hit10", "Plausible-hit@10", "Rate"),
        ("mrr", "MRR", "MRR"),
    ]
    for paired, name in [
        (False, "Fig1_semantic_vs_baseline_TEST"),
        (True, "Fig2_semantic_model_choice_DEV_TEST"),
    ]:
        chosen = MODELS[1:] if paired else MODELS
        labels = MODEL_LABELS[1:] if paired else MODEL_LABELS
        colors = MODEL_COLORS[1:] if paired else MODEL_COLORS
        fig, axes = plt.subplots(1, 3, figsize=(174 / 25.4, 70 / 25.4), sharey=True)
        fig.subplots_adjust(
            left=0.19,
            right=0.98,
            top=0.83,
            bottom=0.30 if paired else 0.20,
            wspace=0.27,
        )
        y = np.arange(len(chosen))
        for letter, ax, (field, title, unit) in zip("abc", axes, fields, strict=True):
            for split, offset in ([("dev", -0.18), ("test", 0.18)] if paired else [("test", 0)]):
                values = metrics[metrics.split.eq(split)].set_index("model").loc[chosen, field]
                ax.barh(
                    y + offset,
                    values,
                    height=0.30 if paired else 0.52,
                    color="white" if paired and split == "test" else colors,
                    edgecolor=colors,
                    hatch="///" if paired and split == "test" else None,
                    linewidth=0.65,
                )
            ax.set_xlim(0, 1.05)
            ax.set_xticks([0, 0.5, 1])
            if field != "mrr":
                ax.xaxis.set_major_formatter(PercentFormatter(1))
            ax.set_xlabel(unit, fontsize=9)
            ax.set_yticks(y, labels)
            axes_style(ax, f"({letter}) {title}")
        axes[0].set_ylim(len(chosen) - 0.5, -0.5)
        if paired:
            fig.legend(
                handles=[
                    Patch(facecolor=COLORS["grey"], label="DEV (n=150)"),
                    Patch(
                        facecolor="white",
                        edgecolor=COLORS["grey"],
                        hatch="///",
                        label="TEST (n=100)",
                    ),
                ],
                loc="lower center",
                ncol=2,
                fontsize=9,
                frameon=False,
                bbox_to_anchor=(0.57, 0.015),
            )
        outputs[name] = save(fig, name, output)
    return outputs


def threshold_figure(curve, output):
    provenance = json.loads((EVIDENCE / "provenance.json").read_text(encoding="utf-8"))
    delta = provenance["selected_delta"][0]
    require(delta == 0.028, "SI reconstruction check failed: delta == 0.028")
    dev = curve[curve.split.eq("dev")]
    require(
        np.isclose(dev.loc[dev.utility.idxmax(), "delta"], delta),
        'SI reconstruction check failed: np.isclose(dev.loc[dev.utility.idxmax(), "delta"], delta)',
    )
    fig, axes = plt.subplots(1, 3, figsize=(174 / 25.4, 68 / 25.4))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.83, bottom=0.29, wspace=0.45)
    fields = [
        ("hit_rate", "Hit rate"),
        ("mean_set_size", "Additional sectors"),
        ("utility", "Utility"),
    ]
    for letter, ax, (field, title) in zip("abc", axes, fields, strict=True):
        for split, color, style in [
            ("dev", COLORS["indigo"], "-"),
            ("test", COLORS["burgundy"], "--"),
        ]:
            data = curve[curve.split.eq(split)].sort_values("delta")
            y = data[field] - (1 if field == "mean_set_size" else 0)
            ax.plot(data.delta, y, color=color, linestyle=style, linewidth=1)
            selected = data[np.isclose(data.delta, delta)].iloc[0]
            ax.plot(
                delta,
                selected[field] - (1 if field == "mean_set_size" else 0),
                marker="o" if split == "dev" else "s",
                color=color,
                markersize=3,
            )
        ax.axvline(delta, color=COLORS["ochre"], linestyle=":", linewidth=1)
        ax.set_xlim(0, 0.05)
        ax.set_xticks([0, 0.025, 0.05], ["0", "0.025", "0.05"])
        ax.set_xlabel("Similarity tolerance, Δ", fontsize=9)
        if field == "hit_rate":
            ax.yaxis.set_major_formatter(PercentFormatter(1))
        axes_style(ax, f"({letter}) {title}")
    fig.legend(
        handles=[
            Line2D([], [], color=COLORS["indigo"], label="DEV (n=90)"),
            Line2D([], [], color=COLORS["burgundy"], linestyle="--", label="TEST (n=60)"),
            Line2D([], [], color=COLORS["ochre"], linestyle=":", label="Selected Δ = 0.028"),
        ],
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.53, 0.025),
        fontsize=9,
        frameon=False,
    )
    return save(fig, "Fig3_delta_selection_utility_high_res", output)


def run():
    data = case_config.path("data_dir", case_config.load())
    output = data / "figures"
    metrics = pd.read_csv(EVIDENCE / "encoder_metrics.csv")
    curve = pd.read_csv(EVIDENCE / "threshold_curve.csv")
    with plt.rc_context(
        {
            **figure_style(),
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "hatch.linewidth": 0.35,
        }
    ):
        outputs = encoder_figures(metrics, output)
        outputs["Fig3_delta_selection_utility_high_res"] = threshold_figure(curve, output)
    record = {
        "inputs": {
            name: hashlib.sha256((EVIDENCE / name).read_bytes()).hexdigest()
            for name in ("encoder_metrics.csv", "threshold_curve.csv", "provenance.json")
        },
        "outputs": outputs,
        "threshold": 0.028,
        "note": "Figures rendered from the original audit's aggregate results. No model inference or reconstruction of manual labels.",
    }
    write_json_atomic(data / "si/matching/rendered.json", record)
    print("Rendered three matching figures from bundled aggregate audit data.")
