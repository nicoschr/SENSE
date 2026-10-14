"""Render the SI productivity figure from checked local inputs and derived shares."""

import hashlib
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator
from matplotlib.transforms import Bbox

from sense.artifacts import write_json_atomic

from . import case_config
from .figure_io import COUNTRY_COLORS, export_figure, figure_style
from .si_common import require


def run():
    DATA = case_config.path("data_dir", case_config.load())
    HERE = DATA / "si/productivity"
    SI = DATA / "figures"
    source = DATA / "Output/PSILCA/psilca_assessments_v4.0.parquet"
    evidence = json.loads((HERE / "evidence.json").read_text(encoding="utf-8"))
    require(
        hashlib.sha256(source.read_bytes()).hexdigest() == evidence["psilca_cache_sha256"],
        'SI reconstruction check failed: hashlib.sha256(source.read_bytes()).hexdigest() == evidence["psilca_cache_sha256"]',
    )
    raw = pd.read_parquet(
        source, columns=["process_index", "source_file", "location", "reference_name", "mean_value"]
    )
    require(
        np.isfinite(raw.mean_value).all() and raw.mean_value.gt(0).all(),
        "SI reconstruction check failed: np.isfinite(raw.mean_value).all() and raw.mean_value.gt(0).all()",
    )
    groups = raw.groupby("process_index", sort=False)
    for field in ["mean_value", "source_file", "location", "reference_name"]:
        require(
            groups[field].nunique().eq(1).all(),
            "SI reconstruction check failed: groups[field].nunique().eq(1).all()",
        )
    data = raw.drop_duplicates("process_index").copy()
    require(
        data.source_file.is_unique, "SI reconstruction check failed: data.source_file.is_unique"
    )
    data["productivity"] = 1 / data.mean_value
    data["relative"] = data.productivity / data.productivity.median()
    data = data.sort_values(
        ["relative", "location", "reference_name", "process_index"]
    ).reset_index(drop=True)
    shares_path = HERE / "selected_country_shares.csv"
    require(
        hashlib.sha256(shares_path.read_bytes()).hexdigest()
        == evidence["selected_country_shares_sha256"],
        'SI reconstruction check failed: hashlib.sha256(shares_path.read_bytes()).hexdigest() == evidence["selected_country_shares_sha256"]',
    )
    shares = pd.read_csv(shares_path).set_index(["scenario", "country"])
    names = {"US": "United States", "RU": "Russia", "CO": "Colombia"}
    markers = {"US": "^", "RU": "o", "CO": "s"}
    style = figure_style()
    style.update({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8})
    with plt.rc_context(style):
        fig = plt.figure(figsize=(174 / 25.4, 182 / 25.4))
        ax = fig.add_axes((0.105, 0.385, 0.875, 0.515))
        fig.text(
            0.105,
            0.965,
            "a  Productivity implied by PSILCA labour intensities",
            fontsize=9,
            weight="bold",
            va="top",
        )
        ax.set_yscale("log")
        ax.scatter(data.index, data.relative, s=9, color="#C3C3C3", linewidths=0, zorder=2)
        for country in names:
            d = data[data.location.eq(country)]
            ax.scatter(
                d.index,
                d.relative,
                s=32,
                marker=markers[country],
                color=COUNTRY_COLORS[country],
                edgecolors="white",
                linewidths=0.3,
                alpha=0.9,
                zorder=3,
            )
        ax.axhline(1, color="#909090", linestyle=(0, (3, 3)), linewidth=0.6, zorder=0)
        ax.annotate(
            "Database median",
            (len(data) * 0.015, 1),
            xytext=(0, 4),
            textcoords="offset points",
            color="#737373",
            va="bottom",
        )
        ax.set_xlim(-0.025 * len(data), 1.025 * len(data))
        ax.set_xticks([])
        ax.yaxis.set_major_locator(NullLocator())
        ax.yaxis.set_minor_locator(NullLocator())
        ax.set_ylabel("Relative labour productivity (log scale)", labelpad=9)
        ax.set_xlabel(
            "Country–sector records, ordered from lower to higher productivity →", labelpad=8
        )
        handles = [
            Line2D(
                [],
                [],
                marker="o",
                color="#C3C3C3",
                linestyle="none",
                markersize=4,
                label="All country–sector records",
            )
        ]
        handles += [
            Line2D(
                [],
                [],
                marker=markers[c],
                color=COUNTRY_COLORS[c],
                linestyle="none",
                markersize=6,
                label=names[c],
            )
            for c in names
        ]
        fig.legend(
            handles=handles,
            loc="upper left",
            bbox_to_anchor=(0.105, 0.925),
            ncol=4,
            borderaxespad=0,
            handletextpad=0.5,
            columnspacing=1.5,
            fontsize=8,
        )
        fig.text(
            0.105,
            0.315,
            "b  Country shares in the US coal-hydrogen diagnostic",
            fontsize=9,
            weight="bold",
            va="top",
        )
        table_ax = fig.add_axes((0.105, 0.095, 0.875, 0.175))
        table_ax.axis("off")
        columns = [
            "Country / supply chain",
            "Monetary\noutput",
            "Actual\nscore",
            "Equal labour\nintensity",
            "Equal risk\nfactors",
        ]
        metrics = ["monetary_output", "actual", "equal_intensity", "equal_risk"]
        table_rows = []
        for country, scenario, label in [
            ("RU", "baseline", "Russia / original"),
            ("CO", "regionalized", "Colombia / regionalized"),
        ]:
            row = shares.loc[(scenario, country)]
            table_rows.append(
                [
                    label,
                    *[
                        f"{row[m]:.2f}%" if m == "monetary_output" else f"{row[m]:.1f}%"
                        for m in metrics
                    ],
                ]
            )
        table = table_ax.table(
            cellText=table_rows,
            colLabels=columns,
            colWidths=[0.30, 0.175, 0.175, 0.175, 0.175],
            cellLoc="center",
            colLoc="center",
            bbox=Bbox.from_bounds(0, 0, 1, 1),
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        for (row, column), cell in table.get_celld().items():
            cell.set_edgecolor("#A0A0A0")
            cell.set_linewidth(0.4)
            cell.visible_edges = "B" if row else "BT"
            if row == 0:
                cell.set_text_props(weight="bold")
            if column == 0:
                cell.set_text_props(ha="left")
                cell.PAD = 0.02
            if row > 0 and column == 0:
                cell.set_facecolor("white")
        fig.text(
            0.105,
            0.052,
            "Each percentage uses its own comparison total. Equalization changes one factor at a time.",
            fontsize=8,
        )
        fig.text(
            0.105,
            0.032,
            "Treatment fallback hours remain fixed when equalizing monetary labour intensities.",
            fontsize=8,
        )
        output = SI / "hydrogen_productivity_diagnostic"
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output.with_suffix(".png"), dpi=300)
        export_figure(fig, output)
    rendered = {
        ext: hashlib.sha256(output.with_suffix(f".{ext}").read_bytes()).hexdigest()
        for ext in ["pdf", "svg"]
    }
    write_json_atomic(
        HERE / "rendered.json",
        {"figure": "productivity diagnostic", "size_mm": [174, 182], "hashes": rendered},
    )
