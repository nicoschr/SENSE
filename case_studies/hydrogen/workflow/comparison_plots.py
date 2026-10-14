"""Plot regionalization and validation snapshots without accessing Brightway."""

import matplotlib.pyplot as plt
import numpy as np
import pycountry
from matplotlib.ticker import MaxNLocator

from .figure_io import COLORS


def _location_label(code):
    """Expand recognized ISO locations; retain other database geography codes."""
    labels = {
        "GLO": "Global",
        "RoW": "Rest of world",
        "Europe without Switzerland": "Europe excl. Switzerland",
        "RU": "Russia",
        "KR": "South Korea",
        "TW": "Taiwan",
        "IR": "Iran",
        "CN-FJ": "Fujian, China",
        "CN-SC": "Sichuan, China",
    }
    if code in labels:
        return labels[code]
    country = pycountry.countries.get(alpha_2=code)
    if country:
        return country.name
    subdivision = pycountry.subdivisions.get(code=code)
    return f"{subdivision.name}, {code[:2]}" if subdivision else code


def plot_regionalization(values, *, top_n=5, min_fus=30, width_mm=174, height_mm=125):
    """Return one panel of extreme-location boxplots and intervening median points.

    ``values`` contains one finite ``log_ratio`` (log10 regionalized/base) per FU
    and its ``location``. Retain locations with at least ``min_fus`` rows, then show
    the ``top_n`` lowest and highest medians as boxes. Intermediate locations remain
    ordered median points in a compressed, bracketed band, not a pooled sample.

    Boxes span quartiles and whiskers extend to observations within 1.5 IQR;
    fliers are omitted. Statistics stay in log space and every whisker is visible.
    Dimensions are millimetres. Do not mutate inputs, save files or display figures.
    """
    if min_fus < 1 or top_n < 1:
        raise ValueError("min_fus and top_n must be positive.")
    if not np.isfinite(values["log_ratio"].to_numpy(float)).all():
        raise ValueError("Regionalization values must be finite.")
    grouped = values.groupby("location")["log_ratio"]
    summary = grouped.agg(median="median", n="size").query("n >= @min_fus")
    summary = summary.sort_values("median", kind="stable")
    if summary.empty:
        raise ValueError("No locations meet min_fus.")

    # If the tails overlap, display each location once as a box.
    if len(summary) <= 2 * top_n:
        selected = list(summary.index)
        positions = np.arange(len(selected), dtype=float)
        middle = summary.iloc[0:0]
    else:
        selected = [*summary.index[:top_n], *summary.index[-top_n:]]
        middle = summary.iloc[top_n:-top_n]
        band_height = 4.0
        positions = np.r_[np.arange(top_n), np.arange(top_n) + top_n + band_height]

    fig, ax = plt.subplots(figsize=(width_mm / 25.4, height_mm / 25.4), layout="constrained")
    ax.boxplot(
        [grouped.get_group(location).to_numpy() for location in selected],
        positions=positions,
        orientation="horizontal",
        widths=0.6,
        patch_artist=True,
        showfliers=False,
        whis=1.5,
        manage_ticks=False,
        boxprops={
            "facecolor": COLORS["distribution"],
            "edgecolor": COLORS["indigo"],
            "linewidth": 0.8,
        },
        medianprops={"color": COLORS["burgundy"], "linewidth": 1.3},
        whiskerprops={"color": COLORS["grey"], "linewidth": 0.8},
        capprops={"color": COLORS["grey"], "linewidth": 0.8},
    )
    ax.set_yticks(
        positions, [f"{_location_label(loc)} (n={summary.loc[loc, 'n']:,})" for loc in selected]
    )
    ax.tick_params(axis="y", length=0)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.65)
    if not middle.empty:
        bottom, top = top_n - 0.15, top_n + band_height - 0.85
        y = np.linspace(bottom, top, len(middle)) if len(middle) > 1 else [(bottom + top) / 2]
        ax.scatter(middle["median"], y, s=8, color=COLORS["burgundy"], zorder=3)
        # The bracket describes vertical rank compression; x values are unmodified.
        trans = ax.get_yaxis_transform()
        ax.plot(
            [-0.012, -0.035, -0.035, -0.012],
            [bottom, bottom, top, top],
            transform=trans,
            color=COLORS["grey"],
            linewidth=0.7,
            clip_on=False,
        )
        ax.text(
            -0.055,
            (bottom + top) / 2,
            f"Other {len(middle)} locations\n(medians only)",
            transform=trans,
            ha="right",
            va="center",
        )

    ax.axvline(0, color=COLORS["grey"], linestyle="--", linewidth=0.8, zorder=0)
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.set_ylim(-0.8, positions[-1] + 0.8)
    ax.margins(x=0.06)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.set_xlabel(r"Change in social score: $\log_{10}$(regionalized / base)")
    return fig


def plot_validation(
    tables,
    metadata,
    *,
    factor_neutral=False,
    aggregation="geometric",
    width_mm=174,
    height_mm=150,
    x_limits=(-4, 4),
    n_bins=40,
):
    """Plot one aggregation's benchmark in three stacked geography panels.

    Densities are normalized by the full positive-pair sample, including tails
    outside the displayed range. Each geography has its own density scale.

    ``tables[aggregation]`` contains ``dataset_id``, ``geo_class`` and ``log_diff``.
    ``metadata[aggregation]['factor']`` holds the recorded monetary multiplier.
    ``factor_neutral`` subtracts
    log10(factor) from the ratios, leaving the saved tables unchanged. ``x_limits`` are log10
    ratios; dimensions are millimetres. Reject empty/nonfinite panel samples with
    ``ValueError``. Return a Figure without saving or displaying it.
    """
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(width_mm / 25.4, height_mm / 25.4),
        sharex=True,
        squeeze=False,
        layout="constrained",
    )
    edges = np.linspace(x_limits[0], x_limits[1], n_bins + 1)
    groups = [
        ("All locations", None),
        ("Countries / states", "Country/State"),
        ("Regions / other", "Region/Other"),
    ]
    data = tables[aggregation]
    factor = metadata[aggregation]["factor"]
    shift = np.log10(factor) if factor_neutral else 0.0
    for row, (label, geo_class) in enumerate(groups):
        ax = axes[row, 0]
        subset = data if geo_class is None else data.loc[data["geo_class"] == geo_class]
        values = subset["log_diff"].to_numpy(float) - shift
        if not len(values) or not np.isfinite(values).all():
            raise ValueError(f"Missing/nonfinite validation values for {aggregation}: {label}")
        counts, _ = np.histogram(values, bins=edges)
        density = counts / (len(values) * np.diff(edges))
        peak = float(density.max())
        ax.stairs(
            density,
            edges,
            fill=True,
            edgecolor=COLORS["indigo"],
            facecolor=COLORS["distribution"],
            linewidth=0.8,
        )
        median = float(np.median(values))
        ax.axvline(0, color=COLORS["grey"], linestyle="--", linewidth=0.7)
        ax.axvline(median, color=COLORS["burgundy"], linewidth=1)
        tails = int(np.count_nonzero((values < edges[0]) | (values > edges[-1])))
        letter = chr(ord("a") + row)
        ax.set_title(f"({letter})  {label}", loc="left", fontsize=8, pad=8)
        ax.text(
            0.02,
            0.96,
            f"Activities: {subset['dataset_id'].nunique():,}\n"
            f"Median: {median:.2f}\nOutside range: {tails / len(values):.1%}",
            transform=ax.transAxes,
            va="top",
        )
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(0.65)
        ax.grid(axis="y")
        ax.set_axisbelow(True)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.set_xlim(x_limits)
        ax.set_xticks(np.arange(np.ceil(x_limits[0]), x_limits[1] + 1, 2))
        ax.set_ylabel("Probability density")
        ax.set_ylim(0, peak * 1.15 if peak > 0 else 1)
    label = r"$\log_{10}$(SENSE / SOCA)"
    if factor_neutral:
        label = r"$\log_{10}$(factor-neutral SENSE / SOCA)"
    for ax in axes[-1]:
        ax.set_xlabel(label)
    return fig
