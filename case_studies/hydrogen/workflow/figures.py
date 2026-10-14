"""Render saved figure inputs without database access."""

import matplotlib.pyplot as plt

from .case_config import load
from .case_config import path as configured_path
from .comparison_plots import plot_regionalization, plot_validation
from .contribution_data import prepare_contributions
from .contribution_plot import (
    build_display_table,
    plot_contributions,
    positive_display_table,
    sum_activity_locations,
    sum_country_activities,
)
from .figure_io import export_figure, figure_style, load_inputs
from .paired_results import plot_paired, select_panels
from .result_archive import load_results
from .study_protocol import calculation_context, load_protocol


def hydrogen():
    """Execute this stage using the configured study inputs."""
    CFG = load()
    SPEC = load_protocol()

    scores, metadata = load_results(
        configured_path("data_dir", CFG) / "paired_hydrogen/results.zip",
        context=calculation_context(CFG, SPEC),
    )
    selected = select_panels(scores, SPEC, metadata["technologies"])
    selected.to_csv(
        configured_path("data_dir", CFG) / "paired_hydrogen/figure_scores.csv", index=False
    )

    with plt.rc_context(figure_style()):
        fig = plot_paired(selected, SPEC, metadata["technologies"])
        stem = configured_path("data_dir", CFG) / "figures/hydrogen_paired"
        stem.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(stem.with_suffix(".png"), dpi=300)
        export_figure(fig, stem, preview=False)


def contributions():
    """Render fatal-accident contributions by activity and country for the United States."""
    cfg, spec = load(), load_protocol()
    data_dir = configured_path("data_dir", cfg)
    figure_spec = {**spec, "panels": [p for p in spec["panels"] if p["id"] == "fatal_accidents"]}
    data, metadata = prepare_contributions(
        data_dir / "paired_hydrogen/results.zip",
        context=calculation_context(cfg, spec),
        spec=figure_spec,
        production_country="US",
    )
    data.to_parquet(
        data_dir / "paired_hydrogen/contributions_by_activity_location.parquet", index=False
    )
    table, legend = build_display_table(data)
    table.to_csv(data_dir / "paired_hydrogen/us_contribution_shares.csv", index=False)
    displayed = positive_display_table(table)
    sum_activity_locations(displayed, legend).to_csv(
        data_dir / "paired_hydrogen/us_activity_contribution_shares.csv", index=False
    )
    countries, _, _ = sum_country_activities(displayed)
    countries.to_csv(data_dir / "paired_hydrogen/us_country_contribution_shares.csv", index=False)
    with plt.rc_context({**figure_style(), "hatch.linewidth": 0.35}):
        fig = plot_contributions(table, legend, spec, metadata["technologies"])
        stem = data_dir / "figures/hydrogen_us_contributions"
        stem.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(stem.with_suffix(".png"), dpi=300)
        export_figure(fig, stem, preview=False)


def regionalization():
    """Execute this stage using the configured study inputs."""
    SPEC = load_protocol()
    CASE_CFG = load()
    DATA_DIR = configured_path("data_dir", CASE_CFG)
    CONTEXT = {
        "versions": CASE_CFG["versions"],
        "projects": CASE_CFG["brightway"],
        "social_layers": SPEC["social_layers"],
        "sense": SPEC["sense"],
    }
    FIGURE_DIR = DATA_DIR / "figures"

    tables, metadata = load_inputs(DATA_DIR / "figure_inputs/regionalization.zip", context=CONTEXT)

    with plt.rc_context(figure_style()):
        fig = plot_regionalization(
            tables["values"], top_n=5, min_fus=30, width_mm=174, height_mm=125
        )
        export_figure(fig, FIGURE_DIR / "regionalization_effect", preview=False)


def benchmark():
    """Execute this stage using the configured study inputs."""
    CASE_CFG = load()
    SPEC = load_protocol()
    DATA_DIR = configured_path("data_dir", CASE_CFG)
    CONTEXT = {
        "versions": CASE_CFG["versions"],
        "projects": CASE_CFG["brightway"],
        "social_layers": SPEC["social_layers"],
        "sense": SPEC["sense"],
    }
    FIGURE_DIR = DATA_DIR / "figures"

    tables, metadata = load_inputs(DATA_DIR / "figure_inputs/validation.zip", context=CONTEXT)

    WIDTH_MM, HEIGHT_MM = 174, 150
    X_LIMITS = (-4, 4)
    N_BINS = 40

    with plt.rc_context(figure_style()):
        fig = plot_validation(
            tables,
            metadata,
            width_mm=WIDTH_MM,
            height_mm=HEIGHT_MM,
            x_limits=X_LIMITS,
            n_bins=N_BINS,
        )
        export_figure(fig, FIGURE_DIR / "validation", preview=False)

    with plt.rc_context(figure_style()):
        fig = plot_validation(
            tables,
            metadata,
            factor_neutral=True,
            width_mm=WIDTH_MM,
            height_mm=HEIGHT_MM,
            x_limits=X_LIMITS,
            n_bins=N_BINS,
        )
        export_figure(fig, FIGURE_DIR / "validation_factor_neutral", preview=False)

    with plt.rc_context(figure_style()):
        fig = plot_validation(
            tables,
            metadata,
            aggregation="arithmetic",
            width_mm=WIDTH_MM,
            height_mm=HEIGHT_MM,
            x_limits=X_LIMITS,
            n_bins=N_BINS,
        )
        export_figure(fig, FIGURE_DIR / "validation_arithmetic", preview=False)
