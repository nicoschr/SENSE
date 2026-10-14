"""Render supplementary LaTeX tables from checked analysis outputs."""

import pandas as pd

from . import case_config
from .si_common import require, write_text

ROUTES = [
    ("hydrogen production, biomass gasification", "Biomass gasification"),
    ("hydrogen production, steam methane reforming of biomethane", "Biomethane SMR"),
    ("hydrogen production, PEM electrolysis, green", "Wind PEM"),
    ("hydrogen production, coal gasification", "Coal gasification"),
    ("hydrogen production, steam methane reforming", "Natural-gas SMR"),
]


def missing():
    data = case_config.path("data_dir", case_config.load())
    HERE = data / "si/missing_assessments"
    SI = data / "si"
    coverage = pd.read_csv(HERE / "labour_weighted_coverage.csv")
    coverage = coverage[coverage.indicator.eq("fatal_accidents")].pivot(
        index=["technology", "country"], columns="scenario", values="missing_labour_percent"
    )
    ratios = pd.read_csv(HERE / "paired_ratios.csv")
    ratios = ratios[ratios.indicator.eq("fatal_accidents")].pivot(
        index=["technology", "country"],
        columns="no_data_factor",
        values="regionalized_over_original",
    )
    routes = ROUTES

    lines = [
        r"\begin{table}[tbp]",
        r"\centering",
        r"\small",
        r"\caption{Fatal-accident assessment coverage and sensitivity for the 15 hydrogen pairs. Hours with no assessment denotes the percentage of absolute modelled working hours associated with explicit no-data levels. O and R denote original and regionalized supply chains. Ratios give the regionalized/original score under each no-data factor. A ratio above one indicates a higher modelled score under regionalization. These factors test assumptions and are not uncertainty bounds}",
        r"\label{tab:missing_assessment_sensitivity}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lcrrrrrr}",
        r"\toprule",
        r" & & \multicolumn{2}{c}{\shortstack{Hours with no\\assessment (\%)}} & \multicolumn{4}{c}{Score ratio at no-data factor}\\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-8}",
        r"Route & Country & O & R & 0.1 & 1 & 10 & 100\\",
        r"\midrule",
    ]
    for number, (technology, label) in enumerate(routes):
        for position, country in enumerate(("DE", "CN", "US")):
            identity = (technology, country)
            c, r = coverage.loc[identity], ratios.loc[identity]
            lines.append(
                " & ".join(
                    [
                        label if position == 0 else "",
                        country,
                        f"{c.baseline:.1f}",
                        f"{c.regionalized:.1f}",
                    ]
                    + [f"{r[factor]:.3f}" for factor in (0.1, 1.0, 10.0, 100.0)]
                )
                + r"\\"
            )
        if number < len(routes) - 1:
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    write_text("\n".join(lines), SI / "missing_assessment_sensitivity.tex")
    print("Wrote 15 derived comparisons to SI table.")


def carbon():
    data = case_config.path("data_dir", case_config.load())
    frame = pd.read_csv(data / "si/carbon/carbon_convention_comparison.csv")
    require(
        not frame.duplicated(["technology", "country", "scenario"]).any(),
        "Duplicate carbon comparison result",
    )
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Climate-score ranges across the three production countries and two supply-chain scenarios for each hydrogen route, in kg CO$_2$-eq per kg hydrogen. Conventional denotes the unmodified ReCiPe~2016 midpoint (H) climate method. Explicit denotes the study extension that includes biogenic uptake and release. Each range spans six results}",
        r"\label{tab:carbon_conventions}",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Route & Conventional & Explicit \\",
        r"\midrule",
    ]
    for technology, label in ROUTES:
        group = frame[frame.technology.eq(technology)]
        require(
            len(group) == 6
            and set(zip(group.country, group.scenario, strict=True))
            == {
                (country, scenario)
                for country in ("DE", "CN", "US")
                for scenario in ("baseline", "regionalized")
            },
            "Carbon table requires all six country/scenario results per route",
        )
        values = [
            f"{group[column].min():.2f}--{group[column].max():.2f}"
            for column in ("conventional_climate_score", "explicit_climate_score")
        ]
        lines.append(" & ".join([label, *values]) + " " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    write_text("\n".join(lines), data / "si/carbon_convention_comparison.tex")
    print("Wrote carbon-convention comparison table.")
