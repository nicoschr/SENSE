"""Prepare regionalization figure inputs from existing CSVs."""

from pathlib import Path

import numpy as np
import pandas as pd

from .case_config import ecoinvent_database_name, load
from .case_config import path as configured_path
from .figure_io import reuse_inputs, save_inputs
from .study_protocol import load_protocol
from .study_runs import run_references


def match_base_and_regionalized(
    base_csv: Path,
    reg_csv: Path,
    key_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read score CSVs and join their rows one-to-one on activity metadata.

    By default, use name, reference product, location and unit. Missing key columns
    raise ``ValueError``; duplicate join keys raise pandas' ``MergeError``. Preserve
    literal country codes such as NA. Print coverage and return three DataFrames:
    matched rows with ``_base``/``_reg`` suffixes, base-only keys and regionalized-only keys.
    """
    if key_cols is None:
        key_cols = ["name", "reference product", "location", "unit"]

    base_df = pd.read_csv(base_csv, keep_default_na=False, na_values=[""])
    reg_df = pd.read_csv(reg_csv, keep_default_na=False, na_values=[""])

    for col in key_cols:
        if col not in base_df.columns:
            raise ValueError(f"Column '{col}' missing in base_df")
        if col not in reg_df.columns:
            raise ValueError(f"Column '{col}' missing in reg_df")

    matched = pd.merge(
        base_df,
        reg_df,
        on=key_cols,
        how="inner",
        suffixes=("_base", "_reg"),
        validate="one_to_one",
    )

    outer = pd.merge(
        base_df[key_cols],
        reg_df[key_cols],
        on=key_cols,
        how="outer",
        indicator=True,
    )

    only_in_base = outer[outer["_merge"] == "left_only"]
    only_in_reg = outer[outer["_merge"] == "right_only"]

    print("=== Matching summary ===")
    print(f"Base rows total:          {len(base_df)}")
    print(f"Regionalized rows total:  {len(reg_df)}")
    print(f"Matched activities:       {len(matched)}")
    print(f"Unmatched only in base:   {len(only_in_base)}")
    print(f"Unmatched only in region.:{len(only_in_reg)}")

    if not only_in_base.empty:
        print("\nExample activities only in BASE:")
        print(only_in_base.head())

    if not only_in_reg.empty:
        print("\nExample activities only in REGIONALIZED:")
        print(only_in_reg.head())

    return matched, only_in_base, only_in_reg


def prepare_regionalization_values(
    matched, social_only=True, stakeholders=None, focus_stakeholder=None
):
    """Return ``location`` and median ``log_ratio`` columns, one row per valid FU.

    Read paired ``_base``/``_reg`` columns from ``matched``. By default retain the six
    social stakeholder groups; ``focus_stakeholder`` optionally narrows them. Use
    strictly positive score pairs with finite log10(regionalized/base), dropping FUs
    without a valid pair. Raise ``ValueError`` for missing locations or no selected
    indicator pairs. Do not filter locations or modify the input table.
    """
    if stakeholders is None:
        stakeholders = [
            "workers",
            "society",
            "local community",
            "value chain actors",
            "consumer",
            "children",
        ]

    if "location" not in matched.columns:
        raise ValueError("matched must contain a 'location' column.")

    # ---- Find *_base / *_reg indicator pairs ----
    pairs = []
    for col in matched.columns:
        if not col.endswith("_base"):
            continue
        ind = col[:-5]
        if social_only:
            first = ind.split(" | ")[0].strip().lower()
            if first not in stakeholders:
                continue
        reg_col = ind + "_reg"
        if reg_col in matched.columns:
            pairs.append((col, reg_col, ind))

    if not pairs:
        raise ValueError("No indicator pairs found to compute Δ.")

    # Optional: filter indicators by stakeholder
    if focus_stakeholder is not None:
        fs = focus_stakeholder.strip().lower()
        pairs = [p for p in pairs if p[2].split(" | ")[0].strip().lower() == fs]
        if not pairs:
            raise ValueError(f"No indicator pairs match focus_stakeholder={focus_stakeholder!r}")

    base_cols = [p[0] for p in pairs]
    reg_cols = [p[1] for p in pairs]

    base = matched[base_cols].apply(pd.to_numeric, errors="coerce").to_numpy()
    reg = matched[reg_cols].apply(pd.to_numeric, errors="coerce").to_numpy()

    with np.errstate(divide="ignore", invalid="ignore"):
        logdiff = np.log10(reg / base)

    valid = np.isfinite(logdiff) & (base > 0) & (reg > 0)
    logdiff = np.where(valid, logdiff, np.nan)

    # ---- FU-level aggregation across indicators (dependence-aware) ----
    fu_delta = np.full(len(logdiff), np.nan)
    has_values = valid.any(axis=1)
    fu_delta[has_values] = np.nanmedian(logdiff[has_values], axis=1)
    valid_fu = np.isfinite(fu_delta)

    fu_delta = fu_delta[valid_fu]
    locs = matched.loc[valid_fu, "location"].astype(str).to_numpy()

    return pd.DataFrame({"location": locs, "log_ratio": fu_delta})


def run(*, rebuild=False):
    SPEC = load_protocol()
    CASE_CFG = load()
    DATA_DIR = configured_path("data_dir", CASE_CFG)
    CONTEXT = {
        "versions": CASE_CFG["versions"],
        "projects": CASE_CFG["brightway"],
        "social_layers": SPEC["social_layers"],
        "sense": SPEC["sense"],
    }
    OUTPUT_DIR = DATA_DIR / "case_study_results/hydrogen_case_study_results"
    CURRENT_RUN_SOURCES = run_references(
        CASE_CFG,
        SPEC,
        [
            (ecoinvent_database_name(CASE_CFG), "geometric"),
            (ecoinvent_database_name(CASE_CFG, regionalized=True), "geometric"),
            (CASE_CFG["brightway"]["regioinvent_database"], "geometric"),
        ],
    )
    REBUILD_PLOT_INPUTS = rebuild  # True rebuilds only the inexpensive CSV summary

    from .regionalization import valid_score_pair

    if not valid_score_pair(OUTPUT_DIR, CURRENT_RUN_SOURCES):
        raise ValueError(
            "Regionalization CSVs lack valid completed artifacts/provenance; run regionalization"
        )

    base_csv = OUTPUT_DIR / "my_batch_run_base_total_scores.csv"

    reg_csv = OUTPUT_DIR / "my_batch_run_regionalized_total_scores.csv"

    sources = (
        [
            {"path": str(p), "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
            for p in (base_csv, reg_csv)
        ]
        if base_csv.is_file() and reg_csv.is_file()
        else None
    )

    reuse_summary = reuse_inputs(
        DATA_DIR / "figure_inputs/regionalization.zip",
        context=CONTEXT,
        rebuild=REBUILD_PLOT_INPUTS,
        sources=sources,
    )

    if not reuse_summary:
        matched, only_in_base, only_in_reg = match_base_and_regionalized(base_csv, reg_csv)
        values = prepare_regionalization_values(matched)
        save_inputs(
            DATA_DIR / "figure_inputs/regionalization.zip",
            {"values": values},
            metadata={
                "context": CONTEXT,
                "run_sources": CURRENT_RUN_SOURCES,
                "statistic": "Per-FU median log10(regionalized/base) across positive social indicator pairs",
                "sources": [
                    {"path": str(p), "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
                    for p in (base_csv, reg_csv)
                ],
            },
        )
