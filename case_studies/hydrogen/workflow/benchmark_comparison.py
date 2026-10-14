"""Match and compare direct SENSE and SOCA intensities."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from bw2data import Database
from tqdm import tqdm

from .soca import get_stakeholder_from_categories, norm, parse_sense_flow_name, read_excel_sheet


def build_ao_and_bw_match(ctx, mapping_df, eco_db):
    """Attach unique Brightway matches to SOCA metadata via the activity overview.

    Read ``OVERVIEW_XLSX`` sheet ``Cut-Off AO`` and join on ecoQuery dataset ID.
    Match stripped, lowercase activity name, reference product and geography in
    ``eco_db``, including normalized product units. Return the enriched table with
    ``bw_key_meta`` and candidate counts. Missing or ambiguous matches
    receive no Brightway key.
    """
    from bw2io.units import normalize_units

    linked = mapping_df.dropna(subset=["dataset_id"])
    versions = linked["eco_version"].astype(str).str.removeprefix("v")
    if not versions.eq(str(ctx.cfg["versions"]["ecoinvent"])).all():
        raise ValueError("SOCA source version does not match the configured ecoinvent version")
    if not linked["system_model"].astype(str).str.lower().eq("cutoff").all():
        raise ValueError("SOCA validation requires the cutoff system model")
    print("Reading overview Excel (Cut-Off AO)...")
    cutoff = read_excel_sheet(ctx.overview_xlsx, "Cut-Off AO", keep_default_na=False)
    cutoff["dataset_id"] = cutoff["ecoQuery URL"].str.extract("/dataset/(\\d+)/").astype("Int64")
    cutoff_small = (
        cutoff[
            [
                "dataset_id",
                "Activity UUID",
                "Activity Name",
                "Geography",
                "Reference Product Name",
                "Unit",
            ]
        ]
        .dropna(subset=["dataset_id"])
        .copy()
    )
    cutoff_small["dataset_id"] = cutoff_small["dataset_id"].astype("int64")
    print(f"  AO rows with dataset_id: {len(cutoff_small)}")
    print(f"  AO unique dataset_ids:  {cutoff_small['dataset_id'].nunique()}")
    spold_with_ao = mapping_df.merge(
        cutoff_small, on="dataset_id", how="left", validate="many_to_one"
    )
    print(f"  Merged rows (spold_with_ao): {len(spold_with_ao)}")
    print("Building Brightway index (name, ref product, location)...")
    bw_index = defaultdict(list)
    for act in eco_db:
        name = act.get("name")
        ref_product = act.get("reference product")
        loc = act.get("location")
        if not (name and ref_product and loc):
            continue
        key = (norm(name), norm(ref_product), norm(loc), normalize_units(norm(act.get("unit"))))
        bw_index[key].append(act)
    print(f"  Brightway index size: {len(bw_index)}")
    meta_cols = ["Activity Name", "Reference Product Name", "Geography", "Unit"]
    unique_meta = spold_with_ao[meta_cols].drop_duplicates().reset_index(drop=True)

    def match_bw(row):
        key = (
            norm(row["Activity Name"]),
            norm(row["Reference Product Name"]),
            norm(row["Geography"]),
            normalize_units(norm(row["Unit"])),
        )
        candidates = bw_index.get(key, [])
        if len(candidates) == 1:
            act = candidates[0]
            return pd.Series(
                {"bw_key_meta": act.key, "bw_name_meta": act["name"], "metadata_n_candidates": 1}
            )
        else:
            return pd.Series(
                {
                    "bw_key_meta": None,
                    "bw_name_meta": None,
                    "metadata_n_candidates": len(candidates),
                }
            )

    uniq_match = unique_meta.apply(match_bw, axis=1)
    unique_meta_matched = pd.concat([unique_meta, uniq_match], axis=1)
    spold_with_ao = spold_with_ao.merge(unique_meta_matched, on=meta_cols, how="left")
    n_matched = spold_with_ao["bw_key_meta"].notna().sum()
    print(f"  Brightway activities uniquely matched: {n_matched}\n")
    return spold_with_ao


def extract_bw_social_flows(eco_db, spold_with_ao, sociosphere_db="sociosphere"):
    """Return SENSE social exchanges for uniquely matched activities as a DataFrame.

    Read ``eco_db`` and referenced biosphere databases without modifying them. Keep
    flows marked ``sense_kind='social'`` with a recognized stakeholder category.
    Return exchange amounts per product unit, typed indicator metadata and dataset IDs joined
    through ``spold_with_ao``; do not calculate supply-chain impacts.
    """
    from sense.brightway.repository import reference_production_amount

    target_keys = set(spold_with_ao["bw_key_meta"].dropna())
    print(f"Extracting social flows from Brightway for {len(target_keys)} activities...")
    eco_data = eco_db.load()
    biosphere_data_cache: dict[str, Any] = {}
    flow_meta_cache: dict[tuple, Any] = {}
    rows = []
    for act_key, ds in tqdm(
        eco_data.items(), total=len(eco_data), desc="Scanning technosphere activities"
    ):
        if act_key not in target_keys:
            continue
        act_name = ds.get("name")
        act_loc = ds.get("location")
        act_rp = ds.get("reference product")
        production = reference_production_amount(
            act_key, [exc for exc in ds.get("exchanges", []) if exc.get("type") == "production"]
        )
        for exc in ds.get("exchanges", []):
            if exc.get("type") != "biosphere":
                continue
            flow_db, flow_code = exc["input"]
            if flow_db != sociosphere_db:
                continue
            flow_key = (flow_db, flow_code)
            if flow_key not in flow_meta_cache:
                bio_data = biosphere_data_cache.get(flow_db)
                if bio_data is None:
                    bio_data = Database(flow_db).load()
                    biosphere_data_cache[flow_db] = bio_data
                obj = bio_data.get(flow_key, {})
                if obj.get("sense_kind") != "social":
                    flow_meta_cache[flow_key] = None
                    continue
                stakeholder_group, subgroup = get_stakeholder_from_categories(obj.get("categories"))
                if stakeholder_group is None:
                    flow_meta_cache[flow_key] = None
                    continue
                raw_flow_name = obj.get("name")
                indicator, assessment_type = parse_sense_flow_name(raw_flow_name)
                meta = {
                    "flow_name_raw": raw_flow_name,
                    "indicator": indicator,
                    "type": assessment_type,
                    "flow_unit": obj.get("unit"),
                    "stakeholder_group": stakeholder_group,
                    "subgroup": subgroup,
                    "raw_categories": obj.get("categories"),
                }
                flow_meta_cache[flow_key] = meta
            meta = flow_meta_cache[flow_key]
            if meta is None:
                continue
            rows.append(
                {
                    "activity_key": act_key,
                    "activity_name": act_name,
                    "activity_location": act_loc,
                    "activity_ref_product": act_rp,
                    "stakeholder_group": meta["stakeholder_group"],
                    "subgroup": meta["subgroup"],
                    "indicator": meta["indicator"],
                    "type": meta["type"],
                    "unit": meta["flow_unit"],
                    "amount": exc["amount"] / production,
                    "flow_key": flow_key,
                    "flow_name_raw": meta["flow_name_raw"],
                    "raw_categories": meta["raw_categories"],
                }
            )
    bw_df = pd.DataFrame(rows)
    print(f"  Social exchanges extracted from Brightway: {len(bw_df)}")
    mapping = (
        spold_with_ao[["bw_key_meta", "dataset_id"]]
        .dropna(subset=["bw_key_meta"])
        .drop_duplicates()
        .rename(columns={"bw_key_meta": "activity_key"})
    )
    bw_df = bw_df.merge(mapping, on="activity_key", how="left")
    bw_df["dataset_id"] = bw_df["dataset_id"].astype("Int64")
    n_with_id = bw_df["dataset_id"].notna().sum()
    print(f"  Social exchanges with dataset_id: {n_with_id}\n")
    return bw_df


def summarize_indicators(bw_df, xml_df):
    """Print overlap counts for distinct (indicator, assessment type) pairs."""
    bw_indicators = set(bw_df[["indicator", "type"]].dropna().itertuples(index=False, name=None))
    xml_indicators = set(xml_df[["indicator", "type"]].dropna().itertuples(index=False, name=None))
    print("Indicator overview")
    print("------------------")
    print(f"  Unique indicators in SENSE / Brightway: {len(bw_indicators)}")
    print(f"  Unique indicators in SOCA / XML:        {len(xml_indicators)}")
    print(f"  Indicators present in both:             {len(bw_indicators & xml_indicators)}")
    print(f"  Only in SENSE / Brightway:              {len(bw_indicators - xml_indicators)}")
    print(f"  Only in SOCA / XML:                     {len(xml_indicators - bw_indicators)}\n")


def factor_neutral_comparison(compare_pos, factor):
    """Remove only the uniform SENSE price factor from the same matched pairs.

    This sensitivity does not establish SOCA's monetary basis or alter any
    Brightway exchanges, matching choices, aggregation, or SOCA values.
    """
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError("Monetary factor must be finite and positive.")
    neutral = compare_pos.copy()
    neutral["sense_value"] = neutral["sense_value"] / factor
    neutral["log_diff"] = np.log10(neutral["sense_value"]) - np.log10(neutral["soca_value"])
    return neutral


def comparison_summary(compare_pos):
    """Return full-sample counts and multiplicative agreement summaries.

    Require nonempty, finite ``log_diff`` values. Report the median log ratio, its
    back-transformed ratio and ``10**mean(abs(log_diff))`` as the typical absolute
    factor difference. Raise ``ValueError`` for an invalid sample.
    """
    values = compare_pos["log_diff"]
    if values.empty or not np.isfinite(values).all():
        raise ValueError("Validation requires nonempty, finite comparison values.")
    return {
        "pairs": int(len(values)),
        "median_log10_ratio": float(values.median()),
        "median_ratio": float(10 ** values.median()),
        "typical_absolute_factor": float(10 ** values.abs().mean()),
    }


def build_compare(bw_df, xml_df):
    """Return matched positive social totals and log10(SENSE/SOCA) by typed indicator.

    Drop missing identifiers/types and missing SOCA weighted hours. Sum SENSE
    ``amount`` and SOCA ``medium_risk_opportunity_hours`` by dataset ID, indicator
    stakeholder, subgroup and assessment type, then compare matching keys.
    Report unmatched identities before selecting the intersection. Keep strictly positive totals
    on both sides and add ``sense_value``, ``soca_value`` and ``log_diff`` columns.
    Print comparison summaries without modifying the input DataFrames.
    """
    print("Preparing data for comparison...")
    bw_df = bw_df.dropna(
        subset=["dataset_id", "stakeholder_group", "subgroup", "indicator", "type"]
    ).copy()
    xml_df = xml_df.dropna(
        subset=[
            "dataset_id",
            "stakeholder_group",
            "subgroup",
            "indicator",
            "type",
            "medium_risk_opportunity_hours",
        ]
    ).copy()
    bw_df["dataset_id"] = bw_df["dataset_id"].astype("int64")
    xml_df["dataset_id"] = xml_df["dataset_id"].astype("int64")
    bw_ids = set(bw_df["dataset_id"].unique())
    xml_ids = set(xml_df["dataset_id"].unique())
    both_ids = bw_ids & xml_ids
    print(f"  Activities with social data in BOTH systems: {len(both_ids)}")
    bw_df = bw_df[bw_df["dataset_id"].isin(both_ids)].copy()
    xml_df = xml_df[xml_df["dataset_id"].isin(both_ids)].copy()
    bw_agg = (
        bw_df.groupby(
            ["dataset_id", "stakeholder_group", "subgroup", "indicator", "type"], dropna=False
        )["amount"]
        .sum()
        .reset_index()
        .rename(columns={"amount": "sense_value"})
    )
    xml_agg = (
        xml_df.groupby(
            ["dataset_id", "stakeholder_group", "subgroup", "indicator", "type"], dropna=False
        )["medium_risk_opportunity_hours"]
        .sum()
        .reset_index()
        .rename(columns={"medium_risk_opportunity_hours": "soca_value"})
    )
    compare = bw_agg.merge(
        xml_agg,
        on=["dataset_id", "stakeholder_group", "subgroup", "indicator", "type"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    print("  Indicator identity coverage:", compare["_merge"].value_counts().to_dict())
    compare = compare.loc[compare["_merge"] == "both"].drop(columns="_merge")
    print(f"  Typed (dataset_id, indicator) pairs present in BOTH systems: {len(compare)}")
    mask_pos = (compare["sense_value"] > 0) & (compare["soca_value"] > 0)
    compare_pos = compare[mask_pos].copy()
    compare_pos["log_diff"] = np.log10(compare_pos["sense_value"]) - np.log10(
        compare_pos["soca_value"]
    )
    print(f"  Pairs with positive values in BOTH systems: {len(compare_pos)}")
    print("\nSummary of log10(SENSE / SOCA) (SOCA as reference):")
    print(compare_pos["log_diff"].describe())
    typical_factor = 10 ** compare_pos["log_diff"].abs().mean()
    print(f"\nTypical absolute factor difference ≈ {typical_factor:.2f}x\n")
    return compare_pos


def summarize_activity_coverage_and_export(bw_df, xml_df, spold_with_ao, output_path: Path):
    """Print activity coverage and overwrite an Excel audit of one-sided coverage.

    An activity is present if it has at least one extracted social exchange, whether
    or not a positive indicator pair survives comparison. Use ``spold_with_ao`` for
    metadata and write ``Only_SENSE`` and ``Only_SOCA`` sheets to ``output_path``.
    """
    all_ids = set(spold_with_ao["dataset_id"].dropna().astype("int64"))
    bw_ids = set(bw_df["dataset_id"].dropna().astype("int64"))
    xml_ids = set(xml_df["dataset_id"].dropna().astype("int64"))
    both_included = bw_ids & xml_ids
    only_sense = bw_ids - xml_ids
    only_soca = xml_ids - bw_ids
    excluded_both = all_ids - (bw_ids | xml_ids)
    print("Activity-level coverage (dataset_id):")
    print("-------------------------------------")
    print(f"  Total activities in AO/spold universe:     {len(all_ids)}")
    print(f"  Included in BOTH sense and SOCA:          {len(both_included)}")
    print(f"  Only sense (SENSE/Brightway):             {len(only_sense)}")
    print(f"  Only SOCA (EcoSpold XML):                  {len(only_soca)}")
    print(f"  Excluded in BOTH (no social flows):        {len(excluded_both)}\n")
    meta_cols = ["dataset_id", "Activity Name", "Geography", "Reference Product Name", "Unit"]
    meta = spold_with_ao[meta_cols].drop_duplicates()
    only_sense_df = (
        meta[meta["dataset_id"].isin(only_sense)]
        .copy()
        .sort_values(["Activity Name", "Reference Product Name", "Geography"])
    )
    only_soca_df = (
        meta[meta["dataset_id"].isin(only_soca)]
        .copy()
        .sort_values(["Activity Name", "Reference Product Name", "Geography"])
    )
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        only_sense_df.to_excel(writer, sheet_name="Only_SENSE", index=False)
        only_soca_df.to_excel(writer, sheet_name="Only_SOCA", index=False)
    print(f"Saved lists of activities only present in one DB to: {output_path}\n")


def prepare_validation_plot_data(ctx, compare_pos, spold_with_ao):
    """Return dataset ID, geography class and log ratio for each positive pair.

    Read geography classifications from ``OVERVIEW_XLSX``. Join one geography per
    dataset from ``spold_with_ao``; reject ambiguous joins or missing geography.
    Retain pair multiplicity, even though indicator/type labels are not exported.
    """
    geography = spold_with_ao[["dataset_id", "Geography"]].dropna().drop_duplicates()
    geography["dataset_id"] = geography["dataset_id"].astype("int64")
    pairs = compare_pos[["dataset_id", "log_diff"]].copy()
    pairs["dataset_id"] = pairs["dataset_id"].astype("int64")
    pairs = pairs.merge(geography, on="dataset_id", how="left", validate="many_to_one")
    if pairs["Geography"].isna().any():
        raise ValueError("Validation pairs are missing geography metadata.")
    geo = read_excel_sheet(ctx.overview_xlsx, "Geographies", dtype=str, keep_default_na=False)
    countries = set(
        geo.loc[
            geo["Geography Classification"].str.strip().isin(["Country", "Province or State"]),
            "Shortname",
        ].str.strip()
    )
    pairs["geo_class"] = np.where(
        pairs["Geography"].str.strip().isin(countries), "Country/State", "Region/Other"
    )
    return pairs[["dataset_id", "geo_class", "log_diff"]]
