"""Orchestrate the SOCA benchmark with explicit run provenance."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from bw2data import Database, projects

from sense.utils.helpers import slug

from .benchmark_comparison import (
    build_ao_and_bw_match,
    build_compare,
    comparison_summary,
    extract_bw_social_flows,
    factor_neutral_comparison,
    prepare_validation_plot_data,
    summarize_activity_coverage_and_export,
    summarize_indicators,
)
from .case_config import ecoinvent_database_name, load
from .case_config import path as configured_path
from .figure_io import reuse_inputs
from .soca import load_or_parse_spold, spold_signature
from .study_protocol import load_protocol
from .study_runs import run_references


def load_validation_monetary_audit(ctx, project_name, mean="geometric"):
    """Return (factor, audit path, audit dict) for the configured validation run.

    Read the immutable monetary audit for this exact project and database.
    Raise ``ValueError`` if it does not describe default PSILCA-based conversion
    for the configured versions or its factor is not finite and positive.
    """
    ei_version = str(ctx.cfg["versions"]["ecoinvent"])
    psilca_version = str(ctx.cfg["versions"]["psilca"])
    from .study_runs import require_run

    if project_name != ctx.cfg["brightway"]["project"]:
        raise ValueError("Benchmark must use the configured shared project")
    run_directory = require_run(ctx.cfg, ctx.spec, ctx.technosphere_db, mean)
    diagnostics = json.loads((run_directory / "diagnostics.json").read_text(encoding="utf-8"))
    if diagnostics.get("processing_issues"):
        raise ValueError("Validation requires a SENSE run without processing issues")
    audit_path = (
        run_directory
        / "Monetary"
        / f"monetary_conversion_{slug(ctx.technosphere_db)}_v{ei_version}_v{psilca_version}.json"
    )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (
        str(audit.get("ecoinvent_version")) != ei_version
        or str(audit.get("psilca_version")) != psilca_version
        or audit.get("psilca_enabled") is not True
        or (audit.get("external_productivity_conversion", {}).get("configured") is not False)
    ):
        raise ValueError("Monetary audit does not describe the default PSILCA validation run.")
    factor = float(audit["psilca_price_conversion"]["factor"])
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError("Recorded monetary factor must be finite and positive.")
    return (factor, audit_path, audit)


def main(ctx, project_name=None, savepath=None, mean="geometric"):
    """Return a plotting table and provenance for one existing validation project.

    Select ``project_name`` or the configured default; read the monetary audit,
    SOCA exports, activity overview and SENSE exchanges. Do not modify Brightway
    inventories. Always overwrite ``COVERAGE_EXCEL``. If ``savepath`` is supplied,
    use its stem for a ``_summary.json`` audit of default and factor-neutral values.
    This function does not render figures.
    """
    factor, monetary_audit_path, monetary_audit = load_validation_monetary_audit(
        ctx, project_name or ctx.project_name, mean
    )
    print("=== SENSE vs SOCA validation ===")
    print(f"Brightway project:    {project_name or ctx.project_name}")
    print(f"Technosphere DB:      {ctx.technosphere_db}")
    print(f"Soca EcoSpold folder: {ctx.spold_folder}")
    print(f"Overview Excel:       {ctx.overview_xlsx}")
    print(f"Cache dir:            {ctx.cache_dir}\n")
    if not ctx.spold_folder.exists():
        raise FileNotFoundError(
            f"SPOLD_FOLDER does not exist: {ctx.spold_folder}\nSet the corresponding path in config.local.yaml."
        )
    if not ctx.overview_xlsx.exists():
        raise FileNotFoundError(
            f"OVERVIEW_XLSX not found: {ctx.overview_xlsx}\nSet the corresponding path in config.local.yaml."
        )
    _project = project_name or ctx.project_name
    projects.set_current(_project)
    eco = Database(ctx.technosphere_db)
    print("Using Brightway project:", projects.current)
    print("Using technosphere DB  :", eco.name, "\n")
    mapping_df, xml_df = load_or_parse_spold(ctx)
    spold_with_ao = build_ao_and_bw_match(ctx, mapping_df, eco)
    bw_df = extract_bw_social_flows(eco, spold_with_ao, ctx.spec["social_layers"][mean])
    summarize_indicators(bw_df, xml_df)
    summarize_activity_coverage_and_export(
        bw_df,
        xml_df,
        spold_with_ao,
        ctx.coverage_excel.with_name(f"{ctx.coverage_excel.stem}_{mean}.xlsx"),
    )
    compare_pos = build_compare(bw_df, xml_df)
    neutral = factor_neutral_comparison(compare_pos, factor)
    summaries = {
        "default": comparison_summary(compare_pos),
        "factor_neutral": comparison_summary(neutral),
    }
    print(f"\nMonetary-factor sensitivity: SENSE divided by {factor:.9f}; SOCA unchanged.")
    print(pd.DataFrame.from_dict(summaries, orient="index").to_string())
    print("A median near 1 does not imply agreement for every activity or region.")
    if savepath:
        summary_path = Path(savepath).with_name(Path(savepath).stem + "_summary.json")
        summary_path.write_text(
            json.dumps(
                {
                    "project": _project,
                    "technosphere_database": ctx.technosphere_db,
                    "monetary_audit_path": str(monetary_audit_path),
                    "monetary_audit": monetary_audit,
                    "comparisons": summaries,
                    "note": "Factor-neutral sensitivity, not a verified SOCA currency correction. Statistics use all positive pairs; plots show [-4, 4].",
                },
                indent=2,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        print(f"Saved comparison summaries to: {summary_path.resolve()}")
    return (
        prepare_validation_plot_data(ctx, compare_pos, spold_with_ao),
        {
            "factor": factor,
            "monetary_audit": monetary_audit,
            "monetary_audit_path": str(monetary_audit_path),
            "project": _project,
            "comparisons": summaries,
            "soca_signature": spold_signature(ctx.spold_folder, ctx.spold_pattern),
            "geographies_excel": str(ctx.overview_xlsx),
        },
    )


def run(*, rebuild=False):
    """Prepare both benchmark aggregations; reuse a compatible saved snapshot."""
    from bw2data import databases

    from .figure_io import save_inputs
    from .study_runs import require_run

    cfg, spec = load(), load_protocol()
    data = configured_path("data_dir", cfg)
    database = ecoinvent_database_name(cfg)
    context = {
        "versions": cfg["versions"],
        "projects": cfg["brightway"],
        "social_layers": spec["social_layers"],
        "sense": spec["sense"],
    }
    requests = [(database, mean) for mean in ("arithmetic", "geometric")]
    for requested, mean in requests:
        require_run(cfg, spec, requested, mean)
    sources = run_references(cfg, spec, requests)
    from sense.utils.cache import file_hash

    sources.append(
        {
            "soca_signature": spold_signature(configured_path("soca_dir", cfg), "*.xml"),
            "overview_sha256": file_hash(configured_path("ecoinvent_geographies_excel", cfg)),
        }
    )
    snapshot = data / "figure_inputs/validation.zip"
    if reuse_inputs(snapshot, context=context, rebuild=rebuild, sources=sources):
        return
    if cfg["brightway"]["project"] not in projects:
        raise ValueError("Run check-backgrounds or build-backgrounds first")
    projects.set_current(cfg["brightway"]["project"])
    if database not in databases:
        raise ValueError("Run check-backgrounds or build-backgrounds first")
    for mean in ("arithmetic", "geometric"):
        require_run(cfg, spec, database, mean)
    cache = data / "cache/soca"
    cache.mkdir(parents=True, exist_ok=True)
    ctx = SimpleNamespace(
        cfg=cfg,
        spec=spec,
        project_name=cfg["brightway"]["project"],
        technosphere_db=database,
        spold_folder=configured_path("soca_dir", cfg),
        spold_pattern="*.xml",
        overview_xlsx=configured_path("ecoinvent_geographies_excel", cfg),
        coverage_excel=data / "sense_soca_activity_coverage.xlsx",
        use_cache=True,
        force_rebuild_cache=False,
        cache_dir=cache,
        cache_mapping=cache / "mapping_df.parquet",
        cache_xml=cache / "xml_df.parquet",
        cache_meta=cache / "spold_signature.json",
    )
    tables, metadata = {}, {"context": context, "sources": sources}
    for mean in ("arithmetic", "geometric"):
        tables[mean], metadata[mean] = main(
            ctx, project_name=ctx.project_name, mean=mean, savepath=data / f"validation_{mean}.svg"
        )
    save_inputs(snapshot, tables, metadata)
