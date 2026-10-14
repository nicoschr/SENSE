"""Offline counterfactuals for archived US coal hydrogen; no Brightway writes."""

import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy import sparse

from sense.artifacts import write_json_atomic
from sense.core.averaging import average_sector_mappings_w_prices
from sense.core.matching import match_from_stores
from sense.core.treatment import fallback_overrides
from sense.utils.helpers import slug
from sense.weighting import CountryWeightProvider, attach_country_weights

from .result_archive import _read_table
from .si_common import (
    cached_coefficients,
    load_inputs,
    method_index,
    require,
    save_coefficients,
    write_csv,
)

INDICATOR = "Rate of fatal accidents at workplace"


def compact_lookup(output):
    raw = pd.read_parquet(output / "PSILCA/psilca_assessments_v4.0.parquet")
    require(
        np.isfinite(raw.mean_value).all() and raw.mean_value.gt(0).all(),
        "SI reconstruction check failed: np.isfinite(raw.mean_value).all() and raw.mean_value.gt(0).all()",
    )
    require(
        raw.groupby("process_index").mean_value.nunique().eq(1).all(),
        'SI reconstruction check failed: raw.groupby("process_index").mean_value.nunique().eq(1).all()',
    )
    # Drop unrelated indicators without dropping any source process or changing
    # its working-hour intensity. A dummy indicator retains processes lacking fatal
    # data; it is never used as a fatal-accident exchange or in any final result.
    processes = raw.drop_duplicates("process_index")
    fatal = raw[
        raw.indicator.eq(INDICATOR)
        & raw.assessment_type.eq("risk")
        & raw.category.eq("Workers")
        & raw.subcategory.eq("Health and safety")
    ]
    fatal_groups = dict(tuple(fatal.groupby("process_index", sort=False)))
    rows = []
    for row in processes.itertuples():
        ex = [
            {
                "indicator": "Diagnostic labour intensity",
                "type": "risk",
                "risk level": "medium",
                "meanValue": row.mean_value,
                "unit": "h",
                "category": "Diagnostic",
                "subcategory": "Diagnostic",
            }
        ]
        for f in fatal_groups.get(row.process_index, pd.DataFrame()).itertuples():
            ex.append(
                {
                    "indicator": f.indicator,
                    "type": f.assessment_type,
                    "risk level": f.risk_level,
                    "meanValue": f.mean_value,
                    "unit": f.unit,
                    "category": f.category,
                    "subcategory": f.subcategory,
                }
            )
        rows.append(
            {
                "country_key": row.location.strip().lower(),
                "sector_key": row.reference_name.strip().lower(),
                "PSILCA_process": row.source_file,
                "PSILCA_Exchanges": ex,
            }
        )
    return pd.DataFrame(rows)


def provider_from_audit(run):
    audit = json.loads(next((run / "Weighting").glob("*.json")).read_text(encoding="utf-8"))
    require(audit["method"] == "gdp", 'SI reconstruction check failed: audit["method"] == "gdp"')
    observations = [
        {"country_code": c, "year": x["year"], "gdp_current_usd": x["gdp_current_usd"]}
        for c, x in audit["country_observations"].items()
    ]
    if not observations:
        # Single-country foregrounds don't consult GDP. Provider still uses its
        # single-country path and needs no observations.
        return CountryWeightProvider(method="gdp", reference_year=audit["reference_year"])
    return CountryWeightProvider.from_dataframe(
        pd.DataFrame(observations),
        reference_year=audit["reference_year"],
        max_year_gap=audit["nearest_year_window"],
        source="archived observations used by this exact run",
    )


def extract(records):
    rows = []
    for r in records:
        fatal = [e for e in r["Averaged_Exchanges"] if e.get("indicator") == INDICATOR]
        if len(fatal) > 1:
            raise ValueError("Multiple fatal-accident outputs")
        # Inspect the package output schema rather than guessing here.
        rows.append((r["Activity code"], r["Working_hours"], fatal))
    return rows


def coefficients(activities, runs, output):
    lookup = compact_lookup(output)
    frames = []
    for database, subset in activities.groupby("database", sort=False):
        run = Path(runs[database])
        config = json.loads((run / "inputs.json").read_text(encoding="utf-8"))["config"]
        conversion = json.loads(
            next((run / "Monetary").glob("*.json")).read_text(encoding="utf-8")
        )["psilca_price_conversion"]["factor"]
        reference_path = run / "InputSnapshots/treatment_working_hours_reference.json"
        reference = (
            json.loads(reference_path.read_text(encoding="utf-8"))["reference"]
            if reference_path.exists()
            else {}
        )
        require(
            config["similarity_delta"] == 0.028 and config["social_flow_mean"] == "geometric",
            'SI reconstruction check failed: config["similarity_delta"] == .028 and config["social_flow_mean"] == "geometric"',
        )
        require(
            config["productivity_baseline_path"] is None and config["rounding_digits"] is None,
            'SI reconstruction check failed: config["productivity_baseline_path"] is None and config["rounding_digits"] is None',
        )
        require(
            not config["psilca_process_overrides"],
            'SI reconstruction check failed: not config["psilca_process_overrides"]',
        )
        provider = provider_from_audit(run)
        needed = set(subset.code)
        snapshots = json.loads(
            (run / "InputSnapshots/priced_activities.json").read_text(encoding="utf-8")
        )
        visited: set[str] = set()
        for position, entry in enumerate(snapshots):
            selected = [a for a in entry["Activities"] if a["code"] in needed]
            if not selected:
                continue
            codes = {a["code"] for a in selected}
            require(not codes & visited, "SI reconstruction check failed: not codes & visited")
            visited.update(codes)
            matched = match_from_stores(
                str(output / "EmbeddingStore/ecoinvent" / f"{slug(database)}_v3.12"),
                str(output / "EmbeddingStore/psilca/v4.0"),
                entry["ISO2"],
                activity_entry={**entry, "Activities": selected},
                threshold=0.028,
                minimum_similarity=config["minimum_similarity"],
            )
            matched["country_key"] = matched.PSILCA_country.str.strip().str.lower()
            matched["sector_key"] = matched.Matched_sector.str.strip().str.lower()
            enriched = matched.merge(
                lookup, on=["country_key", "sector_key"], how="left", validate="many_to_many"
            )
            require(
                enriched.PSILCA_process.notna().all(),
                "SI reconstruction check failed: enriched.PSILCA_process.notna().all()",
            )
            enriched = attach_country_weights(enriched, provider)
            # Reconstruct actual physical coefficients, including physical-hour
            # treatment fallbacks. Keep these hours fixed when equalizing the
            # monetary labour intensities because no monetary output is assumed.
            direct = fallback_overrides(selected, reference)
            neutral_weights = dict.fromkeys(config["level_weights"], 1.0)
            modes = {
                "actual_factor": (config["level_weights"], None),
                "equal_intensity_factor": (
                    config["level_weights"],
                    dict.fromkeys(codes - direct.keys(), 1.0),
                ),
                "equal_risk_factor": (neutral_weights, None),
                "both_equal_factor": (neutral_weights, dict.fromkeys(codes - direct.keys(), 1.0)),
            }
            table = None
            for mode, (weights, overrides) in modes.items():
                records = average_sector_mappings_w_prices(
                    enriched.copy(),
                    weights,
                    mean_method="geometric",
                    hours_override=overrides,
                    psilca_price_factor=conversion,
                    productivity_price_factor=conversion,
                    direct_hours_override=direct,
                )
                values = []
                for code, hours, fatal in extract(records):
                    f = fatal[0] if fatal else None
                    # Package outputs meanValue in a Brightway-ready record.
                    amount = float(f["meanValue"]) if f else 0.0
                    values.append(
                        {
                            "code": code,
                            mode: amount,
                            **({"direct_hours": hours} if mode == "actual_factor" else {}),
                        }
                    )
                current = pd.DataFrame(values).set_index("code")
                table = current if table is None else table.join(current, validate="one_to_one")
            if table is None:
                raise ValueError(f"No productivity reconstruction rows for {database}")
            table["database"] = database
            frames.append(table.reset_index())
            if position % 10 == 0 or len(visited) == len(needed):
                print(
                    f"{database}: {len(visited)}/{len(needed)} activities reconstructed", flush=True
                )
        require(
            visited == needed, f"Missing {len(needed-visited)} archived activities in {database}"
        )
        print(f"Finished {database}", flush=True)
    result = pd.concat(frames, ignore_index=True)
    require(
        not result.duplicated(["database", "code"]).any(),
        'SI reconstruction check failed: not result.duplicated(["database", "code"]).any()',
    )
    return result


def run(*, rebuild=False):
    inputs = load_inputs()
    OUTPUT, ARCHIVE = inputs.output, inputs.archive
    HERE = REVIEW = inputs.data / "si/productivity"
    HERE.mkdir(parents=True, exist_ok=True)
    with ZipFile(ARCHIVE) as z:
        manifest = inputs.manifest
        activities = _read_table(z, "activities")
        methods = _read_table(z, "methods")
        units = _read_table(z, "functional_units")
        units = units[
            units.country.eq("US") & units.technology.eq("hydrogen production, coal gasification")
        ]
        require(len(units) == 2, "SI reconstruction check failed: len(units) == 2")
        hi = int(methods.loc[methods.kind.eq("working_hours"), "method_index"].item())
        fi = method_index(methods, INDICATOR)
        saved = []
        for u in units.itertuples():
            matrix = sparse.load_npz(io.BytesIO(z.read(u.contribution_file))).tocsr()
            h, f = matrix.getrow(hi), matrix.getrow(fi)
            require(
                not set(f.indices) - set(h.indices),
                "SI reconstruction check failed: not set(f.indices) - set(h.indices)",
            )
            s = activities.iloc[h.indices].copy()
            s["scenario"] = u.scenario
            s["working_hours_contribution"] = h.data
            s["actual"] = f.toarray().ravel()[h.indices]
            saved.append(s)
    contributions = pd.concat(saved, ignore_index=True)
    involved = contributions.drop_duplicates(["database", "code"])
    source_manifest = json.loads(
        (OUTPUT / "PSILCA/psilca_assessments_v4.0.manifest.json").read_text(encoding="utf-8")
    )
    source_hash = hashlib.sha256(
        (OUTPUT / "PSILCA/psilca_assessments_v4.0.parquet").read_bytes()
    ).hexdigest()
    require(
        source_hash == source_manifest["artifact_hashes"]["assessments"],
        'SI reconstruction check failed: source_hash == source_manifest["artifact_hashes"]["assessments"]',
    )
    monetary_bases = set()
    prices = []
    for db in involved.database.unique():
        run = Path(manifest["metadata"]["run_directories"][db])
        require(
            json.loads((run / "InputSnapshots/psilca_manifest.json").read_text(encoding="utf-8"))
            == source_manifest,
            'SI reconstruction check failed: json.loads((run / "InputSnapshots/psilca_manifest.json").read_text(encoding="utf-8")) == source_manifest',
        )
        money = json.loads(next((run / "Monetary").glob("*.json")).read_text(encoding="utf-8"))[
            "psilca_price_conversion"
        ]
        monetary_bases.add((money["target_basis"]["currency"], money["target_basis"]["year"]))
        for entry in json.loads(
            (run / "InputSnapshots/priced_activities.json").read_text(encoding="utf-8")
        ):
            for a in entry["Activities"]:
                fallback = a.get("calculation route") == "treatment_fallback"
                prices.append(
                    {
                        "database": db,
                        "code": a["code"],
                        "converted_price": 0.0 if fallback else float(a["price"]) * money["factor"],
                        "treatment_fallback": fallback,
                    }
                )
    require(len(monetary_bases) == 1, "SI reconstruction check failed: len(monetary_bases) == 1")
    cache = HERE / "coefficients_private.parquet"
    coeff = None if rebuild else cached_coefficients(cache, inputs.signature)
    if coeff is None:
        coeff = coefficients(involved, manifest["metadata"]["run_directories"], OUTPUT)
        save_coefficients(coeff, cache, inputs.signature)
    d = contributions.merge(coeff, on=["database", "code"], validate="many_to_one")
    d = d.merge(pd.DataFrame(prices), on=["database", "code"], validate="many_to_one")
    require(
        len(d) == len(contributions) and d.actual_factor.notna().all(),
        "SI reconstruction check failed: len(d) == len(contributions) and d.actual_factor.notna().all()",
    )
    require(
        np.isfinite(d.converted_price).all() and d.converted_price.ge(0).all(),
        "SI reconstruction check failed: np.isfinite(d.converted_price).all() and d.converted_price.ge(0).all()",
    )
    # This Brightway generation stores processed exchange amounts as float32
    # (bw2data/data_store.py). Reproduce that storage conversion for the two
    # original biosphere coefficients before recovering the inventory solution.
    stored_hours = d.direct_hours.to_numpy().astype(np.float32).astype(float)
    stored_fatal = d.actual_factor.to_numpy().astype(np.float32).astype(float)
    require((stored_hours > 0).all(), "SI reconstruction check failed: (stored_hours > 0).all()")
    d["required_production"] = d.working_hours_contribution / stored_hours
    d["monetary_output"] = d.required_production * d.converted_price
    d["reconstructed_actual"] = d.required_production * stored_fatal
    for name, factor in [
        ("equal_intensity", "equal_intensity_factor"),
        ("equal_risk", "equal_risk_factor"),
        ("both_equal", "both_equal_factor"),
    ]:
        d[name] = d.required_production * d[factor].to_numpy().astype(np.float32).astype(float)
    # This is deliberately a strict per-activity check, not just matching totals.
    ok = np.isclose(d.actual, d.reconstructed_actual, rtol=1e-10, atol=1e-13)
    if not ok.all():
        d.loc[~ok].to_csv(HERE / "reconstruction_failures_private.csv", index=False)
        raise ValueError(f"Reconstruction failed for {int((~ok).sum())} activity contributions")
    monetary = ~d.treatment_fallback & d.actual_factor.gt(0)
    require(
        np.allclose(d.both_equal_factor[monetary], d.converted_price[monetary]),
        "SI reconstruction check failed: np.allclose(d.both_equal_factor[monetary], d.converted_price[monetary])",
    )
    d["plot_country"] = d.location.map(
        lambda x: x.split("-")[0] if len(x.split("-")[0]) == 2 else x
    )
    d.to_parquet(HERE / "activity_counterfactuals_private.parquet", index=False)
    metrics = ["monetary_output", "actual", "equal_intensity", "equal_risk", "both_equal"]
    summaries, totals = [], []
    for scenario, g in d.groupby("scenario"):
        for metric in metrics:
            denominator = g[metric].abs().sum()
            net = g[metric].sum()
            totals.append(
                {
                    "scenario": scenario,
                    "metric": metric,
                    "net_total": net,
                    "absolute_total": denominator,
                    "negative_share_percent": -g.loc[g[metric].lt(0), metric].sum()
                    / denominator
                    * 100,
                }
            )
            for country, cg in g.groupby("plot_country"):
                summaries.append(
                    {
                        "scenario": scenario,
                        "metric": metric,
                        "country": country,
                        "net_share_percent": cg[metric].sum() / net * 100,
                        "signed_share_of_absolute_percent": cg[metric].sum() / denominator * 100,
                        "absolute_share_percent": cg[metric].abs().sum() / denominator * 100,
                    }
                )
    summary = pd.DataFrame(summaries)
    summary.to_csv(HERE / "country_shares.csv", index=False)
    selected = (
        summary[summary.country.isin(["RU", "CO", "US"])]
        .pivot(index=["scenario", "country"], columns="metric", values="absolute_share_percent")
        .reset_index()
    )
    selected_path = REVIEW / "selected_country_shares.csv"
    write_csv(selected, selected_path)
    pd.DataFrame(totals).to_csv(HERE / "scenario_totals_private.csv", index=False)
    errors = np.abs(d.actual - d.reconstructed_actual)
    audit = {
        "archive_manifest_sha256": hashlib.sha256(
            json.dumps(manifest, sort_keys=True).encode()
        ).hexdigest(),
        "psilca_cache_sha256_verified_against_all_used_run_snapshots": source_hash,
        "common_monetary_basis": list(next(iter(monetary_bases))),
        "scope": "US coal hydrogen, original and regionalized, all nonzero SENSE working-hour contributions",
        "activity_contributions_checked": len(d),
        "maximum_absolute_reconstruction_error": float(errors.max()),
        "relative_l1_reconstruction_error": float(errors.sum() / d.actual.abs().sum()),
        "uncovered_fatal_monetary_absolute_share_percent": float(
            d.loc[d.actual_factor.eq(0), "monetary_output"].abs().sum()
            / d.monetary_output.abs().sum()
            * 100
        ),
        "risk_neutralization": "all assessment multipliers including no-data set to 1; missing indicator exchanges remain missing",
        "hour_neutralization": "monetary-path source-process hour intensities set to 1 before aggregation. Physical treatment fallback hours held fixed.",
        "country_definition": "inventory activity location; subnational codes collapsed; composite regions retained",
        "monetary_scope": "SENSE-covered priced activities only. No monetary output is assigned to treatments using physical-hour fallbacks. All treatments remain in actual and counterfactual social scores.",
        "monetary_recovery": "recover required production from archived working-hours contribution / float32(reconstructed direct hours). Multiply by converted price for priced activities and verify each fatal contribution against float32(reconstructed direct fatal coefficient).",
        "treatment_fallback_absolute_score_share_percent": {
            scenario: float(
                100 * g.loc[g.treatment_fallback, "actual"].abs().sum() / g.actual.abs().sum()
            )
            for scenario, g in d.groupby("scenario")
        },
        "runs": manifest["metadata"]["run_directories"],
    }
    write_json_atomic(HERE / "audit.json", audit)
    public_evidence = {
        **audit,
        "psilca_cache_sha256": source_hash,
        "selected_country_shares_sha256": hashlib.sha256(selected_path.read_bytes()).hexdigest(),
        "archive_run_ids": {
            db: Path(path).name for db, path in manifest["metadata"]["run_directories"].items()
        },
    }
    inputs.check_unchanged()
    write_json_atomic(REVIEW / "evidence.json", public_evidence)
    print(
        summary[summary.country.isin(["RU", "CO", "US"])]
        .pivot(index=["scenario", "country"], columns="metric", values="absolute_share_percent")
        .round(4)
        .to_string()
    )
    print("Reconstruction:", audit["relative_l1_reconstruction_error"], flush=True)
