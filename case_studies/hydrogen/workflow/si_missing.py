"""Audit archived hydrogen scores without changing Brightway or licensed inputs.

This snapshot has one record for each selected indicator in every source process.
The audit fails if that condition changes instead of treating absent records as
explicit no-data levels. Process coefficients stay in the ignored SENSE Data tree.
"""

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

from . import si_productivity as reconstruction
from .result_archive import _read_table
from .si_common import (
    cached_coefficients,
    load_inputs,
    method_index,
    require,
    save_coefficients,
    write_csv,
)

INDICATORS = {
    "child_employment": "Children in employment, total",
    "fatal_accidents": "Rate of fatal accidents at workplace",
}
FACTORS = (0.1, 1.0, 10.0, 100.0)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def source_table(output):
    raw = pd.read_parquet(output / "PSILCA/psilca_assessments_v4.0.parquet")
    processes = raw.drop_duplicates("process_index").copy()
    require(
        raw.groupby("process_index").mean_value.nunique().eq(1).all(),
        'SI reconstruction check failed: raw.groupby("process_index").mean_value.nunique().eq(1).all()',
    )
    require(
        processes.mean_value.gt(0).all() and np.isfinite(processes.mean_value).all(),
        "SI reconstruction check failed: processes.mean_value.gt(0).all() and np.isfinite(processes.mean_value).all()",
    )
    for key, name in INDICATORS.items():
        observations = raw[
            raw.indicator.eq(name) & raw.category.eq("Workers") & raw.assessment_type.eq("risk")
        ]
        require(
            len(observations) == len(processes),
            "SI reconstruction check failed: len(observations) == len(processes)",
        )
        require(
            observations.process_index.is_unique,
            "SI reconstruction check failed: observations.process_index.is_unique",
        )
        levels = observations.set_index("process_index").risk_level
        require(
            set(levels).issubset(
                {"no", "very low", "low", "medium", "high", "very high", "no data"}
            ),
            'SI reconstruction check failed: set(levels).issubset({"no", "very low", "low", "medium", "high", "very high", "no data"})',
        )
        processes[key] = processes.process_index.map(levels)
        processes[key + "_subcategory"] = processes.process_index.map(
            observations.set_index("process_index").subcategory
        )
    processes["country_key"] = processes.location.str.strip().str.lower()
    processes["sector_key"] = processes.reference_name.str.strip().str.lower()
    return processes


def weights(frame, provider):
    countries = frame.groupby("Activity code", sort=False).PSILCA_country.agg(
        lambda x: tuple(dict.fromkeys(x))
    )
    resolutions = {cs: provider.resolve_iso2(cs).weights for cs in set(countries)}
    by_code = countries.map(resolutions)
    return np.array(
        [
            by_code[code].get(country, 0.0)
            for code, country in zip(frame["Activity code"], frame.PSILCA_country, strict=True)
        ]
    )


def coefficients(involved, manifest, sources, output):
    outputs = []
    for database, subset in involved.groupby("database", sort=False):
        run = Path(manifest["metadata"]["run_directories"][database])
        config = json.loads((run / "inputs.json").read_text(encoding="utf-8"))["config"]
        require(
            config["similarity_delta"] == 0.028 and config["social_flow_mean"] == "geometric",
            'SI reconstruction check failed: config["similarity_delta"] == .028 and config["social_flow_mean"] == "geometric"',
        )
        require(
            config["productivity_baseline_path"] is None and config["rounding_digits"] is None,
            'SI reconstruction check failed: config["productivity_baseline_path"] is None and config["rounding_digits"] is None',
        )
        require(
            config["level_weights"]["no data"] == 0.1 and not config["psilca_process_overrides"],
            'SI reconstruction check failed: config["level_weights"]["no data"] == .1 and not config["psilca_process_overrides"]',
        )
        provider = reconstruction.provider_from_audit(run)
        conversion = json.loads(
            next((run / "Monetary").glob("*.json")).read_text(encoding="utf-8")
        )["psilca_price_conversion"]["factor"]
        reference_path = run / "InputSnapshots/treatment_working_hours_reference.json"
        reference = (
            json.loads(reference_path.read_text(encoding="utf-8"))["reference"]
            if reference_path.exists()
            else {}
        )
        needed = set(subset.code)
        visited: set[str] = set()
        snapshots = json.loads(
            (run / "InputSnapshots/priced_activities.json").read_text(encoding="utf-8")
        )
        package_checked = False
        for entry in snapshots:
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
            joined = matched.merge(
                sources, on=["country_key", "sector_key"], how="left", validate="many_to_many"
            )
            require(
                joined.process_index.notna().all(),
                "SI reconstruction check failed: joined.process_index.notna().all()",
            )
            joined["Country_weight"] = weights(joined, provider)
            joined["hours"] = joined.mean_value
            direct = fallback_overrides(selected, reference)
            # Fixed physical hours replace monetary source intensities for
            # treatment recipients. Unit profiles retain the same risk levels.
            joined.loc[joined["Activity code"].isin(direct), "hours"] = 1.0
            for key in INDICATORS:
                joined[key + "_missing"] = joined[key].eq("no data").astype(float)
                joined[key + "_missing_hours"] = joined.hours * joined[key + "_missing"]
                joined[key + "_log"] = np.log(
                    joined.hours * joined[key].map(config["level_weights"])
                )
            fields = ["hours"] + [
                key + suffix
                for key in INDICATORS
                for suffix in ("_missing", "_missing_hours", "_log")
            ]
            grouped = joined.groupby(["Activity code", "PSILCA_country"], sort=False)
            country = grouped[fields].mean()
            country["weight"] = grouped.Country_weight.first()
            total_weights = country.groupby(level=0).weight.transform("sum")
            country.loc[total_weights.eq(0), "weight"] = 1.0
            weighted = country[fields].mul(country.weight, axis=0).groupby(level=0).sum()
            weighted = weighted.div(country.groupby(level=0).weight.sum(), axis=0)
            for key in INDICATORS:
                weighted[key + "_actual"] = np.exp(weighted.pop(key + "_log"))
            if not package_checked:
                check = joined[joined["Activity code"].isin(list(codes)[:5])].copy()
                check["PSILCA_process"] = check.source_file
                check["PSILCA_Exchanges"] = [
                    [
                        {
                            "indicator": name,
                            "type": "risk",
                            "risk level": getattr(row, key),
                            "meanValue": row.hours,
                            "unit": "h",
                            "category": "Workers",
                            "subcategory": getattr(row, key + "_subcategory"),
                        }
                        for key, name in INDICATORS.items()
                    ]
                    for row in check.itertuples()
                ]
                check["Price"] = 1.0
                for factor in (0.1, 100.0):
                    level_weights = {**config["level_weights"], "no data": factor}
                    records = average_sector_mappings_w_prices(
                        check,
                        level_weights,
                        mean_method="geometric",
                        direct_hours_override=dict.fromkeys(direct, 1.0),
                    )
                    for record in records:
                        row = weighted.loc[record["Activity code"]]
                        require(
                            np.isclose(record["Working_hours"], row.hours, rtol=1e-12),
                            'SI reconstruction check failed: np.isclose(record["Working_hours"], row.hours, rtol=1e-12)',
                        )
                        for exchange in record["Averaged_Exchanges"]:
                            key = next(
                                k for k, name in INDICATORS.items() if name == exchange["indicator"]
                            )
                            expected = (
                                row[key + "_actual"] * (factor / 0.1) ** row[key + "_missing"]
                            )
                            require(
                                np.isclose(exchange["meanValue"], expected, rtol=1e-12),
                                'SI reconstruction check failed: np.isclose(exchange["meanValue"], expected, rtol=1e-12)',
                            )
                package_checked = True
            scales = {
                a["code"]: (
                    direct[a["code"]] if a["code"] in direct else float(a["price"]) * conversion
                )
                for a in selected
            }
            weighted["physical_scale"] = weighted.index.map(scales)
            weighted["treatment_fallback"] = weighted.index.isin(direct)
            weighted["database"] = database
            outputs.append(weighted.reset_index().rename(columns={"Activity code": "code"}))
        require(visited == needed, "SI reconstruction check failed: visited == needed")
        print(f"{database}: verified {len(visited)} activity coefficients", flush=True)
    return pd.concat(outputs, ignore_index=True)


def run(*, rebuild=False):
    inputs = load_inputs()
    DATA, OUTPUT = inputs.data, inputs.output
    PRIVATE = REVIEW = DATA / "si/missing_assessments"
    PRIVATE.mkdir(parents=True, exist_ok=True)
    REVIEW.mkdir(parents=True, exist_ok=True)
    archive = DATA / "paired_hydrogen/results.zip"
    source = OUTPUT / "PSILCA/psilca_assessments_v4.0.parquet"
    source_hash = sha(source)
    source_manifest = json.loads((source.with_suffix(".manifest.json")).read_text(encoding="utf-8"))
    require(
        source_hash == source_manifest["artifact_hashes"]["assessments"],
        'SI reconstruction check failed: source_hash == source_manifest["artifact_hashes"]["assessments"]',
    )
    saved = []
    with ZipFile(archive) as z:
        manifest = inputs.manifest
        activities, methods, units = [
            _read_table(z, name) for name in ("activities", "methods", "functional_units")
        ]
        hi = int(methods.loc[methods.kind.eq("working_hours"), "method_index"].item())
        indexes = {key: method_index(methods, name) for key, name in INDICATORS.items()}
        for unit in units.itertuples():
            matrix = sparse.load_npz(io.BytesIO(z.read(unit.contribution_file))).tocsr()
            h = matrix.getrow(hi)
            rows = activities.iloc[h.indices].copy()
            for key, index in indexes.items():
                score = matrix.getrow(index)
                require(
                    not set(score.indices) - set(h.indices),
                    "SI reconstruction check failed: not set(score.indices) - set(h.indices)",
                )
                rows[key + "_saved"] = score.toarray().ravel()[h.indices]
            rows["working_hours"] = h.data
            rows["fu_index"], rows["country"], rows["technology"], rows["scenario"] = (
                unit.fu_index,
                unit.country,
                unit.technology,
                unit.scenario,
            )
            saved.append(rows)
    contributions = pd.concat(saved, ignore_index=True)
    involved = contributions.drop_duplicates(["database", "code"])
    for database in involved.database.unique():
        run = Path(manifest["metadata"]["run_directories"][database])
        require(
            json.loads((run / "InputSnapshots/psilca_manifest.json").read_text(encoding="utf-8"))
            == source_manifest,
            'SI reconstruction check failed: json.loads((run / "InputSnapshots/psilca_manifest.json").read_text(encoding="utf-8")) == source_manifest',
        )
    cache = PRIVATE / "coefficients_private.parquet"
    coeff = None if rebuild else cached_coefficients(cache, inputs.signature)
    if coeff is None:
        coeff = coefficients(involved, manifest, source_table(OUTPUT), OUTPUT)
        save_coefficients(coeff, cache, inputs.signature)
    d = contributions.merge(coeff, on=["database", "code"], validate="many_to_one")
    require(d.hours.notna().all(), "SI reconstruction check failed: d.hours.notna().all()")
    stored_hours = (d.physical_scale * d.hours).to_numpy().astype(np.float32).astype(float)
    require(
        np.isfinite(stored_hours).all() and (stored_hours > 0).all(),
        "SI reconstruction check failed: np.isfinite(stored_hours).all() and (stored_hours > 0).all()",
    )
    d["required_production"] = d.working_hours / stored_hours
    coverage, sensitivity, errors = [], [], {}
    grouping = ["fu_index", "technology", "country", "scenario"]
    for key in INDICATORS:
        baseline = (
            (d.physical_scale * d[key + "_actual"]).to_numpy().astype(np.float32).astype(float)
        )
        reconstructed = d.required_production * baseline
        discrepancy = np.abs(reconstructed - d[key + "_saved"])
        require(
            np.allclose(reconstructed, d[key + "_saved"], rtol=1e-10, atol=1e-13),
            'SI reconstruction check failed: np.allclose(reconstructed, d[key + "_saved"], rtol=1e-10, atol=1e-13)',
        )
        errors[key] = float(discrepancy.max())
        d["missing_labour"] = d.working_hours.abs() * d[key + "_missing_hours"] / d.hours
        for identity, group in d.groupby(grouping):
            coverage.append(
                {
                    **dict(zip(grouping, identity, strict=True)),
                    "indicator": key,
                    "missing_labour_percent": 100
                    * group.missing_labour.sum()
                    / group.working_hours.abs().sum(),
                    "absent_record_labour_percent": 0.0,
                    "unknown_level_labour_percent": 0.0,
                }
            )
        for factor in FACTORS:
            amounts = d.physical_scale * d[key + "_actual"] * (factor / 0.1) ** d[key + "_missing"]
            d["counterfactual"] = d.required_production * amounts.to_numpy().astype(
                np.float32
            ).astype(float)
            for identity, group in d.groupby(grouping):
                sensitivity.append(
                    {
                        **dict(zip(grouping, identity, strict=True)),
                        "indicator": key,
                        "no_data_factor": factor,
                        "score": float(group.counterfactual.sum()),
                        "baseline_score": float(group[key + "_saved"].sum()),
                    }
                )
    write_csv(pd.DataFrame(coverage), REVIEW / "labour_weighted_coverage.csv")
    sensitivity = pd.DataFrame(sensitivity)
    write_csv(sensitivity, REVIEW / "sensitivity_scores.csv")
    pairs = sensitivity.pivot(
        index=["technology", "country", "indicator", "no_data_factor"],
        columns="scenario",
        values="score",
    ).reset_index()
    pairs["regionalized_over_original"] = pairs.regionalized / pairs.baseline
    write_csv(pairs, REVIEW / "paired_ratios.csv")
    evidence = {
        "archive_sha256": sha(archive),
        "source_sha256": source_hash,
        "archive_created_utc": manifest["created_utc"],
        "script_sha256": sha(Path(__file__)),
        "functional_units": len(units),
        "activity_contributions_checked": len(d),
        "distinct_activities_checked": len(involved),
        "maximum_absolute_reconstruction_errors": errors,
        "source_record_coverage": "One record for each selected indicator in every source process. No absent or unknown levels.",
        "labour_coverage": "Absolute archived working hours, partitioned by matched source-process levels using the original arithmetic labour and country weights.",
        "sensitivity": "No-data factor changed before the original geometric aggregation. Inventory, prices, matches and other factors fixed. Float32 exchange storage reproduced.",
        "factors": FACTORS,
        "limits": "Not uncertainty intervals. Coverage is within SENSE-mapped labour and does not establish representativeness of populated records.",
        "runs": manifest["metadata"]["run_directories"],
    }
    inputs.check_unchanged()
    write_json_atomic(REVIEW / "evidence.json", evidence)
    print(
        pd.DataFrame(coverage)
        .groupby("indicator")
        .missing_labour_percent.agg(["min", "max"])
        .to_string()
    )
    print(pairs[pairs.indicator.eq("fatal_accidents")].to_string(index=False))
