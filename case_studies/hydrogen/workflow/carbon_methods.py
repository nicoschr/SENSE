"""Explicit biogenic-carbon accounting for the hydrogen climate indicator."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import bw2data as bd
import pandas as pd

SOURCE_FAMILY = "ReCiPe 2016 v1.03, midpoint (H)"
EXPLICIT_FAMILY = SOURCE_FAMILY + " (explicit biogenic carbon)"
CLIMATE = ("climate change", "global warming potential (GWP100)")
CONVENTION = "explicit-biogenic-carbon-v1"


def explicit_factors(source: dict, flows: dict) -> dict:
    """Complete CO2 accounting and include oxidation of biogenic methane.

    Stock-change factors are retained. The resource-correction flow corrects
    allocated uptake; it is not an emission. Non-fossil methane uses the source
    method's fossil-methane factor, including its subsequent CO2 formation.
    """
    result = dict(source)
    identities = {}
    for key, flow in flows.items():
        name = flow["name"]
        if "spatialized" in key[0]:
            name = name.rsplit(", ", 1)[0]
        identities[key] = (name, tuple(flow.get("categories", ())), flow.get("unit"))
    fossil = {
        (cats, unit): source[key]
        for key, (name, cats, unit) in identities.items()
        if name == "Methane, fossil" and key in source
    }
    factors = {
        "Carbon dioxide, in air": -1.0,
        "Carbon dioxide, non-fossil": 1.0,
        "Carbon dioxide, non-fossil, resource correction": -1.0,
    }
    found = set()
    for key, (name, cats, unit) in identities.items():
        if name in factors:
            if unit != "kilogram":
                raise ValueError(f"Unexpected carbon-flow unit: {key}: {unit}")
            result[key] = factors[name]
            found.add(name)
        elif name == "Methane, non-fossil":
            if (cats, unit) not in fossil:
                raise ValueError(f"Missing corresponding fossil-methane factor: {key}")
            result[key] = fossil[cats, unit]
    if found != set(factors):
        raise ValueError(f"Missing biogenic carbon flows: {set(factors) - found}")
    return result


def prepare(data_dir: Path, biosphere_database: str) -> dict:
    """Restore logged legacy edits, then create separately named study methods.

    Restoration is restricted to the exact additions recorded by the former
    uptake-only patch. Original rows are backed up before any method is written.
    No inventory exchanges or SENSE methods are changed.
    """
    output = data_dir / "troubleshooting"
    output.mkdir(parents=True, exist_ok=True)
    flows = {a.key: a.as_dict() for a in bd.Database(biosphere_database)}
    if "biosphere3_spatialized_flows" in bd.databases:
        flows.update({a.key: a.as_dict() for a in bd.Database("biosphere3_spatialized_flows")})
    originals, restored = {}, {}
    legacy = output / "added_negative_co2_flows.xlsx"
    if legacy.exists():
        for row in pd.read_excel(legacy).to_dict("records"):
            method = ast.literal_eval(row["method"])
            if method not in bd.methods:
                raise ValueError(f"Legacy patch refers to an absent method: {method}")
            rows = bd.Method(method).load()
            matches = [
                key
                for key, flow in flows.items()
                if flow["name"] == row["added_flow_name"]
                and " :: ".join(flow.get("categories", ())) == row["added_flow_categories"]
                and key[0] == biosphere_database
            ]
            if len(matches) != 1:
                raise ValueError(f"Ambiguous legacy carbon flow: {row}")
            key = matches[0]
            old = dict(rows).get(key)
            if old is None:
                continue  # already restored
            if old != float(row["copied_cf"]):
                raise ValueError(f"Legacy factor was subsequently changed: {method}, {key}")
            originals[method] = rows
            restored[method] = [r for r in rows if r[0] != key]

    planned, changes = {}, []
    for suffix in ("", " regionalized"):
        source_name = (SOURCE_FAMILY + suffix, *CLIMATE)
        target = (EXPLICIT_FAMILY + suffix, *CLIMATE)
        if source_name not in bd.methods:
            raise ValueError(f"Missing source climate method: {source_name}")
        source = dict(restored.get(source_name, bd.Method(source_name).load()))
        updated = explicit_factors(source, flows)
        planned[target] = (source_name, list(updated.items()))
        for key, value in updated.items():
            if source.get(key, 0) != value:
                changes.append(
                    {
                        "method": target,
                        "flow": key,
                        "name": flows[key]["name"],
                        "before": source.get(key, 0),
                        "after": value,
                    }
                )

    if originals:
        backup = output / "legacy_carbon_patch_backup.json"
        if not backup.exists():
            backup.write_text(
                json.dumps([{"method": k, "rows": v} for k, v in originals.items()], indent=2)
                + "\n"
            )
    for method, rows in restored.items():
        bd.Method(method).write(rows)
    for target, (source_key, rows) in planned.items():
        method = bd.Method(target)
        method.register(
            unit=bd.methods[source_key]["unit"],
            source_method=list(source_key),
            carbon_accounting=CONVENTION,
        )
        method.write(rows)
    report = {
        "convention": CONVENTION,
        "changes_from_source": changes,
        "source_family": SOURCE_FAMILY,
        "study_family": EXPLICIT_FAMILY,
    }
    (output / "explicit_biogenic_carbon.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Prepared explicit biogenic-carbon methods; restored {len(restored)} legacy edits.")
    return report
