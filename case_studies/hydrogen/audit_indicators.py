"""Report worker-assessment availability without equating it with observation quality."""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

from sense.data.psilca import _parse_assessment_name

KNOWN_LEVELS = {"no", "very low", "low", "medium", "high", "very high"}


def summarize(records: pd.DataFrame) -> pd.DataFrame:
    """Count process coverage and conservative country-sector-label coverage.

    Industry and commodity processes can share a label. A label counts as assessed
    only if every represented process has a recognized assessment for the indicator.
    A recognized assessment may still be a PSILCA proxy, not a primary observation.
    """
    if records.empty:
        raise ValueError("No PSILCA worker records found")
    identity = ["location", "reference_name", "source_file"]
    processes = records[identity].drop_duplicates()
    rows = []
    groups = ["location", "subcategory", "indicator", "assessment_type"]
    for key, group in records.groupby(groups, dropna=False):
        country = processes[processes.location.eq(key[0])]
        flags = (
            group.assign(assessed=group.risk_level.isin(KNOWN_LEVELS))
            .groupby(identity)
            .assessed.all()
        )
        merged = country.merge(flags.rename("assessed"), on=identity, how="left")
        merged["assessed"] = merged.assessed.eq(True)
        labels = merged.groupby("reference_name").assessed.all()
        present = group.source_file.nunique()
        rows.append(
            {
                **dict(zip(groups, key, strict=True)),
                "country_sector_labels": len(labels),
                "assessed_sector_labels": int(labels.sum()),
                "unassessed_sector_labels": int((~labels).sum()),
                "country_processes": len(country),
                "assessed_processes": int(merged.assessed.sum()),
                "absent_processes": len(country) - present,
                "unassessed_processes": int((~merged.assessed).sum()),
                "assessment_coverage_fraction": float(merged.assessed.mean()),
            }
        )
    return pd.DataFrame(rows)


def audit(source: Path, output: Path) -> pd.DataFrame:
    """Read the prepared Parquet cache or a raw EcoSpold01 directory."""
    source = Path(source)
    if source.is_file() and source.suffix == ".parquet":
        records = pd.read_parquet(
            source,
            columns=[
                "location",
                "reference_name",
                "source_file",
                "category",
                "subcategory",
                "indicator",
                "assessment_type",
                "risk_level",
            ],
        )
        records = records[records.category.eq("Workers")].copy()
    else:
        rows = []
        ns = {"e": "http://www.EcoInvent.org/EcoSpold01"}
        for path in sorted(source.glob("process_*.xml")):
            root = ET.parse(path).getroot()
            geography = root.find(".//e:geography", ns)
            reference = root.find(".//e:referenceFunction", ns)
            if geography is None or reference is None:
                continue
            for exc in root.findall(".//e:exchange", ns):
                if exc.get("category") != "Workers":
                    continue
                indicator, level, assessment = _parse_assessment_name(exc.get("name", ""))
                rows.append(
                    {
                        "location": geography.get("location"),
                        "reference_name": reference.get("name"),
                        "source_file": path.name,
                        "category": "Workers",
                        "subcategory": exc.get("subCategory"),
                        "indicator": indicator,
                        "assessment_type": assessment,
                        "risk_level": level,
                    }
                )
        records = pd.DataFrame(rows)
    summary = summarize(records)
    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output / "worker_indicator_coverage.csv", index=False)
    return summary


def audit_study(cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Record coverage of the fixed indicators alongside every manuscript calculation."""
    from case_studies.hydrogen.workflow.case_config import path
    from case_studies.hydrogen.workflow.study_protocol import load_protocol
    from sense.artifacts import write_json_atomic
    from sense.utils.cache import file_hash

    spec = load_protocol()
    data = path("data_dir", cfg)
    source = data / "Output/PSILCA" / f"psilca_assessments_v{cfg['versions']['psilca']}.parquet"
    if not source.is_file():
        raise FileNotFoundError("PSILCA assessment cache is missing; run sense-backgrounds first")
    output = data / "paired_hydrogen/indicator_audit"
    coverage = audit(source, output)
    indicators = [p["method"][-2] for p in spec["panels"][1:]]
    selected = coverage[coverage.indicator.isin(indicators) & coverage.assessment_type.eq("risk")]
    for indicator in indicators:
        for country in spec["countries"]:
            row = selected[selected.indicator.eq(indicator) & selected.location.eq(country)]
            if len(row) != 1:
                raise ValueError(
                    f"Missing or ambiguous indicator coverage record: {country}, {indicator}. "
                    f"Inspect {output}."
                )
    domestic_gaps = selected[
        selected.location.isin(spec["countries"]) & selected.unassessed_sector_labels.gt(0)
    ][["location", "indicator", "country_sector_labels", "unassessed_sector_labels"]]
    for row in domestic_gaps.itertuples(index=False):
        print(
            f"[DATA GAP] {row.location}: {row.indicator}: "
            f"{row.unassessed_sector_labels}/{row.country_sector_labels} sector labels "
            "lack assessments. Scores retain the configured no-data factor; "
            "supply-chain coverage and sensitivity still need evaluation."
        )
    evidence = {
        "design": spec["design"],
        "source": str(source.resolve()),
        "sha256": file_hash(source),
        "manual": spec["indicator_review"]["manual"],
        "scope": "Exported assessment availability, including PSILCA proxies; "
        "not primary-observation quality or supply-chain labour-weighted coverage.",
        "missing_data_factor": spec["sense"]["level_weights"]["no data"],
        "domestic_assessment_gaps": domestic_gaps.to_dict(orient="records"),
        "interpretation": "Indicator selection does not establish robust comparisons. "
        "Supply-chain labour-weighted coverage and missing-data sensitivity are not "
        "evaluated by this source audit.",
    }
    write_json_atomic(output / "evidence.json", evidence)
    selected.to_csv(output / "selected_indicator_coverage.csv", index=False)
    print(f"Saved worker-indicator source coverage: {output}")
    return selected, evidence


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, nargs="?")
    parser.add_argument("output", type=Path, nargs="?")
    args = parser.parse_args(argv)
    if args.source is None and args.output is None:
        from case_studies.hydrogen.workflow.case_config import load

        audit_study(load())
    elif args.source is not None and args.output is not None:
        audit(args.source, args.output)
    else:
        parser.error("Supply both source and output, or neither for the fixed study")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
