"""Complete hydrogen results, independent of presentation choices and Brightway."""

from __future__ import annotations

import hashlib
import io
import itertools
import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd
from scipy import sparse


def calculation_signature() -> str:
    """Invalidate calculations when computation changes, not when plotting changes.

    Protocol values that affect calculation are checked separately through context.
    The full protocol, including the original panel choice, remains in provenance.
    """
    root = Path(__file__).parent
    paths = [
        root / name
        for name in (
            "calculate.py",
            "carbon_methods.py",
            "result_archive.py",
            "study_checks.py",
            "study_inputs.py",
            "study_runs.py",
            "case_config.py",
        )
    ] + [root.parent / "audit_indicators.py"]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def activity_metadata(key: tuple, activity) -> dict:
    """Keep exact activity identity and native ISIC Rev.4 classification for grouping.

    Missing or conflicting classifications remain unclassified. No industry is
    inferred from a name or from PSILCA matching; all original classifications stay
    available as JSON for inspection.
    """
    classifications = activity.get("classifications") or []
    labels = sorted(
        {
            str(item[1]).strip()
            for item in classifications
            if isinstance(item, (list, tuple))
            and len(item) >= 2
            and re.match(r"^ISIC\s+rev\.?\s*4\b", str(item[0]), re.IGNORECASE)
        }
    )
    status = "missing" if not labels else "ambiguous" if len(labels) > 1 else "unparseable"
    code = sector = division = "Unclassified"
    if len(labels) == 1:
        match = re.fullmatch(r"(\d{2,4})(?:\s*:\s*(.*))?", labels[0])
        if match:
            code, description = match.groups()
            division = code[:2]
            sector = code + (": " + description if description else "")
            status = "classified"
    return {
        "database": key[0],
        "code": key[1],
        "name": activity.get("name", ""),
        "reference_product": activity.get("reference product", ""),
        "location": activity.get("location", ""),
        "unit": activity.get("unit", ""),
        "isic_code": code,
        "isic_sector": sector,
        "isic_division": division,
        "isic_status": status,
        "classifications": json.dumps(classifications, ensure_ascii=False),
    }


def capture_methods(
    registered: dict,
    environmental_family: str,
    social_layer: str,
    *,
    climate_method: tuple | None = None,
) -> pd.DataFrame:
    """Select a complete environmental family and exactly one SENSE social layer."""
    rows: list[dict] = []
    for key in sorted(registered):
        if key[0] == environmental_family:
            kind = "environmental"
        elif key[:2] == ("SENSE", social_layer):
            if len(key) == 6 and key[-1] in {"risk", "opportunity"}:
                kind = "social_" + key[-1]
            elif key[2:4] == ("Social", "Working hours"):
                kind = "working_hours"
            elif key[2:4] == ("Social", "Working hours by ISIC"):
                kind = "working_hours_isic"
            else:
                raise ValueError(f"Unrecognized SENSE method: {key}")
        else:
            continue
        if climate_method and key == (
            environmental_family,
            "climate change",
            "global warming potential (GWP100)",
        ):
            key = climate_method
        unit = registered[key].get("unit")
        if not unit:
            raise ValueError(f"Missing method unit: {key}")
        encoded = json.dumps(list(key))
        rows.append(
            {
                "method_index": len(rows),
                "method_id": hashlib.sha256(encoded.encode()).hexdigest(),
                "method": encoded,
                "kind": kind,
                "unit": unit,
            }
        )
    table = pd.DataFrame(rows)
    if table.empty or not {"environmental", "social_risk", "working_hours"} <= set(table.kind):
        raise ValueError("Expected environmental, social-risk and total working-hours methods")
    return table


def characterization_rows(lca, methods: pd.DataFrame):
    """Cache one sparse characterization row per method, without dense CF matrices."""
    rows = []
    for key in methods.method:
        lca.switch_method(tuple(json.loads(key)))
        # Column sums also preserve totals for a non-diagonal linear CF matrix.
        rows.append(sparse.csr_matrix(lca.characterization_matrix.sum(axis=0)))
    return sparse.vstack(rows, format="csr")


def activity_contributions(characterization, inventory):
    """Return every signed direct activity contribution, including the upstream system.

    Implicit sparse zeros are exact zeros, not a cutoff. Columns are inventory
    activity columns; these are not recursively overlapping supply-chain subtrees.
    """
    values = (characterization @ inventory).tocsr()
    values.sum_duplicates()
    values.eliminate_zeros()
    if not np.isfinite(values.data).all():
        raise ValueError("Non-finite activity contributions")
    totals = np.asarray(values.sum(axis=1)).ravel()
    if not np.isfinite(totals).all():
        raise ValueError("Non-finite method scores")
    return values, totals


def _write_table(archive, name, table):
    with archive.open(name + ".csv", "w") as raw:
        with io.TextIOWrapper(raw, encoding="utf-8", newline="") as stream:
            table.to_csv(stream, index=False)


def _read_table(archive, name):
    with archive.open(name + ".csv") as stream:
        return pd.read_csv(
            stream,
            keep_default_na=False,
            na_values=[""],
            float_precision="round_trip",
            dtype={
                "code": str,
                "isic_code": str,
                "isic_division": str,
                "activity_code": str,
                "database": str,
                "location": str,
            },
        )


class ResultsWriter:
    """Write all methods and contributors atomically; preserve the last complete result."""

    def __init__(self, path: Path, methods: pd.DataFrame, activities: pd.DataFrame, metadata: dict):
        self.path = Path(path)
        self.methods = methods
        self.activities = activities
        self.metadata = metadata
        self.units: list[dict] = []
        self.scores: list[pd.DataFrame] = []

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=self.path.parent, suffix=".tmp", delete=False) as tmp:
            self.temporary = Path(tmp.name)
        self.archive = ZipFile(self.temporary, "w", compression=ZIP_DEFLATED)
        return self

    def add(self, unit: dict, contributions, totals):
        index = len(self.units)
        if contributions.shape != (len(self.methods), len(self.activities)):
            raise ValueError("Contribution dimensions do not match methods and activities")
        actual = np.asarray(contributions.sum(axis=1)).ravel()
        if not np.isfinite(contributions.data).all() or not np.isfinite(totals).all():
            raise ValueError("Non-finite results")
        if not np.allclose(actual, totals, rtol=1e-12, atol=1e-12):
            raise ValueError("Contributions do not reconcile with method totals")
        unit = {**unit, "fu_index": index, "contribution_file": f"contributions/{index:03d}.npz"}
        buffer = io.BytesIO()
        # Outer ZIP compression handles storage; no second compression pass.
        sparse.save_npz(buffer, contributions, compressed=False)
        self.archive.writestr(unit["contribution_file"], buffer.getbuffer())
        self.units.append(unit)
        scores = self.methods.copy()
        for key, value in unit.items():
            if key != "contribution_file":
                scores[key] = value
        scores["score"] = totals
        self.scores.append(scores)

    def finish(self, coverage: pd.DataFrame) -> pd.DataFrame:
        context = self.metadata["context"]
        expected = set(
            itertools.product(
                context["technologies"], context["countries"], ("baseline", "regionalized")
            )
        )
        actual = [(u["technology"], u["country"], u["scenario"]) for u in self.units]
        if set(actual) != expected or len(actual) != len(expected):
            raise ValueError("Incomplete or duplicate hydrogen reference flows")
        if self.methods.method_index.tolist() != list(range(len(self.methods))):
            raise ValueError("Method order does not match contribution rows")
        if self.activities.activity_index.tolist() != list(range(len(self.activities))):
            raise ValueError("Activity order does not match contribution columns")
        scores = pd.concat(self.scores, ignore_index=True)
        for name, table in {
            "scores": scores,
            "methods": self.methods,
            "activities": self.activities,
            "functional_units": pd.DataFrame(self.units),
            "indicator_coverage": coverage,
        }.items():
            _write_table(self.archive, name, table)
        manifest = {
            "schema": 1,
            "calculation_signature": calculation_signature(),
            "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "metadata": self.metadata,
        }
        self.archive.writestr("manifest.json", json.dumps(manifest, allow_nan=False, indent=2))
        self.archive.close()
        os.replace(self.temporary, self.path)
        print(f"Saved complete scores and activity contributions: {self.path}")
        return scores

    def __exit__(self, *_):
        self.archive.close()
        self.temporary.unlink(missing_ok=True)


def load_results(path: Path, *, context: dict) -> tuple[pd.DataFrame, dict]:
    """Load scores for plotting, validating computation but allowing new panel choices."""
    if not Path(path).is_file():
        raise FileNotFoundError("No complete hydrogen results; run calculate once before plotting")
    with ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        if (
            manifest.get("schema") != 1
            or manifest.get("calculation_signature") != calculation_signature()
        ):
            raise ValueError("Saved results predate calculation changes; rerun calculate")
        if manifest["metadata"]["context"] != context:
            raise ValueError("Saved results use another study calculation; rerun calculate")
        return _read_table(archive, "scores"), manifest["metadata"]


def load_contributions(path: Path, fu_index: int):
    """Read an archived result without Brightway, returning matrix and labelled axes.

    Rows follow methods.method_index; columns follow activities.activity_index.
    The final returned dictionary identifies the hydrogen reference flow.
    """
    with ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        if manifest.get("schema") != 1:
            raise ValueError("Unsupported hydrogen result archive")
        units = _read_table(archive, "functional_units")
        matches = units[units.fu_index.eq(fu_index)]
        if len(matches) != 1:
            raise ValueError(f"Unknown reference flow: {fu_index}")
        unit = matches.iloc[0].to_dict()
        values = sparse.load_npz(io.BytesIO(archive.read(unit["contribution_file"])))
        methods, activities = _read_table(archive, "methods"), _read_table(archive, "activities")
        if values.shape != (len(methods), len(activities)):
            raise ValueError("Contribution dimensions do not match archived axes")
        return values, methods, activities, unit


def contributor_table(path: Path, fu_index: int, method: tuple) -> pd.DataFrame:
    """Return a filterable activity table for one method and hydrogen reference flow.

    Includes zero, negative and unclassified contributions. Group by native ISIC
    columns or activity location using ordinary pandas operations. Contribution
    units come from the method; the activity's reference-product unit is separate.
    """
    values, methods, activities, _ = load_contributions(path, fu_index)
    matches = methods[methods.method.map(lambda key: tuple(json.loads(key)) == tuple(method))]
    if len(matches) != 1:
        raise ValueError(f"Method absent or ambiguous in result archive: {method}")
    row = matches.iloc[0]
    return activities.assign(
        contribution=values.getrow(int(row.method_index)).toarray().ravel(),
        contribution_unit=row.unit,
    )
