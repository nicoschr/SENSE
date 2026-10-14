"""Input and provenance checks shared by the supplementary analyses."""

import json
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from zipfile import ZipFile

from sense.artifacts import read_json, write_json_atomic
from sense.utils.cache import file_hash, stable_hash

from . import case_config


def require(condition, message):
    """Keep scientific preconditions active even under python -O."""
    if not condition:
        raise ValueError(message)


def run_id(value):
    """Read a recorded run ID across operating systems."""
    return PureWindowsPath(str(value)).name


@dataclass
class Inputs:
    data: Path
    archive: Path
    output: Path
    manifest: dict
    runs: dict
    source: Path
    source_manifest: dict
    source_hash: str
    archive_hash: str
    signature: str

    def check_unchanged(self):
        require(
            file_hash(self.archive) == self.archive_hash, "Hydrogen archive changed during analysis"
        )


def load_inputs():
    """Resolve archived runs under the configured root, not a sibling checkout."""
    cfg = case_config.load()
    data = case_config.path("data_dir", cfg)
    output = data / "Output"
    archive = data / "paired_hydrogen/results.zip"
    archive_hash = file_hash(archive)
    with ZipFile(archive) as zipped:
        manifest = json.loads(zipped.read("manifest.json"))
    source = output / "PSILCA/psilca_assessments_v4.0.parquet"
    source_hash = file_hash(source)
    source_manifest = read_json(source.with_suffix(".manifest.json"))
    require(
        source_hash == source_manifest["artifact_hashes"]["assessments"],
        "PSILCA cache hash mismatch",
    )
    runs, evidence = {}, {}
    for database, recorded in manifest["metadata"]["run_directories"].items():
        directory = output / "Runs" / run_id(recorded)
        complete = read_json(directory / "complete.json")
        identity = read_json(directory / "inputs.json")
        require(complete["database"] == database, "Archive/run database mismatch")
        require(complete["project"] == cfg["brightway"]["project"], "Archive/run project mismatch")
        require(complete["versions"] == cfg["versions"], "Archive/run version mismatch")
        require(
            complete["sociosphere"] == "sociosphere",
            "SI diagnostics require the geometric social layer",
        )
        require(
            all(complete.get(k) == v for k, v in identity.items()),
            "Run completion/input identity mismatch",
        )
        config = identity["config"]
        validate_config(config)
        # Verify every completed artifact before accepting the archived inputs.
        for relative, checksum in complete["artifacts"].items():
            member = PureWindowsPath(relative)
            require(
                not member.is_absolute() and ".." not in member.parts, "Unsafe run artifact path"
            )
            path = directory.joinpath(*member.parts).resolve()
            require(path.is_relative_to(directory.resolve()), "Run artifact escapes its directory")
            require(
                path.is_file() and file_hash(path) == checksum, f"Changed run artifact: {relative}"
            )
        snapshots = directory / "InputSnapshots"
        require(
            read_json(snapshots / "psilca_manifest.json") == source_manifest,
            "PSILCA source differs from archived mapping",
        )
        for name in ("priced_activities", "psilca_manifest"):
            item = identity["inputs"].get(name)
            require(item is not None, f"Missing archived input: {name}")
            require(
                file_hash(snapshots / (name + ".json")) == item["sha256"],
                f"Changed archived input: {name}",
            )
        treatment = identity["inputs"].get("treatment_working_hours_reference")
        if treatment is not None:
            require(
                file_hash(snapshots / "treatment_working_hours_reference.json")
                == treatment["sha256"],
                "Changed treatment-hour reference",
            )
        else:
            activities = read_json(snapshots / "priced_activities.json")
            require(
                not any(
                    a.get("calculation route") == "treatment_fallback"
                    for entry in activities
                    for a in entry["Activities"]
                ),
                "Treatment recipients lack an archived working-hour reference",
            )
        diagnostic = read_json(directory / "diagnostics.json")
        require(not diagnostic.get("processing_issues"), "Archived mapping has processing issues")
        runs[database] = str(directory)
        evidence[database] = complete
    require(bool(runs), "Archive has no mapping-run evidence")
    manifest["metadata"]["run_directories"] = runs
    # Include analysis and package code, not just the changed entry-point file.
    import sense

    code = {
        str(p.relative_to(Path(sense.__file__).parent)): file_hash(p)
        for p in Path(sense.__file__).parent.rglob("*.py")
    }
    # Plot/table styling does not change reconstructed coefficients.
    code.update(
        {
            name: file_hash(Path(__file__).parent / name)
            for name in ("si_common.py", "si_productivity.py", "si_missing.py")
        }
    )
    signature = stable_hash(
        {"archive": archive_hash, "source": source_hash, "runs": evidence, "code": code}
    )
    return Inputs(
        data,
        archive,
        output,
        manifest,
        runs,
        source,
        source_manifest,
        source_hash,
        archive_hash,
        signature,
    )


def validate_config(config):
    require(
        config["similarity_delta"] == 0.028 and config["social_flow_mean"] == "geometric",
        "SI diagnostic requires the fixed matching and aggregation settings",
    )
    require(
        config["productivity_baseline_path"] is None,
        "SI diagnostic requires PSILCA labour intensities",
    )
    require(
        config["rounding_digits"] is None and not config["psilca_process_overrides"],
        "SI reconstruction does not support rounding or source-process overrides",
    )
    require(
        config["zero_price_policy"] == "treatment_working_hours",
        "SI diagnostic requires the physical treatment-hour policy",
    )


def method_index(methods, indicator):
    """Select a unique worker risk method, keeping risk/opportunity identities separate."""
    expected = {
        "Children in employment, total": "Child labour",
        "Rate of fatal accidents at workplace": "Health and safety",
    }
    key = ("SENSE", "sociosphere", "Workers", expected[indicator], indicator, "risk")
    selected = methods[methods.method.map(lambda value: tuple(json.loads(value)) == key)]
    require(len(selected) == 1, f"Archive requires exactly one method: {key}")
    return int(selected.method_index.iloc[0])


def cached_coefficients(path, signature):
    """Accept only a content-verified cache with matching input/code identity."""
    receipt = path.with_suffix(".manifest.json")
    if not path.exists() or not receipt.exists():
        return None
    metadata = read_json(receipt)
    if metadata.get("signature") != signature or metadata.get("sha256") != file_hash(path):
        return None
    import pandas as pd

    return pd.read_parquet(path)


def save_coefficients(frame, path, signature):
    from sense.artifacts import atomic_output_path

    with atomic_output_path(path) as temporary:
        frame.to_parquet(temporary, index=False)
    write_json_atomic(
        path.with_suffix(".manifest.json"), {"signature": signature, "sha256": file_hash(path)}
    )


def write_csv(frame, path):
    from sense.artifacts import atomic_output_path

    with atomic_output_path(path) as temporary:
        frame.to_csv(temporary, index=False)


def write_text(text, path):
    from sense.artifacts import atomic_output_path

    with atomic_output_path(path) as temporary:
        temporary.write_text(text, encoding="utf-8")
