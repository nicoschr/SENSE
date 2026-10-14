"""Explicit, independently addressable SENSE runs in one Brightway project."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from . import case_config, study_protocol


def registry_path(cfg):
    return case_config.path("data_dir", cfg) / "study_runs.json"


def run_references(cfg, spec, requests):
    """Read layer-specific run identities for cache invalidation, without Brightway."""
    path = registry_path(cfg)
    records = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    result = []
    for database, mean in requests:
        key = json.dumps([cfg["brightway"]["project"], database, spec["social_layers"][mean]])
        if key not in records:
            raise ValueError(
                f"Missing study run {database}/{mean}; run the corresponding sense-backgrounds, sense-arithmetic or sense-foregrounds stage"
            )
        result.append({"database": database, "mean": mean, "run_directory": records[key]})
    return result


def record_run(cfg, spec, database, mean, results):
    from sense.artifacts import write_json_atomic

    if not results or any(row.get("issues") for row in results):
        raise RuntimeError("SENSE run is empty or reported issues; no study run was registered")
    directories = {row["run_directory"] for row in results}
    if len(directories) != 1:
        raise ValueError("Expected one immutable run directory")
    path = registry_path(cfg)
    records = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    key = json.dumps([cfg["brightway"]["project"], database, spec["social_layers"][mean]])
    records[key] = str(Path(directories.pop()).resolve())
    write_json_atomic(path, records)


def require_run(cfg, spec, database, mean="geometric") -> Path:
    """Resolve the requested social layer, never a database's generic latest run."""
    path = registry_path(cfg)
    key = json.dumps([cfg["brightway"]["project"], database, spec["social_layers"][mean]])
    if not path.exists() or key not in (records := json.loads(path.read_text(encoding="utf-8"))):
        raise ValueError(
            f"No registered {mean} study run for {database}; run the corresponding sense-backgrounds, sense-arithmetic or sense-foregrounds stage"
        )
    directory = Path(records[key])
    manifest = json.loads((directory / "complete.json").read_text(encoding="utf-8"))
    expected = {
        "schema": 3,
        "versions": {
            "ecoinvent": str(cfg["versions"]["ecoinvent"]),
            "psilca": str(cfg["versions"]["psilca"]),
        },
        "project": cfg["brightway"]["project"],
        "database": database,
        "sociosphere": spec["social_layers"][mean],
        "config": json.loads(
            json.dumps(asdict(study_protocol.sense_config(spec, mean, database)), default=str)
        ),
    }
    if any(manifest.get(k) != v for k, v in expected.items()):
        raise ValueError(f"Registered run does not match the current study: {database}/{mean}")
    import sense
    from sense.utils.cache import directory_fingerprint, file_hash, stable_hash

    source = Path(sense.__file__).parent
    source_hash = stable_hash(
        {str(p.relative_to(source)): file_hash(p) for p in sorted(source.rglob("*.py"))}
    )
    if manifest.get("source_hash") != source_hash:
        raise ValueError("Registered SENSE run predates current source code; rerun it")
    for item in manifest["inputs"].values():
        if not Path(item["path"]).is_file() or file_hash(Path(item["path"])) != item["sha256"]:
            raise ValueError(f"Registered run input changed: {item['path']}")
    for item in manifest.get("source_directories", {}).values():
        source_dir = Path(item["path"])
        if (
            not source_dir.is_dir()
            or directory_fingerprint(source_dir, item["suffixes"]) != item["fingerprint"]
        ):
            raise ValueError(f"Registered source directory changed: {source_dir}")
    from sense.data.zolca_reader import archive_fingerprint

    for item in manifest.get("source_archives", {}).values():
        archive = Path(item["path"])
        if (
            not archive.is_file()
            or archive_fingerprint(archive) != item["fingerprint"]
            or file_hash(source / "resources/zolca/reader.jar") != item["reader_sha256"]
        ):
            raise ValueError("Registered PSILCA backup or native reader changed; rerun SENSE")
    for name, checksum in manifest["artifacts"].items():
        artifact = (directory / name).resolve()
        if (
            not artifact.is_relative_to(directory.resolve())
            or not artifact.is_file()
            or file_hash(artifact) != checksum
        ):
            raise ValueError(f"Registered run artifact changed: {name}")
    diagnostics = json.loads((directory / "diagnostics.json").read_text(encoding="utf-8"))
    if diagnostics.get("processing_issues"):
        raise ValueError("Registered run contains processing issues")
    from sense.brightway.state import validate_database_state

    validate_database_state(manifest.get("database_state"))
    return directory


def run_layer(cfg, spec, database, mean="geometric"):
    from sense import run_sense

    results = run_sense(
        project=cfg["brightway"]["project"],
        technosphere_db=database,
        sociosphere_db=spec["social_layers"][mean],
        price_source_db=case_config.ecoinvent_database_name(cfg),
        config=study_protocol.sense_config(spec, mean, database),
        psilca_path=case_config.psilca_source(cfg),
        ecoinvent_version=cfg["versions"]["ecoinvent"],
        psilca_version=cfg["versions"]["psilca"],
        data_dir=case_config.path("data_dir", cfg),
        ecoinvent_geographies_excel=case_config.path("ecoinvent_geographies_excel", cfg),
    )
    record_run(cfg, spec, database, mean, results)
    return results
