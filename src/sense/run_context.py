"""Typed paths and immutable identifiers for one SENSE execution."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from sense.artifacts import read_json, write_json_atomic
from sense.utils.cache import directory_fingerprint, file_hash, stable_hash
from sense.utils.helpers import slug


def _latest_path(output: Path, project: str, database: str) -> Path:
    return output / "Latest" / f"{stable_hash([project, database])}.json"


def latest_run_directory(output: Path, project: str, database: str) -> Path:
    """Resolve the last completed run for an exact project/database identity."""
    manifest = read_json(_latest_path(output, project, database))
    if manifest.get("project") != project or manifest.get("database") != database:
        raise ValueError("Run pointer does not match the requested project/database")
    return Path(manifest["run_directory"])


@dataclass
class RunEvidence:
    """Keep immutable input and completion manifests separate from reusable caches."""

    directory: Path
    output: Path
    identity: dict[str, Any]

    @classmethod
    def start(
        cls,
        output: Path,
        project: str,
        database: str,
        sociosphere: str,
        config: Any,
        inputs: dict[str, Path],
        *,
        snapshot_inputs: set[str] | None = None,
        source_directories: dict[str, Path] | None = None,
        source_archives: dict[str, Path] | None = None,
        versions: dict[str, str | None] | None = None,
    ) -> RunEvidence:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
        directory = output / "Runs" / f"{stamp}-{uuid4().hex[:12]}"
        directory.mkdir(parents=True, exist_ok=False)
        recorded_inputs = {}
        for name, path in inputs.items():
            if not path.is_file():
                continue
            original = path.resolve()
            if name in (snapshot_inputs or set()):
                path = directory / "InputSnapshots" / (name + path.suffix)
                path.parent.mkdir(exist_ok=True)
                shutil.copyfile(original, path)
            recorded_inputs[name] = {
                "path": str(path.resolve()),
                "sha256": file_hash(path),
                "source_path": str(original),
            }
        source = Path(__file__).parent
        from sense.data.embeddings import model_revision
        from sense.data.zolca_reader import archive_fingerprint

        identity = {
            "schema": 3,
            "versions": versions or {},
            "project": project,
            "database": database,
            "sociosphere": sociosphere,
            "run_directory": str(directory.resolve()),
            "started_utc": stamp,
            "config": json.loads(json.dumps(asdict(config), default=str)),
            "encoder": (
                {
                    "model": config.embedding_model,
                    "revision": model_revision(config.embedding_model, config.embedding_revision),
                }
                if config.use_psilca
                else None
            ),
            "source_hash": stable_hash(
                {
                    str(path.relative_to(source)): file_hash(path)
                    for path in sorted(source.rglob("*.py"))
                }
            ),
            "inputs": recorded_inputs,
            "source_archives": {
                name: {
                    "path": str(path.resolve()),
                    "fingerprint": archive_fingerprint(path),
                    "reader_sha256": file_hash(source / "resources/zolca/reader.jar"),
                }
                for name, path in (source_archives or {}).items()
            },
            "source_directories": {
                name: {
                    "path": str(path.resolve()),
                    "suffixes": [".xml"],
                    "fingerprint": directory_fingerprint(path, [".xml"]),
                }
                for name, path in (source_directories or {}).items()
            },
        }
        write_json_atomic(directory / "inputs.json", identity)
        return cls(directory, output, identity)

    def snapshot_input(self, name: str, original: Path) -> None:
        """Record an input resolved during preflight before inventory writes."""
        path = self.directory / "InputSnapshots" / (name + original.suffix)
        path.parent.mkdir(exist_ok=True)
        shutil.copyfile(original, path)
        self.identity["inputs"][name] = {
            "path": str(path.resolve()),
            "sha256": file_hash(path),
            "source_path": str(original.resolve()),
        }
        write_json_atomic(self.directory / "inputs.json", self.identity)

    def complete(self, *, database_state: dict[str, Any] | None = None) -> None:
        """Publish a completion record and pointer only after all artifacts succeed."""
        manifest = {
            **self.identity,
            "database_state": database_state,
            "completed_utc": datetime.now(UTC).isoformat(),
            "artifacts": {
                str(path.relative_to(self.directory)): file_hash(path)
                for path in sorted(self.directory.rglob("*"))
                if path.is_file()
            },
        }
        write_json_atomic(self.directory / "complete.json", manifest)
        write_json_atomic(
            _latest_path(self.output, self.identity["project"], self.identity["database"]), manifest
        )


@dataclass(frozen=True)
class RunPaths:
    """Canonical locations for cache and audit artifacts produced by a run."""

    data_dir: Path
    output_dir: Path
    psilca_output_dir: Path
    ecoinvent_output_dir: Path
    ecoinvent_tag: str
    psilca_tag: str
    psilca_assessments: Path
    psilca_sector_list: Path
    psilca_nonpositive_report: Path
    ecoinvent_activity_list: Path
    location_mapping: Path
    ecoinvent_embeddings: Path
    psilca_embeddings: Path
    run_diagnostics: Path

    @classmethod
    def create(
        cls,
        *,
        data_dir: Path,
        technosphere_database: str,
        ecoinvent_version: str,
        psilca_version: str | None,
        use_psilca: bool = True,
    ) -> RunPaths:
        """Construct all paths from the run's stable identifiers."""

        for name, value in (
            ("ecoinvent_version", ecoinvent_version),
            ("psilca_version", psilca_version),
        ):
            if value is None and name == "psilca_version" and not use_psilca:
                continue
            if (
                not isinstance(value, str)
                or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+-]*", value) is None
            ):
                raise ValueError(
                    f"{name} must be a non-empty version label without path separators"
                )

        root = Path(data_dir)
        output = root / "Output"
        psilca_output = output / "PSILCA"
        ecoinvent_output = output / "Ecoinvent"
        embedding_root = output / "EmbeddingStore"
        mode_suffix = "" if use_psilca else "_productivity_only"
        ecoinvent_tag = f"{slug(technosphere_database)}_v{ecoinvent_version}{mode_suffix}"
        psilca_tag = f"v{psilca_version}" if use_psilca else "disabled"
        return cls(
            data_dir=root,
            output_dir=output,
            psilca_output_dir=psilca_output,
            ecoinvent_output_dir=ecoinvent_output,
            ecoinvent_tag=ecoinvent_tag,
            psilca_tag=psilca_tag,
            psilca_assessments=psilca_output / f"psilca_assessments_{psilca_tag}.parquet",
            psilca_sector_list=psilca_output / f"psilca_css_list_{psilca_tag}.json",
            psilca_nonpositive_report=(
                psilca_output / f"psilca_zero_or_negative_exchanges_{psilca_tag}.json"
            ),
            ecoinvent_activity_list=(
                ecoinvent_output / f"ecoinvent_activities_with_prices_{ecoinvent_tag}.json"
            ),
            location_mapping=(
                ecoinvent_output
                / f"ecoinvent_location_mapping_v{ecoinvent_version}_{psilca_tag}.xlsx"
            ),
            ecoinvent_embeddings=embedding_root / "ecoinvent" / ecoinvent_tag,
            psilca_embeddings=embedding_root / "psilca" / psilca_tag,
            run_diagnostics=(
                output / "Diagnostics" / f"run_diagnostics_{ecoinvent_tag}_{psilca_tag}.json"
            ),
        )
