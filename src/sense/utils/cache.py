"""Small, deterministic cache-validation helpers used by the pipeline."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from sense.artifacts import write_json_atomic


def stable_hash(value: Any) -> str:
    """Return a SHA-256 digest for JSON-serializable data."""

    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_hash(path: Path) -> str:
    """Hash a file's complete contents without loading it all into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_fingerprint(root: Path, suffixes: Iterable[str]) -> str:
    """Fingerprint a data directory using relative path, size, and nanosecond mtime.

    This avoids re-reading multi-gigabyte licensed databases on every cached run,
    while detecting normal additions, removals, replacements, and edits.

    Args:
        root: Directory whose matching files should be fingerprinted.
        suffixes: Case-insensitive file suffixes to include.

    Returns:
        Stable digest of matching file metadata.
    """

    root = Path(root)
    normalized_suffixes = {suffix.lower() for suffix in suffixes}
    entries = []
    for path in sorted(root.iterdir(), key=lambda candidate: candidate.name):
        if not path.is_file() or path.suffix.lower() not in normalized_suffixes:
            continue
        stat = path.stat()
        entries.append((path.name, stat.st_size, stat.st_mtime_ns))
    return stable_hash(entries)


def read_manifest(path: Path) -> dict[str, Any] | None:
    """Read a JSON manifest, returning ``None`` when absent or invalid."""

    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def write_manifest(path: Path, data: dict[str, Any]) -> None:
    """Write a deterministic JSON cache manifest."""

    write_json_atomic(Path(path), data, sort_keys=True)
