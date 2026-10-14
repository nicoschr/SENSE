"""Reliable readers and writers for SENSE-generated artifacts."""

from __future__ import annotations

import json
import os
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

_REPLACE_ATTEMPTS = 10
_REPLACE_INITIAL_DELAY = 0.05


def _replace_with_retry(source: Path, destination: Path) -> None:
    """Atomically install ``source`` at ``destination``, tolerating transient Windows locks.

    ``os.replace`` is atomic on every supported platform, but on Windows it raises
    ``PermissionError`` while another process holds the destination open — typically
    an antivirus scanner or the search indexer inspecting a file written moments
    earlier. Retry with exponential backoff before giving up. On POSIX the first
    attempt succeeds or raises immediately.
    """

    delay = _REPLACE_INITIAL_DELAY
    for attempt in range(_REPLACE_ATTEMPTS):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if os.name != "nt" or attempt == _REPLACE_ATTEMPTS - 1:
                raise
            time.sleep(delay)
            delay *= 2


def read_json(path: Path) -> Any:
    """Read and decode a UTF-8 JSON artifact."""

    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


@contextmanager
def atomic_output_path(path: Path) -> Iterator[Path]:
    """Yield a sibling temporary path and atomically install it on success."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.stem}.",
        suffix=destination.suffix,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        yield temporary
        _replace_with_retry(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_json_atomic(
    path: Path,
    value: Any,
    *,
    indent: int | None = 2,
    sort_keys: bool = False,
) -> None:
    """Atomically replace a JSON artifact after a complete durable write."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                value,
                handle,
                indent=indent,
                ensure_ascii=False,
                sort_keys=sort_keys,
            )
            handle.flush()
            os.fsync(handle.fileno())
        _replace_with_retry(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
