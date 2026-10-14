"""Exclude concurrent SENSE writers from Brightway and shared cache directories."""

from __future__ import annotations

from functools import wraps
from pathlib import Path
from threading import Lock

from filelock import FileLock, Timeout

_PROCESS_LOCK = Lock()


def exclusive_run(function):
    """Keep Brightway's process-global project state and artifact writes sequential.

    File locks cover separate interpreters. A process lock also protects threads,
    which share Brightway's active project. Fail immediately rather than waiting
    invisibly behind a multi-hour run. Locks release even after an exception.
    """

    @wraps(function)
    def guarded(**kwargs):
        from bw2data import projects

        root = Path(kwargs.get("data_dir") or Path.cwd() / "Data").expanduser()
        root.mkdir(parents=True, exist_ok=True)
        if not _PROCESS_LOCK.acquire(blocking=False):
            raise RuntimeError("Another SENSE run is active in this Python process")
        try:
            with FileLock(Path(projects._base_data_dir) / ".sense-run.lock", timeout=0):
                with FileLock(root / ".sense-run.lock", timeout=0):
                    return function(**kwargs)
        except Timeout as error:
            raise RuntimeError(
                "Another SENSE run owns the Brightway directory or data_dir"
            ) from error
        finally:
            _PROCESS_LOCK.release()

    return guarded
