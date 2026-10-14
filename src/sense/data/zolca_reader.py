"""Private, read-only transport for native openLCA PSILCA backups.

Only the public Java/Derby reader is bundled. Licensed records stay in the local
subprocess pipe and existing SENSE cache. No GUI, IPC service, download, or full
archive extraction is needed.
"""

from __future__ import annotations

import base64
import os
import re
import shutil
import stat
import subprocess
import tempfile
import threading
import zipfile
from collections.abc import Generator
from importlib.resources import as_file, files
from pathlib import Path, PurePosixPath

from sense.utils.cache import stable_hash

READER_PROTOCOL = 1
READER_TIMEOUT_SECONDS = 1800


def archive_fingerprint(path: Path) -> str:
    """Validate a native backup and hash its ZIP directory, without reading records."""
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"PSILCA backup not found: {path}")
    if path.suffix.lower() != ".zolca":
        raise ValueError("Use the PSILCA openLCA .zolca backup, not the library ZIP")
    try:
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            names = set()
            directory = []
            for entry in entries:
                name = entry.filename
                relative = PurePosixPath(name)
                if (
                    relative.is_absolute()
                    or ".." in relative.parts
                    or "\\" in name
                    or ":" in name
                    or name in names
                    or entry.flag_bits & 1
                    or stat.S_ISLNK(entry.external_attr >> 16)
                ):
                    raise ValueError("Unsafe or encrypted PSILCA backup entry rejected")
                names.add(name)
                directory.append((name, entry.file_size, entry.compress_size, entry.CRC))
            if "service.properties" not in names or "seg0/c10.dat" not in names:
                raise ValueError(
                    "Not a native openLCA Derby backup. Supply the .zolca database, "
                    "or use a PSILCA EcoSpold01 XML folder. Library ZIPs are not equivalent."
                )
    except zipfile.BadZipFile:
        raise ValueError("PSILCA .zolca backup is not a valid ZIP archive") from None
    return stable_hash(sorted(directory))


def java_executable() -> str:
    """Locate and check Java 21+, with an actionable error for pip-only installs."""
    home = os.environ.get("JAVA_HOME")
    candidate = Path(home) / "bin" / ("java.exe" if os.name == "nt" else "java") if home else None
    candidates = [str(candidate)] if candidate and candidate.is_file() else []
    on_path = shutil.which("java")
    if on_path and on_path not in candidates:
        candidates.append(on_path)
    guidance = "Install Java 21 with `conda install -c conda-forge openjdk=21`, then reactivate your environment. XML inputs do not require Java."
    for java in candidates:
        try:
            result = subprocess.run(
                [java, "-version"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=15,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        version = re.search(r'(?:openjdk|java) version "(\d+)', result.stderr + result.stdout)
        if not result.returncode and version and int(version[1]) >= 21:
            return java
    raise RuntimeError("The .zolca importer requires Java 21 or later. " + guidance)


def _encoded_path(path: Path) -> str:
    return base64.b64encode(str(path.resolve()).encode("utf-8")).decode("ascii")


def reader_lines(path: Path) -> Generator[list[str], None, None]:
    """Yield the private pipe protocol, closing the JVM and all temporaries on error.

    The archive opens read-only. Java has a bounded 128 MiB archive block cache,
    a 768 MiB heap limit, and a private temporary directory for Derby sort files.
    Raw Java errors are intentionally suppressed because they may contain labels.
    """
    path = Path(path).expanduser().resolve()
    before = path.stat()
    archive_fingerprint(path)
    java = java_executable()
    resource = files("sense.resources").joinpath("zolca", "reader.jar")
    with (
        as_file(resource) as reader,
        tempfile.TemporaryDirectory(prefix="sense-zolca-") as temporary,
    ):
        root = Path(temporary)
        command = [
            java,
            "-Xmx768m",
            "-jar",
            reader.name,
            _encoded_path(path),
            _encoded_path(root),
            "read",
        ]
        timed_out = threading.Event()
        with (root / "reader.stderr").open("wb") as errors:
            try:
                child = subprocess.Popen(
                    command,
                    cwd=reader.parent,
                    stdout=subprocess.PIPE,
                    stderr=errors,
                    text=True,
                    encoding="utf-8",
                    bufsize=1,
                )
            except OSError:
                raise RuntimeError("Unable to start the bundled PSILCA .zolca reader") from None

            def expire() -> None:
                timed_out.set()
                if child.poll() is None:
                    child.kill()

            timer = threading.Timer(READER_TIMEOUT_SECONDS, expire)
            timer.daemon = True
            timer.start()
            try:
                assert child.stdout is not None
                for line in child.stdout:
                    yield line.rstrip("\r\n").split("\t")
                if child.wait() or timed_out.is_set():
                    raise RuntimeError(
                        "PSILCA .zolca read failed or timed out. The backup may be unsupported, "
                        "incomplete, or require more temporary disk space. No partial cache was "
                        "published. Use the original database backup or an EcoSpold01 XML folder. "
                        "Database records and Java error details are suppressed."
                    )
                after = path.stat()
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise RuntimeError(
                        "PSILCA backup changed during import. No cache was published."
                    )
            finally:
                timer.cancel()
                if child.poll() is None:
                    child.kill()
                child.wait()
                if child.stdout is not None:
                    child.stdout.close()
