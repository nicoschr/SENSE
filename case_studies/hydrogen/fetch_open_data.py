#!/usr/bin/env python3
"""Download and verify the open input used by the hydrogen case study."""

from __future__ import annotations

import hashlib
import sys
import urllib.error
import urllib.request
from pathlib import Path

from case_studies.hydrogen.workflow.case_config import (
    TRADE_DATABASE_PATH,
)
from sense.artifacts import atomic_output_path

TRADE_DATABASE_URL = "https://zenodo.org/records/20596474/files/trade_data.db?download=1"
TRADE_DATABASE_SHA256 = "0e178ea34a06da863c826d643d22b700f7431b9088ef85f4abe665270e3f786e"
CHUNK_SIZE = 1 << 20
PROGRESS_INTERVAL = 64 << 20


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fetch_verified(
    url: str,
    sha256: str,
    destination: Path,
    *,
    label: str,
    size_hint: str,
) -> None:
    """Download ``url`` to ``destination`` unless a checksum-verified copy exists."""
    if destination.is_file():
        print(f"Verifying existing file: {destination}")
        if _sha256(destination) == sha256:
            print(f"{label} is already present and verified.")
            return
        print("Existing file has the wrong checksum; downloading a verified replacement.")

    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "SENSE-hydrogen-case-study/1.1"},
    )
    print(f"Downloading {label} to {destination} ({size_hint})...")
    digest = hashlib.sha256()
    downloaded = 0
    next_report = PROGRESS_INTERVAL
    try:
        with atomic_output_path(destination) as partial:
            with (
                urllib.request.urlopen(request, timeout=300) as response,
                partial.open("wb") as stream,
            ):
                while chunk := response.read(CHUNK_SIZE):
                    stream.write(chunk)
                    digest.update(chunk)
                    downloaded += len(chunk)
                    if downloaded >= next_report:
                        print(f"  {downloaded / (1 << 30):.1f} GB downloaded")
                        next_report += PROGRESS_INTERVAL
            if digest.hexdigest() != sha256:
                raise RuntimeError(f"Downloaded {label} failed SHA-256 verification.")
    except (OSError, urllib.error.URLError) as error:
        raise RuntimeError(f"{label} download failed: {error}") from error

    print(f"{label} downloaded and SHA-256 verified.")


def fetch_trade_database(destination: Path = TRADE_DATABASE_PATH) -> None:
    """Download the pinned Regioinvent trade database unless already verified."""
    _fetch_verified(
        TRADE_DATABASE_URL,
        TRADE_DATABASE_SHA256,
        destination,
        label="Regioinvent trade database",
        size_hint="about 2.4 GB",
    )


def main() -> int:
    """Fetch all automatically downloadable case-study inputs."""
    try:
        fetch_trade_database()
    except RuntimeError as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
