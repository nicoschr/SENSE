"""Map existing Brightway databases. Edit the settings below, then run this file.

From the repository root: python examples/apply_sense.py
This changes SENSE-owned exchanges in the selected databases. It does not import
inventories or create a new study project. Relative paths start at the working
directory. See README.md for custom databases and missing-input instructions.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import bw2data as bd

from sense import SenseConfig, run_sense
from sense.exceptions import SenseInputRequiredError

# Edit these settings to match your existing project and downloaded files.
PROJECT = "my-sense-study"
DATABASES = ["ecoinvent-3.12-cutoff"]
PRICE_SOURCE_DB = "ecoinvent-3.12-cutoff"
PSILCA_PATH = Path("Data/psilca/psilca.zolca")
GEOGRAPHIES = Path("Data/Database-Overview-for-ecoinvent-v3.12.xlsx")
ECOINVENT_VERSION = "3.12"
PSILCA_VERSION = "4.0"
DATA_DIR = Path("Data")
SOCIOSPHERE_DB = "sociosphere"  # Created by SENSE. Keep one name for the whole model.
CONFIG = SenseConfig()  # Start with defaults. See README.md to change parameters.


def main() -> int:
    """Run the selected databases sequentially and stop when user input is needed."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        if PROJECT not in bd.projects:
            raise ValueError(
                f"Project {PROJECT!r} is absent. Check the name and Brightway data directory. "
                f"Available projects: {[project.name for project in bd.projects]}"
            )
        bd.projects.set_current(PROJECT)
        if not DATABASES or len(set(DATABASES)) != len(DATABASES):
            raise ValueError("DATABASES must list each intended process database once")
        required = set(DATABASES)
        if PRICE_SOURCE_DB is not None:
            required.add(PRICE_SOURCE_DB)
        missing = required.difference(bd.databases)
        if missing:
            raise ValueError(
                f"Databases absent from {PROJECT!r}: {sorted(missing)}. "
                f"Available databases: {list(bd.databases)}"
            )
        for database in DATABASES:
            print(f"\nMapping {database}", flush=True)
            results = run_sense(
                project=PROJECT,
                technosphere_db=database,
                sociosphere_db=SOCIOSPHERE_DB,
                price_source_db=PRICE_SOURCE_DB,
                psilca_path=PSILCA_PATH,
                ecoinvent_version=ECOINVENT_VERSION,
                psilca_version=PSILCA_VERSION,
                ecoinvent_geographies_excel=GEOGRAPHIES,
                data_dir=DATA_DIR,
                config=CONFIG,
            )
            if not results or any(row.get("issues") for row in results):
                raise RuntimeError(f"Resolve processing issues for {database!r} before continuing")
            print("Diagnostics:", Path(results[0]["run_directory"]) / "diagnostics.json")
        print(f"\nMapping complete. Brightway methods start with ('SENSE', {SOCIOSPHERE_DB!r}).")
        return 0
    except SenseInputRequiredError as error:
        print(f"\nSENSE needs input: {error}", file=sys.stderr)
        print(
            "Fill the required entries in that file, then rerun this script. "
            "See README.md: Custom databases and foregrounds.",
            file=sys.stderr,
        )
        return 2
    except Exception as error:
        print(f"\nSENSE stopped: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
