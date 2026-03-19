"""Simple CLI for SENSE package.

Provides minimal commands for showing info and running the pipeline.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from . import __version__
from .config import SenseConfig
from .pipeline import run_sense


def _parse_args(argv: List[str] | None = None):
    p = argparse.ArgumentParser(prog="sense", description="SENSE social-LCA toolkit CLI")
    p.add_argument("--version", action="version", version=f"SENSE {__version__}")

    sub = p.add_subparsers(dest="cmd", required=False)

    info = sub.add_parser("info", help="Show package information")

    run = sub.add_parser("run", help="Run the pipeline (requires minimal args)")
    run.add_argument("--project", required=True, help="Brightway project name")
    run.add_argument("--technosphere-db", required=True, help="Technosphere DB name (ecoinvent)")
    run.add_argument("--biosphere-db", required=True, help="Biosphere DB name (e.g., biosphere3)")
    run.add_argument("--ecoinvent-dir", required=True, help="Path to ecoinvent EcoSpold02 datasets folder")
    run.add_argument("--psilca-dir", required=True, help="Path to PSILCA EcoSpold01 folder")
    run.add_argument("--ecoinvent-version", default="3.11", help="Ecoinvent version label")
    run.add_argument("--psilca-version", default="4.0", help="PSILCA version label")
    run.add_argument("--foreground-active", action="store_true", help="Attach flows to a foreground DB instead")
    run.add_argument("--foreground-db", default=None, help="Foreground DB name, if using foreground mode")
    run.add_argument("--background-dbs", default=None, help="Comma-separated list of background DBs for pricing")
    run.add_argument("--save-intermediate", action="store_true", help="Save JSON intermediate artifacts")

    return p.parse_args(argv)


def main(argv: List[str] | None = None):
    args = _parse_args(argv)

    if args.cmd in (None, "info"):
        print(f"SENSE {__version__}")
        print("A Python toolkit to integrate PSILCA social data into ecoinvent via Brightway2.")
        print("Use `sense run --help` for pipeline options.")
        return 0

    if args.cmd == "run":
        cfg = SenseConfig()
        cfg.save_intermediate = bool(args.save_intermediate)

        bg = []
        if args.background_dbs:
            bg = [s.strip() for s in args.background_dbs.split(",") if s.strip()]
        else:
            bg = [args.technosphere_db]

        results = run_sense(
            project=args.project,
            technosphere_db=args.technosphere_db,
            biosphere_db=args.biosphere_db,
            sociosphere_db="sociosphere",
            foreground_active=bool(args.foreground_active),
            foreground_database=args.foreground_db,
            background_databases=bg,
            ecoinvent_dir=Path(args.ecoinvent_dir),
            psilca_dir=Path(args.psilca_dir),
            ecoinvent_version=args.ecoinvent_version,
            psilca_version=args.psilca_version,
            config=cfg,
        )

        # Basic summary print
        print("SENSE run completed. Summary per ISO2:")
        for r in results:
            print(f" - {r.get('ISO2')}: matches={r.get('num_matches')} social_added={r.get('social_added')}")

        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
