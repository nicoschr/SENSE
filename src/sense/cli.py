"""Command-line entry point for SENSE."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from . import __version__
from .config import SenseConfig
from .exceptions import SenseError


def run_sense(**kwargs):
    """Load and invoke the pipeline lazily so metadata commands stay lightweight."""

    from .pipeline import run_sense as execute

    return execute(**kwargs)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="sense", description="SENSE integrated LCA toolkit CLI")
    p.add_argument("--version", action="version", version=f"SENSE {__version__}")

    sub = p.add_subparsers(dest="cmd", required=False)

    sub.add_parser("info", help="Show package information")

    run = sub.add_parser("run", help="Run the pipeline (requires minimal args)")
    run.add_argument("--project", required=True, help="Brightway project name")
    run.add_argument("--technosphere-db", required=True, help="Technosphere DB name (ecoinvent)")
    run.add_argument(
        "--price-source-db", help="Reference background for prices and treatment working hours"
    )
    run.add_argument(
        "--include-empty-inventories",
        action="store_true",
        help="Allow direct social mapping of activities without physical inputs or environmental exchanges",
    )
    run.add_argument(
        "--zero-price-policy",
        choices=("treatment_working_hours", "legacy_price_floor", "zero"),
        default="treatment_working_hours",
        help="Handling of eligible zero-priced activities",
    )
    run.add_argument(
        "--treatment-working-hours-percentile",
        type=float,
        default=20.0,
        help="Percentile of direct hours by physical unit; 0 disables unpriced treatment inventories",
    )
    run.add_argument(
        "--price-floor-percentile",
        type=float,
        default=20.0,
        help="Imported-price percentile used only with legacy_price_floor",
    )
    run.add_argument(
        "--sociosphere-db", default="sociosphere", help="SENSE flow and method namespace"
    )
    source = run.add_mutually_exclusive_group()
    source.add_argument(
        "--psilca-path",
        type=Path,
        help="PSILCA .zolca backup (recommended) or EcoSpold01 XML folder; required unless --no-psilca",
    )
    source.add_argument("--psilca-dir", type=Path, help="Legacy alias for --psilca-path")
    run.add_argument("--ecoinvent-version", default="3.12", help="Ecoinvent version label")
    run.add_argument("--psilca-version", default="4.0", help="PSILCA version label")
    run.add_argument(
        "--no-psilca",
        action="store_true",
        help="Run working-hours-only from an external productivity baseline",
    )
    run.add_argument(
        "--productivity-baseline",
        type=Path,
        help="Parquet productivity baseline (required with --no-psilca)",
    )
    run.add_argument(
        "--productivity-year",
        type=int,
        default=2020,
        help="Productivity reference year (default: 2020)",
    )
    run.add_argument(
        "--productivity-eur-per-usd",
        type=float,
        help=(
            "Ecoinvent price-currency units per international USD "
            "(required with --no-psilca; use 1.0 for an explicitly identical basis)"
        ),
    )
    run.add_argument(
        "--ecoinvent-geographies-excel",
        type=Path,
        help="Path to the ecoinvent Database Overview workbook",
    )
    run.add_argument(
        "--data-dir",
        type=Path,
        default=Path("Data"),
        help="Directory for SENSE caches, worklists, and audits (default: ./Data)",
    )
    run.add_argument(
        "--save-intermediate", action="store_true", help="Save JSON intermediate artifacts"
    )
    run.add_argument("--verbose", action="store_true", help="Show detailed progress and tracebacks")

    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the command-line interface and return its process exit code."""

    args = _parse_args(argv)

    if args.cmd in (None, "info"):
        print(f"SENSE {__version__}")
        print("Social and environmental nexus for sustainability evaluation.")
        print("Use `sense run --help` for pipeline options.")
        return 0

    if args.cmd == "run":
        logging.basicConfig(
            level=logging.DEBUG if args.verbose else logging.INFO,
            format="%(levelname)s: %(message)s",
        )
        try:
            cfg = SenseConfig(
                use_psilca=not args.no_psilca,
                enable_social_flows=not args.no_psilca,
                productivity_baseline_path=args.productivity_baseline,
                productivity_year=args.productivity_year,
                productivity_eur_per_usd=args.productivity_eur_per_usd,
                save_intermediate=bool(args.save_intermediate),
                zero_price_policy=args.zero_price_policy,
                treatment_working_hours_percentile=args.treatment_working_hours_percentile,
                price_floor_percentile=args.price_floor_percentile,
                exclude_empty_inventories=not args.include_empty_inventories,
            )
            results = run_sense(
                project=args.project,
                technosphere_db=args.technosphere_db,
                sociosphere_db=args.sociosphere_db,
                price_source_db=args.price_source_db,
                psilca_path=args.psilca_path,
                psilca_dir=args.psilca_dir,
                ecoinvent_version=args.ecoinvent_version,
                psilca_version=None if args.no_psilca else args.psilca_version,
                config=cfg,
                data_dir=args.data_dir,
                ecoinvent_geographies_excel=args.ecoinvent_geographies_excel,
            )
        except SenseError as exc:
            print(f"SENSE stopped: {exc}", file=sys.stderr)
            return 2
        except Exception as exc:
            if args.verbose:
                logging.getLogger(__name__).exception("SENSE failed")
                return 1
            print(f"SENSE failed: {exc}", file=sys.stderr)
            return 1

        # Basic summary print
        if not results:
            print("SENSE failed: no regions were processed.", file=sys.stderr)
            return 1

        print("SENSE run completed. Summary per ISO2:")
        for r in results:
            print(
                f" - {r.get('ISO2')}: matches={r.get('num_matches')} social_added={r.get('social_added')}"
            )

        return 1 if any(result.get("issues") for result in results) else 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
