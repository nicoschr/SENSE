"""Run the optional manuscript workflow from the repository root.

Each stage runs in a fresh process, in order, and stops the workflow on failure.
Use --list or --dry-run to inspect execution without loading study data.
"""

from __future__ import annotations

import argparse
import importlib
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# module, function, keyword arguments, description
STAGES = {
    "check-backgrounds": ("backgrounds", "run", {}, "Check/reuse the three connected backgrounds"),
    "build-backgrounds": (
        "backgrounds",
        "run",
        {"build": True},
        "Import and regionalize in a fresh project",
    ),
    "sense-backgrounds": ("social", "backgrounds", {}, "Geometric SENSE on all three backgrounds"),
    "foregrounds": ("foregrounds", "run", {}, "Build/reuse and check paired hydrogen foregrounds"),
    "sense-foregrounds": (
        "social",
        "foregrounds",
        {},
        "Match prices and map both foreground scenarios",
    ),
    "calculate": ("calculate", "run", {}, "Calculate the 30 hydrogen functional units"),
    "figure-hydrogen": (
        "figures",
        "hydrogen",
        {},
        "Render the paired hydrogen figure from saved scores",
    ),
    "figure-contributions": (
        "figures",
        "contributions",
        {},
        "Render US fatal-accident contributions by activity and country",
    ),
    "sense-arithmetic": (
        "social",
        "arithmetic",
        {},
        "Arithmetic SENSE on original ecoinvent for benchmarking",
    ),
    "regionalization": (
        "regionalization",
        "run",
        {},
        "Calculate/reuse the database-wide regionalization comparison",
    ),
    "regionalization-summary": (
        "regionalization_summary",
        "run",
        {},
        "Prepare regionalization figure inputs from CSVs",
    ),
    "benchmark": ("benchmark", "run", {}, "Compare both SENSE layers with SOCA"),
    "figure-regionalization": (
        "figures",
        "regionalization",
        {},
        "Render the regionalization figure from saved inputs",
    ),
    "figure-benchmark": (
        "figures",
        "benchmark",
        {},
        "Render geometric, arithmetic and monetary-sensitivity figures",
    ),
    "figure-si-matching": (
        "si_matching",
        "run",
        {},
        "Render three matching figures from aggregate audit data",
    ),
    "table-si-coverage": ("si_coverage", "run", {}, "Render the SOCA comparison coverage table"),
    "si-productivity": (
        "si_productivity",
        "run",
        {},
        "Reconstruct the US coal-hydrogen productivity diagnostic",
    ),
    "figure-si-productivity": (
        "si_productivity_plot",
        "run",
        {},
        "Render the productivity distribution and country diagnostic",
    ),
    "si-missing-assessments": (
        "si_missing",
        "run",
        {},
        "Calculate missing-assessment coverage and sensitivities",
    ),
    "table-si-missing-assessments": (
        "si_tables",
        "missing",
        {},
        "Render the missing-assessment sensitivity table",
    ),
    "si-carbon": (
        "si_carbon",
        "run",
        {},
        "Compare climate conventions using the current archived inventory",
    ),
    "table-si-carbon": ("si_tables", "carbon", {}, "Render the carbon-convention comparison table"),
}
GROUPS = {
    "hydrogen": (
        "check-backgrounds",
        "sense-backgrounds",
        "foregrounds",
        "sense-foregrounds",
        "calculate",
        "figure-hydrogen",
        "figure-contributions",
    ),
    "comparisons": (
        "check-backgrounds",
        "sense-arithmetic",
        "benchmark",
        "figure-benchmark",
        "table-si-coverage",
        "regionalization",
        "regionalization-summary",
        "figure-regionalization",
    ),
    "figures": (
        "figure-hydrogen",
        "figure-contributions",
        "figure-regionalization",
        "figure-benchmark",
        "figure-si-matching",
        "figure-si-productivity",
    ),
    "si": (
        "figure-si-matching",
        "si-productivity",
        "figure-si-productivity",
        "si-missing-assessments",
        "table-si-missing-assessments",
        "si-carbon",
        "table-si-carbon",
    ),
    "si-outputs": (
        "figure-si-matching",
        "figure-si-productivity",
        "table-si-missing-assessments",
        "table-si-carbon",
    ),
}
GROUPS["all"] = GROUPS["hydrogen"] + GROUPS["si"] + GROUPS["comparisons"][1:]
GROUPS["recalculate-hydrogen"] = (
    "check-backgrounds",
    "sense-backgrounds",
    "sense-foregrounds",
    "calculate",
    "figure-hydrogen",
    "figure-contributions",
)
REBUILDABLE = {
    "regionalization",
    "regionalization-summary",
    "benchmark",
    "si-productivity",
    "si-missing-assessments",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", nargs="?", choices=(*STAGES, *GROUPS))
    parser.add_argument(
        "--list", action="store_true", help="List stages and groups without accessing data"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print execution order without running it"
    )
    parser.add_argument(
        "--rebuild", action="store_true", help="Refresh saved results for a single comparison stage"
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.list:
        for name, (_, _, _, description) in STAGES.items():
            print(f"{name:25} {description}")
        for name, stages in GROUPS.items():
            print(f"{name:25} {' -> '.join(stages)}")
        return 0
    if not args.stage:
        parser.error("choose a stage/group or use --list")
    if args.rebuild and args.stage not in REBUILDABLE:
        parser.error("--rebuild applies only to " + ", ".join(sorted(REBUILDABLE)))
    if args.worker:
        if args.stage not in STAGES or args.dry_run:
            parser.error("worker requires one executable stage")
        module, function, keywords, _ = STAGES[args.stage]
        keywords = dict(keywords)
        if args.rebuild:
            keywords["rebuild"] = True
        # Imports, configuration and Brightway access happen only at execution.
        getattr(importlib.import_module(f".workflow.{module}", __package__), function)(**keywords)
        return 0
    sequence = GROUPS.get(args.stage, (args.stage,))
    for stage in sequence:
        command = [sys.executable, "-m", "case_studies.hydrogen.run", stage, "--worker"]
        if args.rebuild:
            command.append("--rebuild")
        print(
            f"{'Would run' if args.dry_run else 'Running'}: {stage} — {STAGES[stage][3]}",
            flush=True,
        )
        if args.dry_run:
            continue
        started = time.monotonic()
        result = subprocess.run(
            command, cwd=ROOT, env={**os.environ, "MPLBACKEND": "Agg"}, check=False
        )
        if result.returncode:
            print(f"Stopped at {stage}; exit status {result.returncode}.", file=sys.stderr)
            return result.returncode if result.returncode > 0 else 1
        print(f"Completed {stage} in {(time.monotonic() - started) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
