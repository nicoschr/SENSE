"""Validate the local environment and private inputs for the hydrogen case study."""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import subprocess
import sys
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from zipfile import BadZipFile

from case_studies.hydrogen.workflow.case_config import (
    TRADE_DATABASE_PATH,
    ecoinvent_credentials,
    load,
    path,
    psilca_source,
)

CASE_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = CASE_DIR.parents[1]


@dataclass(frozen=True)
class RequiredInput:
    """One file or populated directory required by the workflow."""

    label: str
    path: Path
    patterns: tuple[str, ...] = ()

    def is_available(self) -> bool:
        if not self.path.exists():
            return False
        if not self.patterns:
            return self.path.is_file()
        return self.path.is_dir() and any(
            next(self.path.glob(pattern), None) is not None for pattern in self.patterns
        )


# Executed in a child interpreter: import the modules whose native libraries can
# fail at load time, then run one sparse solve through bw2calc's bound solver and
# report which backend it is.
_RUNTIME_CHECK = """
import importlib, sys
failed = []
for name in (
    "numpy", "scipy", "torch", "sentence_transformers",
    "bw2data", "bw2calc", "bw2io", "regioinvent", "geopandas",
):
    try:
        importlib.import_module(name)
    except BaseException as error:  # noqa: BLE001 - report anything that stops the import
        failed.append(f"{name}: {type(error).__name__}: {error}")
if failed:
    print("FAILED " + " | ".join(failed))
    sys.exit(1)
import numpy as np
from scipy.sparse import csc_matrix
import bw2calc.lca as lca
solution = lca.spsolve(csc_matrix(np.eye(2)), np.array([2.0, 3.0]))
assert np.allclose(solution, [2.0, 3.0]), solution
print("SOLVER " + lca.factorized.__module__)
"""

REQUIRED_MODULES = (
    "brightway2",
    "bw2analyzer",
    "bw2calc",
    "bw2data",
    "bw2io",
    "geopandas",
    "matplotlib",
    "numpy",
    "openpyxl",
    "pandas",
    "regioinvent",
    "sense",
    "sentence_transformers",
    "tqdm",
    "yaml",
)


def _module_available(name: str) -> bool:
    """Handle an absent parent namespace for optional modules."""
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False


def _status(ok: bool, label: str, detail: str = "") -> None:
    suffix = f": {detail}" if detail else ""
    print(f"[{'OK' if ok else 'MISSING'}] {label}{suffix}")


def _distribution_version(distribution: str) -> str | None:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def _check_environment(cfg: dict, *, require_credentials: bool = True) -> bool:
    valid = True
    python_ok = sys.version_info[:3] == (3, 11, 15)
    _status(python_ok, "tested Python version", sys.version.split()[0])
    valid &= python_ok

    for module in REQUIRED_MODULES:
        found = importlib.util.find_spec(module) is not None
        _status(found, f"Python module {module}")
        valid &= found

    regioinvent_version = _distribution_version("regioinvent")
    regioinvent_ok = regioinvent_version == "1.4.0.2"
    _status(regioinvent_ok, "Regioinvent 1.4.0.2", regioinvent_version or "not installed")
    valid &= regioinvent_ok

    if psilca_source(cfg).suffix.lower() == ".zolca":
        from sense.data.zolca_reader import java_executable

        try:
            java = java_executable()
        except RuntimeError as error:
            _status(False, "Java 21+ for native PSILCA import", str(error))
            valid = False
        else:
            _status(True, "Java 21+ for native PSILCA import", java)

    # `find_spec` above only locates modules. Native-library conflicts can make an
    # import abort Python or fail at load time, so import the heavy modules for real
    # and exercise bw2calc's solver in a child process, where a broken installation
    # cannot pass and cannot take the preflight down with it.
    try:
        runtime_check = subprocess.run(
            [sys.executable, "-c", _RUNTIME_CHECK],
            capture_output=True,
            text=True,
            check=False,
            timeout=180,
        )
    except subprocess.TimeoutExpired:
        _status(False, "Runtime imports and sparse solve", "timed out after 180 seconds")
        return False
    runtime_lines = runtime_check.stdout.strip().splitlines()
    runtime_ok = runtime_check.returncode == 0 and any(
        line.startswith("SOLVER ") for line in runtime_lines
    )
    if runtime_ok:
        runtime_detail = "imports and solve passed"
    else:
        failures = [line[len("FAILED ") :] for line in runtime_lines if line.startswith("FAILED ")]
        stderr_tail = runtime_check.stderr.strip().splitlines()[-1:]
        runtime_detail = (
            failures[0] if failures else " ".join(stderr_tail) or "child process failed"
        )
    _status(runtime_ok, "Runtime imports and sparse solve", runtime_detail)
    valid &= runtime_ok

    # Performance, not correctness: SuperLU gives the same results, slowly. Report
    # which solver bw2calc actually bound rather than asserting a package name.
    if runtime_ok:
        solver_module = next(
            line[len("SOLVER ") :] for line in runtime_lines if line.startswith("SOLVER ")
        )
        if solver_module.startswith("pypardiso"):
            _status(True, "fast sparse solver", "Intel MKL PARDISO (pypardiso)")
        elif _module_available("scikits.umfpack"):
            _status(True, "fast sparse solver", "SciPy + UMFPACK (scikit-umfpack)")
        else:
            print(
                "[WARN] fast sparse solver: SciPy SuperLU - results are correct but LCA "
                "calculations will be much slower. Install pypardiso (x86-64) or "
                "scikit-umfpack (macOS); see the platform environment files."
            )

    if importlib.util.find_spec("sense") is not None:
        import sense
        from sense import run_sense

        parameters = inspect.signature(run_sense).parameters
        source = Path(sense.__file__).resolve()
        sense_ok = (
            sense.__version__ == "1.1.0"
            and "sociosphere_db" in parameters
            and "price_source_db" in parameters
            and source.is_relative_to(REPOSITORY_ROOT)
        )
        _status(sense_ok, "SENSE 1.1 editable checkout", f"{sense.__version__} at {source}")
        valid &= sense_ok

    if require_credentials:
        username, password = ecoinvent_credentials(cfg)
        _status(bool(username), "ecoinvent username", "local config or environment")
        _status(bool(password), "ecoinvent password", "local config or environment")
        valid &= bool(username and password)

    return valid


def _check_inputs(requirements: tuple[RequiredInput, ...]) -> bool:
    valid = True
    for requirement in requirements:
        available = requirement.is_available()
        if available and requirement.path.suffix.lower() == ".zolca":
            from sense.data.zolca_reader import archive_fingerprint

            try:
                archive_fingerprint(requirement.path)
            except (OSError, ValueError) as error:
                _status(False, requirement.label, str(error))
                valid = False
                continue
        _status(available, requirement.label, str(requirement.path))
        valid &= available
    return valid


def _check_study_workbook(cfg: dict) -> bool:
    """Check study selections before any expensive background construction/mapping."""
    if not _module_available("openpyxl"):
        _status(False, "hydrogen study selections", "install openpyxl to validate the workbook")
        return False
    from case_studies.hydrogen.workflow.study_inputs import load_study_inputs

    try:
        selections = load_study_inputs(
            path("hydrogen_inventory", cfg), ecoinvent_version=cfg["versions"]["ecoinvent"]
        )
    except (OSError, ValueError, KeyError, BadZipFile) as error:
        _status(False, "hydrogen study selections", str(error))
        return False
    _status(
        True,
        "hydrogen study selections",
        f"ecoinvent {selections['ecoinvent_version']}; "
        f"{len(selections['activity_specs'])} functional-unit selections",
    )
    return True


def _main_inputs(cfg: dict, *, reuse_backgrounds: bool = False) -> tuple[RequiredInput, ...]:
    version = cfg["versions"]["ecoinvent"]
    psilca_version = cfg["versions"]["psilca"]
    return (
        RequiredInput(
            f"PSILCA {psilca_version} source",
            psilca_source(cfg),
            () if psilca_source(cfg).suffix.lower() == ".zolca" else ("*.xml",),
        ),
        RequiredInput(
            f"ecoinvent {version} Database Overview",
            path("ecoinvent_geographies_excel", cfg),
        ),
        *(
            (RequiredInput("Regioinvent trade database", TRADE_DATABASE_PATH),)
            if not reuse_backgrounds
            else ()
        ),
        RequiredInput("hydrogen foreground inventory", path("hydrogen_inventory", cfg)),
        RequiredInput("methanol foreground inventory", path("methanol_inventory", cfg)),
    )


def _validation_inputs(cfg: dict) -> tuple[RequiredInput, ...]:
    try:
        soca_dir = path("soca_dir", cfg)
    except ValueError:
        # `soca_dir` is optional for hydrogen; only the benchmark needs it.
        soca_dir = CASE_DIR / "<paths.soca_dir is not set in config.local.yaml>"
    return (
        RequiredInput(
            "SOCA v5 EcoSpold01 processes (ecoinvent 3.12)",
            soca_dir,
            ("*.xml",),
        ),
    )


def main(argv: list[str] | None = None) -> int:
    """Run setup checks and return a shell-compatible status code."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reuse-backgrounds",
        action="store_true",
        help="reuse the existing project: no import credentials or trade download required; check-backgrounds checks its databases",
    )
    parser.add_argument(
        "--include-validation",
        action="store_true",
        help="also check ecoinvent/SOCA v5 compatibility and the SOCA input used by the benchmark",
    )
    args = parser.parse_args(argv)

    try:
        cfg = load()
    except (FileNotFoundError, ValueError) as error:
        _status(False, "local configuration", str(error))
        return 1

    print("Environment")
    valid = _check_environment(cfg, require_credentials=not args.reuse_backgrounds)
    print("\nMain case-study inputs")
    valid &= _check_inputs(_main_inputs(cfg, reuse_backgrounds=args.reuse_backgrounds))
    valid &= _check_study_workbook(cfg)
    if args.include_validation:
        print("\nOptional validation inputs")
        ecoinvent_version = str(cfg["versions"]["ecoinvent"])
        compatible = ecoinvent_version == "3.12"
        _status(
            compatible,
            "SOCA v5 background compatibility",
            f"ecoinvent {ecoinvent_version}",
        )
        valid &= compatible
        valid &= _check_inputs(_validation_inputs(cfg))

    if valid:
        print("\nSetup is complete. Run python -m case_studies.hydrogen.run --list.")
        return 0
    print("\nSetup is incomplete. Resolve every MISSING item before running the study.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
