"""Exercise stage dispatch and dependency boundaries without licensed data."""

import ast
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from case_studies.hydrogen import run


def test_study_paths_roundtrip_with_windows_default_encoding(tmp_path, monkeypatch):
    from case_studies.hydrogen.workflow import study_protocol, study_runs

    read_text = Path.read_text

    def windows_read_text(path, encoding=None, errors=None):
        return read_text(path, encoding=encoding or "cp1252", errors=errors)

    monkeypatch.setattr(Path, "read_text", windows_read_text)
    spec = study_protocol.load_protocol()
    cfg = {"brightway": {"project": "test"}, "paths": {"data_dir": str(tmp_path)}}
    directory = tmp_path / "Schröder — 核"
    study_runs.record_run(cfg, spec, "ecoinvent", "geometric", [{"run_directory": str(directory)}])
    references = study_runs.run_references(cfg, spec, [("ecoinvent", "geometric")])
    assert references[0]["run_directory"] == str(directory.resolve())


def test_listing_and_dry_run_need_no_study_configuration(monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError("data or subprocess access during preview")

    monkeypatch.setattr(run.importlib, "import_module", forbidden)
    monkeypatch.setattr(run.subprocess, "run", forbidden)
    assert run.main(["--list"]) == 0
    assert run.main(["all", "--dry-run"]) == 0
    output = capsys.readouterr().out
    assert "figure-benchmark" in output and "sense-arithmetic" in output


def test_sequence_stops_on_failure_and_keeps_background_build_explicit(monkeypatch):
    calls = []

    def execute(command, **kwargs):
        calls.append(command[3])
        assert kwargs["cwd"] == run.ROOT
        assert kwargs["env"]["MPLBACKEND"] == "Agg"
        return SimpleNamespace(returncode=7 if command[3] == "foregrounds" else 0)

    monkeypatch.setattr(run.subprocess, "run", execute)
    assert run.main(["all"]) == 7
    assert calls == ["check-backgrounds", "sense-backgrounds", "foregrounds"]
    assert all("build-backgrounds" not in group for group in run.GROUPS.values())
    assert "sense-arithmetic" not in run.GROUPS["hydrogen"]
    assert all(stage.startswith("figure-") for stage in run.GROUPS["figures"])


def test_worker_dispatches_explicit_rebuild(monkeypatch):
    calls = []
    monkeypatch.setattr(
        run.importlib,
        "import_module",
        lambda *args: SimpleNamespace(run=lambda **kw: calls.append(kw)),
    )
    assert run.main(["benchmark", "--rebuild", "--worker"]) == 0
    assert calls == [{"rebuild": True}]
    with pytest.raises(SystemExit):
        run.main(["hydrogen", "--rebuild"])


def test_complete_workflow_exports_hydrogen_and_benchmark_before_regionalization(monkeypatch):
    calls = []

    def execute(command, **kwargs):
        calls.append(command[3])
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(run.subprocess, "run", execute)
    assert run.main(["all"]) == 0
    assert calls.index("calculate") < calls.index("figure-contributions")
    assert calls.index("figure-contributions") < calls.index("regionalization")
    assert calls.index("sense-arithmetic") < calls.index("benchmark")
    assert calls.index("benchmark") < calls.index("figure-benchmark")
    assert calls.index("figure-benchmark") < calls.index("regionalization")
    assert calls[-1] == "figure-regionalization"


def test_workflow_imports_do_not_load_configuration_or_run_calculations(tmp_path):
    for dependency in ("regioinvent", "bw2analyzer", "bw2calc", "matplotlib"):
        pytest.importorskip(dependency)
    # A fresh interpreter catches import-time globals that unit-test imports could mask.
    code = """
import importlib
from case_studies.hydrogen.workflow import case_config, study_protocol

def forbidden(*args, **kwargs):
    raise AssertionError("import attempted study configuration or a calculation")
case_config.load = forbidden
study_protocol.load_protocol = forbidden
for name in ("backgrounds", "foregrounds", "social", "calculate", "regionalization",
             "regionalization_summary", "benchmark", "figures", "si_common",
             "si_matching", "si_productivity", "si_productivity_plot", "si_missing",
             "si_carbon", "si_tables", "si_coverage"):
    importlib.import_module("case_studies.hydrogen.workflow." + name)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=run.ROOT,
        env={**os.environ, "BRIGHTWAY2_DIR": str(tmp_path), "MPLBACKEND": "Agg"},
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_figure_import_and_export_do_not_require_brightway_or_ipython(tmp_path):
    pytest.importorskip("matplotlib")
    code = """
import sys
for name in ("bw2data", "bw2calc", "brightway2", "IPython"):
    sys.modules[name] = None
from case_studies.hydrogen.workflow import figures
from case_studies.hydrogen.workflow.figure_io import export_figure
from matplotlib import pyplot as plt
from pathlib import Path
fig, ax = plt.subplots()
ax.plot([0, 1], [0, 1])
export_figure(fig, Path(sys.argv[1]) / "headless")
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)],
        cwd=run.ROOT,
        env={**os.environ, "MPLBACKEND": "Agg"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "headless.pdf").is_file()
    assert (tmp_path / "headless.svg").is_file()


def test_tutorial_is_syntax_valid_and_uses_only_public_api():
    path = run.ROOT / "examples/getting_started.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    code = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    for source in code:
        compile(ast.parse(source), str(path), "exec")
    assert "case_studies" not in "\n".join(code)
    assert "from sense import SenseConfig, run_sense" in "\n".join(code)
