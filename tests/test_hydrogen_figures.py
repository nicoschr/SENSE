"""Contracts for figure snapshots and calculation-free plotting."""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from case_studies.hydrogen.workflow import figure_io

WORKFLOW = Path(__file__).resolve().parents[1] / "case_studies/hydrogen/workflow"


def test_figure_snapshot_roundtrip_context_and_atomic_failure(tmp_path):
    target = tmp_path / "inputs.zip"
    context = {"versions": {"ecoinvent": "3.12"}}
    data = pd.DataFrame({"location": ["NA", "NULL"], "value": [1.3108611656722589, np.nan]})
    with pytest.raises(FileNotFoundError, match="preparation"):
        figure_io.load_inputs(target, context=context)
    figure_io.save_inputs(target, {"values": data}, {"context": context})
    tables, metadata = figure_io.load_inputs(target, context=context)
    pd.testing.assert_frame_equal(tables["values"], data)
    assert metadata == {"context": context}
    assert figure_io.reuse_inputs(target, context=context)
    assert not figure_io.reuse_inputs(target, context=context, rebuild=True)
    assert not figure_io.reuse_inputs(target, context=context, sources=[{"mtime_ns": 42}])
    with pytest.raises(ValueError, match="rebuild flag"):
        figure_io.reuse_inputs(target, context={})
    with pytest.raises(ValueError, match="another project/version"):
        figure_io.load_inputs(target, context={})
    original = target.read_bytes()
    with pytest.raises(ValueError):
        figure_io.save_inputs(target, {"values": data}, {"factor": np.nan})
    assert target.read_bytes() == original
    assert list(tmp_path.iterdir()) == [target]


def test_comparison_figures_keep_full_sample_statistics_and_canvas(tmp_path, monkeypatch):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from case_studies.hydrogen.workflow.comparison_plots import (
        plot_regionalization,
        plot_validation,
    )

    values = pd.DataFrame(
        {"location": ["NA"] * 4 + ["CH"] * 4, "log_ratio": [-2, -1, 0, 1, 3, 4, 5, 6]}
    )
    with plt.rc_context(figure_io.figure_style()):
        original = values.copy(deep=True)
        fig = plot_regionalization(values, min_fus=1)
        assert len(fig.axes) == 1
        ax = fig.axes[0]
        assert len(ax.patches) == 2  # Overlapping tails display each location once.
        for i, (median, quartiles) in enumerate(
            zip([-0.5, 4.5], [(-1.25, 0.25), (3.75, 5.25)], strict=True)
        ):
            np.testing.assert_allclose(ax.lines[5 * i + 4].get_xdata(), [median, median])
            vertices = ax.patches[i].get_path().vertices[:, 0]
            np.testing.assert_allclose([vertices.min(), vertices.max()], quartiles)
            assert ax.get_xlim()[0] < quartiles[0] < quartiles[1] < ax.get_xlim()[1]
            assert ax.lines[5 * i + 4].get_color() == figure_io.COLORS["burgundy"]
        assert ax.get_xlim()[1] >= 6  # Do not clip whiskers as in the old manuscript.
        assert ax.get_yticklabels()[0].get_text() == "Namibia (n=4)"
        assert ax.xaxis.label.get_fontfamily() == [figure_io.figure_style()["font.family"]]
        assert ax.xaxis.label.get_fontsize() == 7
        pd.testing.assert_frame_equal(values, original)
        figure_io.export_figure(fig, tmp_path / "regionalization", preview_scale=2)
        svg = (tmp_path / "regionalization.svg").read_text(encoding="utf-8")
        assert 'width="493.228346pt"' in svg  # 174 mm, not a cropped bounding box
        assert (tmp_path / "regionalization.pdf").is_file()

    ranked = pd.DataFrame(
        {
            "location": np.repeat([f"region{i}" for i in range(11)], 2),
            "log_ratio": np.repeat(np.arange(11), 2),
        }
    )
    fig = plot_regionalization(ranked, min_fus=2, top_n=2)
    assert len(fig.axes[0].patches) == 4
    np.testing.assert_allclose(fig.axes[0].collections[0].get_offsets()[:, 0], np.arange(2, 9))
    assert any("Other 7 locations" in text.get_text() for text in fig.axes[0].texts)
    plt.close(fig)

    pairs = pd.DataFrame(
        {
            "dataset_id": [1, 1, 2, 2],
            "geo_class": ["Country/State"] * 2 + ["Region/Other"] * 2,
            "log_diff": [0.0, 10.0, 0.0, 10.0],
        }
    )
    tables = dict.fromkeys(("arithmetic", "geometric"), pairs)
    metadata = {mean: {"factor": 10.0} for mean in tables}
    fig = plot_validation(tables, metadata)
    assert "Median: 5.00" in fig.axes[0].texts[0].get_text()
    assert "Outside range: 50.0%" in fig.axes[0].texts[0].get_text()
    assert len(fig.axes) == 3
    assert [ax.get_title(loc="left") for ax in fig.axes] == [
        "(a)  All locations",
        "(b)  Countries / states",
        "(c)  Regions / other",
    ]
    # Each visible bin includes half the full sample, with a bin width of 0.2.
    assert fig.axes[0].get_ylim()[1] > 2.5
    plt.close(fig)
    fig = plot_validation(
        {"geometric": pairs}, {"geometric": metadata["geometric"]}, factor_neutral=True
    )
    assert "Median: 4.00" in fig.axes[0].texts[0].get_text()
    assert pairs["log_diff"].tolist() == [0.0, 10.0, 0.0, 10.0]
    plt.close(fig)


def test_plot_stages_execute_from_snapshots_without_preparation(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    from case_studies.hydrogen.workflow import figures

    cfg = {"versions": {}, "brightway": {"project": "test"}}
    monkeypatch.setattr(figures, "load", lambda: cfg)
    monkeypatch.setattr(figures, "configured_path", lambda *args: tmp_path)
    spec = figures.load_protocol()
    monkeypatch.setattr(figures, "load_protocol", lambda **kwargs: spec)

    def load(path, **kwargs):
        return {"values": None}, {}

    monkeypatch.setattr(figures, "load_inputs", load)
    monkeypatch.setattr(figures, "load_results", lambda *a, **kw: (None, {"technologies": []}))
    monkeypatch.setattr(figures, "select_panels", lambda *a: pd.DataFrame({"score": [1.0]}))
    (tmp_path / "paired_hydrogen").mkdir()
    calls = []
    monkeypatch.setattr(
        figures, "export_figure", lambda *args, **kwargs: calls.append(args[1].name)
    )
    for name in ("plot_paired", "plot_regionalization", "plot_validation"):
        monkeypatch.setattr(
            figures, name, lambda *args, **kwargs: SimpleNamespace(savefig=lambda *a, **kw: None)
        )
    figures.hydrogen()
    figures.regionalization()
    figures.benchmark()
    assert calls == [
        "hydrogen_paired",
        "regionalization_effect",
        "validation",
        "validation_factor_neutral",
        "validation_arithmetic",
    ]


def test_regionalization_preparation_starts_from_existing_csvs(tmp_path, monkeypatch):
    from case_studies.hydrogen.workflow import figure_io as notebook_io
    from case_studies.hydrogen.workflow import regionalization_summary as summary
    from case_studies.hydrogen.workflow import study_protocol

    cfg = {
        "versions": {"ecoinvent": "3.12"},
        "brightway": {"project": "test", "regioinvent_database": "Regioinvent"},
    }
    monkeypatch.setattr(summary, "load", lambda: cfg)
    monkeypatch.setattr(summary, "configured_path", lambda *args: tmp_path)
    monkeypatch.setattr(summary, "run_references", lambda *args: [])
    output = tmp_path / "case_study_results/hydrogen_case_study_results"
    output.mkdir(parents=True)
    (output / "regionalization_runs.json").write_text("[]")
    base = pd.DataFrame(
        {
            "name": ["activity", "no positive pairs"],
            "reference product": ["product"] * 2,
            "location": ["NA"] * 2,
            "unit": ["kg"] * 2,
            "Workers | example": [1.0, 0.0],
        }
    )
    regionalized = base.copy()
    regionalized["Workers | example"] = [10.0, 0.0]
    publish_regionalization_test_files(output, base, regionalized)
    summary.run()
    tables, _ = notebook_io.load_inputs(
        tmp_path / "figure_inputs/regionalization.zip",
        context={
            "versions": cfg["versions"],
            "projects": cfg["brightway"],
            "social_layers": study_protocol.load_protocol()["social_layers"],
            "sense": study_protocol.load_protocol()["sense"],
        },
    )
    assert tables["values"].to_dict("records") == [{"location": "NA", "log_ratio": 1.0}]


def test_regionalization_skips_database_work_only_with_matching_run_receipt(tmp_path, monkeypatch):
    from case_studies.hydrogen.workflow import regionalization as module

    monkeypatch.setattr(
        module,
        "load",
        lambda: {
            "versions": {"ecoinvent": "3.12", "psilca": "4"},
            "brightway": {"project": "test", "regioinvent_database": "regio"},
        },
    )
    monkeypatch.setattr(module, "configured_path", lambda *args: tmp_path)
    monkeypatch.setattr(module, "load_protocol", lambda: {"social_layers": {}, "sense": {}})
    monkeypatch.setattr(module, "run_references", lambda *args: [])
    monkeypatch.setattr(module, "require_run", lambda *args: None)

    def forbidden(*args, **kwargs):
        raise AssertionError("calculation required")

    monkeypatch.setattr(module, "set_project", forbidden)
    with pytest.raises(AssertionError, match="calculation required"):
        module.run()
    output = tmp_path / "case_study_results/hydrogen_case_study_results"
    output.mkdir(parents=True)
    for label in ("base", "regionalized"):
        pd.DataFrame({"score": [1]}).to_csv(
            output / f"my_batch_run_{label}_total_scores.csv", index=False
        )
    (output / "regionalization_runs.json").write_text("[]")
    # Matched run IDs and two existing malformed files do not prove completion.
    with pytest.raises(AssertionError, match="calculation required"):
        module.run()
    table = pd.DataFrame(
        {
            "name": ["Invented production"],
            "reference product": ["Invented A"],
            "location": ["DE"],
            "unit": ["kg"],
            "database": ["base"],
            "Workers | example": [1.0],
        }
    )
    publish_regionalization_test_files(output, table, table.copy())
    module.run()
    with pytest.raises(AssertionError, match="calculation required"):
        module.run(rebuild=True)
    path = output / "my_batch_run_base_total_scores.csv"
    path.write_text("truncated\n")
    with pytest.raises(AssertionError, match="calculation required"):
        module.run()


def publish_regionalization_test_files(output, base, regionalized):
    """Invented completed score files for stage-boundary tests, not LCA fixtures."""
    from case_studies.hydrogen.workflow.regionalization import SCORE_METADATA
    from sense.artifacts import write_json_atomic
    from sense.utils.cache import file_hash, stable_hash

    artifacts = {}
    for label, table in (("base", base), ("regionalized", regionalized)):
        table = table.copy()
        table["database"] = label
        columns = SCORE_METADATA + [column for column in table if column not in SCORE_METADATA]
        table = table[columns]
        path = output / f"my_batch_run_{label}_total_scores.csv"
        table.to_csv(path, index=False)
        metadata = pd.read_csv(path, keep_default_na=False)[SCORE_METADATA]
        write_json_atomic(
            path.with_suffix(".manifest.json"),
            {
                "schema": 1,
                "sha256": file_hash(path),
                "rows": len(table),
                "columns": columns,
                "population": stable_hash(metadata.to_dict("records")),
                "paired_population": stable_hash(
                    metadata.drop(columns="database").to_dict("records")
                ),
            },
        )
        artifacts[path.name] = file_hash(path)
    write_json_atomic(
        output / "regionalization_runs.json", {"schema": 2, "runs": [], "artifacts": artifacts}
    )


def test_validation_geographies_preserve_namibia_and_pair_values(tmp_path):
    import numpy as np

    root = Path(__file__).resolve().parents[1]
    source = (root / "case_studies/hydrogen/workflow/benchmark_comparison.py").read_text(
        encoding="utf-8"
    )
    function = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "prepare_validation_plot_data"
    )
    workbook = tmp_path / "geographies.xlsx"
    pd.DataFrame(
        {
            "Shortname": ["NA", "US", "US-CO", "RER"],
            "Geography Classification": ["Country", "Country", "Province or State", "Region"],
        }
    ).to_excel(workbook, sheet_name="Geographies", index=False)
    scope = {"pd": pd, "np": np, "OVERVIEW_XLSX": workbook, "read_excel_sheet": pd.read_excel}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), "<validation-geography>", "exec"),
        scope,
    )
    pairs = pd.DataFrame({"dataset_id": [1, 1, 2, 3, 4], "log_diff": [0.1, -0.2, 0, 0.3, -0.4]})
    geography = pd.DataFrame(
        {"dataset_id": [1, 2, 3, 4], "Geography": ["NA", "US", "US-CO", "RER"]}
    )
    result = scope["prepare_validation_plot_data"](
        __import__("types").SimpleNamespace(overview_xlsx=workbook), pairs, geography
    )
    assert result["geo_class"].tolist() == ["Country/State"] * 4 + ["Region/Other"]
    pd.testing.assert_frame_equal(result[["dataset_id", "log_diff"]], pairs)
