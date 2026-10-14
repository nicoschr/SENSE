"""Exercise the Python environment and synthetic LCA without licensed data."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

CODE = r"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow
import torch
import sentence_transformers
import bw2io
import regioinvent
import geopandas
import bw2data as bd
import bw2calc.lca as backend
from bw2calc import LCA
from scipy.sparse import csc_matrix
from sense import SenseConfig
from matplotlib import pyplot as plt
from case_studies.hydrogen.workflow.figure_io import export_figure

assert SenseConfig().similarity_delta == 0.028
parquet_path = Path(OUTPUT_DIR) / "setup-smoke.parquet"
expected = pd.DataFrame({"activity": ["synthetic"], "working_hours": [2.0]})
expected.to_parquet(parquet_path)
pd.testing.assert_frame_equal(pd.read_parquet(parquet_path), expected)
print("PASS Parquet round trip", pyarrow.__version__)
if EXPECTED_SOLVER == "pardiso":
    assert backend.factorized.__module__.startswith("pypardiso")
else:
    import scikits.umfpack
    from scipy.sparse.linalg import use_solver
    use_solver(useUmfpack=True)
solve = backend.factorized(csc_matrix(np.diag([2.0, 3.0])))
assert np.allclose(solve(np.array([4.0, 9.0])), [2.0, 3.0])

bd.projects.set_current("setup-smoke")
bd.Database("biosphere").write({("biosphere", "flow"): {
    "name": "Synthetic flow", "unit": "kg", "type": "emission",
    "categories": ("air",),
}})
bd.Database("technosphere").write({("technosphere", "activity"): {
    "name": "Synthetic activity", "reference product": "Synthetic product",
    "unit": "kg", "location": "GLO", "type": "process",
    "exchanges": [
        {"input": ("technosphere", "activity"), "amount": 1, "type": "production"},
        {"input": ("biosphere", "flow"), "amount": 2, "type": "biosphere"},
    ],
}})
method = bd.Method(("Setup smoke",))
method.register(unit="kg equivalent")
method.write([(("biosphere", "flow"), 3.0)])
lca = LCA({("technosphere", "activity"): 1}, method.name)
lca.lci(factorize=True)
lca.lcia()
assert np.isclose(lca.score, 6.0), lca.score
lca.redo_lci({("technosphere", "activity"): 2})
lca.lcia()
assert np.isclose(lca.score, 12.0), lca.score
fig, ax = plt.subplots()
ax.bar(["Synthetic LCA"], [lca.score])
export_figure(fig, Path(OUTPUT_DIR) / "setup-smoke")
plt.close(fig)
for suffix in (".pdf", ".svg"):
    assert (Path(OUTPUT_DIR) / ("setup-smoke" + suffix)).stat().st_size > 100
print("PASS", sys.executable, EXPECTED_SOLVER, "LCA scores 6 and 12; PDF/SVG exported")
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver", choices=("pardiso", "umfpack"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    source = f"EXPECTED_SOLVER = {args.solver!r}\nOUTPUT_DIR = {str(output)!r}\n" + CODE
    with TemporaryDirectory(prefix="sense-smoke-bw-") as project_dir:
        env = {**os.environ, "BRIGHTWAY2_DIR": project_dir, "MPLBACKEND": "Agg"}
        subprocess.run([sys.executable, "-c", source], cwd=root, env=env, check=True, timeout=240)
    print(f"Python imports, factorized LCA and figure export passed: {output}")


if __name__ == "__main__":
    main()
