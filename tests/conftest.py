"""Isolate Brightway and headless rendering before test-module collection."""

from __future__ import annotations

import os
from tempfile import TemporaryDirectory

_previous = {
    name: os.environ.get(name) for name in ("BRIGHTWAY2_DIR", "MPLBACKEND", "MPLCONFIGDIR")
}
_root = TemporaryDirectory(prefix="sense-tests-")
os.environ["BRIGHTWAY2_DIR"] = _root.name
os.environ["MPLBACKEND"] = "Agg"
os.environ["MPLCONFIGDIR"] = os.path.join(_root.name, "matplotlib")


def pytest_unconfigure(config):
    """Close database handles before Windows attempts to remove temporary files."""
    import sys

    if "bw2data" in sys.modules:
        from bw2data import config, projects

        # Inventory and parameter databases both follow the current project.
        for _, database in config.sqlite3_databases:
            database.db.close()
        projects.db.db.close()
    import gc

    gc.collect()
    for name, previous in _previous.items():
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous
    _root.cleanup()
