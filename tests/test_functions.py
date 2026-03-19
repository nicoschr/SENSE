import importlib.util
import unittest
from pathlib import Path
import sys


def load_module_from_src(rel_path: str, name: str = None):
    """Import a module from `src/` using its package-style path.

    This ensures the module's __module__ corresponds to the package name
    (e.g. `sense.config`) so decorators like `dataclass` can resolve types
    via `sys.modules` correctly.
    """
    repo = Path(__file__).resolve().parents[1]
    src_root = repo / "src"
    # Ensure src is on sys.path for normal package imports
    src_str = str(src_root)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)

    # Derive module name if not provided. Accept either "sense/config.py" or "sense.config"
    rel = rel_path.replace("\\", "/")
    if rel.endswith(".py"):
        rel = rel[:-3]
    module_name = rel.replace("/", ".")

    try:
        return importlib.import_module(module_name)
    except Exception:
        # Fallback to loading from file location if import fails
        src_file = src_root / Path(rel_path)
        if not src_file.exists():
            raise FileNotFoundError(f"Module file not found: {src_file}")
        name = name or module_name
        spec = importlib.util.spec_from_file_location(name, str(src_file))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod


# Run Brightway test DB setup if available (mirrors other project's pattern)
try:
    from tests.sample_database import setup_test_db

    setup_test_db()
except Exception:
    # If Brightway or required deps are not available in the current environment,
    # we continue so unit tests that don't need BW can still run.
    pass


class TestSensePackage(unittest.TestCase):
    def test_config_defaults(self):
        mod = load_module_from_src("sense/config.py", "sense_config_test")
        self.assertTrue(hasattr(mod, "SenseConfig"))
        cfg = mod.SenseConfig()
        # Basic sanity checks for default attribute types
        self.assertIsInstance(cfg.embedding_model, str)
        self.assertIsInstance(cfg.similarity_delta, float)
        self.assertIsInstance(cfg.enable_social_flows, bool)

    def test_utils_helpers_exist(self):
        mod = load_module_from_src("sense/utils/helpers.py", "sense_utils_test")
        self.assertTrue(hasattr(mod, "norm_code") and callable(mod.norm_code))
        self.assertTrue(hasattr(mod, "slug") and callable(mod.slug))

    def test_data_psilca_functions(self):
        mod = load_module_from_src("sense/data/psilca.py", "sense_psilca_test")
        self.assertTrue(hasattr(mod, "combine_psilca_css_and_extract_social_exchanges"))
        self.assertTrue(hasattr(mod, "embed_psilca_to_store"))
        self.assertTrue(callable(mod.combine_psilca_css_and_extract_social_exchanges))
        self.assertTrue(callable(mod.embed_psilca_to_store))

    def test_data_ecoinvent_functions(self):
        mod = load_module_from_src("sense/data/ecoinvent.py", "sense_ecoinvent_test")
        self.assertTrue(hasattr(mod, "map_ecoinvent_locations"))
        self.assertTrue(hasattr(mod, "fetch_ecoinvent_activities_from_ecospold_files"))
        self.assertTrue(hasattr(mod, "embed_ecoinvent_to_store"))
        self.assertTrue(callable(mod.map_ecoinvent_locations))


if __name__ == "__main__":
    unittest.main()
