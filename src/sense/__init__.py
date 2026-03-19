"""
SENSE - Social LCA Extension for Ecoinvent

SENSE (Social Ecoinvent No-Sector Extension) is a Python framework that integrates
social life cycle assessment (S-LCA) data from the PSILCA database into ecoinvent
activities within Brightway2.

Main entry points:
    - run_sense(): Run the full SENSE pipeline
    - SenseConfig: Configuration dataclass for tuning pipeline behavior

Example usage:
    >>> from sense import run_sense, SenseConfig
    >>> from pathlib import Path
    >>>
    >>> cfg = SenseConfig()
    >>> results = run_sense(
    ...     project="my_project",
    ...     technosphere_db="ecoinvent-3.11-cutoff",
    ...     biosphere_db="biosphere3",
    ...     sociosphere_db="sociosphere",
    ...     foreground_active=False,
    ...     foreground_database=None,
    ...     background_databases=["ecoinvent-3.11-cutoff"],
    ...     ecoinvent_dir=Path("Data/ecoinvent 3.11_cutoff_ecoSpold02/datasets"),
    ...     psilca_dir=Path("Data/PSILCA/EcoSpold01"),
    ...     ecoinvent_version="3.11",
    ...     psilca_version="4.0",
    ...     config=cfg,
    ... )
"""

__version__ = "0.1.0"
__author__ = "SENSE Contributors"

from sense.config import SenseConfig
from sense.pipeline import run_sense

__all__ = [
    "run_sense",
    "SenseConfig",
    "__version__",
]
