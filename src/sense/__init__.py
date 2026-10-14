"""Public interface for SENSE."""

from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sense.pipeline import run_sense as run_sense

try:
    __version__ = version("sense-lca")
except PackageNotFoundError:  # Source tree without an installed distribution.
    __version__ = "1.1.0"
__author__ = "SENSE Contributors"

from sense.config import SenseConfig
from sense.exceptions import SenseError, SenseInputRequiredError, SenseRegionError


def __getattr__(name: str) -> Any:
    """Load the Brightway-dependent pipeline only when it is requested."""

    if name == "run_sense":
        from sense.pipeline import run_sense

        return run_sense
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Include lazily exported names in interactive discovery."""

    return sorted(set(globals()) | set(__all__))


__all__ = [
    "run_sense",
    "SenseConfig",
    "SenseError",
    "SenseInputRequiredError",
    "SenseRegionError",
    "__version__",
]
