"""Public exception hierarchy for predictable SENSE failure handling."""

from __future__ import annotations

from pathlib import Path


class SenseError(RuntimeError):
    """Base class for expected SENSE pipeline failures."""


class SenseInputRequiredError(SenseError):
    """Raised after SENSE creates a template that requires user input."""

    def __init__(self, message: str, path: Path):
        super().__init__(f"{message}: {path}")
        self.path = Path(path)


class SenseRegionError(SenseError):
    """Raised when processing a specific ecoinvent region fails."""

    def __init__(self, region: str, message: str):
        super().__init__(f"Region {region}: {message}")
        self.region = region
