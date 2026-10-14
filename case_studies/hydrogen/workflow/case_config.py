"""Load the hydrogen case study's ignored, machine-specific configuration."""

from __future__ import annotations

import functools
import logging
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

CASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = CASE_DIR / "config.local.yaml"
TRADE_DATABASE_PATH = CASE_DIR / "Data" / "regioinvent" / "trade_data.db"

# `paths.soca_dir` is deliberately absent: SOCA is read only by the benchmark and by
# `check_setup.py --include-validation`, which resolve it when needed. Users without
# SOCA can run the hydrogen workflow with the key null or removed.
_REQUIRED_VALUES = (
    ("brightway", "project"),
    ("brightway", "regioinvent_database"),
    ("versions", "ecoinvent"),
    ("versions", "psilca"),
    ("paths", "data_dir"),
    ("paths", "ecoinvent_geographies_excel"),
    ("paths", "hydrogen_inventory"),
    ("paths", "methanol_inventory"),
)


def enable_sense_progress() -> None:
    """Show SENSE's existing INFO-level progress events in the current process."""
    logger = logging.getLogger("sense")
    if not any(getattr(handler, "_sense_progress", False) for handler in logger.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        handler._sense_progress = True  # type: ignore[attr-defined]
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


@functools.cache
def load(config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Parse and cache a local configuration file."""
    config_path = Path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(
            f"{config_path} not found. Copy config.yaml.template to "
            "config.local.yaml and set your paths and credentials."
        )

    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{config_path} must contain a YAML mapping.")
    for keys in _REQUIRED_VALUES:
        value(*keys, config=loaded)
    psilca_source(loaded)
    if loaded["versions"] != {"ecoinvent": "3.12", "psilca": "4.0"}:
        raise ValueError("The manuscript case study uses ecoinvent 3.12 and PSILCA 4.0.")
    credentials = loaded.get("ecoinvent_credentials")
    if not isinstance(credentials, Mapping):
        raise ValueError("ecoinvent_credentials must be a YAML mapping.")
    for key in ("username", "password"):
        entry = credentials.get(key)
        if entry is not None and not isinstance(entry, str):
            raise ValueError(f"ecoinvent_credentials.{key} must be a string or null.")
    return loaded


def value(*keys: str, config: Mapping[str, Any] | None = None) -> Any:
    """Return a required nested configuration value."""
    node: Any = config if config is not None else load()
    dotted = ".".join(keys)
    try:
        for key in keys:
            node = node[key]
    except (KeyError, TypeError) as error:
        raise ValueError(f"Missing required configuration value: {dotted}") from error
    if node is None or (isinstance(node, str) and not node.strip()):
        raise ValueError(f"Configuration value cannot be empty: {dotted}")
    return node


def path(key: str, config: Mapping[str, Any] | None = None) -> Path:
    """Resolve one ``paths`` entry relative to the case-study directory."""
    configured = Path(str(value("paths", key, config=config))).expanduser()
    return configured if configured.is_absolute() else CASE_DIR / configured


def psilca_source(config: Mapping[str, Any] | None = None) -> Path:
    """Resolve the native PSILCA path, accepting older XML-only configurations."""
    cfg = config if config is not None else load()
    paths = cfg.get("paths", {})
    supplied = [key for key in ("psilca_path", "psilca_dir") if paths.get(key)]
    if len(supplied) != 1:
        raise ValueError(
            "Set paths.psilca_path to your .zolca backup. Legacy paths.psilca_dir is also accepted, but do not set both."
        )
    return path(supplied[0], cfg)


def ecoinvent_database_name(
    config: Mapping[str, Any] | None = None, *, regionalized: bool = False
) -> str:
    """Return the canonical base or Regioinvent-created ecoinvent database name."""
    base = f"ecoinvent-{value('versions', 'ecoinvent', config=config)}-cutoff"
    return f"{base} - regionalized" if regionalized else base


def ecoinvent_credentials(
    config: Mapping[str, Any] | None = None,
) -> tuple[str | None, str | None]:
    """Read credentials from local config, falling back to environment variables."""
    cfg = config if config is not None else load()
    credentials = cfg.get("ecoinvent_credentials") or {}
    if not isinstance(credentials, Mapping):
        raise ValueError("ecoinvent_credentials must be a YAML mapping.")
    username = credentials.get("username") or os.getenv("ECOINVENT_USERNAME")
    password = credentials.get("password") or os.getenv("ECOINVENT_PASSWORD")
    return username, password


def require_ecoinvent_credentials(
    config: Mapping[str, Any] | None = None,
) -> tuple[str, str]:
    """Return configured credentials or fail before starting an import."""
    username, password = ecoinvent_credentials(config)
    if not username or not password:
        raise RuntimeError(
            "Set ecoinvent_credentials in config.local.yaml or set "
            "ECOINVENT_USERNAME and ECOINVENT_PASSWORD in the environment."
        )
    return username, password
