"""Synchronize LCIA methods owned by one SENSE sociosphere database."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from bw2data import Database, Method, methods


def _owned(metadata: dict, database: str, kind: str) -> bool:
    return metadata.get("sense_database") == database and metadata.get("sense_kind") == kind


def _sync_methods(database: str, kind: str, desired: dict[tuple, tuple[str, list]]) -> None:
    """Validate ownership before replacing methods; remove only owned stale keys."""
    for key in desired:
        if key in methods and not _owned(methods[key], database, kind):
            raise ValueError(f"Refusing to overwrite a method not owned by SENSE: {key!r}")
    existing = {key for key in methods if _owned(methods[key], database, kind)}
    for key, (unit, cfs) in desired.items():
        cfs = sorted(cfs)
        method = Method(key)
        if key not in methods:
            method.register(
                unit=unit,
                sense_database=database,
                sense_kind=kind,
                description=f"SENSE {kind} method for {database}.",
            )
        else:
            if method.metadata.get("unit") == unit and sorted(method.load()) == cfs:
                continue
            method.metadata = {**method.metadata, "unit": unit}
        method.write(cfs)
    for key in existing - desired.keys():
        Method(key).deregister()


def remove_impact_categories(biosphere_db_name: str, kind: str) -> int:
    """Remove methods of one kind owned by the specified sociosphere."""
    count = sum(_owned(methods[key], biosphere_db_name, kind) for key in methods)
    _sync_methods(biosphere_db_name, kind, {})
    return count


def add_social_impact_categories(biosphere_db_name: str) -> None:
    """Create unit-CF methods for typed social flows in this sociosphere.

    Keys are ``('SENSE', database, category, subcategory, indicator, type)``.
    User flows without explicit SENSE ownership metadata are not included.
    """
    grouped: dict[tuple, list[Any]] = defaultdict(list)
    for flow in Database(biosphere_db_name):
        if flow.get("sense_kind") != "social":
            continue
        match = re.match(r"^(.*) \((risk|opportunity)\), .+$", flow.get("name") or "")
        if match is None:
            continue
        cats: Any = flow.get("categories") or ()
        category, subcategory = cats[:2] if len(cats) >= 2 else ("Social", "Unknown")
        grouped[("SENSE", biosphere_db_name, category, subcategory, *match.groups())].append(flow)
    desired = {}
    for key, flows in grouped.items():
        units = {str(flow.get("unit") or "") for flow in flows}
        if len(units) != 1:
            raise ValueError(f"Mixed units for {key}: {sorted(units)}")
        desired[key] = (units.pop(), [(flow.key, 1.0) for flow in flows])
    _sync_methods(biosphere_db_name, "social", desired)


def add_working_hours_impact_category(biosphere_db_name: str) -> None:
    """Synchronize the aggregate working-hours method for this sociosphere."""
    cfs = [
        (flow.key, 1.0)
        for flow in Database(biosphere_db_name)
        if flow.get("sense_kind") == "working_hours"
    ]
    key = ("SENSE", biosphere_db_name, "Social", "Working hours", "working hours")
    _sync_methods(biosphere_db_name, "working_hours", {key: ("h", cfs)} if cfs else {})


def add_working_hours_impact_categories_by_isic(biosphere_db_name: str) -> None:
    """Synchronize per-ISIC working-hours methods, including empty-state cleanup."""
    grouped: dict[str, list] = defaultdict(list)
    for flow in Database(biosphere_db_name):
        if flow.get("sense_kind") != "working_hours_isic":
            continue
        match = re.match(r"^working hours, isic(\d{2}), ", str(flow.get("name") or ""))
        if match:
            grouped[match[1]].append((flow.key, 1.0))
    desired = {
        ("SENSE", biosphere_db_name, "Social", "Working hours by ISIC", f"ISIC {isic}"): ("h", cfs)
        for isic, cfs in grouped.items()
    }
    _sync_methods(biosphere_db_name, "working_hours_isic", desired)
