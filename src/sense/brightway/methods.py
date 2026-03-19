"""
Brightway2 impact category (Method) creation for SENSE.

This module provides functions for creating LCIA methods from social flows.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Set

from bw2data import Method, Database, methods
from tqdm import tqdm


def add_social_impact_categories(biosphere_db_name: str) -> None:
    """
    Create one impact method per social indicator from the sociosphere database.
    
    Scans the sociosphere database for social flows and creates LCIA methods
    for each unique indicator. Each method includes all flows for that indicator
    across all regions and types (risk/opportunity).
    
    Method key format: (category, subcategory, indicator_name)
    
    Args:
        biosphere_db_name: Name of the sociosphere database containing social flows
        
    Note:
        Only processes flows with the SENSE comment marker.
        Working hours flows are handled separately.
        
    Examples:
        >>> add_social_impact_categories("sociosphere")
    """
    biosphere = Database(biosphere_db_name)

    # (cat, subcat, indicator_name) -> set(units)
    indicator_units: Dict[tuple, Set[str]] = defaultdict(set)

    for flow in biosphere:
        # Only use flows created by SENSE
        comment = flow.get("comment") or ""
        if "Imported from PSILCA sector matching (country-resolved, typed)" not in comment:
            continue

        cats = flow.get("categories") or ()
        if not cats:
            continue

        name = flow.get("name") or ""
        # Skip working-hours flows
        if name.startswith("working hours, "):
            continue
        if " (" not in name:
            continue

        indicator_name = name.split(" (", 1)[0]
        cat, subcat = (cats[0], cats[1]) if len(cats) >= 2 else ("Social", "Unknown")
        unit = flow.get("unit") or ""

        indicator_units[(cat, subcat, indicator_name)].add(unit)

    if not indicator_units:
        print("⚠️ No social indicator flows found in sociosphere DB; no methods created.")
        return

    # Create/overwrite methods and attach CFs
    for (cat, subcat, indicator_name), units in tqdm(
        indicator_units.items(), desc="⚙️ Creating social impact methods"
    ):
        method_key = (cat, subcat, indicator_name)

        # Choose a representative unit
        chosen_unit = next(iter(units)) if units else ""
        if len(units) > 1:
            print(
                f"⚠️ Mixed units for {method_key}: {sorted(units)}. "
                f"Using '{chosen_unit}' for the method unit."
            )

        # Overwrite if exists
        if method_key in methods:
            Method(method_key).deregister()

        method = Method(method_key)
        method.register(
            unit=chosen_unit or "",
            description=(
                "Auto-generated from SENSE social flows in sociosphere "
                "(all regions, both risk and opportunity)."
            ),
        )

        # Collect all matching flows
        prefix = f"{indicator_name} ("
        cfs = [
            (flow.key, 1.0)
            for flow in biosphere
            if str(flow.get("name", "")).startswith(prefix)
        ]

        if not cfs:
            print(
                f"⚠️ No matching flows found for indicator {method_key}; "
                "method will be empty."
            )
            continue

        method.write(cfs)


def add_working_hours_impact_category(biosphere_db_name: str) -> None:
    """
    Create a single impact method that sums all working hours flows.
    
    Creates a method that aggregates all 'working hours, <region>' flows
    with characterization factor = 1.
    
    Method key: ('Social', 'Working hours', 'working hours')
    
    Args:
        biosphere_db_name: Name of the sociosphere database
        
    Examples:
        >>> add_working_hours_impact_category("sociosphere")
    """
    biosphere = Database(biosphere_db_name)

    method_key = ("Social", "Working hours", "working hours")

    # Overwrite if exists
    if method_key in methods:
        Method(method_key).deregister()

    method = Method(method_key)
    method.register(
        unit="h",
        description="Total working hours (sum of all 'working hours, <region>' flows)",
    )

    cfs = []
    for flow in biosphere:
        name = str(flow.get("name", ""))
        if name.startswith("working hours, "):
            cfs.append((flow.key, 1.0))

    if not cfs:
        print("⚠️ No 'working hours, <region>' flows found in biosphere DB; method will be empty.")
        return

    method.write(cfs)
