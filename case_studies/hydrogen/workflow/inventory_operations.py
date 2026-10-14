"""Import and make explicit inventory/method edits for the hydrogen study."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import bw2data as bd
import pandas as pd
from bw2io import ExcelImporter

from .foreground_import import normalize_foreground_database

SOCIAL_COMPARTMENTS = {
    "Children",
    "Consumer",
    "Local Community",
    "Society",
    "Workers",
    "Value Chain Actors",
}


def _ensure_project(project: str) -> None:
    """Set the current Brightway project."""
    bd.projects.set_current(project)


def _require_new_database(db_name: str) -> None:
    """Require an unused output name before constructing an inventory."""
    if db_name in bd.databases:
        raise ValueError(f"Database already exists: {db_name}; use a new output name")


def import_foreground_from_excel(
    excel_path: Path,
    target_database: str,
    biosphere_db: str,
    technosphere_db: str,
    extra_match_dbs: Sequence[str] | None = None,
    unlinked_csv: Path | None = None,
) -> None:
    """Import and link a workbook, stopping before writing if exchanges remain unlinked.

    Normalize legacy database references in memory, then match internal foreground,
    extra foreground, technosphere and biosphere links in that order. Inventory
    identities must already match the configured ecoinvent version. If supplied,
    ``unlinked_csv`` receives unresolved exchanges. The workbook is not modified.
    """
    excel_path = Path(excel_path)
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel not found: {excel_path}")

    imp = ExcelImporter(str(excel_path))
    normalization = normalize_foreground_database(imp.data, target_database)
    imp.db_name = target_database
    print(
        f"Normalized {normalization.source_databases} -> {target_database!r}; "
        f"retained {normalization.internal_exchanges} internal and reset "
        f"{normalization.external_exchanges} external exchanges for matching."
    )
    imp.apply_strategies()

    # Foreground-to-foreground
    imp.match_database(fields=["name", "unit", "reference product", "location"])

    # Extra DB(s) (e.g., another foreground DB)
    if extra_match_dbs:
        for dbn in extra_match_dbs:
            imp.match_database(dbn, fields=["name", "unit", "location", "reference product"])

    # Background technosphere
    imp.match_database(technosphere_db, fields=["name", "unit", "location", "reference product"])

    # Biosphere
    imp.match_database(
        biosphere_db,
        fields=["name", "categories", "location"],
        kind="biosphere",
        relink=True,
    )

    imp.statistics()

    unlinked = list(imp.unlinked)
    if unlinked:
        detail = ""
        if unlinked_csv is not None:
            pd.DataFrame(unlinked).to_csv(unlinked_csv, index=False)
            detail = f" Details were written to {unlinked_csv}."
        raise RuntimeError(
            f"Foreground import has {len(unlinked)} unique unlinked exchanges."
            f"{detail} The database was not written."
        )
    print("✅ All exchanges successfully linked.")

    # Write DB
    imp.write_database()
    print("✅ Foreground database installed successfully.")


def remove_self_referencing_technosphere_exchanges(db_name: str) -> int:
    """Delete exchanges whose input compares equal to their activity key.

    Modify the database in place and return the number of removed exchanges.
    """
    db = bd.Database(db_name)
    removed = 0
    for act in db:
        modified = False
        for exc in list(act.technosphere()):
            if exc.input.key == act.key:
                exc.delete()
                removed += 1
                modified = True
        if modified:
            act.save()
    return removed


def replace_technosphere_exchange_in_activity(
    db_name: str,
    target_activity_name: str,
    target_reference_product: str,
    old_input_names: Sequence[str],
    old_input_reference_product: str,
    new_input_name: str,
    new_input_reference_product: str,
) -> tuple[int, int]:
    """Replace named inputs with a same-location provider, preserving amounts.

    Modify input keys in place, retaining amounts and uncertainty metadata.
    Missing/ambiguous providers fail. Return updated and already-correct counts.
    """
    db = bd.Database(db_name)

    def find_activity(name: str, ref_product: str, location: str):
        hits = [
            act
            for act in db
            if act.get("name") == name
            and act.get("reference product") == ref_product
            and act.get("location") == location
        ]
        if len(hits) == 1:
            return hits[0]
        if len(hits) == 0:
            return None
        raise ValueError(f"Multiple matches for ({name}, {ref_product}, {location}) in {db_name}")

    replacements = 0
    skipped = 0

    for act in db:
        if act.get("name") != target_activity_name:
            continue
        if act.get("reference product") != target_reference_product:
            continue

        loc = act.get("location")
        new_act = find_activity(new_input_name, new_input_reference_product, loc)
        if not new_act:
            raise ValueError(f"Replacement provider is missing for {loc}: {new_input_name}")

        modified = False
        for exc in list(act.technosphere()):
            input_act = bd.get_activity(exc.input)
            if (
                input_act.get("name") in set(old_input_names)
                and input_act.get("reference product") == old_input_reference_product
            ):
                exc["input"] = new_act.key
                exc.save()
                modified = True

        if modified:
            act.save()
            replacements += 1
        else:
            if not any(exc.input.key == new_act.key for exc in act.technosphere()):
                raise ValueError(f"Neither old nor replacement input was found for {act.key}")
            skipped += 1

    return replacements, skipped
