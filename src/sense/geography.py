"""Shared ecoinvent geography normalization."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from sense.utils.helpers import ECOINVENT_NAME_TO_ISO2

_COMPOSITE_GEO_TYPES = {
    "Aluminium Industry",
    "ecoinvent Custom Area",
    "ecoinvent Region",
    "Electricity Grid",
    "UN Region",
    "UN Subregion",
    "World",
}


def read_ecoinvent_geographies(workbook: Path) -> pd.DataFrame:
    """Read ecoinvent geographies and add normalized ISO2 constituents."""

    frame = pd.read_excel(Path(workbook), sheet_name="Geographies", keep_default_na=False)
    required = {
        "Shortname",
        "Name",
        "Geography Classification",
        "Contained and Overlapping Geographies",
    }
    if not required.issubset(frame.columns):
        raise ValueError(
            f"Geography workbook is missing columns: {sorted(required - set(frame.columns))}"
        )

    def simplify(row: pd.Series) -> str | None:
        classification = str(row.get("Geography Classification", ""))
        shortname = str(row.get("Shortname", ""))
        name = str(row.get("Name", ""))
        contained = str(row.get("Contained and Overlapping Geographies", ""))
        if name in ECOINVENT_NAME_TO_ISO2:
            return ECOINVENT_NAME_TO_ISO2[name]
        if "Country" in classification:
            return shortname
        if "Province or State" in classification:
            return shortname[:2] if len(shortname) > 2 and shortname[2] == "-" else shortname
        if any(kind in classification for kind in _COMPOSITE_GEO_TYPES):
            codes = [code.strip() for code in contained.split(";")]
            resolved = [code for code in codes if re.fullmatch(r"[A-Z]{2}", code)]
            return "; ".join(resolved) or None
        return None

    normalized = frame.copy()
    normalized["simplified_ISO2"] = normalized.apply(simplify, axis=1)
    unresolved_grid = (normalized["Geography Classification"] == "Electricity Grid") & (
        normalized["simplified_ISO2"].isna()
        | (normalized["simplified_ISO2"].fillna("").str.strip() == "")
    )
    normalized.loc[unresolved_grid, "simplified_ISO2"] = normalized.loc[
        unresolved_grid, "Shortname"
    ].str[:2]
    normalized.loc[normalized["Name"] == "Namibia", "Shortname"] = "NA"
    return normalized
