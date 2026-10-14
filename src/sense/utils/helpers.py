"""Small normalization helpers shared across SENSE."""

from __future__ import annotations

import re
from typing import Any


def norm_code(x: Any) -> str | None:
    """Return a stripped activity code, unwrapping one-item tuples and lists."""
    if x is None:
        return None
    if isinstance(x, (tuple, list)) and len(x) == 1:
        return str(x[0]).strip()
    return str(x).strip()


def safe_name(name: str) -> str:
    """Replace separators and whitespace with underscores for file names."""
    s = str(name)
    for ch in "/\\:- ":
        s = s.replace(ch, "_")
    return s


def extract_isic2(raw_isic: str) -> str | None:
    """Extract an ISIC two-digit division, or return ``None`` if unavailable."""
    if not raw_isic or not isinstance(raw_isic, str):
        return None
    # Take the part before the colon (or the whole string)
    numeric = raw_isic.split(":")[0].strip()
    # Extract only digits
    digits = re.sub(r"[^0-9]", "", numeric)
    if len(digits) < 2:
        return None
    return digits[:2]


def slug(s: str) -> str:
    """Return a lowercase, filesystem-safe slug."""
    s = s.lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_]+", "-", s)
    return s.strip("-_")


# Ecoinvent location names that need manual ISO2 assignment.
# Single source of truth — imported by both ecoinvent.py and productivity/geography.py.
ECOINVENT_NAME_TO_ISO2 = {
    "Clipperton Island": "FR",
    "Canary Islands": "ES",
    "France, including overseas territories": "FR",
    "Australia, including overseas territories": "AU",
    "Puerto Rico": "US",
    "China without Inner Mongol": "CN",
    "Canada without Alberta": "CA",
    "Canada without Alberta and Quebec": "CA",
    "Canada without Quebec": "CA",
    "Russia (Europe)": "RU",
    "Russia (Asia)": "RU",
    "Coral Sea Islands": "AU",
    "Kosovo": "XK",
    "Serranilla Bank": "CO",
    "Spratly Islands": "PH",
    "Dhekelia Sovereign Base Area": "CY",
    "Serbia and Montenegro": "RS",
    "Cyprus No Mans Area": "CY",
    "Siachen Glacier": "IN",
    "Akrotiri Sovereign Base Area": "GB",
    "Bajo Nuevo Bank (Petrel Is.)": "CO",
    "Taiwan": "TW",
    "Scarborough Reef": "PH",
    "Northern Cyprus": "CY",
    "US Naval Base Guantanamo Bay": "CU",
    "Somaliland": "SO",
    "Québec, Hydro-Québec distribution network": "CA",
    "Namibia": "NA",
}
