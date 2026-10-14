"""Preserve inventory metadata and resolve direct environmental geography when cloning."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import bw2data as bd

from sense.brightway.repository import reference_production_amount


def copy_exchange(exchange: Any, input_key: tuple[str, str] | None = None) -> dict[str, Any]:
    """Copy amounts, uncertainty and formulas, removing the old output identity."""
    data: dict[str, Any] = deepcopy(exchange.as_dict())
    data.pop("output", None)
    if input_key is not None:
        data["input"] = input_key
        # Brightway links by input. Remove redundant metadata from the old link.
        for field in ("name", "database", "code", "location", "reference product"):
            data.pop(field, None)
    return data


def clone_activity(
    activity: Any,
    key: tuple[str, str],
    location: str,
    source_exchanges: list[Any] | None = None,
) -> dict[str, Any]:
    """Copy source metadata and production without changing its inventory basis."""
    data: dict[str, Any] = deepcopy(activity.as_dict())
    for field in ("id", "database", "code", "exchanges"):
        data.pop(field, None)
    data["location"] = location
    production = (
        list(activity.production())
        if source_exchanges is None
        else [exc for exc in source_exchanges if exc.get("type") == "production"]
    )
    reference_production_amount(tuple(activity.key), production)
    data["exchanges"] = [copy_exchange(exc, key) for exc in production]
    if not production:
        data["exchanges"] = [{"input": key, "amount": 1.0, "type": "production"}]
    return data


class BiosphereLocalizer:
    """Index each biosphere database once and resolve equivalent regional flows.

    Regioinvent encodes geography as a final ``, location`` suffix in flow names.
    Other databases may store an explicit location. Ordinary non-spatial flows
    retain their identity. Missing or ambiguous spatial counterparts stop cloning.
    """

    def __init__(self) -> None:
        self._indices: dict[str, dict[tuple, list[Any]]] = {}

    @staticmethod
    def _identity(flow: Any) -> tuple[str, str | None]:
        name = str(flow.get("name") or "")
        location = flow.get("location")
        if "spatialized" in flow.key[0] and ", " in name:
            name, location = name.rsplit(", ", 1)
        elif location and name.endswith(f", {location}"):
            name = name[: -len(f", {location}")]
        return name, location

    def exchange(self, exchange: Any, location: str) -> dict[str, Any]:
        """Copy an exchange and relocate a spatial biosphere input if required."""
        if exchange.get("type") != "biosphere":
            return copy_exchange(exchange)
        flow = exchange.input
        name, original_location = self._identity(flow)
        if not original_location or original_location == location:
            return copy_exchange(exchange)
        database = flow.key[0]
        if database not in self._indices:
            index: dict[tuple, list[Any]] = {}
            for candidate in bd.Database(database):
                base, region = self._identity(candidate)
                identity = (
                    base,
                    tuple(candidate.get("categories") or ()),
                    candidate.get("unit"),
                    region,
                )
                index.setdefault(identity, []).append(candidate)
            self._indices[database] = index
        identity = (name, tuple(flow.get("categories") or ()), flow.get("unit"), location)
        matches = self._indices[database].get(identity, [])
        if len(matches) != 1:
            raise ValueError(
                f"Expected one spatial counterpart for {flow.key!r} in {location!r}, "
                f"found {len(matches)}"
            )
        return copy_exchange(exchange, tuple(matches[0].key))
