"""Prepare legacy foreground workbooks for the hydrogen case-study import."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class NormalizationSummary:
    """Counts from normalizing one foreground workbook in memory."""

    source_databases: tuple[str, ...]
    internal_exchanges: int
    external_exchanges: int


def _identity(record: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        record.get("name"),
        record.get("reference product"),
        record.get("unit"),
        record.get("location"),
    )


def _clear_link(record: MutableMapping[str, Any]) -> None:
    for key in ("database", "code", "input"):
        record.pop(key, None)


def normalize_foreground_database(
    datasets: Sequence[MutableMapping[str, Any]],
    target_database: str,
) -> NormalizationSummary:
    """Rename internal links and clear legacy external links before matching.

    The source workbooks were exported against prospective ecoinvent databases.
    Exact activity identities determine which exchanges are internal to the
    workbook. All other technosphere and biosphere links are cleared so the
    importer can match them against the databases selected by the case study.
    The function changes only in-memory importer records, never the workbook.
    """
    if not target_database.strip():
        raise ValueError("target_database cannot be empty.")
    if not datasets:
        raise ValueError("The foreground workbook contains no activities.")

    source_databases = tuple(
        sorted({str(dataset["database"]) for dataset in datasets if dataset.get("database")})
    )
    identity_counts = Counter(_identity(dataset) for dataset in datasets)
    duplicate_identities = [identity for identity, count in identity_counts.items() if count > 1]
    if duplicate_identities:
        formatted = ", ".join(repr(identity) for identity in duplicate_identities)
        raise ValueError(
            "The foreground workbook contains duplicate activity identities and "
            f"cannot be linked unambiguously: {formatted}"
        )

    activity_identities = set(identity_counts)
    internal_exchanges = 0
    external_exchanges = 0

    for dataset in datasets:
        dataset["database"] = target_database
        for exchange in dataset.get("exchanges", []):
            exchange_type = exchange.get("type")
            _clear_link(exchange)
            if exchange_type == "production" or (
                exchange_type == "technosphere" and _identity(exchange) in activity_identities
            ):
                exchange["database"] = target_database
                internal_exchanges += 1
            else:
                external_exchanges += 1

    return NormalizationSummary(
        source_databases=source_databases,
        internal_exchanges=internal_exchanges,
        external_exchanges=external_exchanges,
    )
