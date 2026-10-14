"""Normalize native PSILCA social assessments into the existing SENSE cache."""

from __future__ import annotations

import base64
import logging
import math
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from uuid import UUID

from sense.artifacts import atomic_output_path, write_json_atomic
from sense.data.psilca import _parse_assessment_name, _psilca_schema, _write_parquet_batch
from sense.data.zolca_reader import READER_PROTOCOL, reader_lines

logger = logging.getLogger(__name__)


def _text(value: str) -> str:
    return base64.b64decode(value, validate=True).decode("utf-8")


def normalized_rows(lines: Iterator[list[str]]) -> Iterator[dict[str, Any]]:
    """Preserve original hours, duplicates, missing values and assessment labels.

    Native social-aspect values are used directly. Library coefficients normalized
    by the reference exchange are not a substitute. The reference-product name
    comes from the original quantitative reference, not the process name.
    """
    categories: dict[int, tuple[int, str]] = {}
    locations: dict[int, str] = {}
    processes: dict[int, tuple[str, int, str, str]] = {}
    units: dict[int, str] = {}
    groups: dict[int, int] = {}
    quantities: dict[int, int] = {}
    flows = []
    indicators: dict[int, tuple[str, int, int]] = {}
    lookup: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    ordinals = {}
    started = metadata_complete = finished = False
    count = 0
    current = None
    exchange_index = 0
    seen_owners = set()

    def category_pair(ident: int) -> tuple[str, str]:
        if ident == 0:
            return "", ""
        parent, leaf = categories[ident]
        return categories[parent][1] if parent else "", leaf

    try:
        for fields in lines:
            if finished:
                raise ValueError("Records after reader completion")
            tag = fields[0]
            if tag == "BEGIN":
                if started or fields != ["BEGIN", str(READER_PROTOCOL)]:
                    raise ValueError("Reader protocol mismatch")
                started = True
                continue
            if not started:
                raise ValueError("Reader protocol header missing")
            if tag in {"C", "L", "P", "U", "G", "Q", "F", "I"} and metadata_complete:
                raise ValueError("Late metadata")
            if tag == "C":
                categories[int(fields[1])] = (int(fields[2]), _text(fields[3]))
            elif tag == "L":
                locations[int(fields[1])] = _text(fields[2]).strip().upper()
            elif tag == "P":
                ident = int(fields[1])
                if ident in processes:
                    raise ValueError("Duplicate process identifier")
                # UUID-based identity stays compatible with EcoSpold source-file overrides.
                uuid = str(UUID(_text(fields[2])))
                processes[ident] = (uuid, int(fields[3]), _text(fields[4]), _text(fields[5]))
            elif tag == "U":
                units[int(fields[1])] = _text(fields[2])
            elif tag == "G":
                groups[int(fields[1])] = int(fields[2])
            elif tag == "Q":
                quantities[int(fields[1])] = int(fields[2])
            elif tag == "F":
                flows.append((_text(fields[1]), int(fields[2]), int(fields[3])))
            elif tag == "I":
                indicators[int(fields[1])] = (_text(fields[2]), int(fields[3]), int(fields[4]))
            elif tag == "ASSESSMENTS_BEGIN":
                if metadata_complete:
                    raise ValueError("Duplicate assessment header")
                metadata_complete = True
                uuids = [row[0] for row in processes.values()]
                if len(set(uuids)) != len(uuids):
                    raise ValueError("Duplicate process UUID")
                ordinals = {uuid: i for i, uuid in enumerate(sorted(uuids))}
                names = {row[0] for row in indicators.values()}
                for name, category_id, quantity_id in flows:
                    indicator, level, kind = _parse_assessment_name(name)
                    if indicator not in names:
                        continue
                    unit = units[groups[quantities[quantity_id]]]
                    if unit != "h":
                        raise ValueError("Social assessment flow is not expressed in hours")
                    category, subcategory = category_pair(category_id)
                    enum = level.upper().replace(" ", "_")
                    if enum not in {"NO_DATA", "NOT_APPLICABLE"}:
                        enum += "_" + kind.upper()
                    lookup[(indicator, category, subcategory, enum)].append(
                        {
                            "indicator": indicator,
                            "assessment_type": kind,
                            "risk_level": level,
                            "unit": unit,
                            "category": category,
                            "subcategory": subcategory,
                        }
                    )
            elif tag == "A":
                if not metadata_complete:
                    raise ValueError("Assessments before metadata")
                owner = int(fields[1])
                if owner != current:
                    if owner in seen_owners:
                        raise ValueError("Noncontiguous assessment process")
                    seen_owners.add(owner)
                    current = owner
                    exchange_index = 0
                uuid, location_id, product, library = processes[owner]
                if library or not product:
                    raise ValueError("Original reference product/assessments unavailable")
                indicator, category_id, unit_id = indicators[int(fields[2])]
                if units[unit_id] != "h":
                    raise ValueError("Indicator activity unit is not hours")
                category, subcategory = category_pair(category_id)
                choices = lookup.get((indicator, category, subcategory, _text(fields[4])), [])
                if len(choices) != 1:
                    raise ValueError("Missing or ambiguous assessment-flow mapping")
                value = math.nan if fields[3] == "nan" else float.fromhex(fields[3])
                if exchange_index > 32767:
                    raise ValueError("Assessment ordinal exceeds cache schema")
                yield {
                    "process_index": ordinals[uuid],
                    "exchange_index": exchange_index,
                    "source_file": f"process_{uuid}.xml",
                    "location": locations.get(location_id, "unknown"),
                    "reference_name": product.replace("/", "-").replace("\\", "-"),
                    "mean_value": value,
                    **choices[0],
                }
                exchange_index += 1
                count += 1
            elif tag == "END":
                if not metadata_complete or int(fields[1]) != count or count == 0:
                    raise ValueError("Empty or incomplete assessment stream")
                finished = True
            else:
                raise ValueError("Unexpected reader record")
        if not finished:
            raise ValueError("Reader did not complete")
    except (ValueError, KeyError, IndexError, UnicodeError):
        # Never include licensed labels or raw protocol records in an exception.
        raise ValueError(
            "Unsupported or inconsistent PSILCA .zolca social-assessment schema. "
            "No partial cache was published. Use an original PSILCA database backup "
            "or the compatible EcoSpold01 XML export. Record details are suppressed."
        ) from None


def extract_zolca(
    path: Path, combined_path: Path, css_path: Path, zero_report_path: Path | None = None
) -> None:
    """Stream a native backup into the same Parquet and sector artifacts as XML."""
    import pyarrow.parquet as pq

    if combined_path.suffix.lower() not in {".parquet", ".pq"}:
        raise ValueError("combined_output_path must use a .parquet or .pq suffix")
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    css_path.parent.mkdir(parents=True, exist_ok=True)
    sectors: dict[str, set[str]] = defaultdict(set)
    buffer = []
    nonpositive = []
    count = zeros = negatives = 0
    logger.info("Reading PSILCA .zolca backup locally (Java 21+, read-only)...")
    lines = reader_lines(path)
    try:
        with atomic_output_path(combined_path) as temporary:
            with pq.ParquetWriter(
                temporary, _psilca_schema(), compression="zstd", use_dictionary=True
            ) as writer:
                for row in normalized_rows(lines):
                    count += 1
                    location = row["location"]
                    if location != "unknown" and row["reference_name"]:
                        sectors[location].add(row["reference_name"])
                    buffer.append(row)
                    value = row["mean_value"]
                    if value <= 0:
                        zeros += int(value == 0)
                        negatives += int(value < 0)
                        nonpositive.append(
                            {
                                "File": row["source_file"],
                                "Location": location,
                                "Reference name": row["reference_name"],
                                "Indicator": row["indicator"],
                                "Assessment type": row["assessment_type"],
                                "Risk level": row["risk_level"],
                                "Mean value": value,
                                "Reason": "zero" if value == 0 else "negative",
                            }
                        )
                    if len(buffer) >= 100_000:
                        _write_parquet_batch(writer, buffer)
                        buffer.clear()
                _write_parquet_batch(writer, buffer)
    finally:
        lines.close()
    write_json_atomic(css_path, {c: sorted(names) for c, names in sectors.items()})
    if zero_report_path:
        write_json_atomic(zero_report_path, nonpositive)
    logger.info("PSILCA exchanges: %d total, %d zero, %d negative", count, zeros, negatives)
