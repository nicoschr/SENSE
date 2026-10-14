"""Parse PSILCA assessments and build country-sector embedding stores."""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from sense.artifacts import atomic_output_path, write_json_atomic
from sense.data.embeddings import _sha, prep_doc, save_embeddings, store_is_complete

logger = logging.getLogger(__name__)

PSILCA_CACHE_SCHEMA_VERSION = 3
_PSILCA_COLUMNS = (
    "process_index",
    "exchange_index",
    "source_file",
    "location",
    "reference_name",
    "indicator",
    "assessment_type",
    "risk_level",
    "mean_value",
    "unit",
    "category",
    "subcategory",
)


def _to_list(x):
    """Ensure x is a list (wrap single items)."""
    if x is None:
        return []
    return x if isinstance(x, list) else [x]


def _parse_assessment_name(name: str) -> tuple[str, str, str]:
    """Split a PSILCA exchange name into indicator, level, and assessment type."""

    value = (name or "").strip()
    if ";" not in value:
        return value, "no data", "risk"
    indicator, suffix = (part.strip() for part in value.split(";", 1))
    words = suffix.lower().split()
    if words and words[-1] in {"risk", "opportunity"}:
        return indicator, " ".join(words[:-1]).strip() or "no data", words[-1]
    return indicator, suffix.lower() or "no data", "risk"


def _write_parquet_batch(writer: Any, rows: list[dict[str, Any]]) -> None:
    """Append one normalized batch to an open Arrow Parquet writer."""

    if not rows:
        return
    import pyarrow as pa

    writer.write_table(pa.Table.from_pylist(rows, schema=writer.schema))


def _psilca_schema():
    """Shared, format-independent schema for XML and native database imports."""
    import pyarrow as pa

    return pa.schema(
        [
            ("process_index", pa.int32()),
            ("exchange_index", pa.int16()),
            ("source_file", pa.string()),
            ("location", pa.string()),
            ("reference_name", pa.string()),
            ("indicator", pa.string()),
            ("assessment_type", pa.string()),
            ("risk_level", pa.string()),
            ("mean_value", pa.float64()),
            ("unit", pa.string()),
            ("category", pa.string()),
            ("subcategory", pa.string()),
        ],
        metadata={"sense_schema": str(PSILCA_CACHE_SCHEMA_VERSION)},
    )


def validate_psilca_source(path: Path) -> Path:
    """Accept a native .zolca backup or an existing EcoSpold01 XML directory."""
    path = Path(path).expanduser()
    if path.is_dir():
        if not any(p.is_file() and p.suffix.lower() == ".xml" for p in path.iterdir()):
            raise FileNotFoundError(f"No PSILCA XML files found in {path}")
    else:
        from sense.data.zolca_reader import archive_fingerprint

        archive_fingerprint(path)
    return path


def combine_psilca_css_and_extract_social_exchanges(
    input_folder: str,
    combined_output_path: str,
    css_list_output_path: str,
    zero_report_path: str | None = None,
) -> None:
    """
    Parse PSILCA EcoSpold01 files into a compact assessment cache.

    The normalized rows preserve social risks, social opportunities, and
    working-hour intensities for each country-sector process.

    Additionally, counts and exports exchanges with zero or negative meanValue
    to a separate report file for inspection.

    Args:
        input_folder: Path to folder containing PSILCA EcoSpold01 XML files
        combined_output_path: Output Parquet path.
        css_list_output_path: Path for the country-sector list JSON output
        zero_report_path: Optional path for the zero/negative exchanges report

    Note:
        Requires the `xmltodict` package for XML parsing.
    """
    import pyarrow.parquet as pq
    import xmltodict
    from tqdm import tqdm

    combined_path = Path(combined_output_path)
    css_path = Path(css_list_output_path)
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    css_path.parent.mkdir(parents=True, exist_ok=True)

    if combined_path.suffix.lower() not in {".parquet", ".pq"}:
        raise ValueError("combined_output_path must use a .parquet or .pq suffix")
    schema = _psilca_schema()
    buffered_rows: list[dict[str, Any]] = []
    zero_or_negative_records: list[dict[str, Any]] = []
    country_sectors: defaultdict[str, set[str]] = defaultdict(set)
    total_exchanges = 0
    zero_count = 0
    negative_count = 0
    parse_errors = []

    xml_files = sorted(f for f in os.listdir(input_folder) if f.lower().endswith(".xml"))
    if not xml_files:
        raise FileNotFoundError(f"No PSILCA XML files found in {input_folder}")
    logger.info("Processing %d PSILCA XML files", len(xml_files))

    with atomic_output_path(combined_path) as temporary_path:
        with pq.ParquetWriter(
            temporary_path, schema, compression="zstd", use_dictionary=True
        ) as writer:
            for process_index, filename in enumerate(
                tqdm(xml_files, desc="Parsing PSILCA XML", unit="file", dynamic_ncols=True)
            ):
                file_path = os.path.join(input_folder, filename)
                try:
                    with open(file_path, encoding="utf-8") as f:
                        data = xmltodict.parse(f.read())

                    dataset = (data.get("ecoSpold") or {}).get("dataset") or {}
                    meta_info = dataset.get("metaInformation") or {}
                    ref_func = (meta_info.get("processInformation") or {}).get(
                        "referenceFunction"
                    ) or {}
                    location = (
                        (meta_info.get("processInformation") or {})
                        .get("geography", {})
                        .get("@location", "unknown")
                    )

                    exchanges = _to_list((dataset.get("flowData") or {}).get("exchange"))
                    normalized_exchanges: list[dict[str, Any]] = []
                    product_name = (
                        ref_func.get("@name", "unnamed").replace("/", "-").replace("\\", "-")
                    )

                    for exchange_index, exch in enumerate(exchanges):
                        if not isinstance(exch, dict):
                            continue
                        unit = exch.get("@unit")
                        output_group = (exch.get("outputGroup") or "").strip()
                        mv_raw = exch.get("@meanValue")
                        try:
                            mv = float(mv_raw) if mv_raw is not None else float("nan")
                        except (TypeError, ValueError):
                            mv = float("nan")

                        # Keep all hours in outputGroup 4, including 0 and negatives
                        if unit == "h" and output_group == "4":
                            total_exchanges += 1
                            indicator, risk_level, assessment_type = _parse_assessment_name(
                                exch.get("@name") or ""
                            )
                            normalized = {
                                "process_index": process_index,
                                "exchange_index": exchange_index,
                                "source_file": filename,
                                "location": location,
                                "reference_name": product_name,
                                "indicator": indicator,
                                "assessment_type": assessment_type,
                                "risk_level": risk_level,
                                "mean_value": mv,
                                "unit": unit,
                                "category": exch.get("@category", ""),
                                "subcategory": exch.get("@subCategory", ""),
                            }
                            normalized_exchanges.append(normalized)

                            if mv == 0:
                                zero_count += 1
                                zero_or_negative_records.append(
                                    {
                                        "File": filename,
                                        "Location": location,
                                        "Reference name": ref_func.get("@name", "unnamed"),
                                        "Indicator": indicator,
                                        "Assessment type": assessment_type,
                                        "Risk level": risk_level,
                                        "Mean value": mv,
                                        "Reason": "zero",
                                    }
                                )
                            elif mv < 0:
                                negative_count += 1
                                zero_or_negative_records.append(
                                    {
                                        "File": filename,
                                        "Location": location,
                                        "Reference name": ref_func.get("@name", "unnamed"),
                                        "Indicator": indicator,
                                        "Assessment type": assessment_type,
                                        "Risk level": risk_level,
                                        "Mean value": mv,
                                        "Reason": "negative",
                                    }
                                )

                    # Skip processes with no relevant social exchanges
                    if not normalized_exchanges:
                        continue
                    if location != "unknown" and product_name:
                        country_sectors[location].add(product_name)
                    buffered_rows.extend(normalized_exchanges)
                    if len(buffered_rows) >= 100_000:
                        _write_parquet_batch(writer, buffered_rows)
                        buffered_rows.clear()

                except Exception as e:
                    parse_errors.append((filename, str(e)))

            _write_parquet_batch(writer, buffered_rows)

            if parse_errors:
                examples = "; ".join(
                    f"{filename}: {message}" for filename, message in parse_errors[:5]
                )
                raise RuntimeError(
                    f"Failed to parse {len(parse_errors)} of {len(xml_files)} PSILCA XML "
                    f"files. No partial cache was written. Examples: {examples}"
                )
            if total_exchanges == 0:
                raise ValueError(
                    f"No PSILCA processes with hour exchanges in outputGroup 4 were found in {input_folder}"
                )

    logger.info("Saved PSILCA assessment cache to %s", combined_path)

    # Country-sector mapping
    country_sector_lists = {c: sorted(sectors) for c, sectors in country_sectors.items()}
    write_json_atomic(css_path, country_sector_lists)
    logger.info("Saved PSILCA sector list to %s", css_path)

    # Zero/negative report
    logger.info(
        "PSILCA exchanges: %d total, %d zero, %d negative",
        total_exchanges,
        zero_count,
        negative_count,
    )

    if zero_or_negative_records and zero_report_path:
        write_json_atomic(Path(zero_report_path), zero_or_negative_records)
        logger.info(
            "Exported %d zero/negative exchanges to %s",
            len(zero_or_negative_records),
            zero_report_path,
        )


def embed_psilca_to_store(
    psilca_sector_json: str,
    out_dir: str,
    model_name: str = "BAAI/bge-base-en-v1.5",
    batch_size: int = 256,
    force: bool = False,
    revision: str | None = None,
) -> None:
    """
    Embed PSILCA sectors and save to disk.

    Creates one embedding store per PSILCA country:
    ``{out_dir}/psilca_{ISO2}.npy`` plus its manifest.

    Skips already-existing stores to avoid redundant computation.

    Args:
        psilca_sector_json: Path to the CSS list JSON file
        out_dir: Directory to save embedding stores
        model_name: Name of the sentence-transformer model to use
        batch_size: Batch size for embedding computation
        force: Rebuild stores even when their manifests match.

    Note:
        Requires the `sentence-transformers` package.
    """
    os.makedirs(out_dir, exist_ok=True)

    with open(psilca_sector_json, encoding="utf-8") as f:
        country_to_sectors = json.load(f)

    from sense.data.embeddings import model_revision

    resolved_revision = model_revision(model_name, revision)
    common_meta = {
        "model": model_name,
        "revision": resolved_revision,
        "prompt": "",
        "normalized": True,
        "source": "psilca_sectors",
    }

    missing = []
    for country, sectors in country_to_sectors.items():
        if not sectors:
            continue
        base_name = f"psilca_{country}"
        texts = [prep_doc(sector) for sector in sectors]
        ids = [f"{country}::{sector}" for sector in sectors]
        meta = {
            **common_meta,
            "input_hash": _sha(json.dumps(texts, ensure_ascii=False)),
        }
        if force or not store_is_complete(out_dir, base_name, expected_ids=ids, expected_meta=meta):
            missing.append((country, texts, ids, meta))

    if not missing:
        logger.info("All PSILCA per-country embedding stores are current in %s", out_dir)
        return

    logger.info("Creating %d PSILCA country embedding stores", len(missing))
    logger.info("Loading embedding model '%s'...", model_name)
    from tqdm import tqdm

    from sense.data.embeddings import load_encoder

    model = load_encoder(model_name, revision=resolved_revision)
    for country, texts, ids_raw, meta in tqdm(
        missing,
        desc="Embedding PSILCA countries",
        unit="country",
        dynamic_ncols=True,
    ):
        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            batch_size=batch_size,
            show_progress_bar=False,
        ).astype("float32")
        save_embeddings(
            out_dir,
            f"psilca_{country}",
            embeddings,
            pd.Series(ids_raw, name="id"),
            meta,
        )

    logger.info("Finished PSILCA per-country embedding stores in %s", out_dir)
