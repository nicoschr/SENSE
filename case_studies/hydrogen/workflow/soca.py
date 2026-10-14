"""Parse and cache licensed SOCA exports without executable serialization."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
from tqdm import tqdm

NS1 = {"spold": "http://www.EcoInvent.org/EcoSpold01"}
STAKEHOLDER_GROUPS = {
    "Society",
    "Workers",
    "Local Community",
    "Value Chain Actors",
    "Children",
    "Consumer",
}
LEVEL_WEIGHTS = {
    "no": 0.001,
    "very low": 0.01,
    "low": 0.1,
    "medium": 1,
    "high": 10,
    "very high": 100,
    "no data": 0.1,
}


def norm(x):
    """Normalize strings: strip, lower; keep None as None."""
    return None if x is None else str(x).strip().lower()


def level_to_weight(level):
    """Return the configured case-insensitive level weight, or ``None`` if unmapped."""
    if not isinstance(level, str):
        return None
    return LEVEL_WEIGHTS.get(level.strip().lower())


def get_stakeholder_from_categories(categories):
    """Return (stakeholder, subgroup) from a recognized Brightway category.

    Accept a string or a sequence, using the first recognized stakeholder prefix.
    A category without a subgroup returns ``None`` for that field; an unrecognized
    category returns ``(None, None)``.
    """
    if not categories:
        return (None, None)
    if isinstance(categories, (list, tuple)):
        cat_elems = [str(c) for c in categories]
    else:
        cat_elems = [str(categories)]
    if len(cat_elems) >= 2 and cat_elems[0] in STAKEHOLDER_GROUPS:
        return (cat_elems[0], cat_elems[1])
    candidate = None
    for c in cat_elems:
        for sh in STAKEHOLDER_GROUPS:
            if c.startswith(sh):
                candidate = c
                break
        if candidate is not None:
            break
    if candidate is None:
        return (None, None)
    if " - " in candidate:
        stakeholder_group, subgroup = candidate.split(" - ", 1)
        stakeholder_group = stakeholder_group.strip()
        subgroup = subgroup.strip()
    else:
        stakeholder_group = candidate.strip()
        subgroup = None
    if stakeholder_group not in STAKEHOLDER_GROUPS:
        return (None, None)
    return (stakeholder_group, subgroup)


def parse_sense_flow_name(flow_name):
    """Return the indicator and trailing risk/opportunity type from a SENSE name.

    Strip only the final ``(risk|opportunity), location`` suffix, preserving internal
    parentheses. An unmatched name is returned stripped with type ``None``;
    ``None`` input returns ``(None, None)``.
    """
    if flow_name is None:
        return (None, None)
    s = str(flow_name).strip()
    match = re.match("^(.*) \\((risk|opportunity)\\), .+$", s, flags=re.IGNORECASE)
    if match is None:
        return (s, None)
    return (match.group(1).rstrip(), match.group(2).lower())


def parse_spold_name_field(raw_name):
    """Return (indicator, level, assessment type) from a SOCA exchange name.

    Split at the first semicolon and recognize a final risk/opportunity token.
    For example, ``Internet freedom scores; very low risk`` yields
    ``('Internet freedom scores', 'very low', 'risk')``. Bare ``no data`` after the
    semicolon maps to level ``no data`` and type ``risk``. Missing components return
    ``None``; an unrecognized suffix remains a level with no assessment type.
    """
    if raw_name is None:
        return (None, None, None)
    txt = str(raw_name).strip()
    if not txt:
        return (None, None, None)
    parts = txt.split(";", 1)
    indicator = parts[0].strip()
    rest = parts[1].strip() if len(parts) > 1 else ""
    if not rest:
        return (indicator or None, None, None)
    if rest.lower() == "no data":
        return (indicator or None, "no data", "risk")
    tokens = rest.split()
    if not tokens:
        return (indicator or None, None, None)
    last = tokens[-1].lower()
    if last in {"risk", "opportunity"}:
        type_ = last
        level = " ".join(tokens[:-1]).strip() or None
    else:
        type_ = None
        level = rest or None
    return (indicator or None, level, type_)


def read_excel_sheet(path: Path, sheet_name: str, **read_kwargs):
    """Read an overview worksheet, forwarding keyword arguments to pandas.

    Re-raise file and worksheet errors with case-study input guidance.
    """
    try:
        return pd.read_excel(path, sheet_name=sheet_name, **read_kwargs)
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"Excel file not found: {path}\nSet paths.ecoinvent_geographies_excel in config.local.yaml."
        ) from error
    except ValueError as e:
        raise ValueError(
            f"Sheet '{sheet_name}' not found in {path}.\nPlease make sure you're using the ecoinvent overview file with the expected sheet names ('Cut-Off AO', 'Geographies')."
        ) from e


def spold_signature(folder: Path, pattern: str) -> str:
    """Hash source contents, paths and parser schema for SOCA cache checks.

    Content hashes detect edits even when file counts and timestamps are retained.
    """
    from sense.utils.cache import file_hash, stable_hash

    return str(
        stable_hash(
            {
                "schema": 2,
                "folder": str(folder.resolve()),
                "pattern": pattern,
                "files": [
                    (str(path.relative_to(folder)), file_hash(path))
                    for path in sorted(folder.glob(pattern))
                    if path.is_file()
                ],
            }
        )
    )


def load_or_parse_spold(ctx):
    """Return SOCA metadata and social-flow DataFrames from cache or XML.

    Use non-executable Parquet tables and verify both checksums before reuse.
    Publish the manifest last, so an interrupted replacement cannot mix caches.
    """
    from sense.artifacts import atomic_output_path, write_json_atomic
    from sense.utils.cache import file_hash, read_manifest

    sig = spold_signature(ctx.spold_folder, ctx.spold_pattern)
    if ctx.use_cache and (not ctx.force_rebuild_cache):
        if ctx.cache_mapping.exists() and ctx.cache_xml.exists() and ctx.cache_meta.exists():
            manifest = read_manifest(ctx.cache_meta) or {}
            if (
                manifest.get("source") == sig
                and manifest.get("mapping") == file_hash(ctx.cache_mapping)
                and manifest.get("xml") == file_hash(ctx.cache_xml)
            ):
                print(f"Loading cached EcoSpold parses from: {ctx.cache_dir}")
                mapping_df = pd.read_parquet(ctx.cache_mapping)
                xml_df = pd.read_parquet(ctx.cache_xml)
                print(f"  Cached mapping_df rows: {len(mapping_df)}")
                print(f"  Cached xml_df rows:     {len(xml_df)}\n")
                return (mapping_df, xml_df)
    mapping_df, xml_df = process_spold_folder(ctx)
    if ctx.use_cache:
        print(f"Saving EcoSpold cache to: {ctx.cache_dir}")
        checksums = {}
        for name, frame, destination in (
            ("mapping", mapping_df, ctx.cache_mapping),
            ("xml", xml_df, ctx.cache_xml),
        ):
            with atomic_output_path(destination) as temporary:
                frame.to_parquet(temporary, index=False)
                checksums[name] = file_hash(temporary)
        write_json_atomic(ctx.cache_meta, {"source": sig, **checksums})
        print("  Cache saved.\n")
    return (mapping_df, xml_df)


def parse_spold_file(path: Path):
    """Return one SOCA file's metadata dict and list of social-exchange dicts.

    Read EcoSpold01 XML and extract the dataset ID from its ecoQuery URL. Without
    an ID, return metadata but no social rows. For recognized stakeholder exchanges,
    record raw amounts and level-weighted medium risk/opportunity hours; invalid
    amounts or unknown weights produce ``None`` for the weighted value. XML parsing
    errors propagate to the caller.
    """
    text = path.read_text(encoding="utf-8", errors="ignore")
    m = re.search("https://ecoquery\\.ecoinvent\\.org/([^/]+)/([^/]+)/dataset/(\\d+)/", text)
    if m:
        version, system_model, dataset_id = m.groups()
        dataset_id = int(dataset_id)
    else:
        version = system_model = None
        dataset_id = None
    root = ET.fromstring(text)
    rf = root.find(".//spold:referenceFunction", NS1)
    geo = root.find(".//spold:geography", NS1)
    product = rf.get("name") if rf is not None else None
    unit = rf.get("unit") if rf is not None else None
    location = geo.get("location") if geo is not None else None
    meta_row = {
        "file": str(path),
        "file_name": path.name,
        "dataset_id": dataset_id,
        "eco_version": version,
        "system_model": system_model,
        "product": product,
        "unit": unit,
        "location": location,
    }
    social_rows = []
    if dataset_id is not None:
        for exc in root.findall(".//spold:exchange", NS1):
            cat = exc.get("category")
            if cat not in STAKEHOLDER_GROUPS:
                continue
            subcat = exc.get("subCategory")
            raw_name = exc.get("name")
            exc_unit = exc.get("unit")
            mean_val_str = exc.get("meanValue")
            try:
                mean_value = float(mean_val_str) if mean_val_str is not None else None
            except ValueError:
                mean_value = None
            indicator, level, type_ = parse_spold_name_field(raw_name)
            w = level_to_weight(level)
            medium_hours = mean_value * w if mean_value is not None and w is not None else None
            social_rows.append(
                {
                    "dataset_id": dataset_id,
                    "file_name": path.name,
                    "eco_version": version,
                    "system_model": system_model,
                    "product": product,
                    "location": location,
                    "stakeholder_group": cat,
                    "subgroup": subcat,
                    "indicator": indicator,
                    "level": level,
                    "type": type_,
                    "unit": exc_unit,
                    "mean_value": mean_value,
                    "level_weight": w,
                    "medium_risk_opportunity_hours": medium_hours,
                    "raw_name": raw_name,
                    "exchange_number": exc.get("number"),
                }
            )
    return (meta_row, social_rows)


def process_spold_folder(ctx):
    """Parse configured SOCA files into metadata and social-flow DataFrames.

    Read sorted ``SPOLD_PATTERN`` matches under ``SPOLD_FOLDER`` and retain records
    with ecoQuery dataset IDs. Raise ``FileNotFoundError`` if no files match.
    """
    print("Parsing EcoSpold01 files...")
    spold_files = sorted(ctx.spold_folder.glob(ctx.spold_pattern))
    if not spold_files:
        raise FileNotFoundError(
            f"No files matching {ctx.spold_pattern!r} in {ctx.spold_folder.resolve()}\nPlease set SPOLD_FOLDER and SPOLD_PATTERN correctly in the USER CONFIG section."
        )
    meta_rows = []
    social_rows = []
    for path in tqdm(spold_files, desc="EcoSpold files"):
        meta_row, s_rows = parse_spold_file(path)
        meta_rows.append(meta_row)
        social_rows.extend(s_rows)
    mapping_df = pd.DataFrame(meta_rows)
    mapping_df = mapping_df.dropna(subset=["dataset_id"]).copy()
    mapping_df["dataset_id"] = mapping_df["dataset_id"].astype("int64")
    xml_df = pd.DataFrame(social_rows)
    if not xml_df.empty:
        xml_df["dataset_id"] = xml_df["dataset_id"].astype("Int64")
    print(f"  EcoSpold files with dataset_id: {len(mapping_df)}")
    print(f"  Unique dataset_ids in XML:      {mapping_df['dataset_id'].nunique()}")
    print(f"  Social exchanges from XML:      {len(xml_df)}\n")
    return (mapping_df, xml_df)
