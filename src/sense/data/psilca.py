"""
PSILCA data processing for SENSE.

This module provides functions for:
- Parsing PSILCA EcoSpold01 XML files
- Extracting social exchanges
- Creating embeddings for PSILCA sectors
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from sense.data.embeddings import save_embeddings, prep_doc, store_is_complete


def _to_list(x):
    """Ensure x is a list (wrap single items)."""
    if x is None:
        return []
    return x if isinstance(x, list) else [x]


def combine_psilca_css_and_extract_social_exchanges(
    input_folder: str,
    combined_output_path: str,
    css_list_output_path: str,
    zero_report_path: Optional[str] = None,
) -> None:
    """
    Combine PSILCA EcoSpold01 XML files into one JSON, keeping all social exchanges.
    
    This function parses all PSILCA XML files and extracts:
    - Country-Specific Sectors (CSS) with their social exchanges
    - Working hours with risk levels for each sector
    
    Additionally, counts and exports exchanges with zero or negative meanValue
    to a separate report file for inspection.
    
    Args:
        input_folder: Path to folder containing PSILCA EcoSpold01 XML files
        combined_output_path: Path for the combined PSILCA JSON output
        css_list_output_path: Path for the country-sector list JSON output
        zero_report_path: Optional path for the zero/negative exchanges report
        
    Note:
        Requires the `xmltodict` package for XML parsing.
    """
    import xmltodict
    
    if os.path.exists(combined_output_path) and os.path.exists(css_list_output_path):
        print("⚠️ Both output files already exist. Skipping processing.")
        return

    os.makedirs(os.path.dirname(combined_output_path), exist_ok=True)
    os.makedirs(os.path.dirname(css_list_output_path), exist_ok=True)
    if zero_report_path:
        os.makedirs(os.path.dirname(zero_report_path), exist_ok=True)

    all_data = []
    zero_or_negative_records = []
    total_exchanges = 0
    zero_count = 0
    negative_count = 0

    xml_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".xml")]
    print(f"📂 Processing {len(xml_files)} XML files...\n")

    for filename in tqdm(xml_files, desc="🔄 Parsing files"):
        file_path = os.path.join(input_folder, filename)
        try:
            with open(file_path, encoding="utf-8") as f:
                data = xmltodict.parse(f.read())

            dataset = (data.get('ecoSpold') or {}).get('dataset') or {}
            meta_info = dataset.get('metaInformation') or {}
            ref_func = (meta_info.get('processInformation') or {}).get('referenceFunction') or {}
            location = (meta_info.get('processInformation') or {}).get('geography', {}).get('@location', 'unknown')

            exchanges = _to_list((dataset.get('flowData') or {}).get('exchange'))
            filtered_exchanges = []

            for exch in exchanges:
                if not isinstance(exch, dict):
                    continue
                unit = exch.get('@unit')
                output_group = (exch.get('outputGroup') or '').strip()
                mv_raw = exch.get('@meanValue')
                if mv_raw in (None, '', 'nan'):
                    continue
                try:
                    mv = float(mv_raw)
                except ValueError:
                    continue

                # Keep all hours in outputGroup 4, including 0 and negatives
                if unit == 'h' and output_group == '4':
                    total_exchanges += 1
                    filtered_exchanges.append(exch)

                    if mv == 0:
                        zero_count += 1
                        zero_or_negative_records.append({
                            "File": filename,
                            "Location": location,
                            "Reference name": ref_func.get('@name', 'unnamed'),
                            "Exchange": exch,
                            "Reason": "zero"
                        })
                    elif mv < 0:
                        negative_count += 1
                        zero_or_negative_records.append({
                            "File": filename,
                            "Location": location,
                            "Reference name": ref_func.get('@name', 'unnamed'),
                            "Exchange": exch,
                            "Reason": "negative"
                        })

            # Skip processes with no relevant social exchanges
            if not filtered_exchanges:
                continue

            product_name = (ref_func.get('@name', 'unnamed')
                            .replace("/", "-").replace("\\", "-"))
            entry = {
                "File": filename,
                "Location": location,
                "Reference name": product_name,
                "MetaInformation": meta_info,
                "FilteredExchanges": filtered_exchanges
            }
            all_data.append(entry)

        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")

    # Write combined PSILCA CSS with all social exchanges
    with open(combined_output_path, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Saved combined data to: {combined_output_path}")

    # Country-sector mapping
    country_sectors = defaultdict(set)
    for entry in all_data:
        loc = entry.get("Location", "unknown")
        sector = entry.get("Reference name", "Unnamed sector")
        if loc != "unknown" and sector:
            country_sectors[loc].add(sector)
    country_sectors = {c: sorted(sectors) for c, sectors in country_sectors.items()}

    with open(css_list_output_path, "w", encoding="utf-8") as f:
        json.dump(country_sectors, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved sector list to: {css_list_output_path}")

    # Zero/negative report
    print("\n📊 Social exchange statistics:")
    print(f"   Total exchanges processed: {total_exchanges}")
    print(f"   Zero-value exchanges: {zero_count}")
    print(f"   Negative-value exchanges: {negative_count}")

    if zero_or_negative_records and zero_report_path:
        with open(zero_report_path, "w", encoding="utf-8") as f:
            json.dump(zero_or_negative_records, f, indent=2, ensure_ascii=False)
        print(f"⚠️ Exported {len(zero_or_negative_records)} zero/negative exchanges to: {zero_report_path}")
    elif not zero_or_negative_records:
        print("✅ No zero or negative social exchanges found.")


def embed_psilca_to_store(
    psilca_sector_json: str,
    out_dir: str,
    model_name: str = "BAAI/bge-base-en-v1.5",
    batch_size: int = 256,
    per_country: bool = True,
) -> None:
    """
    Embed PSILCA sectors and save to disk.
    
    Creates embedding stores for PSILCA sectors, either:
    - Per-country: {out_dir}/psilca_{ISO3}.npy + manifest.json
    - Combined: {out_dir}/psilca_all.npy + manifest.json
    
    Skips already-existing stores to avoid redundant computation.
    
    Args:
        psilca_sector_json: Path to the CSS list JSON file
        out_dir: Directory to save embedding stores
        model_name: Name of the sentence-transformer model to use
        batch_size: Batch size for embedding computation
        per_country: If True, create separate stores per country; else combined
        
    Note:
        Requires the `sentence-transformers` package.
    """
    from sentence_transformers import SentenceTransformer
    
    os.makedirs(out_dir, exist_ok=True)
    done_flag = os.path.join(out_dir, "psilca_done.flag")

    with open(psilca_sector_json, "r", encoding="utf-8") as f:
        country_to_sectors = json.load(f)

    model = SentenceTransformer(model_name)
    meta = {
        "model": model_name,
        "prompt": "",
        "normalized": True,
        "source": "psilca_sectors",
    }

    if per_country:
        # Pre-scan to find missing stores
        missing = []
        for iso3, sectors in country_to_sectors.items():
            if not sectors:
                continue
            base_name = f"psilca_{iso3}"
            if not store_is_complete(out_dir, base_name):
                missing.append((iso3, sectors))

        if not missing:
            print(f"✅ All PSILCA per-country stores already present in {out_dir}. Nothing to do.")
            open(done_flag, "w").close()
            return

        # Only embed missing countries
        for iso3, sectors in tqdm(missing, desc="📦 Embedding PSILCA sectors (missing only)"):
            texts = [prep_doc(s) for s in sectors]
            embeddings = model.encode(
                texts,
                normalize_embeddings=True,
                convert_to_numpy=True,
                batch_size=batch_size,
                show_progress_bar=False,
            ).astype("float32")

            ids = pd.Series([f"{iso3}::{s}" for s in sectors], name="id")
            save_embeddings(out_dir, f"psilca_{iso3}", embeddings, ids, meta)

        open(done_flag, "w").close()
        print(f"✅ Finished PSILCA per-country embeddings → {out_dir}")

    else:
        # Single combined store
        base_name = "psilca_all"
        if store_is_complete(out_dir, base_name):
            print(f"✅ PSILCA combined store already present in {out_dir}. Nothing to do.")
            open(done_flag, "w").close()
            return

        # Build combined embeddings
        ids_all, texts_all = [], []
        for iso3, sectors in country_to_sectors.items():
            if not sectors:
                continue
            ids_all += [f"{iso3}::{s}" for s in sectors]
            texts_all += [prep_doc(s) for s in sectors]

        embeddings = model.encode(
            texts_all,
            normalize_embeddings=True,
            convert_to_numpy=True,
            batch_size=batch_size,
            show_progress_bar=True,
        ).astype("float32")

        save_embeddings(out_dir, base_name, embeddings, pd.Series(ids_all), meta)
        open(done_flag, "w").close()
        print(f"✅ Finished PSILCA combined embeddings → {out_dir}")
