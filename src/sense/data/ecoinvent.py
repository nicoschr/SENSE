"""
Ecoinvent data processing for SENSE.

This module provides functions for:
- Mapping ecoinvent locations to PSILCA countries
- Extracting activities and prices from EcoSpold02 files
- Creating embeddings for ecoinvent activities
"""

from __future__ import annotations

import json
import os
import pickle
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from sense.data.embeddings import save_embeddings, prep_query_bge, BGE_QUERY_PROMPT, store_is_complete
from sense.utils.helpers import norm_code


def map_ecoinvent_locations(
    ecoinvent_excel_path: str,
    psilca_sector_json_path: str,
) -> pd.DataFrame:
    """
    Map ecoinvent geographies to PSILCA countries.
    
    This function translates all ecoinvent locations into sets of PSILCA-compatible
    ISO2 country codes. The mapping depends on geography classification:
    
    - Countries present in PSILCA stay as-is (US -> US)
    - Subnational regions get assigned to their parent country (US-CO -> US)
    - Supranational regions get assigned all countries they contain (RER -> DE;FR;IT;...)
    - Countries not in PSILCA get assigned to the smallest overlapping region
    - Some locations require manual overrides
    
    Args:
        ecoinvent_excel_path: Path to ecoinvent Database Overview Excel file
        psilca_sector_json_path: Path to PSILCA sector JSON (country -> sectors)
        
    Returns:
        DataFrame with columns including 'Shortname', 'simplified_ISO2', 'PSILCA_mapping'
    """
    # Load ecoinvent locations Excel
    ecoinvent_locations_df = pd.read_excel(ecoinvent_excel_path, sheet_name="Geographies")

    # Load PSILCA ISO2 -> sectors mapping from JSON
    with open(psilca_sector_json_path, encoding='utf-8') as f:
        psilca_sector_dict = json.load(f)

    # Insert simplified_ISO2 column
    shortname_index = ecoinvent_locations_df.columns.get_loc('Shortname')
    ecoinvent_locations_df.insert(shortname_index + 1, 'simplified_ISO2', None)

    def simplify_iso2(row):
        geo_type = str(row['Geography Classification'])
        shortname = str(row['Shortname'])
        contained = str(row['Contained and Overlapping Geographies'])

        # Countries stay as is
        if 'Country' in geo_type:
            return shortname
        # Subnational regions get assigned to their parent country
        elif 'Province or State' in geo_type:
            if shortname[2] == '-':
                return shortname[:2]
            else:
                return shortname
        # Supranational regions get assigned all countries they contain
        elif any(term in geo_type for term in [
            'Aluminium Industry', 'ecoinvent Custom Area', 'ecoinvent Region',
            'Electricity Grid', 'UN Region', 'UN Subregion', 'World'
        ]):
            codes = [code.strip() for code in contained.split(';')]
            iso2_codes = [code for code in codes if re.fullmatch(r'[A-Z]{2}', code)]
            return '; '.join(iso2_codes)
        return None

    ecoinvent_locations_df['simplified_ISO2'] = ecoinvent_locations_df.apply(simplify_iso2, axis=1)

    # Clean up electricity grid data
    mask = (
        (ecoinvent_locations_df['Geography Classification'] == 'Electricity Grid') &
        (ecoinvent_locations_df['simplified_ISO2'].isna() | 
         (ecoinvent_locations_df['simplified_ISO2'].str.strip() == ''))
    )
    ecoinvent_locations_df.loc[mask, 'simplified_ISO2'] = \
        ecoinvent_locations_df.loc[mask, 'Shortname'].str[:2]

    # Manual overrides
    manual_overrides = {
        'Clipperton Island': 'FR', 'Canary Islands': 'ES',
        'France, including overseas territories': 'FR',
        'Australia, including overseas territories': 'AU', 'Puerto Rico': 'US',
        'China without Inner Mongol': 'CN', 'Canada without Alberta': 'CA',
        'Canada without Alberta and Quebec': 'CA', 'Canada without Quebec': 'CA',
        'Russia (Europe)': 'RU', 'Russia (Asia)': 'RU', 'Coral Sea Islands': 'AU',
        'Kosovo': 'XK', 'Serranilla Bank': 'CO', 'Spratly Islands': 'PH',
        'Dhekelia Sovereign Base Area': 'CY', 'Serbia and Montenegro': 'RS',
        'Cyprus No Mans Area': 'CY', 'Siachen Glacier': 'IN',
        'Akrotiri Sovereign Base Area': 'GB', 'Bajo Nuevo Bank (Petrel Is.)': 'CO',
        'Taiwan': 'TW', 'Scarborough Reef': 'PH', 'Northern Cyprus': 'CY',
        'US Naval Base Guantanamo Bay': 'CU', 'Somaliland': 'SO',
        'Québec, Hydro-Québec distribution network': 'CA', 'Namibia': 'NA'
    }

    ecoinvent_locations_df['simplified_ISO2'] = ecoinvent_locations_df.apply(
        lambda row: manual_overrides.get(row['Name'], row['simplified_ISO2']),
        axis=1
    )

    ecoinvent_locations_df.loc[
        ecoinvent_locations_df["Name"] == "Namibia", "Shortname"
    ] = "NA"

    # Get PSILCA countries
    psilca_iso2_set = set(psilca_sector_dict.keys())

    # Build region map for multi-country regions
    region_map = {
        idx: set(code.strip() for code in str(row['simplified_ISO2']).split(';'))
        for idx, row in ecoinvent_locations_df.iterrows()
        if row['simplified_ISO2'] and ';' in str(row['simplified_ISO2'])
    }

    def map_to_psilca(row):
        iso_string = row['simplified_ISO2']
        if not iso_string or pd.isna(iso_string):
            return None, None

        iso_list = [code.strip() for code in iso_string.split(';') if code.strip()]
        iso_set = set(iso_list)

        # Case: single country
        if len(iso_list) == 1:
            iso2 = iso_list[0]
            if iso2 in psilca_iso2_set:
                return iso2, None

            # Find smallest region containing this country + at least one PSILCA country
            candidates = [
                (idx, region_map[idx])
                for idx in region_map
                if iso2 in region_map[idx]
            ]
            candidates.sort(key=lambda x: len(x[1]))

            for idx, region in candidates:
                psilca_countries = [code for code in region if code in psilca_iso2_set]
                if psilca_countries:
                    removed = [code for code in region if code not in psilca_iso2_set]
                    region_name = ecoinvent_locations_df.loc[idx, 'Shortname']
                    info = f"Used region: {region_name}, removed: {', '.join(removed)}"
                    return '; '.join(psilca_countries), info

            return None, f"No PSILCA region found for {iso2}"

        # Case: region (multiple countries)
        else:
            filtered = [code for code in iso_list if code in psilca_iso2_set]
            if filtered:
                removed = [code for code in iso_list if code not in psilca_iso2_set]
                info = f"Removed: {', '.join(removed)}" if removed else None
                return '; '.join(filtered), info
            else:
                candidates = [
                    (idx, region_map[idx])
                    for idx in region_map
                    if iso_set & region_map[idx] and any(code in psilca_iso2_set for code in region_map[idx])
                ]
                candidates.sort(key=lambda x: len(x[1]))

                for idx, region in candidates:
                    psilca_countries = [code for code in region if code in psilca_iso2_set]
                    if psilca_countries:
                        removed = [code for code in region if code not in psilca_iso2_set]
                        region_name = ecoinvent_locations_df.loc[idx, 'Shortname']
                        info = f"Used overlapping region: {region_name}, removed: {', '.join(removed)}"
                        return '; '.join(psilca_countries), info

                return None, f"No overlapping PSILCA region found for region: {iso_string}"

    ecoinvent_locations_df[['PSILCA_mapping', 'PSILCA_mapping_info']] = \
        ecoinvent_locations_df.apply(map_to_psilca, axis=1, result_type='expand')

    return ecoinvent_locations_df


def fetch_ecoinvent_activities_from_ecospold_files(
    folder_path: str,
    mapping_df: pd.DataFrame,
    output_path: str,
    technosphere: str,
    background_databases: Any,
    foreground: bool,
    missing_price_report_path: Optional[str] = None,
    exclude_terms: Optional[Sequence[str]] = None,
    exclude_terms_starting_with: Optional[Sequence[str]] = None,
    price_floor_percentile: float = 20.0,
) -> pd.DataFrame:
    """
    Build a positive-price activity table for a Brightway2 database.
    
    Estimates missing prices from EcoSpold files and maps activities to PSILCA regions.
    
    Pre-check order (foreground mode):
      0) Explicit price? If activity has price>0 → use and stop.
      1) In a loop (cycle)? → flag `foreground_cycle`, skip pricing this node.
      2) Has technosphere inputs? If none → flag `foreground_no_inputs`, skip.
      3) Has a path to any background DB? If not → flag `no_path_to_background`, skip.

    Pricing rules:
      - Inputs that are no-input activities are ignored (0 contribution).
      - Inputs that are in cycles or have no path to background are blocking.
      - Gross-only pricing: ignore negative-amount BG exchanges (credits).
      - Recursively accumulate priced background leaves; require total > 0.
    
    Args:
        folder_path: Path to folder containing EcoSpold02 (.spold) files
        mapping_df: DataFrame mapping ISO2 codes to PSILCA regions
        output_path: Path for the resulting JSON output
        technosphere: Name of the technosphere database to analyze
        background_databases: Background database name(s) for price estimation
        foreground: If True, treat technosphere as a foreground database
        missing_price_report_path: Optional path for price report
        exclude_terms: Terms that exclude activities from output if found anywhere
        exclude_terms_starting_with: Terms that exclude if found as prefix
        price_floor_percentile: Percentile for unit-level price floor (0 to disable)
        
    Returns:
        DataFrame with activity data including prices
        
    Note:
        Requires Brightway2 (`bw2data`) to be installed and a project to be active.
    """
    from bw2data import Database

    # ---------------------------------------------------------------------------------
    # Detailed function overview:
    # Builds a per-activity positive-price table for a Brightway2 technosphere
    # database. Extraction strategy:
    #  - Load explicit prices from activities (foreground explicit prices)
    #  - Build `flat_price_lookup` from EcoSpold .spold files (background prices)
    #  - For foreground activities, either compute a one-shot BG-only sum when
    #    all inputs are background, or resolve mixed FG/BG activities by
    #    iterative recursion using already-resolved FG children.
    #
    # The function records diagnostics in `not_found_records` and writes the
    # final per-ISO2 JSON to `output_path`.
    # ---------------------------------------------------------------------------------

    # Default exclude terms
    if exclude_terms is None:
        exclude_terms = []
    if exclude_terms_starting_with is None:
        exclude_terms_starting_with = []
    
    # EcoSpold v2 namespace
    ns = {'eco': 'http://www.EcoInvent.org/EcoSpold02'}

    def clean(text: str) -> str:
        return re.sub(r'\s+', ' ', (text or "").replace('\xa0', ' ').strip().lower())

    def normalize_name(name: str) -> str:
        return re.sub(r'\s+', ' ', (name or "").lower().strip().replace(',', '').replace('-', ''))

    # Pre-lowercase for efficient checks
    exclude_prefixes_l = tuple(t.lower() for t in exclude_terms_starting_with)
    exclude_substrings_l = tuple(t.lower() for t in exclude_terms)

    def is_skipped_name(name: str) -> bool:
        s = (name or "").lower()
        for pref in exclude_prefixes_l:
            if s.startswith(pref):
                return True
        for term in exclude_substrings_l:
            if term in s:
                return True
        return False

    # Normalize background DBs to a set
    if isinstance(background_databases, (list, tuple, set)):
        bg_dbs: Set[str] = set(background_databases)
    else:
        bg_dbs = {str(background_databases)}

    # PSILCA mappings
    iso2_to_psilca: Dict[str, str] = mapping_df.set_index('Shortname')['PSILCA_mapping'].to_dict()
    base_psilca_sets: Dict[str, Set[str]] = {
        "RoW": set(iso2_to_psilca.get("GLO", "").split("; ")),
        "RoE": set(iso2_to_psilca.get("RER", "").split("; ")),
    }

    # Load database
    techno_db = Database(technosphere)

    activity_meta: Dict[str, Dict[str, Any]] = {}
    estimated_prices: Dict[str, float] = {}
    not_found_records: List[Dict[str, Any]] = []
    
    # Storage for foreground analysis
    act_by_code: Dict[str, Any] = {}
    inputs_by_code: Dict[str, List[Any]] = {}
    codes_in_cycles: Set[str] = set()
    scc_membership: Dict[str, List[str]] = {}
    codes_with_no_inputs: Set[str] = set()
    codes_no_path_to_bg: Set[str] = set()

    def _sb_tags(code: str, reason: str) -> Dict[str, Any]:
        """Mark true source blockers."""
        if not foreground:
            return {"is_source_blocker": False, "reason_tag": "", "source_blocker_type": ""}

        ins = inputs_by_code.get(code, [])
        has_fg = any(getattr(exc.input, 'database', None) == technosphere for exc in ins)
        has_bg = any(getattr(exc.input, 'database', None) in bg_dbs for exc in ins)

        if reason in ("foreground_no_inputs", "no_path_to_background", "foreground_cycle"):
            return {"is_source_blocker": False, "reason_tag": "", "source_blocker_type": ""}

        if (reason in ("foreground_total_nonpositive", "foreground_zero_inputs_no_unit_p20", 
                       "nonpositive_after_cleanup") and not has_fg and has_bg):
            return {"is_source_blocker": True, "reason_tag": "source_blocker", 
                    "source_blocker_type": "bg_only_root"}

        if reason in ("foreground_unresolvable_inputs",) and not has_fg and has_bg:
            return {"is_source_blocker": True, "reason_tag": "source_blocker", 
                    "source_blocker_type": "bg_missing_price"}

        return {"is_source_blocker": False, "reason_tag": "", "source_blocker_type": ""}

    # Fast path for fully priced foreground
    has_all_prices = foreground and all((act.get("price", 0) or 0) > 0 for act in techno_db)
    
    if has_all_prices:
        for act in techno_db:
            code = norm_code(act["code"])
            estimated_prices[code] = float(act["price"])
            activity_meta[code] = {
                "name": act["name"],
                "reference product": act["reference product"],
                "unit": (act["unit"] or "").lower(),
                "location": act["location"],
            }
        isic_lookup: Dict[Tuple[str, str], Set[str]] = {}
    else:
        # ------------------ Parse EcoSpold files (cached) ------------------
        # Collect price observations from EcoSpold02 files. Results are stored
        # in `product_price_map` which maps (unit, product) -> list(prices).
        # If a `spold_cache.pkl` exists we load `product_price_map` and
        # `isic_lookup` from the cache to avoid re-parsing thousands of files.
        cache_file = os.path.join(folder_path, "spold_cache.pkl")
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                product_price_map, isic_lookup = pickle.load(f)
        else:
            product_price_map: Dict[Tuple[str, str], List[float]] = defaultdict(list)
            isic_lookup: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
            spold_files = [os.path.join(folder_path, f) 
                          for f in os.listdir(folder_path) if f.endswith(".spold")]

            for path in tqdm(spold_files, desc="📄 Parsing .spold files"):
                try:
                    root = ET.parse(path).getroot()
                    act_name = root.findtext('.//eco:activity/eco:activityName', 
                                            default='', namespaces=ns).strip()

                    ref = next(
                        (ie for ie in root.findall(".//eco:intermediateExchange", ns)
                         if (ie.findtext("eco:outputGroup", default='', namespaces=ns) or "").strip() == "0"),
                        None
                    )
                    if not ref:
                        continue

                    ref_name = (ref.findtext("eco:name", default='', namespaces=ns) or "").strip().lower()
                    unit = (ref.findtext("eco:unitName", default='', namespaces=ns) or "").strip().lower()

                    price = next(
                        (float(p.attrib["amount"]) for p in ref.findall("eco:property", ns)
                         if (p.findtext("eco:name", default='', namespaces=ns) or "").lower() == "price"
                         and p.attrib.get("amount") is not None),
                        None
                    )
                    if ref_name and price is not None:
                        product_price_map[(unit, ref_name)].append(price)

                    isic = next(
                        (c.findtext("eco:classificationValue", default='', namespaces=ns).strip()
                         for c in root.findall(".//eco:classification", ns)
                         if "isic" in (c.findtext("eco:classificationSystem", default='', 
                                                  namespaces=ns) or "").lower()),
                        None
                    )
                    if ref_name and isic:
                        isic_lookup[(clean(act_name), clean(ref_name))].add(isic)
                except Exception:
                    continue

            with open(cache_file, "wb") as f:
                pickle.dump((product_price_map, isic_lookup), f)

        # ------------------ Build cleaned lookup (unit-level price floor) ------------------
        # Convert raw price lists into an average price per (unit, product).
        # Optionally compute a per-unit percentile floor (`unit_price_floor`) to
        # replace zero observations with a conservative estimate. The result
        # `flat_price_lookup` maps normalized product names -> average price.
        avg_prices: Dict[str, Dict[str, float]] = defaultdict(dict)
        for (unit, product), prices in product_price_map.items():
            valid = [p for p in prices if p is not None]
            if valid:
                avg_prices[unit][product] = sum(valid) / len(valid)

        unit_price_floor: Dict[str, float] = {}
        q = None
        if price_floor_percentile is not None and price_floor_percentile > 0:
            q = float(price_floor_percentile) / 100.0

        for unit, products in avg_prices.items():
            vals = [v for v in products.values() if v is not None and v > 0]
            if q is not None and vals:
                unit_price_floor[unit] = float(pd.Series(vals).quantile(q))
            else:
                unit_price_floor[unit] = float("nan")

        zero_no_unitp20_products: Set[str] = set()
        price_lookup_raw: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        for (unit, product), prices in product_price_map.items():
            cleaned: List[float] = []
            saw_zero_no_p20 = False
            for p in prices:
                if p is None:
                    continue
                if p == 0:
                    ufloor = unit_price_floor.get(unit, None)
                    if ufloor is not None and not pd.isna(ufloor) and ufloor > 0:
                        cleaned.append(float(ufloor))
                    else:
                        cleaned.append(0.0)
                        saw_zero_no_p20 = True
                else:
                    cleaned.append(float(p))
            if cleaned:
                avg = sum(cleaned) / len(cleaned)
                price_lookup_raw[unit][product] = avg
                if saw_zero_no_p20 and avg == 0.0:
                    zero_no_unitp20_products.add(normalize_name(product))

        flat_price_lookup: Dict[str, float] = {
            normalize_name(product): price
            for unit_dict in price_lookup_raw.values()
            for product, price in unit_dict.items()
        }

        # ------------------ Foreground pre-checks ------------------
        # When `foreground` is True we build per-code structures, detect
        # strongly-connected components (cycles), find activities without
        # technosphere inputs, and compute reachability to background DBs.
        # These pre-checks allow the pricing stages to skip or flag
        # problematic nodes early and to support the recursive resolver.
        fg_adj: Dict[str, Set[str]] = defaultdict(set)
        
        if foreground:
            for act in techno_db:
                c = norm_code(act['code'])
                act_by_code[c] = act
                ins = list(act.technosphere())
                inputs_by_code[c] = ins
                for exc in ins:
                    if exc.input['database'] == technosphere:
                        fg_adj[c].add(norm_code(exc.input['code']))

            # Cycle detection via Tarjan SCC
            index, stack, onstack = 0, [], set()
            indices, lowlinks = {}, {}
            sccs: List[List[str]] = []

            def _strongconnect(v: str):
                nonlocal index
                indices[v] = index
                lowlinks[v] = index
                index += 1
                stack.append(v)
                onstack.add(v)
                for w in fg_adj.get(v, ()):
                    if w not in indices:
                        _strongconnect(w)
                        lowlinks[v] = min(lowlinks[v], lowlinks[w])
                    elif w in onstack:
                        lowlinks[v] = min(lowlinks[v], indices[w])
                if lowlinks[v] == indices[v]:
                    comp = []
                    while True:
                        w = stack.pop()
                        onstack.remove(w)
                        comp.append(w)
                        if w == v:
                            break
                    sccs.append(comp)

            for v in act_by_code.keys():
                if v not in indices:
                    _strongconnect(v)

            for comp in sccs:
                if len(comp) > 1:
                    for c in comp:
                        codes_in_cycles.add(c)
                        scc_membership[c] = comp[:]
                elif len(comp) == 1:
                    c = comp[0]
                    if c in fg_adj and c in fg_adj[c]:
                        codes_in_cycles.add(c)
                        scc_membership[c] = comp[:]

            codes_with_no_inputs = {c for c, ins in inputs_by_code.items() if len(ins) == 0}

            # Background reachability
            has_direct_bg = {
                c: any(exc.input['database'] in bg_dbs for exc in ins)
                for c, ins in inputs_by_code.items()
            }
            bg_reachable = dict(has_direct_bg)
            changed = True
            while changed:
                changed = False
                for c, ins in inputs_by_code.items():
                    if bg_reachable.get(c, False):
                        continue
                    fg_children = [
                        norm_code(exc.input['code'])
                        for exc in ins if exc.input['database'] == technosphere
                    ]
                    if any(bg_reachable.get(ch, False) for ch in fg_children):
                        bg_reachable[c] = True
                        changed = True

            codes_no_path_to_bg = {
                c for c in act_by_code.keys()
                if not bg_reachable.get(c, False)
            }

            # Pre-flag problematic activities
            for c in codes_in_cycles:
                a = act_by_code[c]
                not_found_records.append({
                    "code": c, "name": a["name"],
                    "reference product": a["reference product"],
                    "unit": (a["unit"] or "").lower(), "location": a["location"],
                    "normalized_key": normalize_name(a["reference product"]),
                    "reason": "foreground_cycle",
                    "in_cycle": True,
                    "cycle_members": "; ".join(scc_membership.get(c, [c])),
                    **_sb_tags(c, "foreground_cycle"),
                })
            for c in codes_with_no_inputs:
                a = act_by_code[c]
                not_found_records.append({
                    "code": c, "name": a["name"],
                    "reference product": a["reference product"],
                    "unit": (a["unit"] or "").lower(), "location": a["location"],
                    "normalized_key": normalize_name(a["reference product"]),
                    "reason": "foreground_no_inputs",
                    **_sb_tags(c, "foreground_no_inputs"),
                })
            for c in codes_no_path_to_bg:
                a = act_by_code[c]
                not_found_records.append({
                    "code": c, "name": a["name"],
                    "reference product": a["reference product"],
                    "unit": (a["unit"] or "").lower(), "location": a["location"],
                    "normalized_key": normalize_name(a["reference product"]),
                    "reason": "no_path_to_background",
                    **_sb_tags(c, "no_path_to_background"),
                })

        # Initialize `unresolved`: activities that need iterative resolution
        # (mixed foreground and background inputs).
        unresolved: Dict[str, Any] = {}

        # ------------------ Step 4: Price estimation for each activity ------------------
        # For each activity we try, in order: explicit price (FG only), direct
        # BG lookup (BG mode), one-shot BG-only sum (FG node whose inputs are
        # all BG), otherwise collect it into `unresolved` for recursive pass.
        # The loop below records detailed diagnostics for any activity that
        # cannot be priced immediately.
        for act in tqdm(techno_db, desc="🔎 Estimating activity prices"):
            code = norm_code(act['code'])
            name = act['name']
            ref_product = act['reference product']
            ref_key = normalize_name(ref_product)

            activity_meta[code] = {
                "name": name,
                "reference product": ref_product,
                "unit": (act['unit'] or "").lower(),
                "location": act['location'],
            }

            # Explicit price (foreground only)
            price_field = act.get("price", 0)
            if foreground and price_field and price_field > 0:
                estimated_prices[code] = float(price_field)
                continue

            # Background mode: direct lookup
            if not foreground:
                if ref_key in flat_price_lookup:
                    ref_price = flat_price_lookup[ref_key]
                    if ref_price is not None and ref_price > 0:
                        estimated_prices[code] = float(ref_price)
                    else:
                        not_found_records.append({
                            "code": code, "name": name, "reference product": ref_product,
                            "unit": (act['unit'] or "").lower(), "location": act['location'],
                            "normalized_key": ref_key,
                            "reason": ("zero_price_no_unit_p20" if ref_key in zero_no_unitp20_products
                                      else "background_no_spold_price"),
                        })
                else:
                    not_found_records.append({
                        "code": code, "name": name, "reference product": ref_product,
                        "unit": (act['unit'] or "").lower(), "location": act['location'],
                        "normalized_key": ref_key, "reason": "background_no_spold_price",
                    })
                continue

            # Foreground pre-check gatekeeping
            if code in codes_in_cycles or code in codes_with_no_inputs or code in codes_no_path_to_bg:
                continue

            inputs = inputs_by_code.get(code, [])
            # If all inputs are background/biosphere3 → one-shot sum
            if inputs and all(exc.input['database'] in bg_dbs or exc.input['database'] == "biosphere3" for exc in inputs):
                total, resolvable = 0.0, True
                zero_nop20_inputs: List[str] = []
                for exc in inputs:
                    if exc.input['database'] not in bg_dbs:
                        continue
                    amt = float(exc['amount'])
                    if amt <= 0:
                        continue  # ignore BG credits
                    inp_key = normalize_name(exc.input['reference product'])
                    inp_price = flat_price_lookup.get(inp_key, None)
                    if inp_price is None:
                        resolvable = False
                        break
                    if inp_price == 0.0 and inp_key in zero_no_unitp20_products:
                        zero_nop20_inputs.append(inp_key)
                    total += amt * float(inp_price)
                if resolvable:
                    if total > 0:
                        estimated_prices[code] = total
                    else:
                        reason = ("foreground_zero_inputs_no_unit_p20" if zero_nop20_inputs
                                  else "foreground_total_nonpositive")
                        not_found_records.append({
                            "code": code, "name": name, "reference product": ref_product,
                            "unit": (act['unit'] or "").lower(), "location": act['location'],
                            "normalized_key": ref_key,
                            "reason": reason,
                            **_sb_tags(code, reason),
                        })
                else:
                    not_found_records.append({
                        "code": code, "name": name, "reference product": ref_product,
                        "unit": (act['unit'] or "").lower(), "location": act['location'],
                        "normalized_key": ref_key, "reason": "foreground_unresolvable_inputs",
                        **_sb_tags(code, "foreground_unresolvable_inputs"),
                    })
                    unresolved[code] = inputs
            else:
                # Mixed foreground inputs → postpone to recursive pass
                unresolved[code] = inputs

        # ------------------ Step 5: Recursive resolution for foreground ------------------
        # Iteratively resolve activities collected in `unresolved` by
        # substituting prices of already-resolved FG children and adding BG
        # contributions. Continue until no new activities are resolved in an
        # iteration. Activities that remain unresolved are recorded in the
        # diagnostics with an appropriate reason (cycle, no-path-to-bg, or
        # recursive_unresolved).
        if foreground:
            while True:
                newly_resolved: Dict[str, float] = {}

                for code, inputs in list(unresolved.items()):
                    meta = activity_meta[code]
                    total, resolvable = 0.0, True
                    zero_nop20_inputs: List[str] = []

                    for exc in inputs:
                        if exc.input['database'] == technosphere:
                            child_code = norm_code(exc.input['code'])

                            # Ignore: no-input children
                            if child_code in codes_with_no_inputs:
                                continue
                            # Blockers: cycles or no-path-to-bg children
                            if child_code in codes_in_cycles or child_code in codes_no_path_to_bg:
                                resolvable = False
                                break

                            inp_price = estimated_prices.get(child_code)
                            if inp_price is None:
                                resolvable = False
                                break
                            amt = float(exc['amount'])
                            if amt <= 0:
                                continue
                            total += amt * float(inp_price)

                        elif exc.input['database'] in bg_dbs:
                            amt = float(exc['amount'])
                            if amt <= 0:
                                continue  # ignore BG credits
                            inp_key = normalize_name(exc.input['reference product'])
                            inp_price = flat_price_lookup.get(inp_key)
                            if inp_price is None:
                                resolvable = False
                                break
                            if inp_price == 0.0 and inp_key in zero_no_unitp20_products:
                                zero_nop20_inputs.append(inp_key)
                            total += amt * float(inp_price)

                        else:
                            # biosphere3 or others → ignore
                            continue

                    if resolvable and total > 0:
                        newly_resolved[code] = total
                    elif resolvable:
                        reason = ("foreground_zero_inputs_no_unit_p20" if zero_nop20_inputs
                                  else "foreground_total_nonpositive")
                        not_found_records.append({
                            "code": code, "name": meta["name"],
                            "reference product": meta["reference product"],
                            "unit": meta["unit"], "location": meta["location"],
                            "normalized_key": normalize_name(meta["reference product"]),
                            "reason": reason,
                            **_sb_tags(code, reason),
                        })
                        unresolved.pop(code, None)
                    # else: keep unresolved (might resolve after others)

                if not newly_resolved:
                    break

                estimated_prices.update(newly_resolved)
                unresolved = {k: v for k, v in unresolved.items() if k not in newly_resolved}

            # Anything still unresolved → cycle, no-path-to-bg, or stuck
            for code in list(unresolved.keys()):
                meta = activity_meta[code]
                reason = ("foreground_cycle" if code in codes_in_cycles else
                          "no_path_to_background" if code in codes_no_path_to_bg else
                          "foreground_recursive_unresolved")
                not_found_records.append({
                    "code": code, "name": meta["name"],
                    "reference product": meta["reference product"],
                    "unit": meta["unit"], "location": meta["location"],
                    "normalized_key": normalize_name(meta["reference product"]),
                    "reason": reason,
                    "in_cycle": code in codes_in_cycles,
                    "cycle_members": "; ".join(scc_membership.get(code, [])) if code in codes_in_cycles else None,
                    **_sb_tags(code, reason),
                })

        # ------------------ Strict guard: move any nonpositive into report ------------------
        # Any activity that ended up with a nonpositive price (None or <= 0)
        # is treated as unpriced for the final output and moved to the
        # diagnostics `not_found_records` with reason `nonpositive_after_cleanup`.
        for code, val in list(estimated_prices.items()):
            if val is None or val <= 0:
                meta = activity_meta[code]
                not_found_records.append({
                    "code": code, "name": meta["name"],
                    "reference product": meta["reference product"],
                    "unit": meta["unit"], "location": meta["location"],
                    "normalized_key": normalize_name(meta["reference product"]),
                    "reason": "nonpositive_after_cleanup",
                    **_sb_tags(code, "nonpositive_after_cleanup"),
                })
                estimated_prices.pop(code, None)

    # Build final DataFrame
    rows: List[Dict[str, Any]] = []
    for code, meta in activity_meta.items():
        price = estimated_prices.get(code)
        if price is not None and price > 0:
            rows.append({
                "ISO2": meta["location"],
                "name": meta["name"],
                "reference product": meta["reference product"],
                "unit": meta["unit"],
                "code": norm_code(code),
                "ISIC rev.4 ecoinvent": (
                    next(iter(isic_lookup.get((clean(meta["name"]), 
                         clean(meta["reference product"])), set())), None)
                    if not foreground else None
                ),
                "price": float(price),
            })
    df = pd.DataFrame(rows)

    # Apply exclusions
    if not df.empty:
        df["name"] = df["name"].astype(str)
        df = df[~df["name"].apply(is_skipped_name)].copy()
        df["code"] = df["code"].apply(norm_code).astype(str)
        df["price"] = pd.to_numeric(df["price"], errors="coerce")

    # Map to PSILCA & save JSON
    if not df.empty:
        key_iso2_map = (
            df[["name", "reference product", "ISO2"]]
            .drop_duplicates()
            .groupby(["name", "reference product"])["ISO2"]
            .agg(set)
            .to_dict()
        )
    else:
        key_iso2_map = {}

    final_output: List[Dict[str, Any]] = []
    for iso2, group in df.groupby("ISO2"):
        if iso2 in base_psilca_sets:
            base = base_psilca_sets[iso2]
            activities = []

            for _, row in group.iterrows():
                key = (row["name"], row["reference product"])
                all_iso2s = key_iso2_map.get(key, set())
                others = all_iso2s - {iso2}
                psilca = "; ".join(sorted(base - others))

                row_dict = row.drop(labels="ISO2").to_dict()
                row_dict["PSILCA_mapping"] = psilca
                activities.append(row_dict)

            final_output.append({"ISO2": iso2, "Activities": activities})
        else:
            psilca = iso2_to_psilca.get(iso2)
            if psilca:
                activities = group.drop(columns="ISO2").to_dict(orient="records")
                final_output.append({"ISO2": iso2, "PSILCA_mapping": psilca, "Activities": activities})

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Saved enriched activity data to: {output_path}")

    # Diagnostics report
    if missing_price_report_path and not_found_records:
        os.makedirs(os.path.dirname(missing_price_report_path), exist_ok=True)
        report_df = pd.DataFrame(not_found_records).drop_duplicates(subset=["code", "reason"])
        ext = os.path.splitext(missing_price_report_path)[1].lower()
        if ext in (".xlsx", ".xls"):
            report_df.to_excel(missing_price_report_path, index=False)
        elif ext == ".json":
            report_df.to_json(missing_price_report_path, orient="records", indent=2, force_ascii=False)
        else:
            report_df.to_csv(missing_price_report_path, index=False, encoding="utf-8")
        print(f"🧾 Wrote 'no price found' report with {len(report_df)} rows to: {missing_price_report_path}")

    return df


def embed_ecoinvent_to_store(
    technosphere_json: str,
    out_dir: str,
    model_name: str = "BAAI/bge-base-en-v1.5",
    batch_size: int = 256,
    target_iso2: Optional[str] = None,
    target_iso2_list: Optional[List[str]] = None,
) -> None:
    """
    Embed ecoinvent activities and save to disk.
    
    Creates embedding stores for ecoinvent activities per ISO2 region:
      - {out_dir}/ecoinvent_{ISO2}.npy
      - {out_dir}/ecoinvent_{ISO2}.manifest.json
    
    Skips already-existing stores to avoid redundant computation.
    
    Args:
        technosphere_json: Path to the activity list JSON file
        out_dir: Directory to save embedding stores
        model_name: Name of the sentence-transformer model to use
        batch_size: Batch size for embedding computation
        target_iso2: Single ISO2 code to embed (deprecated, use target_iso2_list)
        target_iso2_list: List of ISO2 codes to embed
        
    Note:
        Requires the `sentence-transformers` package.
    """
    from sentence_transformers import SentenceTransformer
    
    os.makedirs(out_dir, exist_ok=True)

    # Normalize inputs
    if target_iso2_list is None and target_iso2 is None:
        raise ValueError("Provide either target_iso2 (str) or target_iso2_list (List[str]).")
    if target_iso2_list is None:
        target_iso2_list = [target_iso2]
    else:
        if target_iso2 is not None:
            target_iso2_list = list({*target_iso2_list, target_iso2})

    def safe(name: str) -> str:
        return (str(name)
                .replace("/", "_").replace("\\", "_").replace(":", "_")
                .replace("-", "_").replace(" ", "_"))

    # Find missing ISO2 stores
    missing_iso2s: List[str] = []
    for iso2 in target_iso2_list:
        base_name = f"ecoinvent_{safe(iso2)}"
        if not store_is_complete(out_dir, base_name):
            missing_iso2s.append(iso2)

    if not missing_iso2s:
        print(f"✅ All ecoinvent per-location stores already present in {out_dir}. Nothing to do.")
        return

    # Load data & model once
    with open(technosphere_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    model = SentenceTransformer(model_name)

    # Embed missing ISO2s
    for iso2 in tqdm(missing_iso2s, desc="📦 Embedding ecoinvent activities (missing only)"):
        base_name = f"ecoinvent_{safe(iso2)}"

        rows = []
        for loc in data:
            if loc.get("ISO2") != iso2:
                continue
            for act in loc.get("Activities", []):
                code = (act.get("code") or "").strip()
                name = (act.get("name") or "").strip()
                ref = (act.get("reference product") or "").strip()
                isic_raw = act.get("ISIC rev.4 ecoinvent")
                isic = isic_raw.strip() if isinstance(isic_raw, str) else ""
                text = f"{name}; {ref}; isic rev.4: {isic}".lower().strip()
                rows.append((code, text))

        if not rows:
            continue

        df = pd.DataFrame(rows, columns=["code", "text"]).drop_duplicates("code")

        embeddings = model.encode(
            df["text"].map(prep_query_bge).tolist(),
            normalize_embeddings=True,
            convert_to_numpy=True,
            batch_size=batch_size,
            show_progress_bar=False,
        ).astype("float32")

        meta = {
            "model": model_name,
            "prompt": BGE_QUERY_PROMPT,
            "normalized": True,
            "source": "ecoinvent_activities",
            "iso2": iso2,
        }

        save_embeddings(out_dir, base_name, embeddings, df["code"], meta)
    
    print(f"✅ Finished ecoinvent combined embeddings → {out_dir}")


def clean(text: str) -> str:
    """Lowercase, strip, collapse whitespace."""
    return re.sub(r'\s+', ' ', (text or "").replace('\xa0', ' ').strip().lower())
