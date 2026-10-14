"""Match ecoinvent activities to PSILCA sectors and attach assessments."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sense.data.embeddings import load_embeddings
from sense.utils.helpers import norm_code, safe_name


def _retained_matches(sectors, activities, columns, threshold, *, batch_size=1024):
    """Compute bounded similarity blocks, retaining the original sector-major order."""
    if sectors.ndim != 2 or not len(sectors) or not np.isfinite(sectors).all():
        raise ValueError("Sector embeddings must be non-empty and finite")
    retained: list[tuple[int, int, float, float]] = []
    for offset in range(0, len(columns), batch_size):
        block = activities[columns[offset : offset + batch_size]]
        if not np.isfinite(block).all():
            raise ValueError("Activity embeddings must be finite")
        similarities = sectors @ block.T
        best = similarities.max(axis=0)
        sector_indices, activity_indices = (similarities >= (best - threshold)[None, :]).nonzero()
        retained.extend(
            (int(s), int(a) + offset, float(similarities[s, a]), float(best[a]))
            for s, a in zip(sector_indices, activity_indices, strict=True)
        )
    return sorted(retained, key=lambda row: (row[0], row[1]))


_PSILCA_LOOKUP_COLUMNS = [
    "process_index",
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
]


class PSILCALookupStore:
    """Load only the PSILCA countries required by the current region."""

    def __init__(
        self,
        cache_path: str | Path,
        max_cached_countries: int = 8,
        process_overrides: dict[str, str] | None = None,
    ):
        self.path = Path(cache_path)
        self.process_overrides = process_overrides or {}
        self.max_cached_countries = max(1, int(max_cached_countries))
        self._countries: OrderedDict[str, dict[tuple, list[dict]]] = OrderedDict()

    def for_countries(self, countries: list[str]) -> dict[tuple, list[dict]]:
        """Return (country, sector, source file) entries for the requested countries."""

        result: dict[tuple, list[dict]] = {}
        import pyarrow.parquet as pq

        for country in dict.fromkeys(countries):
            normalized = str(country).strip().upper()
            lookup = self._countries.pop(normalized, None)
            if lookup is None:
                table = pq.read_table(
                    self.path,
                    columns=_PSILCA_LOOKUP_COLUMNS,
                    filters=[("location", "=", normalized)],
                )
                lookup = _lookup_from_arrow_batches(table.to_batches(), self.process_overrides)
            self._countries[normalized] = lookup
            result.update(lookup)
            while len(self._countries) > self.max_cached_countries:
                self._countries.popitem(last=False)
        return result


def match_from_stores(
    ecoinvent_store_dir: str,
    psilca_store_dir: str,
    target_iso2: str,
    *,
    activity_entry: dict[str, Any],
    threshold: float = 0.028,
    minimum_similarity: float | None = 0.5,
) -> pd.DataFrame:
    """
    Match ecoinvent activities to PSILCA sectors using precomputed embeddings.

    Uses cosine similarity between sentence embeddings to find the best matching
    PSILCA sectors for each ecoinvent activity. Supports per-activity allow-lists
    for complex regions like RoW (Rest of World).

    Args:
        ecoinvent_store_dir: Directory containing ecoinvent embeddings
        psilca_store_dir: Directory containing PSILCA embeddings
        target_iso2: ISO2 code to match (e.g., 'BR', 'RoW')
        threshold: Cosine similarity tolerance for multiple matches
        minimum_similarity: Absolute score below which the best available match
            is flagged as low confidence. Low-confidence matches are retained.
            Set to None to disable this diagnostic flag.
        activity_entry: Preloaded activity-list entry for ``target_iso2``.

    Returns:
        DataFrame with columns:
        ["ISO2", "Activity code", "Activity name", "Reference product", "ISIC",
         "Unit", "Price", "PSILCA_country", "Matched_sector", "Semantic_score"]

    """
    # Load metadata & per-activity PSILCA allow-lists
    act_allowed: dict[str, set] = {}
    meta_df = pd.DataFrame(
        columns=[
            "Activity code",
            "Activity name",
            "Reference product",
            "ISIC",
            "Unit",
            "Price",
            "ISO2",
        ]
    )
    entry = activity_entry
    activities = entry.get("Activities", [])
    default_mapping = entry.get("PSILCA_mapping", "") if target_iso2 not in ("RoW", "RoE") else ""

    meta_rows = []
    for activity in activities:
        raw_code = activity.get("code")
        code = norm_code(raw_code)
        if not code:
            continue
        isic_raw = activity.get("ISIC rev.4 ecoinvent")
        mapping = (
            activity.get("PSILCA_mapping", "") if target_iso2 in ("RoW", "RoE") else default_mapping
        )
        act_allowed[code] = {country.strip() for country in mapping.split(";") if country.strip()}
        meta_rows.append(
            {
                "Activity code": code,
                "Activity name": (activity.get("name") or "").strip(),
                "Reference product": (activity.get("reference product") or "").strip(),
                "ISIC": isic_raw.strip() if isinstance(isic_raw, str) else "",
                "Unit": (activity.get("unit") or "").strip(),
                "Price": activity.get("price"),
                "ISO2": target_iso2,
            }
        )

    meta_df = pd.DataFrame(meta_rows)
    countries = sorted({country for values in act_allowed.values() for country in values})

    # Load ecoinvent embeddings for this ISO2
    base_name = f"ecoinvent_{safe_name(target_iso2)}"
    act_emb, act_ids, _ = load_embeddings(ecoinvent_store_dir, base_name)

    # Build normalized-code -> column-index map
    act_ids_list = act_ids.tolist()
    act_ids_norm = [norm_code(c) for c in act_ids_list]
    if len(set(act_ids_norm)) != len(act_ids_norm):
        raise ValueError("Duplicate activity IDs in the embedding store")
    code_to_col_norm = {c_norm: i for i, c_norm in enumerate(act_ids_norm)}

    # Choose activity codes to consider
    if not meta_df.empty:
        missing = set(meta_df["Activity code"]) - set(code_to_col_norm)
        if missing:
            raise ValueError(
                f"Activity embeddings are missing {len(missing)} requested codes; rebuild the cache"
            )
        wanted_codes_norm = [c for c in meta_df["Activity code"].tolist() if c in code_to_col_norm]
    else:
        wanted_codes_norm = act_ids_norm

    if not wanted_codes_norm:
        return pd.DataFrame(
            columns=[
                "ISO2",
                "Activity code",
                "Activity name",
                "Reference product",
                "ISIC",
                "Unit",
                "Price",
                "PSILCA_country",
                "Matched_sector",
                "Semantic_score",
                "Below minimum similarity",
            ]
        )

    col_idx = np.array([code_to_col_norm[c] for c in wanted_codes_norm], dtype=int)
    inv_codes_norm = np.array(wanted_codes_norm, dtype=object)

    # Match against each mapped PSILCA country (ISO2).
    rows: list[dict[str, Any]] = []

    for country in countries:
        try:
            sec_emb, sec_ids, _ = load_embeddings(psilca_store_dir, f"psilca_{country}")
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Missing PSILCA embedding store for mapped country {country!r}"
            ) from exc

        if sec_emb.shape[1] != act_emb.shape[1]:
            raise ValueError(
                "Embedding dimension mismatch for mapped country "
                f"{country!r}: PSILCA={sec_emb.shape[1]}, "
                f"ecoinvent={act_emb.shape[1]}"
            )

        # Activity-level filtering (for RoW/RoE allow-lists)
        if entry is not None:
            mask_cols = np.array(
                [country in act_allowed.get(code, set()) for code in inv_codes_norm], dtype=bool
            )
            if not mask_cols.any():
                continue
            sub_cols = col_idx[mask_cols]
            sub_codes = inv_codes_norm[mask_cols]
        else:
            sub_cols = col_idx
            sub_codes = inv_codes_norm

        if sub_cols.size == 0:
            continue

        sec_ids_np = sec_ids.to_numpy()
        for s, a, score, best_score in _retained_matches(sec_emb, act_emb, sub_cols, threshold):
            try:
                _, sector_name = sec_ids_np[s].split("::", 1)
            except ValueError:
                sector_name = str(sec_ids_np[s])

            rows.append(
                {
                    "ISO2": target_iso2,
                    "Activity code": sub_codes[a],
                    "PSILCA_country": country,
                    "Matched_sector": sector_name,
                    "Semantic_score": round(score, 4),
                    "Below minimum similarity": bool(
                        minimum_similarity is not None and best_score < minimum_similarity
                    ),
                }
            )

    df_map = pd.DataFrame(rows)

    if df_map.empty:
        return pd.DataFrame(
            columns=[
                "ISO2",
                "Activity code",
                "Activity name",
                "Reference product",
                "ISIC",
                "Unit",
                "Price",
                "PSILCA_country",
                "Matched_sector",
                "Semantic_score",
                "Below minimum similarity",
            ]
        )

    # Merge metadata
    df_map["Activity code"] = df_map["Activity code"].astype(str)

    if meta_df.empty:
        df_map["Activity name"] = np.nan
        df_map["Reference product"] = np.nan
        df_map["ISIC"] = np.nan
        df_map["Unit"] = np.nan
        df_map["Price"] = np.nan
        df_out = df_map[
            [
                "ISO2",
                "Activity code",
                "Activity name",
                "Reference product",
                "ISIC",
                "Unit",
                "Price",
                "PSILCA_country",
                "Matched_sector",
                "Semantic_score",
                "Below minimum similarity",
            ]
        ].copy()
        return df_out

    meta_df = meta_df.copy()
    meta_df["Activity code"] = meta_df["Activity code"].astype(str)
    if "Price" in meta_df.columns:
        meta_df["Price"] = pd.to_numeric(meta_df["Price"], errors="coerce")

    df_out = df_map.merge(meta_df, on=["ISO2", "Activity code"], how="left", validate="m:1")

    df_out = df_out[
        [
            "ISO2",
            "Activity code",
            "Activity name",
            "Reference product",
            "ISIC",
            "Unit",
            "Price",
            "PSILCA_country",
            "Matched_sector",
            "Semantic_score",
            "Below minimum similarity",
        ]
    ].copy()

    return df_out


def _lookup_from_arrow_batches(
    batches: Any,
    process_overrides: dict[str, str] | None = None,
) -> dict[tuple, list[dict]]:
    """Preserve each source process, including processes sharing a sector name."""

    def normalize(value: Any) -> str:
        return str(value or "").strip().lower()

    lookup: dict[tuple, list[dict]] = {}
    current_process: int | None = None
    current_key: tuple[str, str] | None = None
    current_exchanges: list[dict[str, Any]] = []
    current_source = ""
    overrides = {k.strip().lower(): v for k, v in (process_overrides or {}).items()}

    def flush() -> None:
        if current_key is not None:
            requested = overrides.get(" | ".join(current_key))
            if requested is not None and current_source != requested:
                return
            process_key = (*current_key, current_source)
            if process_key in lookup:
                raise ValueError(f"Repeated PSILCA source process {process_key!r}")
            lookup[process_key] = list(current_exchanges)

    for batch in batches:
        values = batch.to_pydict()
        for index, process_index in enumerate(values["process_index"]):
            if current_process != process_index:
                flush()
                current_process = int(process_index)
                current_source = str(values["source_file"][index])
                current_key = (
                    normalize(values["location"][index]),
                    normalize(values["reference_name"][index]),
                )
                current_exchanges = []
            current_exchanges.append(
                {
                    "indicator": values["indicator"][index] or "",
                    "type": values["assessment_type"][index] or "risk",
                    "risk level": values["risk_level"][index] or "no data",
                    "meanValue": float(values["mean_value"][index]),
                    "unit": values["unit"][index] or "",
                    "category": values["category"][index] or "",
                    "subcategory": values["subcategory"][index] or "",
                }
            )
    flush()
    return lookup


def add_social_exchanges(
    sector_mapping_df: pd.DataFrame,
    psilca_lookup: dict[tuple, list[dict]],
) -> pd.DataFrame:
    """
    Add PSILCA social exchanges (risks and opportunities) to mapped ecoinvent activities.

    Args:
        sector_mapping_df: DataFrame with ecoinvent-PSILCA mappings
        psilca_lookup: Process lookup from ``PSILCALookupStore``. Legacy
            (country, sector) keys are also accepted for direct API callers.

    Returns:
        One row per matched source process, with PSILCA_process,
        PSILCA_Exchanges and type/level summaries. Identical labels share a
        semantic score but retain separate assessments and aggregation weight.
    """

    def _norm(s):
        return (s or "").strip().lower()

    # Text embeddings are deliberately shared by identical labels. Expand a
    # qualifying label into its distinct source processes before aggregation;
    # this is equivalent to matching every process separately, without encoding
    # the same text twice. Do not flatten assessments or deduplicate processes
    # with identical assessments: each qualifying source is one candidate.
    candidates: dict[tuple, list[tuple[str | None, list[dict]]]] = {}
    for key, exchanges in psilca_lookup.items():
        country, sector = key[:2]
        source = key[2] if len(key) == 3 else None
        candidates.setdefault((_norm(country), _norm(sector)), []).append((source, exchanges))

    positions = []
    processes = []
    exchanges_list = []
    missing_keys = []
    for position, (_, row) in enumerate(sector_mapping_df.iterrows()):
        key = (_norm(row.get("PSILCA_country")), _norm(row.get("Matched_sector")))
        matches = candidates.get(key, [])
        if not matches:
            missing_keys.append(key)
        for source, exchanges in matches:
            if not exchanges:
                missing_keys.append(key)
            positions.append(position)
            processes.append(source)
            exchanges_list.append(exchanges)

    sector_mapping_df = sector_mapping_df.iloc[positions].copy().reset_index(drop=True)
    sector_mapping_df["PSILCA_process"] = processes
    sector_mapping_df["PSILCA_Exchanges"] = exchanges_list

    sector_mapping_df["risk/opportunity types"] = [
        "; ".join(sorted({ex["type"] for ex in exchs})) if exchs else "" for exchs in exchanges_list
    ]
    sector_mapping_df["risk level"] = [
        "; ".join(sorted({ex["risk level"] for ex in exchs})) if exchs else ""
        for exchs in exchanges_list
    ]

    # Diagnostics
    if missing_keys:
        uniq_missing = sorted(set(missing_keys))
        examples = ", ".join(repr(key) for key in uniq_missing[:10])
        raise ValueError(
            f"PSILCA exchange lookup is missing {len(uniq_missing)} matched "
            f"country-sector pair(s): {examples}"
        )

    return sector_mapping_df
