"""Clone and localize foreground recipes with cached supplier matching."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import bw2data as bd
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from tqdm.auto import tqdm

from .inventory_localization import BiosphereLocalizer, clone_activity, copy_exchange
from .inventory_operations import _require_new_database
from .study_checks import technology_candidates

SOCIAL_COMPARTMENTS = {
    "Children",
    "Consumer",
    "Local Community",
    "Society",
    "Workers",
    "Value Chain Actors",
}


_DATABASE_CANDIDATE_INDEXES: dict[str, dict[tuple[str, str, str], list[Any]]] = {}


def build_candidate_index(
    background_db_names: Sequence[str],
) -> dict[tuple[str, str, str], list[Any]]:
    """Pool candidates by (reference product, unit, location), caching each database.

    Preserve database priority and sort activity keys for deterministic tie handling.
    The foreground stage clears this cache before each build.
    """
    pooled: dict[tuple[str, str, str], list[Any]] = {}
    for dbn in background_db_names:
        if dbn not in _DATABASE_CANDIDATE_INDEXES:
            database_index: dict[tuple[str, str, str], list[Any]] = {}
            for act in sorted(bd.Database(dbn), key=lambda activity: activity.key):
                key = (act.get("reference product"), act.get("unit"), act.get("location"))
                database_index.setdefault(key, []).append(act)
            _DATABASE_CANDIDATE_INDEXES[dbn] = database_index

        for key, activities in _DATABASE_CANDIDATE_INDEXES[dbn].items():
            pooled.setdefault(key, []).extend(activities)
    return pooled


class SemanticMatcher:
    """Cache name embeddings for cosine-similarity supplier matching.

    ``best`` returns the highest-scoring candidate name and score. Candidate order
    breaks exact ties; the caller applies its acceptance threshold.
    """

    def __init__(self, model: SentenceTransformer) -> None:
        """Use the supplied encoder and initialize an empty in-memory name cache."""
        self.model = model
        self.embeddings: dict[str, np.ndarray] = {}

    def best(
        self, reference_name: str, candidate_names: Sequence[str]
    ) -> tuple[str | None, float | None]:
        """Return the first highest-scoring name and cosine score, or ``(None, None)``.

        Encode uncached names with the instance model. An empty candidate sequence
        returns no match; this method does not apply a similarity threshold.
        """
        if not candidate_names:
            return None, None

        ordered_names = [reference_name, *candidate_names]
        missing_names = list(
            dict.fromkeys(name for name in ordered_names if name not in self.embeddings)
        )
        if missing_names:
            encoded = self.model.encode(
                missing_names,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            self.embeddings.update(zip(missing_names, encoded, strict=True))

        candidate_embeddings = np.stack([self.embeddings[name] for name in candidate_names])
        scores = util.cos_sim(self.embeddings[reference_name], candidate_embeddings)[0]
        best_index = int(scores.argmax().item())
        return candidate_names[best_index], float(scores[best_index].item())


def localize_foreground_database(
    original_fg_db_name: str,
    output_fg_db_name: str,
    background_db_names: Sequence[str],
    target_locations: Sequence[str],
    similarity_threshold: float,
    matcher: SemanticMatcher,
    mapping_log_xlsx: Path,
    foreground_sources: dict[str, str] | None = None,
) -> None:
    """Clone each foreground activity per location and relink its inputs.

    Redirect internal foreground links first. Otherwise retain already-local inputs
    or choose a same-product/unit/location background candidate whose name similarity
    meets the threshold. Unmatched inputs retain their original exchanges.
    Write the output database and mapping workbook; Existing output databases are rejected. Reuse the supplied matcher across localization steps.
    """
    _require_new_database(output_fg_db_name)

    original_db = bd.Database(original_fg_db_name)
    original_activities = list(original_db)
    original_exchanges = {act.key: list(act.exchanges()) for act in original_activities}
    input_activities: dict[tuple[str, str], Any] = {}

    # pooled candidates
    candidate_index = build_candidate_index(background_db_names)

    # precompute expected localized keys for FG-first redirection
    expected_localized_keys = {
        (output_fg_db_name, act.key[1] + f"_{loc}")
        for act in original_activities
        for loc in target_locations
    }

    log_entries: list[dict] = []
    new_db_data: dict[tuple[str, str], dict] = {}
    biosphere_localizer = BiosphereLocalizer()

    for loc in target_locations:
        for act in tqdm(original_activities, desc=f"Localizing '{original_fg_db_name}' for {loc}"):
            localized_key = (output_fg_db_name, act.key[1] + f"_{loc}")

            new_act_data = clone_activity(act, localized_key, loc, original_exchanges[act.key])

            for exc in original_exchanges[act.key]:
                if exc["type"] == "production":
                    continue

                # Keep non-technosphere exchanges
                if exc["type"] != "technosphere":
                    new_act_data["exchanges"].append(biosphere_localizer.exchange(exc, loc))
                    continue

                input_key = tuple(exc["input"])
                if input_key not in input_activities:
                    input_activities[input_key] = bd.get_activity(input_key)
                input_act = input_activities[input_key]

                # FG-first: redirect to localized FG copy if the input is in the same original FG DB
                if input_act.key[0] == original_fg_db_name:
                    localized_fg_key_in = (output_fg_db_name, input_act.key[1] + f"_{loc}")
                    if localized_fg_key_in in expected_localized_keys:
                        new_act_data["exchanges"].append(copy_exchange(exc, localized_fg_key_in))
                        log_entries.append(
                            {
                                "target_location": loc,
                                "activity": act.get("name"),
                                "product": input_act.get("name"),
                                "original_location": input_act.get("location"),
                                "new_location": loc,
                                "matched_name": input_act.get("name"),
                                "similarity": 1.0,
                                "status": "redirected to localized FG (FG-first)",
                            }
                        )
                        continue

                # Extra foreground dependencies must stay in the same scenario.
                if foreground_sources is not None and input_act.key[0] in foreground_sources:
                    target = (foreground_sources[input_act.key[0]], input_act.key[1] + f"_{loc}")
                    bd.get_activity(target)  # fail on an absent dependency
                    new_act_data["exchanges"].append(copy_exchange(exc, target))
                    log_entries.append(
                        {
                            "target_location": loc,
                            "activity": act.get("name"),
                            "product": input_act.get("name"),
                            "matched_name": input_act.get("name"),
                            "status": "redirected to scenario foreground",
                        }
                    )
                    continue
                # Already localized to target location
                if input_act.get("location") == loc:
                    new_act_data["exchanges"].append(biosphere_localizer.exchange(exc, loc))
                    log_entries.append(
                        {
                            "target_location": loc,
                            "activity": act.get("name"),
                            "product": input_act.get("name"),
                            "original_location": input_act.get("location"),
                            "new_location": input_act.get("location"),
                            "matched_name": input_act.get("name"),
                            "similarity": 1.0,
                            "status": "unchanged (already localized)",
                        }
                    )
                    continue

                # Background candidates by (ref product, unit, target location)
                k = (input_act.get("reference product"), input_act.get("unit"), loc)
                candidates = technology_candidates(
                    input_act.get("name", ""), candidate_index.get(k, [])
                )

                if not candidates:
                    new_act_data["exchanges"].append(biosphere_localizer.exchange(exc, loc))
                    log_entries.append(
                        {
                            "target_location": loc,
                            "activity": act.get("name"),
                            "product": input_act.get("name"),
                            "original_location": input_act.get("location"),
                            "new_location": input_act.get("location"),
                            "matched_name": None,
                            "similarity": None,
                            "status": "unchanged (no BG candidates)",
                        }
                    )
                    continue

                candidate_names = [c.get("name") for c in candidates]
                best_name, score = matcher.best(input_act.get("name"), candidate_names)

                if best_name is None or score is None or score < similarity_threshold:
                    new_act_data["exchanges"].append(biosphere_localizer.exchange(exc, loc))
                    log_entries.append(
                        {
                            "target_location": loc,
                            "activity": act.get("name"),
                            "product": input_act.get("name"),
                            "original_location": input_act.get("location"),
                            "new_location": input_act.get("location"),
                            "matched_name": best_name,
                            "similarity": round(score, 4) if score is not None else None,
                            "status": f"unchanged (low BG similarity: {round(score, 4) if score is not None else 'n/a'})",
                        }
                    )
                    continue

                matched = next(c for c in candidates if c.get("name") == best_name)
                new_act_data["exchanges"].append(copy_exchange(exc, matched.key))
                log_entries.append(
                    {
                        "target_location": loc,
                        "activity": act.get("name"),
                        "product": input_act.get("name"),
                        "original_location": input_act.get("location"),
                        "new_location": matched.get("location"),
                        "matched_name": best_name,
                        "similarity": round(score, 4),
                        "status": "changed (BG match)",
                    }
                )

            new_db_data[localized_key] = new_act_data

    bd.Database(output_fg_db_name).write(new_db_data)
    bd.Database(output_fg_db_name).process()
    print(f"✅ Localized DB written to: {output_fg_db_name}")

    pd.DataFrame(log_entries).to_excel(mapping_log_xlsx, index=False)
    print(f"📝 Mapping log saved to: {mapping_log_xlsx}")


def is_target_activity(act: Any, targets: set[tuple]) -> bool:
    """Match an activity against exact (name, reference product, location) targets."""
    return (act.get("name"), act.get("reference product"), act.get("location")) in targets


def is_social_biosphere_exchange(exc: Any) -> bool:
    """Identify social or working-hours flows by SENSE metadata or category fallback."""
    if exc.get("type") != "biosphere":
        return False

    flow = bd.get_activity(exc.input)
    if flow.get("sense_kind") in {"social", "working_hours", "working_hours_isic"}:
        return True

    # Compatibility fallback for exchanges written by the public v1.0.0 release.
    cats = flow.get("categories") or ()

    if isinstance(cats, str):
        cats = (cats,)
    elif not isinstance(cats, (list, tuple)):
        cats = (str(cats),)

    for raw_cat in cats:
        cat = str(raw_cat).strip()
        if cat in SOCIAL_COMPARTMENTS:
            return True
        for base in SOCIAL_COMPARTMENTS:
            if cat.startswith(base + " -"):
                return True
    return False


def localize_ecoinvent_subset(
    source_db_name: str,
    output_db_name: str,
    background_db_names: Sequence[str],
    target_locations: Sequence[str],
    targets: set[tuple],
    matcher: SemanticMatcher,
    similarity_threshold: float,
    log_xlsx: Path,
) -> None:
    """Clone selected source activities per location, removing social biosphere flows.

    Match background inputs by product, unit, location and name similarity. On a
    missing or low-confidence match, redirect inputs to another localized subset
    activity when possible; otherwise retain the original exchange. Write the
    output database and mapping workbook, requiring a new output name.
    """
    _require_new_database(output_db_name)

    source_db = bd.Database(source_db_name)

    # select target acts once (preserve database iteration order)
    target_acts = [act for act in source_db if is_target_activity(act, targets)]
    target_exchanges = {act.key: list(act.exchanges()) for act in target_acts}
    input_activities: dict[tuple[str, str], Any] = {}
    print(f"Found {len(target_acts)} target activities to localize from '{source_db_name}'.")

    # pooled candidates
    candidate_index = build_candidate_index(background_db_names)

    expected_localized_keys = {
        (output_db_name, act.key[1] + f"_{loc}") for act in target_acts for loc in target_locations
    }

    log_rows: list[dict] = []
    new_db_data: dict[tuple[str, str], dict] = {}
    biosphere_localizer = BiosphereLocalizer()

    for loc in target_locations:
        for act in tqdm(target_acts, desc=f"Localizing subset for {loc}"):
            localized_key = (output_db_name, act.key[1] + f"_{loc}")

            new_act_data = clone_activity(act, localized_key, loc, target_exchanges[act.key])

            for exc in target_exchanges[act.key]:
                if exc.get("type") == "production":
                    continue

                # Remove only social biosphere flows; keep all other non-technosphere exchanges
                if exc.get("type") != "technosphere":
                    if is_social_biosphere_exchange(exc):
                        flow = bd.get_activity(exc.input)
                        log_rows.append(
                            {
                                "target_location": loc,
                                "activity": act.get("name"),
                                "product": flow.get("name"),
                                "original_location": flow.get("location"),
                                "new_location": None,
                                "matched_name": flow.get("name"),
                                "similarity": None,
                                "status": "removed social biosphere flow",
                            }
                        )
                        continue
                    new_act_data["exchanges"].append(biosphere_localizer.exchange(exc, loc))
                    continue

                # technosphere
                input_key = tuple(exc["input"])
                if input_key not in input_activities:
                    input_activities[input_key] = bd.get_activity(input_key)
                input_act = input_activities[input_key]

                # already in target loc
                if input_act.get("location") == loc:
                    new_act_data["exchanges"].append(biosphere_localizer.exchange(exc, loc))
                    log_rows.append(
                        {
                            "target_location": loc,
                            "activity": act.get("name"),
                            "product": input_act.get("name"),
                            "original_location": input_act.get("location"),
                            "new_location": input_act.get("location"),
                            "matched_name": input_act.get("name"),
                            "similarity": 1.0,
                            "status": "unchanged (already localized)",
                        }
                    )
                    continue

                # BG candidates
                k = (input_act.get("reference product"), input_act.get("unit"), loc)
                candidates = technology_candidates(
                    input_act.get("name", ""), candidate_index.get(k, [])
                )

                if not candidates:
                    # subset fallback redirect
                    if input_act.key[0] == source_db_name and is_target_activity(
                        input_act, targets
                    ):
                        redirected = (output_db_name, input_act.key[1] + f"_{loc}")
                        if redirected in expected_localized_keys:
                            new_act_data["exchanges"].append(copy_exchange(exc, redirected))
                            log_rows.append(
                                {
                                    "target_location": loc,
                                    "activity": act.get("name"),
                                    "product": input_act.get("name"),
                                    "original_location": input_act.get("location"),
                                    "new_location": loc,
                                    "matched_name": input_act.get("name"),
                                    "similarity": 1.0,
                                    "status": "redirected to localized subset",
                                }
                            )
                            continue

                    # else keep original
                    new_act_data["exchanges"].append(biosphere_localizer.exchange(exc, loc))
                    log_rows.append(
                        {
                            "target_location": loc,
                            "activity": act.get("name"),
                            "product": input_act.get("name"),
                            "original_location": input_act.get("location"),
                            "new_location": input_act.get("location"),
                            "matched_name": None,
                            "similarity": None,
                            "status": "unchanged (no candidates)",
                        }
                    )
                    continue

                candidate_names = [c.get("name") for c in candidates]
                best_name, score = matcher.best(input_act.get("name"), candidate_names)

                if best_name is None or score is None or score < similarity_threshold:
                    # fallback redirect if subset
                    if input_act.key[0] == source_db_name and is_target_activity(
                        input_act, targets
                    ):
                        redirected = (output_db_name, input_act.key[1] + f"_{loc}")
                        if redirected in expected_localized_keys:
                            new_act_data["exchanges"].append(copy_exchange(exc, redirected))
                            log_rows.append(
                                {
                                    "target_location": loc,
                                    "activity": act.get("name"),
                                    "product": input_act.get("name"),
                                    "original_location": input_act.get("location"),
                                    "new_location": loc,
                                    "matched_name": input_act.get("name"),
                                    "similarity": 1.0,
                                    "status": "redirected to localized subset (low similarity fallback)",
                                }
                            )
                            continue

                    new_act_data["exchanges"].append(biosphere_localizer.exchange(exc, loc))
                    log_rows.append(
                        {
                            "target_location": loc,
                            "activity": act.get("name"),
                            "product": input_act.get("name"),
                            "original_location": input_act.get("location"),
                            "new_location": input_act.get("location"),
                            "matched_name": best_name,
                            "similarity": round(score, 4) if score is not None else None,
                            "status": f"unchanged (low similarity: {round(score, 4) if score is not None else 'n/a'})",
                        }
                    )
                    continue

                matched = next(c for c in candidates if c.get("name") == best_name)
                new_act_data["exchanges"].append(copy_exchange(exc, matched.key))
                log_rows.append(
                    {
                        "target_location": loc,
                        "activity": act.get("name"),
                        "product": input_act.get("name"),
                        "original_location": input_act.get("location"),
                        "new_location": matched.get("location"),
                        "matched_name": best_name,
                        "similarity": round(score, 4),
                        "status": "changed",
                    }
                )

            new_db_data[localized_key] = new_act_data

    bd.Database(output_db_name).write(new_db_data)
    bd.Database(output_db_name).process()
    print(f"✅ Localized subset DB written to: {output_db_name}")

    pd.DataFrame(log_rows).to_excel(log_xlsx, index=False)
    print(f"📝 Log saved to: {log_xlsx}")
