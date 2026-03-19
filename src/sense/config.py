"""
Configuration module for SENSE.

This module contains the SenseConfig dataclass that controls all tunable
parameters of the SENSE social-LCA pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Literal


@dataclass
class SenseConfig:
    """
    Configuration for SENSE's social-LCA pipeline.

    This controls how SENSE:
      - builds embeddings and matches ecoinvent activities to PSILCA sectors
      - converts qualitative PSILCA risk levels into numbers
      - averages social indicators across sectors and countries
      - writes intermediate results and updates Brightway biosphere flows

    Most people can use the default values without changing anything.
    Only change these if you know what you want to do differently.
    
    Examples:
        >>> cfg = SenseConfig()  # Use all defaults
        
        >>> # Only process specific countries
        >>> cfg = SenseConfig(iso2_selection=["CH", "DE"])
        
        >>> # Adjust matching tolerance
        >>> cfg = SenseConfig(similarity_delta=0.02)
    """

    # ------------------------------------------------------------------
    # 1) General behaviour
    # ------------------------------------------------------------------

    #: If True, SENSE writes intermediate JSON files for each ISO2
    #: - "matched + social exchanges" (enriched sector mapping)
    #: - "averaged" indicators per activity
    #: Turn this off (False) if you want a minimal run with no extra files.
    save_intermediate: bool = False

    #: ONLY ONE OF THOSE CAN BE ACTIVE AT A TIME:
    #: Optional list of ISO2 country codes to process.
    #: - None  -> process ALL ISO2s found in the ecoinvent activity list
    #: - ["CH", "DE"] -> only process Switzerland and Germany
    iso2_selection: Optional[List[str]] = None

    #: If set to an ISO2 code (e.g. "BR"), SENSE will skip all ISO2s that
    #: come BEFORE this one (alphabetically) and start from this ISO2.
    #: This is useful when you resume a long run that stopped halfway.
    #: - None -> start from the first ISO2 (default)
    iso2_start_at: Optional[str] = None

    # Control if working hours and social flows are written to brightway.
    enable_social_flows: bool = True
    enable_working_hours_flows: bool = True

    # ------------------------------------------------------------------
    # 2) Embedding & matching
    # ------------------------------------------------------------------

    #: Name of the sentence-embedding model used to represent texts
    #: (ecoinvent activities and PSILCA sectors) as vectors.
    #: Default "BAAI/bge-base-en-v1.5" is a good English general-purpose model.
    #: Change this ONLY if you know another model and want to re-embed everything.
    embedding_model: str = "BAAI/bge-base-en-v1.5"

    #: How many texts to embed at once on the GPU/CPU.
    #: Larger values are faster but need more RAM/VRAM.
    #: If you get out-of-memory errors, try lowering this (e.g. 64 or 32).
    embedding_batch_size: int = 256

    #: Matching tolerance (delta) for semantic similarity.
    #: For each ecoinvent activity and PSILCA country:
    #:   - SENSE finds the best sector similarity score "best"
    #:   - It keeps ALL sectors with similarity >= best - similarity_delta
    #: A smaller value -> fewer sectors kept (stricter)
    #: A larger value  -> more sectors kept (looser)
    #: Default 0.015 is a reasonable balance.
    similarity_delta: float = 0.016

    # ------------------------------------------------------------------
    # 3) Ecoinvent pricing heuristics
    # ------------------------------------------------------------------

    #: Terms that cause an activity to be excluded from the *output table*
    #: if they appear as a PREFIX of the activity name (case-insensitive).
    #: The activity is still used internally for price estimation.
    exclude_terms_starting_with: List[str] = field(
        default_factory=lambda: [
            "market for",
            "market group for",
            "generic market for",
        ]
    )

    #: Terms that cause an activity to be excluded from the *output table*
    #: if they appear ANYWHERE in the activity name (case-insensitive).
    #: Again, activities are still used internally for price estimation.
    exclude_terms: List[str] = field(
        default_factory=lambda: [
            "import from",
            "production mix",
            "consumption mix",
            "cut-off",
            "cut off",
            "cutoff",
            "technology mix",
            "consumption market for",
        ]
    )

    #: Percentile used as a "floor" for unit-level prices, per unit.
    #: SENSE gathers all positive prices for a given unit (e.g. "kg", "kWh"):
    #:   - If price_floor_percentile > 0, it uses that percentile (e.g. 20)
    #:     as a minimum value to replace zero product prices (treatment activities).
    #:   - If price_floor_percentile == 0.0, this behaviour is disabled
    #:     and zeros are left as zeros (treatments dont get risk assigned).
    price_floor_percentile: float = 20.0

    # ------------------------------------------------------------------
    # 4) Averaging & risk mapping
    # ------------------------------------------------------------------

    #: Values smaller than this (but > 0) are counted as "tiny" in diagnostics.
    #: This does NOT affect calculations, only the debugging output
    #: that tells you how many very small indicator values you have.
    tiny_threshold: float = 1e-9

    #: Number of decimal digits to round final indicator values to.
    #: - None   -> no rounding (full precision)
    #: - an int -> e.g. 12 means round(value, 12)
    #: Be careful: rounding too aggressively can turn very small positives into 0.0
    #: and may confuse "zero vs non-zero" diagnostics.
    rounding_digits: Optional[int] = None

    #: Mapping from qualitative PSILCA risk levels to numeric weights.
    #: These weights multiply the PSILCA hours before averaging, so they
    #: directly control how strongly each level contributes to "medium risk hours".
    #: Defaults follow a logarithmic scale (each level ≈ 10x the previous).
    #: You can change these if you want a different interpretation of "high" vs "low".
    level_weights: Dict[str, float] = field(
        default_factory=lambda: {
            # No risk
            "no": 0.001,
            # Very small risk
            "very low": 0.01,
            # Small risk
            "low": 0.1,
            # Baseline level used as 1.0
            "medium": 1.0,
            # High risk
            "high": 10.0,
            # Very high risk
            "very high": 100.0,
            # "no data" means unknown; default is to treat as low
            "no data": 0.1,
        }
    )

    #: Maximum number of per-activity examples to keep in diagnostics
    #: (e.g. activities with missing hours or missing price).
    #: This only affects how much detail you see in summary/output logs.
    max_activity_examples: int = 12

    #: Maximum number of global examples to keep in diagnostics
    #: (e.g. the total number of unique indicators or causes shown).
    max_global_examples: int = 20

    #: Mean used to aggregate social indicator values across matched sectors
    #: and across PSILCA countries. (Working_hours remains arithmetic.)
    #: Options: "geometric" (default), "arithmetic"
    social_flow_mean: Literal["geometric", "arithmetic"] = "geometric"

    # ------------------------------------------------------------------
    # 5) Biosphere integration (Brightway)
    # ------------------------------------------------------------------

    #: If True, SENSE assumes that all required social biosphere flows
    #: already exist in the Brightway biosphere database and will NOT
    #: create new flows (it will only attach exchanges where possible).
    #: If False (default), SENSE creates any missing flows automatically.
    precreated_flows: bool = False

    #: Minimum absolute difference required to update an existing biosphere
    #: exchange amount in Brightway.
    #: - 0.0   -> always update if new_amount != old_amount
    #: - 1e-9  -> only update if |new - old| > 1e-9
    #: Can be useful to reduce tiny numerical changes in databases.
    update_tol: float = 0.0

    # ------------------------------------------------------------------
    # 6) Reporting / match-quality diagnostics
    # ------------------------------------------------------------------

    #: Lower bounds for semantic match quality buckets in report_matches.
    #: Example with default [0.8, 0.7, 0.6, 0.5]:
    #:   >= 0.80        -> "very high" match
    #:   0.70 - 0.79    -> "high" match
    #:   0.60 - 0.69    -> "medium" match
    #:   0.50 - 0.59    -> "low" match
    #:   < 0.50         -> "very low" match
    #: Only affects how results are grouped in reports, not the matching itself.
    match_quality_buckets: List[float] = field(
        default_factory=lambda: [0.8, 0.7, 0.6, 0.5]
    )
