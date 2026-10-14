"""Validated configuration for the SENSE pipeline."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class SenseConfig:
    """Control matching, aggregation, pricing, diagnostics, and Brightway output.

    Defaults define the standard SENSE workflow. Invalid or conflicting options
    raise ``ValueError`` during construction, before a run can modify Brightway.
    """

    # ------------------------------------------------------------------
    # 1) General behaviour
    # ------------------------------------------------------------------

    #: Write enriched matches and averaged activity records for each region.
    save_intermediate: bool = False

    #: Rebuild parsed-data, activity, and embedding caches even when their
    #: validation manifests match. Use after changing inputs outside normal writes.
    rebuild_cache: bool = False

    #: Continue with later regions after one ISO2 fails. The default is strict:
    #: library and CLI runs stop immediately instead of producing partial output.
    continue_on_iso2_error: bool = False

    #: Optional ISO2 region allow-list. Mutually exclusive with ``iso2_start_at``.
    iso2_selection: list[str] | None = None

    #: Resume at this ISO2 region in activity-list order. Mutually exclusive
    #: with ``iso2_selection``.
    iso2_start_at: str | None = None

    #: Use PSILCA for sector matching and social risk/opportunity data. When
    #: disabled, SENSE runs a working-hours-only pipeline from the external
    #: productivity baseline and never reads PSILCA or loads embedding models.
    use_psilca: bool = True

    #: Explicit source filename for ambiguous ``COUNTRY | sector name`` identities.
    #: Conflicting assessments otherwise stop instead of depending on file order.
    psilca_process_overrides: dict[str, str] = field(default_factory=dict)

    #: Write risk- and opportunity-weighted social exchanges and methods.
    #: ``None`` follows ``use_psilca`` (enabled in the integrated workflow,
    #: disabled in the productivity-only workflow).
    enable_social_flows: bool | None = None

    #: Write aggregate working-hour exchanges and their method.
    enable_working_hours_flows: bool = True

    #: Add working-hour flows and methods by ISIC two-digit division. Requires
    #: ``enable_working_hours_flows``.
    enable_isic_working_hours_methods: bool = False

    # ------------------------------------------------------------------
    # 2) Embedding & matching
    # ------------------------------------------------------------------

    #: Sentence-transformer model used for activity-to-sector matching.
    embedding_model: str = "BAAI/bge-base-en-v1.5"

    #: Immutable Hub commit for a custom encoder. Built-in encoders have pins.
    embedding_revision: str | None = None

    #: Texts encoded per batch; lower values reduce peak memory use.
    embedding_batch_size: int = 256

    #: Retain sectors whose cosine similarity is within this distance of the
    #: best match for an activity-country pair.
    similarity_delta: float = 0.028

    #: Absolute cosine-similarity threshold for flagging low-confidence best
    #: matches. Matches below it are retained and reported in run diagnostics;
    #: set to None to disable this diagnostic.
    minimum_similarity: float | None = 0.5

    # ------------------------------------------------------------------
    # 3) Ecoinvent pricing heuristics
    # ------------------------------------------------------------------

    #: Case-insensitive activity-name prefixes excluded from social mapping but
    #: retained in the price-inference graph.
    exclude_terms_starting_with: list[str] = field(
        default_factory=lambda: [
            "market for",
            "market group for",
            "generic market for",
            "production market for",
        ]
    )

    #: Case-insensitive activity-name substrings excluded from social mapping
    #: but retained in the price-inference graph.
    exclude_terms: list[str] = field(
        default_factory=lambda: [
            "production mix",
            "consumption mix",
            "technology mix",
            "consumption market for",
        ]
    )

    #: Complete activity names excluded from social mapping after
    #: case-insensitive whitespace normalization. Activities remain available
    #: to price inference.
    exclude_exact: list[str] = field(default_factory=list)

    #: Exclude activities with neither nonzero technosphere inputs nor nonzero
    #: environmental exchanges. Disable for intentionally empty custom inventories.
    exclude_empty_inventories: bool = True

    #: Zero-price handling. The default assigns direct hours only to eligible
    #: treatments. ``legacy_price_floor`` explicitly reproduces the old pricing
    #: heuristic; ``zero`` leaves all zero prices with zero direct scores.
    zero_price_policy: Literal["treatment_working_hours", "legacy_price_floor", "zero"] = (
        "treatment_working_hours"
    )

    #: Percentile of ordinary direct hours, by reference-product unit, used for
    #: treatments. Reference activities never include fallback recipients.
    #: Zero disables fallback contributions for zero/missing-price treatments.
    treatment_working_hours_percentile: float = 20.0

    #: Price percentile used only with ``zero_price_policy='legacy_price_floor'``.
    #: Set to zero to disable that historical price replacement.
    price_floor_percentile: float = 20.0

    #: Monetary basis expected on ecoinvent reference exchanges imported into
    #: Brightway. Ecoinvent prices are conventionally EUR2005; these fields
    #: allow non-standard inputs to be declared explicitly.
    ecoinvent_price_currency: str = "EUR"
    ecoinvent_price_year: int = 2005

    #: PSILCA's worker-hours activity variable is expressed per USD. SENSE
    #: selects a documented representative year from the ``psilca_version``
    #: passed to ``run_sense``. Override either value only for a custom export
    #: or a different modelling assumption.
    psilca_price_currency: str = "USD"
    psilca_price_year: int | None = None

    #: Optional multiplier converting ecoinvent prices into PSILCA's monetary
    #: basis: ``price_psilca = price_ecoinvent * factor``. When None, SENSE
    #: uses its documented built-in conversion for known PSILCA versions.
    #: A factor is required for monetary bases outside those built-in cases.
    ecoinvent_to_psilca_price_factor: float | None = None

    #: Seed-template directory. ``None`` selects
    #: ``data_dir/Output/SeedPrices``.
    seed_prices_dir: Path | None = None

    #: Location-concordance directory. ``None`` selects
    #: ``data_dir/Output/LocationConcordances``.
    location_concordances_dir: Path | None = None

    # ------------------------------------------------------------------
    # 4) Averaging & risk mapping
    # ------------------------------------------------------------------

    #: Decimal places for final indicators; ``None`` preserves full precision.
    rounding_digits: int | None = None

    #: Multipliers from qualitative PSILCA levels to medium-equivalent hours.
    level_weights: dict[str, float] = field(
        default_factory=lambda: {
            "no": 0.001,
            "very low": 0.01,
            "low": 0.1,
            "medium": 1.0,
            "high": 10.0,
            "very high": 100.0,
            "no data": 0.1,
        }
    )

    #: Mean used for social indicators across sectors and countries. Working
    #: hours always use arithmetic aggregation.
    social_flow_mean: Literal["geometric", "arithmetic"] = "geometric"

    #: Country weighting used when an ecoinvent location maps to multiple
    #: PSILCA countries. ``"gdp"`` uses bundled World Bank GDP first and IMF WEO when
    #: no World Bank observation is usable within the nearest-year window; ``"equal"`` reproduces the historical unweighted method.
    #: The same provider is used by the optional productivity baseline.
    country_weighting: Literal["gdp", "equal"] = "gdp"

    #: Reference year for country weights. ``None`` selects the representative
    #: PSILCA monetary year in the integrated workflow and ``productivity_year``
    #: in the productivity-only workflow.
    country_weight_year: int | None = None

    #: Optional complete replacement for SENSE's bundled GDP sources.
    #: Custom inputs are used as supplied, without automatic IMF supplementation.
    #: The CSV must contain: country_code (ISO3), year, gdp_current_usd.
    country_gdp_path: Path | None = None

    # ------------------------------------------------------------------
    # 5) Brightway integration
    # ------------------------------------------------------------------

    #: Require all needed flows to exist in the sociosphere database instead of
    #: creating missing flows.
    precreated_flows: bool = False

    #: Minimum absolute amount difference required before updating an exchange.
    update_tol: float = 0.0

    # ------------------------------------------------------------------
    # 6) External productivity baseline
    # ------------------------------------------------------------------

    #: Optional Parquet baseline replacing PSILCA working-hour intensities. It
    #: must contain ``country``, ``year``, ``isic4_code``, and
    #: ``productivity_hrs_per_usd``. It is required when PSILCA is disabled and
    #: working-hour output is enabled.
    productivity_baseline_path: Path | None = None

    #: Reference year for productivity lookups. When the baseline has no data
    #: for the exact year, the nearest available year (within ±5) is used.
    productivity_year: int = 2020

    #: Strict by default. The explicit alternative accepts GDP-only omissions
    #: in regional productivity averages and records each omitted country.
    #: Missing retained productivity, no usable GDP, and unmapped locations
    #: still fail preflight, including when PSILCA is enabled.
    productivity_gdp_policy: Literal["strict", "available_countries"] = "strict"

    #: Price-currency units per international USD in ``productivity_year``.
    #: Working hours use ``price / productivity_eur_per_usd``. Runs using an
    #: external productivity baseline must declare this conversion explicitly.
    productivity_eur_per_usd: float | None = None

    #: Explicit productivity substitutes for uncovered ecoinvent locations,
    #: expressed as ``{location: [donor_iso3, ...]}``. Donors use the same
    #: country-weight provider as all other multi-country aggregations. No
    #: scientific proxy is assumed by default.
    productivity_location_proxies: dict[str, list[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Reject invalid configurations before they can produce partial output."""

        for name in (
            "save_intermediate",
            "rebuild_cache",
            "continue_on_iso2_error",
            "use_psilca",
            "enable_working_hours_flows",
            "enable_isic_working_hours_methods",
            "precreated_flows",
            "exclude_empty_inventories",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")
        if self.enable_social_flows is not None and type(self.enable_social_flows) is not bool:
            raise ValueError("enable_social_flows must be a boolean or None")
        for name in ("embedding_model", "ecoinvent_price_currency", "psilca_price_currency"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be non-empty text")
        if self.use_psilca:
            from sense.data.embeddings import model_revision

            model_revision(self.embedding_model, self.embedding_revision)
        for name in (
            "iso2_selection",
            "exclude_terms_starting_with",
            "exclude_terms",
            "exclude_exact",
        ):
            value = getattr(self, name)
            if value is None and name == "iso2_selection":
                continue
            if not isinstance(value, list) or any(
                not isinstance(item, str) or not item.strip() for item in value
            ):
                raise ValueError(f"{name} must be a list of non-empty strings")
            if name == "iso2_selection" and (not value or len(set(value)) != len(value)):
                raise ValueError("iso2_selection must be non-empty and contain no duplicates")
        if self.iso2_start_at is not None and (
            not isinstance(self.iso2_start_at, str) or not self.iso2_start_at.strip()
        ):
            raise ValueError("iso2_start_at must be non-empty text or None")
        for name in ("psilca_process_overrides", "level_weights", "productivity_location_proxies"):
            value = getattr(self, name)
            if not isinstance(value, dict) or any(
                not isinstance(key, str) or not key.strip() for key in value
            ):
                raise ValueError(f"{name} must be a mapping with non-empty text keys")
        if not self.level_weights:
            raise ValueError("level_weights must not be empty")
        for name in (
            "similarity_delta",
            "minimum_similarity",
            "price_floor_percentile",
            "treatment_working_hours_percentile",
            "update_tol",
            "ecoinvent_to_psilca_price_factor",
            "productivity_eur_per_usd",
        ):
            value = getattr(self, name)
            optional = name in {
                "minimum_similarity",
                "ecoinvent_to_psilca_price_factor",
                "productivity_eur_per_usd",
            }
            if value is None and optional:
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be a finite number")
            if (
                name in {"ecoinvent_to_psilca_price_factor", "productivity_eur_per_usd"}
                and value <= 0
            ):
                raise ValueError(f"{name} must be positive")
        for name in (
            "seed_prices_dir",
            "location_concordances_dir",
            "country_gdp_path",
            "productivity_baseline_path",
        ):
            value = getattr(self, name)
            if value is not None:
                if not isinstance(value, (str, Path)) or not str(value).strip():
                    raise ValueError(f"{name} must be a non-empty path or None")
                setattr(self, name, Path(value).expanduser())

        normalized_overrides: dict[str, str] = {}
        for identity, filename in self.psilca_process_overrides.items():
            parts = identity.split(" | ")
            if (
                len(parts) != 2
                or not all(part.strip() for part in parts)
                or not isinstance(filename, str)
                or not filename.strip()
            ):
                raise ValueError(
                    "psilca_process_overrides requires 'COUNTRY | sector name': filename"
                )
            key = " | ".join(part.strip().lower() for part in parts)
            if key in normalized_overrides and normalized_overrides[key] != filename:
                raise ValueError(f"Conflicting PSILCA process selections for {identity!r}")
            normalized_overrides[key] = filename
        self.psilca_process_overrides = normalized_overrides

        if self.iso2_selection is not None and self.iso2_start_at is not None:
            raise ValueError("Set either iso2_selection or iso2_start_at, not both")
        if self.enable_social_flows is None:
            self.enable_social_flows = self.use_psilca
        elif not self.use_psilca and self.enable_social_flows:
            raise ValueError(
                "enable_social_flows requires use_psilca=True because social "
                "risk and opportunity levels come from PSILCA"
            )
        if (
            not self.use_psilca
            and self.enable_working_hours_flows
            and self.productivity_baseline_path is None
        ):
            raise ValueError("PSILCA-free working hours require productivity_baseline_path")
        if self.productivity_location_proxies and self.productivity_baseline_path is None:
            raise ValueError("productivity_location_proxies requires productivity_baseline_path")
        if self.enable_isic_working_hours_methods and not self.enable_working_hours_flows:
            raise ValueError(
                "enable_isic_working_hours_methods requires enable_working_hours_flows"
            )
        if type(self.embedding_batch_size) is not int or self.embedding_batch_size <= 0:
            raise ValueError("embedding_batch_size must be a positive integer")
        if not 0 <= self.similarity_delta <= 2:
            raise ValueError("similarity_delta must be between 0 and 2")
        if self.minimum_similarity is not None and not -1 <= self.minimum_similarity <= 1:
            raise ValueError("minimum_similarity must be between -1 and 1 or None")
        if not 0 <= self.price_floor_percentile <= 100:
            raise ValueError("price_floor_percentile must be between 0 and 100")
        if not 0 <= self.treatment_working_hours_percentile <= 100:
            raise ValueError("treatment_working_hours_percentile must be between 0 and 100")
        if self.zero_price_policy not in {"treatment_working_hours", "legacy_price_floor", "zero"}:
            raise ValueError("Unknown zero_price_policy")
        if self.rounding_digits is not None and (
            type(self.rounding_digits) is not int or self.rounding_digits < 0
        ):
            raise ValueError("rounding_digits must be a non-negative integer or None")
        if self.social_flow_mean not in ("geometric", "arithmetic"):
            raise ValueError("social_flow_mean must be 'geometric' or 'arithmetic'")
        if self.country_weighting not in ("gdp", "equal"):
            raise ValueError("country_weighting must be 'gdp' or 'equal'")
        if self.productivity_gdp_policy not in ("strict", "available_countries"):
            raise ValueError("productivity_gdp_policy must be 'strict' or 'available_countries'")
        if self.productivity_gdp_policy == "available_countries":
            if self.productivity_baseline_path is None:
                raise ValueError("available_countries requires productivity_baseline_path")
            if self.country_weighting != "gdp":
                raise ValueError("available_countries requires country_weighting='gdp'")
        if not math.isfinite(self.update_tol) or self.update_tol < 0:
            raise ValueError("update_tol must be finite and non-negative")
        if not self.ecoinvent_price_currency.strip() or not self.psilca_price_currency.strip():
            raise ValueError("price currencies must be non-empty")
        for field_name, year in (
            ("ecoinvent_price_year", self.ecoinvent_price_year),
            ("psilca_price_year", self.psilca_price_year),
            ("country_weight_year", self.country_weight_year),
            ("productivity_year", self.productivity_year),
        ):
            if year is None and field_name in {"ecoinvent_price_year", "productivity_year"}:
                raise ValueError(f"{field_name} must be a positive integer")
            if year is not None and (type(year) is not int or year <= 0):
                raise ValueError(f"{field_name} must be a positive integer")
        for level, weight in self.level_weights.items():
            if (
                isinstance(weight, bool)
                or not isinstance(weight, (int, float))
                or not math.isfinite(weight)
                or weight <= 0
            ):
                raise ValueError(f"Risk weight for {level!r} must be positive and finite")
        for location, donors in self.productivity_location_proxies.items():
            if not str(location).strip():
                raise ValueError("productivity_location_proxies contains an empty location")
            if not isinstance(donors, list) or not donors:
                raise ValueError(
                    f"productivity_location_proxies[{location!r}] must list at least one donor"
                )
            invalid = [
                donor
                for donor in donors
                if not isinstance(donor, str)
                or len(donor) != 3
                or not donor.isalpha()
                or not donor.isupper()
            ]
            if invalid:
                raise ValueError(
                    f"productivity_location_proxies[{location!r}] contains invalid ISO3 "
                    f"donors: {invalid}"
                )
