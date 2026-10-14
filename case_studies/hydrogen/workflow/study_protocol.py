"""One versioned specification for the paired hydrogen application."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Literal

SCENARIOS = ("baseline", "regionalized")

# Fixed manuscript design. Local configuration contains paths and credentials only.
# Scientific rationale and assumptions: ../REFERENCE.md.
_STUDY = {
    "schema": 1,
    "design": "hydrogen-child-employment-fatal-accidents-delta028-foreground-bookkeeping-v6",
    "countries": ["DE", "CN", "US"],
    "environmental_family": "ReCiPe 2016 v1.03, midpoint (H) regionalized",
    "country_rationale": "Selected by the author before inspecting revised results.",
    "technology_labels": {
        "hydrogen production, biomass gasification": "Biomass gasification",
        "hydrogen production, steam methane reforming of biomethane": "SMR (biomethane)",
        "hydrogen production, PEM electrolysis, green": "Wind PEM electrolysis",
        "hydrogen production, coal gasification": "Coal gasification",
        "hydrogen production, steam methane reforming": "SMR (natural gas)",
    },
    "social_layers": {"geometric": "sociosphere", "arithmetic": "sociosphere_arithmetic"},
    # These workbook biogas supply nodes contain carbon accounting only.
    # Apply to the supporting methanol foregrounds, never to background activities.
    "foreground_exclude_exact": [
        "treatment of industrial wastewater by anaerobic digestion, cut-off with biogenic carbon uptake",
        "treatment of biowaste by anaerobic digestion, cut-off with biogenic carbon uptake",
        "treatment of sewage sludge by anaerobic digestion, cut-off with biogenic carbon uptake",
    ],
    "sense": {
        "exclude_empty_inventories": True,
        "zero_price_policy": "treatment_working_hours",
        "treatment_working_hours_percentile": 20.0,
        "embedding_model": "BAAI/bge-base-en-v1.5",
        "similarity_delta": 0.028,
        "country_weighting": "gdp",
        "level_weights": {
            "no": 0.001,
            "very low": 0.01,
            "low": 0.1,
            "medium": 1.0,
            "high": 10.0,
            "very high": 100.0,
            "no data": 0.1,
        },
    },
    "reference_basis": {
        "amount": 1.0,
        "unit": "kilogram",
        "product": "hydrogen at each route's modelled production gate",
        "comparison": "original versus regionalized supply chains within each route",
    },
    "indicator_review": {
        "manual": "https://www.openlca.org/wp-content/uploads/2025/07/PSILCA_manual_for-v.4_04072025.pdf",
        "section": "4.1.1, printed pages 34-36; 4.1.6, printed pages 51-52",
        "scope": "Sector-based screening with inherited PSILCA attribution and proxy data.",
        "coverage": "Exported assessments, not verified primary observations or labour-weighted coverage.",
        "interpretation": "Risk-weighted modelled labour associated with children-in-employment "
        "and fatal-accident assessments, not child-worker hours, numbers of children, "
        "or predicted fatalities. Children in employment includes non-hazardous employment. "
        "Missing assessments require supply-chain coverage and sensitivity analysis.",
    },
    "panels": [
        {
            "id": "climate",
            "label": "Climate change",
            "method": [
                "ReCiPe 2016 v1.03, midpoint (H) (explicit biogenic carbon) regionalized",
                "climate change",
                "global warming potential (GWP100)",
            ],
        },
        {
            "id": "children_in_employment",
            "label": "Children-in-employment risk screening",
            "method": [
                "SENSE",
                "sociosphere",
                "Workers",
                "Child labour",
                "Children in employment, total",
                "risk",
            ],
        },
        {
            "id": "fatal_accidents",
            "label": "Fatal-accident risk screening",
            "method": [
                "SENSE",
                "sociosphere",
                "Workers",
                "Health and safety",
                "Rate of fatal accidents at workplace",
                "risk",
            ],
        },
    ],
}


def load_protocol() -> dict:
    """Return an independent copy of the fixed, source-reviewed manuscript design."""
    return deepcopy(_STUDY)


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def context(config: dict, spec: dict) -> dict:
    return {
        "versions": config["versions"],
        "project": config["brightway"]["project"],
        "study": spec,
        "study_sha256": fingerprint(spec),
    }


def calculation_context(config: dict, spec: dict) -> dict:
    """Scientific calculation identity; panel selection and labels are presentation."""
    return {
        "versions": config["versions"],
        "project": config["brightway"]["project"],
        "brightway": config["brightway"],
        "foregrounds": {scenario: foregrounds(scenario) for scenario in SCENARIOS},
        "countries": spec["countries"],
        "technologies": sorted(spec["technology_labels"]),
        "reference_basis": spec["reference_basis"],
        "sense": spec["sense"],
        "foreground_exclude_exact": spec["foreground_exclude_exact"],
        "social_layer": spec["social_layers"]["geometric"],
        "environmental_family": spec["environmental_family"],
        "climate_method": spec["panels"][0]["method"],
    }


def foregrounds(scenario: str) -> dict[str, str]:
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}")
    return {
        "hydrogen": f"01_hydrogen_inventories_{scenario}",
        "methanol": f"02_methanol_inventories_{scenario}",
        "subset": f"01_hydrogen_inventories_{scenario}_subset",
    }


def sense_config(
    spec: dict, mean: Literal["geometric", "arithmetic"] = "geometric", database: str | None = None
):
    from sense import SenseConfig

    settings = dict(spec["sense"])
    if database is not None and any(
        database == foregrounds(scenario).get("methanol") for scenario in SCENARIOS
    ):
        settings["exclude_exact"] = [
            *settings.get("exclude_exact", []),
            *spec["foreground_exclude_exact"],
        ]
    if database in {name for scenario in SCENARIOS for name in foregrounds(scenario).values()}:
        settings["iso2_selection"] = spec["countries"]
    return SenseConfig(social_flow_mean=mean, **settings)
