"""
Brightway2 integration modules for SENSE.

This subpackage contains:
- Biosphere flow creation and management
- Impact category (Method) creation
"""

from sense.brightway.flows import (
    add_social_biosphere_flows_and_exchanges,
    add_working_hours_flows_and_exchanges,
)
from sense.brightway.methods import (
    add_social_impact_categories,
    add_working_hours_impact_category,
)

__all__ = [
    "add_social_biosphere_flows_and_exchanges",
    "add_working_hours_flows_and_exchanges",
    "add_social_impact_categories",
    "add_working_hours_impact_category",
]
