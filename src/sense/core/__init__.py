"""
Core processing modules for SENSE.

This subpackage contains:
- Semantic matching between ecoinvent activities and PSILCA sectors
- Sector averaging with price weighting
"""

from sense.core.matching import match_from_stores, add_social_exchanges
from sense.core.averaging import average_sector_mappings_w_prices

__all__ = [
    "match_from_stores",
    "add_social_exchanges",
    "average_sector_mappings_w_prices",
]
