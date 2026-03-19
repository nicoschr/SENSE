"""
Data handling modules for SENSE.

This subpackage contains modules for:
- PSILCA data parsing and processing
- Ecoinvent data extraction and location mapping
- Embedding storage and retrieval
"""

from sense.data.embeddings import (
    save_embeddings,
    load_embeddings,
    store_is_complete,
    prep_query_bge,
    prep_doc,
)
from sense.data.psilca import (
    combine_psilca_css_and_extract_social_exchanges,
    embed_psilca_to_store,
)
from sense.data.ecoinvent import (
    map_ecoinvent_locations,
    fetch_ecoinvent_activities_from_ecospold_files,
    embed_ecoinvent_to_store,
)

__all__ = [
    # Embeddings
    "save_embeddings",
    "load_embeddings",
    "store_is_complete",
    "prep_query_bge",
    "prep_doc",
    # PSILCA
    "combine_psilca_css_and_extract_social_exchanges",
    "embed_psilca_to_store",
    # Ecoinvent
    "map_ecoinvent_locations",
    "fetch_ecoinvent_activities_from_ecospold_files",
    "embed_ecoinvent_to_store",
]
