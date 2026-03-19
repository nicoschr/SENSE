"""
Embedding storage utilities for SENSE.

This module provides functions for saving and loading pre-computed embeddings
to/from disk in an efficient format (.npy for vectors + JSON manifest for metadata).
"""

from __future__ import annotations

import json
import os
import time
import hashlib
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


BGE_QUERY_PROMPT = "Represent this sentence for searching relevant passages: "

def prep_query_bge(text: str) -> str:
    """
    Prepare a *query* string for BGE-style retrieval embeddings.

    - BGE retrieval models are trained with an *instruction-style prefix* that is
      intended for **queries** (short inputs that represent the information need).
    - For “query → passage/document” retrieval, you typically prepend the query
      instruction and leave documents unprefixed (see `prep_doc`).
    - This improves alignment with how the model was trained and often improves
      ranking quality, especially when queries are short or underspecified.

    Args:
        text: Query text (e.g., an activity description).

    Returns:
        The query text with the BGE query instruction prefix prepended and whitespace trimmed.

    Examples:
        >>> BGE_QUERY_PROMPT = "Represent this sentence for searching relevant passages: "
        >>> prep_query_bge("market for electricity, medium voltage")
        'Represent this sentence for searching relevant passages: market for electricity, medium voltage'
    """
    return BGE_QUERY_PROMPT + (text or "").strip()


def prep_doc(text: str) -> str:
    """
    Prepare a *document/passage* string for BGE-style retrieval embeddings.

    Best practice (BGE / FlagEmbedding):
    - For BGE retrieval models, documents/passages generally **do not need an instruction**
      prefix. The recommended setup for query→document retrieval is:
        - apply an instruction to the **query** (see `prep_query_bge`)
        - embed **documents as-is** (this function)
    - Keeping documents unprefixed avoids injecting extra tokens that can blur the
      semantic identity of short labels (like PSILCA sector names) and keeps corpus
      embeddings stable and reusable.

    Args:
        text: Document/passage text (e.g., a PSILCA sector name).

    Returns:
        The cleaned document text with surrounding whitespace removed (no prompt added).

    Examples:
        >>> prep_doc("Agriculture, hunting and related service activities")
        'Agriculture, hunting and related service activities'
    """
    return (text or "").strip()


def _sha(s: str) -> str:
    """Generate a short SHA256 hash of a string."""
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def save_embeddings(
    dirpath: str,
    name: str,
    emb: np.ndarray,
    ids: pd.Series,
    meta: Dict[str, Any],
) -> None:
    """
    Save embeddings to disk.
    
    Creates two files:
      - {name}.npy: float32 embedding vectors
      - {name}.manifest.json: metadata + IDs inline
      
    Args:
        dirpath: Directory to save files to
        name: Base name for the files (without extension)
        emb: Embedding vectors as a 2D numpy array (n_samples x dim)
        ids: Series of identifiers for each embedding row
        meta: Additional metadata to store in the manifest
        
    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> emb = np.random.randn(100, 768).astype('float32')
        >>> ids = pd.Series([f"id_{i}" for i in range(100)])
        >>> save_embeddings("./store", "my_embeddings", emb, ids, {"model": "bge-base"})
    """
    os.makedirs(dirpath, exist_ok=True)
    base = os.path.join(dirpath, name)

    # Save embeddings as numpy array
    np.save(base + ".npy", emb.astype("float32"))

    # Build and save manifest
    manifest = {
        "name": name,
        "rows": int(emb.shape[0]),
        "dim": int(emb.shape[1]),
        "emb_path": os.path.basename(base + ".npy"),
        "storage": "npy_float32",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "meta": meta,
        "ids": ids.tolist(),
    }

    with open(base + ".manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def load_embeddings(
    dirpath: str,
    name: str,
) -> Tuple[np.ndarray, pd.Series, Dict[str, Any]]:
    """
    Load embeddings from disk.
    
    Args:
        dirpath: Directory containing the embedding files
        name: Base name of the files (without extension)
        
    Returns:
        Tuple of (embeddings, ids, manifest):
        - embeddings: numpy array of shape (n_samples, dim)
        - ids: pandas Series of identifiers
        - manifest: full manifest dictionary including metadata
        
    Raises:
        FileNotFoundError: If the manifest or embedding files don't exist
        
    Examples:
        >>> emb, ids, meta = load_embeddings("./store", "my_embeddings")
        >>> print(f"Loaded {len(ids)} embeddings of dimension {emb.shape[1]}")
    """
    base = os.path.join(dirpath, name)
    
    with open(base + ".manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)
    
    emb = np.load(os.path.join(dirpath, manifest["emb_path"]), mmap_mode="r")
    ids = pd.Series(manifest["ids"], name="id")
    
    return emb, ids, manifest


def store_is_complete(out_dir: str, base_name: str) -> bool:
    """
    Check if an embedding store is complete (both files exist).
    
    Args:
        out_dir: Directory to check
        base_name: Base name of the embedding files
        
    Returns:
        True if both .manifest.json and .npy files exist
        
    Examples:
        >>> if not store_is_complete("./store", "my_embeddings"):
        ...     # Need to create embeddings
        ...     pass
    """
    manifest = os.path.join(out_dir, f"{base_name}.manifest.json")
    vecs = os.path.join(out_dir, f"{base_name}.npy")
    return os.path.exists(manifest) and os.path.exists(vecs)
