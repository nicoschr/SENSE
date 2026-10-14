"""Store normalized embedding arrays with validated JSON manifests."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sense.artifacts import atomic_output_path, write_json_atomic
from sense.utils.cache import file_hash

BGE_QUERY_PROMPT = "Represent this sentence for searching relevant passages: "

PINNED_MODEL_REVISIONS = {
    "BAAI/bge-base-en-v1.5": "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a",
    "sentence-transformers/all-MiniLM-L6-v2": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
}


def model_revision(model_name: str, revision: str | None = None) -> str:
    """Require a Hub identifier and immutable revision, even when caches exist."""
    if (
        not isinstance(model_name, str)
        or len(model_name) > 193
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*(/[A-Za-z0-9][A-Za-z0-9._-]*)?", model_name)
        is None
        or ".." in model_name
        or "--" in model_name
        or Path(model_name).exists()
    ):
        raise ValueError("Use a trusted Hugging Face model identifier, not a local model path")
    resolved = revision if revision is not None else PINNED_MODEL_REVISIONS.get(model_name)
    if not isinstance(resolved, str) or re.fullmatch(r"[0-9a-f]{40}", resolved) is None:
        raise ValueError(
            "Custom encoders require embedding_revision: an immutable 40-digit Hub commit"
        )
    return resolved


def load_encoder(model_name: str, revision: str | None = None) -> Any:
    """Load model weights as safetensors, without repository-provided Python code."""
    # Older sentence-transformers releases import custom Python from local paths
    # despite trust_remote_code=False. Reject paths/shadowing before importing it.
    resolved = model_revision(model_name, revision)
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        model_name,
        revision=resolved,
        trust_remote_code=False,
        model_kwargs={"use_safetensors": True},
    )


def prep_query_bge(text: str) -> str:
    """Prefix an activity description with the BGE retrieval instruction."""
    return BGE_QUERY_PROMPT + (text or "").strip()


def prep_doc(text: str) -> str:
    """Strip a PSILCA sector label without adding the query instruction."""
    return (text or "").strip()


def _sha(s: str) -> str:
    """Generate a short SHA256 hash of a string."""
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def save_embeddings(
    dirpath: str,
    name: str,
    emb: np.ndarray,
    ids: pd.Series,
    meta: dict[str, Any],
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

    Each file is replaced atomically; the pair is not a single transaction.
    """
    if Path(name).name != name or name in {"", ".", ".."}:
        raise ValueError("Embedding store name must be a filename")
    emb = np.asarray(emb, dtype=np.float32)
    if emb.ndim != 2 or emb.shape[0] != len(ids) or not np.isfinite(emb).all():
        raise ValueError("Embeddings must be a finite 2D array with one row per ID")
    os.makedirs(dirpath, exist_ok=True)
    base = os.path.join(dirpath, name)

    # Save embeddings as numpy array
    with atomic_output_path(Path(base + ".npy")) as temporary:
        np.save(temporary, emb.astype("float32"))
        checksum = file_hash(temporary)

    # Build and save manifest
    manifest = {
        "name": name,
        "rows": int(emb.shape[0]),
        "dim": int(emb.shape[1]),
        "emb_path": os.path.basename(base + ".npy"),
        "storage": "npy_float32",
        "sha256": checksum,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "meta": meta,
        "ids": ids.tolist(),
    }

    write_json_atomic(Path(base + ".manifest.json"), manifest)


def load_embeddings(
    dirpath: str,
    name: str,
) -> tuple[np.ndarray, pd.Series, dict[str, Any]]:
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

    """
    if Path(name).name != name or name in {"", ".", ".."}:
        raise ValueError("Embedding store name must be a filename")
    base = os.path.join(dirpath, name)

    with open(base + ".manifest.json", encoding="utf-8") as f:
        manifest = json.load(f)

    vector_path = Path(base + ".npy")
    if (
        not isinstance(manifest, dict)
        or manifest.get("emb_path") != vector_path.name
        or manifest.get("sha256") != file_hash(vector_path)
        or not isinstance(manifest.get("ids"), list)
        or not isinstance(manifest.get("meta"), dict)
    ):
        raise ValueError("Invalid or incomplete embedding store; rebuild the cache")
    emb = np.load(vector_path, mmap_mode="r", allow_pickle=False)
    if (
        emb.ndim != 2
        or emb.shape != (manifest.get("rows"), manifest.get("dim"))
        or len(manifest["ids"]) != len(emb)
        or emb.dtype != np.dtype("float32")
        or not np.isfinite(emb).all()
    ):
        raise ValueError("Embedding array does not match its manifest")
    ids = pd.Series(manifest["ids"], name="id")

    return emb, ids, manifest


def store_is_complete(
    out_dir: str,
    base_name: str,
    *,
    expected_ids: Sequence[Any] | None = None,
    expected_meta: dict[str, Any] | None = None,
) -> bool:
    """
    Check if an embedding store is complete (both files exist).

    Args:
        out_dir: Directory to check
        base_name: Base name of the embedding files
        expected_ids: Exact row identifiers required for cache reuse.
        expected_meta: Manifest metadata values required for cache reuse.

    Returns:
        True if both .manifest.json and .npy files exist

    Examples:
        >>> if not store_is_complete("./store", "my_embeddings"):
        ...     # Need to create embeddings
        ...     pass
    """
    manifest = os.path.join(out_dir, f"{base_name}.manifest.json")
    vecs = os.path.join(out_dir, f"{base_name}.npy")
    if not (os.path.exists(manifest) and os.path.exists(vecs)):
        return False

    try:
        vectors, _, stored = load_embeddings(out_dir, base_name)
    except (OSError, ValueError, KeyError, TypeError):
        return False

    stored_ids = stored.get("ids")
    if not isinstance(stored_ids, list):
        return False
    if vectors.ndim != 2 or vectors.shape[0] != len(stored_ids):
        return False
    if stored.get("rows") != vectors.shape[0] or stored.get("dim") != vectors.shape[1]:
        return False
    if expected_ids is not None and stored_ids != list(expected_ids):
        return False

    stored_meta = stored.get("meta") or {}
    if expected_meta is not None and any(
        stored_meta.get(key) != value for key, value in expected_meta.items()
    ):
        return False
    return True
