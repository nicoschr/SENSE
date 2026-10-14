"""Content checks for completed mappings, with verified reusable fingerprints."""

from __future__ import annotations

import hashlib
import logging
from itertools import groupby
from pathlib import Path
from pickle import loads
from typing import Any

from sense.utils.cache import stable_hash

logger = logging.getLogger(__name__)
# Store digests/counts only for inventories. Flow metadata is held per SQLite epoch.
_PHYSICAL: dict[tuple, dict[str, Any]] = {}
_LAYERS: dict[tuple, dict[str, Any]] = {}
_OWNERSHIP: dict[tuple, tuple[dict, list[str]]] = {}


def _epoch(db) -> tuple:
    """Detect writes by this connection, other connections and file replacement."""
    connection = db.connection()
    path = Path(db.database)
    files: list[tuple] = []
    for name in (path, Path(str(path) + "-wal")):
        try:
            stat = name.stat()
            files.append((str(name), stat.st_ino, stat.st_mtime_ns, stat.st_size))
        except OSError:
            files.append((str(name), None))
    return (
        str(path),
        id(connection),
        connection.total_changes,
        db.execute_sql("PRAGMA data_version").fetchone()[0],
        tuple(files),
    )


def _raw_update(digest, values) -> None:
    """Frame byte strings unambiguously without decoding exchange payloads."""
    for value in values:
        raw = value if isinstance(value, bytes) else str(value).encode("utf-8")
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)


def _ownership(db, epoch) -> tuple[dict, list[str]]:
    cached = _OWNERSHIP.get(epoch)
    if cached is not None:
        return cached
    owned = {}
    for database, code, payload in db.execute_sql(
        "SELECT database, code, data FROM activitydataset WHERE type = ?", ("emission",)
    ):
        metadata = loads(payload)
        if metadata.get("sense_managed") is True:
            owned[database, code] = metadata
    counts: dict[str, int] = {}
    for database, _ in owned:
        counts[database] = counts.get(database, 0) + 1
    fully_owned = [
        name
        for name, count in counts.items()
        if db.execute_sql(
            "SELECT COUNT(*) FROM activitydataset WHERE database = ?", (name,)
        ).fetchone()[0]
        == count
    ]
    _OWNERSHIP.clear()
    _OWNERSHIP[epoch] = (owned, fully_owned)
    return owned, fully_owned


def _physical_inventory(db, name, owned, fully_owned, epoch) -> tuple[dict, set[str]]:
    key = (str(db.database), name)
    cached = _PHYSICAL.get(key)
    if cached is not None and cached["epoch"] == epoch:
        return dict(cached["state"]), set(cached["dependencies"])

    def scan(canonical: bool):
        raw = hashlib.sha256()
        digest = hashlib.sha256()
        activities = exchanges = 0
        dependencies = set()
        for code, payload in db.execute_sql(
            "SELECT code, data FROM activitydataset WHERE database = ? ORDER BY code", (name,)
        ):
            _raw_update(raw, (code, payload))
            if canonical:
                digest.update(stable_hash([code, loads(payload)]).encode("ascii"))
            activities += 1
        exclude = ""
        if fully_owned:
            slots = ",".join("?" for _ in fully_owned)
            exclude = f" AND (type != 'biosphere' OR input_database NOT IN ({slots}))"
        rows = db.execute_sql(
            "SELECT output_code, input_database, input_code, type, data "
            f"FROM exchangedataset WHERE output_database = ?{exclude} ORDER BY output_code",
            (name, *fully_owned),
        )
        for code, group in groupby(rows, key=lambda row: row[0]):
            physical = []
            for _, namespace, flow_code, kind, payload in group:
                if kind == "biosphere" and (namespace, flow_code) in owned:
                    continue
                _raw_update(raw, (code, namespace, flow_code, kind, payload))
                if canonical:
                    physical.append(stable_hash(loads(payload)))
                exchanges += 1
                if namespace != name:
                    dependencies.add(namespace)
            if physical:
                digest.update(stable_hash([code, sorted(physical)]).encode("ascii"))
        return (
            raw.hexdigest(),
            {"sha256": digest.hexdigest(), "activities": activities, "exchanges": exchanges},
            dependencies,
        )

    signature, state, dependencies = scan(cached is None)
    if cached is not None:
        if signature == cached["signature"]:
            state = cached["state"]
            dependencies = set(cached["dependencies"])
        else:
            verified, state, dependencies = scan(True)
            if verified != signature:
                raise ValueError(
                    "Inventory changed during state verification; retry after edits finish"
                )
    if len(_PHYSICAL) >= 128:
        _PHYSICAL.clear()
    _PHYSICAL[key] = {
        "epoch": epoch,
        "signature": signature,
        "state": state,
        "dependencies": sorted(dependencies),
    }
    return dict(state), dependencies


def _social_layer(db, database, sociosphere, owned, epoch) -> tuple[str, set[tuple]]:
    key = (str(db.database), database, sociosphere)
    cached = _LAYERS.get(key)
    if cached is not None and cached["epoch"] == epoch:
        return cached["digest"], set(cached["flows"])

    def scan(canonical):
        raw = hashlib.sha256()
        hashes, flows = [], set()
        for code, flow_code, payload in db.execute_sql(
            "SELECT output_code, input_code, data FROM exchangedataset "
            "WHERE output_database = ? AND input_database = ? AND type = ?",
            (database, sociosphere, "biosphere"),
        ):
            flow = (sociosphere, flow_code)
            if flow not in owned:
                continue
            _raw_update(raw, (code, flow_code, payload))
            flows.add(flow)
            if canonical:
                hashes.append(stable_hash([code, loads(payload)]))
        return raw.hexdigest(), stable_hash(sorted(hashes)), flows

    signature, digest, flows = scan(cached is None)
    if cached is not None:
        if signature == cached["signature"]:
            digest = cached["digest"]
        else:
            verified, digest, flows = scan(True)
            if verified != signature:
                raise ValueError(
                    "Social layer changed during state verification; retry after edits finish"
                )
    if len(_LAYERS) >= 128:
        _LAYERS.clear()
    _LAYERS[key] = {
        "epoch": epoch,
        "signature": signature,
        "digest": digest,
        "flows": sorted(flows),
    }
    return digest, flows


def capture_database_state(
    database: str, sociosphere: str, *, extra_databases: tuple[str, ...] = ()
) -> dict[str, Any]:
    """Capture the same content digests while avoiding repeated decoding/hashing.

    SQLite changes invalidate the fast path. Unchanged physical payloads can reuse
    their canonical digest after byte-for-byte verification. Other owned social
    layers are excluded by SQL where their namespace is entirely SENSE-owned.
    All used flow metadata and method factors remain checked on every snapshot.
    """
    from bw2data import Method, databases, methods, projects
    from bw2data.backends.peewee import ExchangeDataset

    db = ExchangeDataset._meta.database
    epoch = _epoch(db)
    with db.atomic():
        owned, fully_owned = _ownership(db, epoch)
        pending = {database, *extra_databases}
        inventories = {}
        while pending:
            name = min(pending)
            pending.remove(name)
            if name in inventories:
                continue
            if name not in databases:
                raise ValueError(f"Mapped inventory dependency {name!r} is absent; rerun SENSE")
            logger.info("Verifying physical inventory state: %s", name)
            state, dependencies = _physical_inventory(db, name, owned, fully_owned, epoch)
            inventories[name] = state
            pending.update(dependencies - inventories.keys())
        logger.info("Verifying mapped social layer: %s / %s", database, sociosphere)
        layer_exchanges, used_flows = _social_layer(db, database, sociosphere, owned, epoch)
        layer_methods = {}
        for key in sorted(methods):
            if key[:2] != ("SENSE", sociosphere):
                continue
            factors = [
                stable_hash(row) for row in Method(key).load() if tuple(row[0]) in used_flows
            ]
            if factors:
                layer_methods[" | ".join(key)] = stable_hash(
                    {"unit": methods[key].get("unit"), "rows": sorted(factors)}
                )
        result = {
            "schema": 1,
            "project": projects.current,
            "database": database,
            "sociosphere": sociosphere,
            "extra_databases": list(extra_databases),
            "inventories": inventories,
            "social_exchanges": layer_exchanges,
            "used_social_flows": stable_hash([(key, owned[key]) for key in sorted(used_flows)]),
            "methods": layer_methods,
        }
    if _epoch(db) != epoch:
        raise ValueError("Database changed during state verification; retry after edits finish")
    logger.info("Inventory and social data verification complete")
    return result


def validate_database_state(recorded: dict[str, Any]) -> None:
    """Reject changed physical inputs, owned exchanges, flows or characterization."""
    from bw2data import projects

    if not isinstance(recorded, dict) or recorded.get("schema") != 1:
        raise ValueError("Completed run lacks a supported inventory-state snapshot; rerun SENSE")
    if recorded.get("project") not in projects:
        raise ValueError("Completed run's Brightway project is absent; rebuild the study")
    projects.set_current(recorded["project"])
    current = capture_database_state(
        recorded["database"],
        recorded["sociosphere"],
        extra_databases=tuple(recorded["extra_databases"]),
    )
    if current != recorded:
        raise ValueError(
            "Brightway inventory or mapped social layer changed after the completed run; rerun SENSE for this database/layer before calculating or reusing results"
        )
