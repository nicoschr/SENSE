"""Transactional persistence for SENSE-managed Brightway exchanges."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

from sense.utils.helpers import slug

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DesiredExchange:
    """One validated biosphere exchange requested by the calculation layer."""

    activity: Any
    flow: Any
    amount: float


def create_missing_flows(
    *,
    flow_specs: dict[tuple[str, str, tuple[str, ...]], dict[str, Any]],
    existing_keys: set[tuple[str, str, tuple[str, ...]]],
    database: Any,
    lookup: dict[tuple[str, str, tuple[str, ...]], Any],
    code_prefix: str,
) -> list[str]:
    """Create missing biosphere flows in one transaction where supported."""

    for key in flow_specs.keys() & existing_keys:
        expected_kind = flow_specs[key].get("sense_kind", code_prefix)
        if lookup[key].get("sense_kind") != expected_kind:
            raise ValueError(f"Refusing to reuse a flow not owned by SENSE: {key!r}")
    missing = [key for key in flow_specs if key not in existing_keys]
    if not missing:
        return []
    if type(database).__module__.startswith("bw2data.") and _supports_sqlite_bulk([]):
        from bw2data import databases
        from bw2data.backends.peewee import ActivityDataset, sqlite3_lci_db
        from bw2data.backends.peewee.utils import dict_as_activitydataset

        database_name = str(database.name)
        rows: list[dict[str, Any]] = []
        keys_and_codes: list[tuple[tuple[str, str, tuple[str, ...]], str]] = []
        for key in missing:
            name, unit, categories = key
            digest = hashlib.sha256(repr(key).encode()).hexdigest()[:12]
            code = f"{code_prefix}_{slug(name)}_{slug(unit)}_{digest}"
            rows.append(
                dict_as_activitydataset(
                    {
                        "database": database_name,
                        "code": code,
                        "name": name,
                        "unit": unit,
                        "categories": categories,
                        "type": "emission",
                        "comment": flow_specs[key]["comment"],
                        "sense_managed": True,
                        "sense_kind": flow_specs[key].get("sense_kind", code_prefix),
                        "sense_schema": 1,
                    }
                )
            )
            keys_and_codes.append((key, code))
        with sqlite3_lci_db.db.atomic():
            for batch in _chunks(rows, 125):
                ActivityDataset.insert_many(batch).execute()
        databases.set_dirty(database_name)
        for key, code in keys_and_codes:
            flow = database.get(code)
            lookup[key] = flow
            existing_keys.add(key)
        return []

    failures: list[str] = []
    for key in missing:
        name, unit, categories = key
        try:
            digest = hashlib.sha256(repr(key).encode()).hexdigest()[:12]
            code = f"{code_prefix}_{slug(name)}_{slug(unit)}_{digest}"
            flow = database.new_activity(
                name=name,
                unit=unit,
                categories=categories,
                type="emission",
                code=code,
                comment=flow_specs[key]["comment"],
                sense_managed=True,
                sense_kind=flow_specs[key].get("sense_kind", code_prefix),
                sense_schema=1,
            )
            flow.save()
            lookup[key] = flow
            existing_keys.add(key)
        except Exception as exc:  # pragma: no cover - backend-specific
            failures.append(f"{name} [{unit}] {categories}: {exc}")
    return failures


def build_desired_exchanges(
    grouped: dict[str, list[dict[str, Any]]],
    techno_lookup: dict[str, Any],
    resolve_flow: Callable[[dict[str, Any]], tuple[Any | None, float | None]],
) -> tuple[list[DesiredExchange], list[str]]:
    """Validate records and collapse them to unique activity-flow pairs."""

    desired: list[DesiredExchange] = []
    errors: list[str] = []
    for code, rows in grouped.items():
        activity = techno_lookup.get(code)
        if activity is None:
            errors.append(f"Technosphere activity not found for code {code!r}")
            continue
        by_flow: dict[tuple[str, str], DesiredExchange] = {}
        for record in rows:
            flow, amount = resolve_flow(record)
            if flow is None or amount is None:
                continue
            numeric_amount = float(amount)
            if not np.isfinite(numeric_amount) or numeric_amount < 0:
                raise ValueError(
                    f"Managed exchange amount must be finite and non-negative: {numeric_amount}"
                )
            if numeric_amount == 0:
                continue
            flow_key = tuple(flow.key)
            previous = by_flow.get(flow_key)
            if previous is not None and not np.isclose(
                previous.amount, numeric_amount, rtol=1e-12, atol=0
            ):
                raise ValueError(
                    f"Conflicting amounts for managed exchange {flow_key!r} "
                    f"on activity {code!r}: {previous.amount} and {numeric_amount}"
                )
            by_flow[flow_key] = DesiredExchange(activity, flow, numeric_amount)
        desired.extend(by_flow.values())
    return desired, errors


def reference_production_amounts(
    activities: dict[Any, Any],
    *,
    preserve_sign: bool = False,
) -> dict[Any, float]:
    """Read reference-throughput magnitudes; implicit production defaults to one.

    A negative single reference exchange represents consumption/treatment of a
    waste. Positive labour/risk intensities scale by its magnitude. The physical
    production sign remains in the technosphere, so activity supply determines
    whether treatment is required or credited. Multiple reference outputs,
    zero production and non-finite production remain unsupported.
    ``preserve_sign`` exposes the same validated references for treatment
    classification. Writers keep the default throughput magnitudes.
    """
    productions: dict[Any, list[dict[str, Any]]] = {key: [] for key in activities}
    values = list(activities.values())
    if values and _supports_sqlite_bulk(values):
        from bw2data.backends.peewee import ExchangeDataset

        by_database: dict[str, list[str]] = {}
        for database, code in activities:
            by_database.setdefault(database, []).append(code)
        for database, all_codes in by_database.items():
            for codes in _chunks(all_codes, 400):
                query = ExchangeDataset.select().where(
                    (ExchangeDataset.output_database == database)
                    & ExchangeDataset.output_code.in_(codes)
                    & (ExchangeDataset.type == "production")
                )
                for exchange in query:
                    productions[(database, exchange.output_code)].append(exchange.data)
    else:
        for key, activity in activities.items():
            if hasattr(activity, "production"):
                productions[key] = [dict(exchange) for exchange in activity.production()]
            else:
                productions[key] = [
                    dict(exchange)
                    for exchange in activity.exchanges()
                    if exchange.get("type") == "production"
                ]

    return {
        key: reference_production_amount(key, exchanges, preserve_sign=preserve_sign)
        for key, exchanges in productions.items()
    }


def empty_inventory_keys(
    activities: list[Any], *, flow_dependencies: set[str] | None = None
) -> set[tuple[str, str]]:
    """Find inventories without physical inputs or environmental exchanges.

    Stream technosphere records first, decoding amounts only until an activity is
    known to be nonempty. Inspect biosphere records only for remaining candidates.
    Read ownership metadata only for referenced flows, not the whole project.
    ``flow_dependencies`` collects namespaces whose metadata affects the decision.
    """
    candidates = {tuple(activity.key): activity for activity in activities}
    if not candidates:
        return set()
    nonempty: set[tuple[str, str]] = set()
    if _supports_sqlite_bulk(list(candidates.values())):
        from pickle import loads

        from bw2data.backends.peewee import ExchangeDataset

        db = ExchangeDataset._meta.database
        by_database: dict[str, set[str]] = {}
        for database, code in candidates:
            by_database.setdefault(database, set()).add(code)
        ownership: dict[tuple[str, str], bool] = {}
        for database, codes in by_database.items():
            # Probe one indexed technosphere exchange per activity. Nearly all
            # producing activities are decided immediately, without visiting their
            # remaining inputs or the much larger SENSE biosphere layers.
            zero_first: list[str] = []
            for group in _chunks(sorted(codes), 400):
                slots = ",".join("?" for _ in group)
                query = db.execute_sql(
                    "SELECT a.code, (SELECT e.data FROM exchangedataset AS e "
                    "WHERE e.output_database = a.database AND e.output_code = a.code "
                    "AND e.type = ? LIMIT 1) FROM activitydataset AS a "
                    f"WHERE a.database = ? AND a.code IN ({slots})",
                    ("technosphere", database, *group),
                )
                for code, payload in query:
                    if payload is None:
                        continue
                    amount = float(loads(payload)["amount"])
                    key = (database, code)
                    if not np.isfinite(amount):
                        raise ValueError(f"Non-finite physical exchange on activity {key!r}")
                    if amount != 0:
                        nonempty.add(key)
                    else:
                        zero_first.append(code)
            # A first zero does not establish emptiness. Check all remaining
            # technosphere inputs for this small unresolved subset.
            for group in _chunks(zero_first, 400):
                slots = ",".join("?" for _ in group)
                query = db.execute_sql(
                    "SELECT output_code, data FROM exchangedataset "
                    "WHERE output_database = ? AND type = ? "
                    f"AND output_code IN ({slots})",
                    (database, "technosphere", *group),
                )
                for code, payload in query:
                    key = (database, code)
                    if key in nonempty:
                        continue
                    amount = float(loads(payload)["amount"])
                    if not np.isfinite(amount):
                        raise ValueError(f"Non-finite physical exchange on activity {key!r}")
                    if amount != 0:
                        nonempty.add(key)
            remaining = sorted(code for code in codes if (database, code) not in nonempty)
            for group in _chunks(remaining, 400):
                placeholders = ",".join("?" for _ in group)
                where = "output_database = ? AND type = ? " f"AND output_code IN ({placeholders})"
                params = (database, "biosphere", *group)
                references = list(
                    db.execute_sql(
                        "SELECT DISTINCT input_database, input_code "
                        f"FROM exchangedataset WHERE {where}",
                        params,
                    )
                )
                missing: dict[str, list[str]] = {}
                for namespace, code in references:
                    if flow_dependencies is not None:
                        flow_dependencies.add(namespace)
                    if (namespace, code) not in ownership:
                        ownership[namespace, code] = False
                        missing.setdefault(namespace, []).append(code)
                for namespace, flow_codes in missing.items():
                    for flow_group in _chunks(flow_codes, 400):
                        flow_slots = ",".join("?" for _ in flow_group)
                        for code, payload in db.execute_sql(
                            "SELECT code, data FROM activitydataset "
                            f"WHERE database = ? AND code IN ({flow_slots})",
                            (namespace, *flow_group),
                        ):
                            ownership[namespace, code] = loads(payload).get("sense_managed") is True
                for code, namespace, flow_code, payload in db.execute_sql(
                    "SELECT output_code, input_database, input_code, data "
                    f"FROM exchangedataset WHERE {where}",
                    params,
                ):
                    key = (database, code)
                    if key in nonempty or ownership[namespace, flow_code]:
                        continue
                    amount = float(loads(payload)["amount"])
                    if not np.isfinite(amount):
                        raise ValueError(f"Non-finite physical exchange on activity {key!r}")
                    if amount != 0:
                        nonempty.add(key)
    else:
        for key, activity in candidates.items():
            for exchange in activity.exchanges():
                if exchange.get("type") not in {"technosphere", "biosphere"}:
                    continue
                if exchange.get("type") == "biosphere":
                    if flow_dependencies is not None:
                        flow_dependencies.add(str(exchange.input.key[0]))
                    if getattr(exchange, "input", {}).get("sense_managed") is True:
                        continue
                amount = float(exchange["amount"])
                if not np.isfinite(amount):
                    raise ValueError(f"Non-finite physical exchange on activity {key!r}")
                if amount != 0:
                    nonempty.add(key)
                    break
    return set(candidates) - nonempty


def reference_production_amount(
    key: Any,
    exchanges: list[Any],
    *,
    preserve_sign: bool = False,
) -> float:
    """Validate a single reference exchange and return its throughput magnitude."""
    if len(exchanges) > 1 or any(tuple(exc.get("input", key)) != key for exc in exchanges):
        raise ValueError(f"Activity {key!r} requires a single reference production output")
    amount = float(exchanges[0]["amount"]) if exchanges else 1.0
    if not np.isfinite(amount) or amount == 0:
        raise ValueError(f"Activity {key!r} requires nonzero finite reference production")
    return amount if preserve_sign else abs(amount)


def synchronize_exchanges(
    *,
    database_name: str,
    grouped: dict[str, list[dict[str, Any]]],
    techno_lookup: dict[str, Any],
    resolve_flow: Callable[[dict[str, Any]], tuple[Any | None, float | None]],
    managed_flow: Callable[[Any], bool] | None,
    managed_flow_keys: set[tuple[str, str]] | None,
    update_tol: float,
) -> tuple[int, int, int]:
    """Synchronize exchanges, using a bulk transaction when the backend supports it."""

    desired, errors = build_desired_exchanges(grouped, techno_lookup, resolve_flow)
    if errors:
        raise RuntimeError(
            f"Brightway exchange synchronization failed ({len(errors)} errors): "
            + "; ".join(errors[:5])
        )
    for item in desired:
        owned = (
            tuple(item.flow.key) in managed_flow_keys
            if managed_flow_keys is not None
            else managed_flow is None or managed_flow(item.flow)
        )
        if not owned:
            raise ValueError(
                f"Refusing to write an exchange to a flow not owned by SENSE: {item.flow.key!r}"
            )
    # Calculations return amounts per reference-product unit. Brightway stores
    # exchanges per activity execution. Use the reference-throughput magnitude
    # for waste treatments too; its sign remains solely in the technosphere.
    factors = reference_production_amounts(
        {getattr(item.activity, "key", id(item.activity)): item.activity for item in desired}
    )
    desired = [
        DesiredExchange(
            item.activity,
            item.flow,
            item.amount * factors[getattr(item.activity, "key", id(item.activity))],
        )
        for item in desired
    ]
    if any(not np.isfinite(item.amount) or item.amount <= 0 for item in desired):
        raise ValueError("Production scaling produced a non-finite or non-positive exchange amount")
    activities = [techno_lookup[code] for code in grouped if code in techno_lookup]
    if managed_flow_keys is not None and _supports_sqlite_bulk(activities):
        return _sync_sqlite_bulk(
            database_name=database_name,
            activities=activities,
            desired=desired,
            managed_flow_keys=managed_flow_keys,
            update_tol=update_tol,
        )
    return _sync_with_public_proxies(
        activities=activities,
        desired=desired,
        managed_flow=managed_flow,
        update_tol=update_tol,
    )


def _supports_sqlite_bulk(activities: list[Any]) -> bool:
    """Return whether activities use Brightway's supported Peewee backend."""

    if activities and not all(
        type(activity).__module__.startswith("bw2data.") for activity in activities
    ):
        return False
    try:
        from bw2data.backends.peewee import ExchangeDataset, sqlite3_lci_db  # noqa: F401
        from bw2data.backends.peewee.utils import dict_as_exchangedataset  # noqa: F401
    except ImportError:
        return False
    return True


def _chunks(values: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _sync_sqlite_bulk(
    *,
    database_name: str,
    activities: list[Any],
    desired: list[DesiredExchange],
    managed_flow_keys: set[tuple[str, str]],
    update_tol: float,
) -> tuple[int, int, int]:
    """Apply a complete diff through Brightway's Peewee backend in one transaction."""

    from bw2data import databases
    from bw2data.backends.peewee import ExchangeDataset, sqlite3_lci_db
    from bw2data.backends.peewee.utils import dict_as_exchangedataset

    output_keys = {tuple(activity.key) for activity in activities}
    desired_by_key = {(tuple(item.activity.key), tuple(item.flow.key)): item for item in desired}
    existing_by_key: dict[tuple[tuple[str, str], tuple[str, str]], list[Any]] = {}
    output_codes = sorted({key[1] for key in output_keys})
    for codes in _chunks(output_codes, 400):
        query = ExchangeDataset.select().where(
            (ExchangeDataset.output_database == database_name)
            & (ExchangeDataset.output_code.in_(codes))
            & (ExchangeDataset.type == "biosphere")
        )
        for exchange in query:
            output_key = (exchange.output_database, exchange.output_code)
            input_key = (exchange.input_database, exchange.input_code)
            if output_key in output_keys and input_key in managed_flow_keys:
                existing_by_key.setdefault((output_key, input_key), []).append(exchange)

    delete_ids: list[int] = []
    insert_rows: list[dict[str, Any]] = []
    added = updated = removed = 0

    for key, item in desired_by_key.items():
        existing = existing_by_key.pop(key, [])
        if not existing:
            added += 1
            insert_rows.append(
                dict_as_exchangedataset(
                    {
                        "input": tuple(item.flow.key),
                        "output": tuple(item.activity.key),
                        "amount": item.amount,
                        "type": "biosphere",
                    }
                )
            )
            continue
        current, *duplicates = existing
        current_amount = float((current.data or {}).get("amount", 0.0))
        delete_ids.extend(exchange.id for exchange in duplicates)
        removed += len(duplicates)
        if not np.isfinite(current_amount) or abs(current_amount - item.amount) > update_tol:
            delete_ids.append(current.id)
            insert_rows.append(
                dict_as_exchangedataset(
                    {
                        "input": tuple(item.flow.key),
                        "output": tuple(item.activity.key),
                        "amount": item.amount,
                        "type": "biosphere",
                    }
                )
            )
            updated += 1

    for exchanges in existing_by_key.values():
        delete_ids.extend(exchange.id for exchange in exchanges)
        removed += len(exchanges)

    if not delete_ids and not insert_rows:
        return added, updated, removed

    with sqlite3_lci_db.db.atomic():
        for identifiers in _chunks(delete_ids, 500):
            ExchangeDataset.delete().where(ExchangeDataset.id.in_(identifiers)).execute()
        for rows in _chunks(insert_rows, 125):
            ExchangeDataset.insert_many(rows).execute()
    databases.set_dirty(database_name)
    return added, updated, removed


def _sync_with_public_proxies(
    *,
    activities: list[Any],
    desired: list[DesiredExchange],
    managed_flow: Callable[[Any], bool] | None,
    update_tol: float,
) -> tuple[int, int, int]:
    """Portable fallback using only public Brightway proxy operations."""

    def activity_identity(activity: Any) -> Any:
        key = getattr(activity, "key", None)
        return tuple(key) if key is not None else id(activity)

    desired_by_activity: dict[Any, dict[tuple[str, str], DesiredExchange]] = {}
    for item in desired:
        desired_by_activity.setdefault(activity_identity(item.activity), {})[
            tuple(item.flow.key)
        ] = item

    added = updated = removed = 0
    errors: list[str] = []
    for activity in activities:
        wanted = desired_by_activity.get(activity_identity(activity), {})
        existing: dict[tuple[str, str], list[Any]] = {}
        for exchange in activity.exchanges():
            if exchange.get("type") == "biosphere":
                existing.setdefault(tuple(exchange.input.key), []).append(exchange)
        for flow_key, item in wanted.items():
            matches = existing.get(flow_key, [])
            if matches:
                current = matches[0]
                if (
                    not np.isfinite(float(current["amount"]))
                    or abs(float(current["amount"]) - item.amount) > update_tol
                ):
                    current["amount"] = item.amount
                    try:
                        current.save()
                        updated += 1
                    except Exception as exc:  # pragma: no cover - backend-specific
                        errors.append(
                            f"Failed to update {getattr(activity, 'key', activity)!r}: {exc}"
                        )
            else:
                try:
                    activity.new_exchange(
                        input=flow_key,
                        amount=item.amount,
                        type="biosphere",
                    ).save()
                    added += 1
                except Exception as exc:  # pragma: no cover - backend-specific
                    errors.append(f"Failed to create {getattr(activity, 'key', activity)!r}: {exc}")
        if managed_flow is None:
            continue
        for flow_key, exchanges in existing.items():
            stale = exchanges[1:] if flow_key in wanted else exchanges
            for exchange in stale:
                try:
                    if managed_flow(exchange.input):
                        exchange.delete()
                        removed += 1
                except Exception as exc:  # pragma: no cover - backend-specific
                    errors.append(f"Failed to remove {getattr(activity, 'key', activity)!r}: {exc}")
    if errors:
        raise RuntimeError(
            f"Brightway exchange synchronization failed ({len(errors)} errors): "
            + "; ".join(errors[:5])
        )
    return added, updated, removed
