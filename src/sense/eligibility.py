"""Shared activity eligibility, independent of prices and labour estimates."""

from collections.abc import Mapping, Sequence
from typing import Any


def normalized_name(value: Any) -> str:
    """Normalize case and whitespace without changing product punctuation."""
    return " ".join(str(value or "").casefold().split())


def exclusion_reason(
    activity: Mapping[str, Any],
    *,
    prefixes: Sequence[str] = (),
    terms: Sequence[str] = (),
    exact: Sequence[str] = (),
    structural: bool = True,
    empty_inventory: bool = False,
) -> str | None:
    """Identify bookkeeping nodes, retaining operational import activities.

    Configured exclusions are authoritative. The structural import rule applies
    to commodity descriptors (or electricity trade descriptors), not to actions
    such as ``evaporation of natural gas, import from ...``. Renaming is based
    on an explicit statement in the activity documentation, never a generic
    ``collection for final disposal`` suffix.
    """
    name = normalized_name(activity.get("name"))
    if name in {normalized_name(value) for value in exact}:
        return "configured_exact_name"
    if any(name.startswith(normalized_name(value)) for value in prefixes):
        return "configured_prefix"
    if any(normalized_name(value) in name for value in terms):
        return "configured_term"
    if not structural:
        return None
    if empty_inventory:
        return "empty_inventory"
    activity_type = normalized_name(activity.get("activity type"))
    if activity_type in {"market activity", "market group", "production mix", "import activity"}:
        return "aggregation_activity_type"
    if "this is a renaming activity" in normalized_name(activity.get("comment")):
        return "documented_renaming"
    if ", import from " in name:
        descriptor = name.split(", import from ", 1)[0]
        product = normalized_name(activity.get("reference product"))
        if descriptor == product or descriptor.startswith("electricity,"):
            return "commodity_import_node"
    return None


def is_treatment(activity: Mapping[str, Any], reference_amount: float) -> bool:
    """Recognize ecoinvent's negative waste reference or an explicit custom role.

    Eligibility must be checked separately: markets and renaming nodes can also
    have negative reference exchanges. Custom databases can declare
    ``sense_activity_role='treatment'`` when they use positive service outputs.
    """
    return reference_amount < 0 or activity.get("sense_activity_role") == "treatment"


def validate_activity_locations(
    activities: Sequence[Any],
    *,
    prefixes: Sequence[str],
    terms: Sequence[str],
    exact: Sequence[str],
) -> None:
    """Reject missing/null/blank locations on eligible processes before mutation."""
    invalid = []
    for activity in activities:
        if exclusion_reason(activity, prefixes=prefixes, terms=terms, exact=exact) is not None:
            continue
        location = activity.get("location")
        if not isinstance(location, str) or not location.strip():
            invalid.append(getattr(activity, "key", activity.get("code")))
    if invalid:
        raise ValueError(
            f"Eligible activities require a non-empty location: {invalid[:8]!r}. "
            "Set their inventory locations before running SENSE. No SENSE exchanges were changed."
        )
