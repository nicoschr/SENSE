"""Optional interactive LCA after mapping: python examples/calculate_scores.py.

Uses the project, database list and social namespace from apply_sense.py. Choose
the activity and compatible methods yourself; nothing is selected automatically.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Callable, Sequence
from typing import Any

import bw2data as bd
from apply_sense import DATABASES, PROJECT, SOCIOSPHERE_DB
from bw2calc import LCA


def choose(items: Sequence[Any], label: Callable[[Any], str], title: str) -> Any:
    """Select explicitly from a bounded list of labelled choices."""
    if not items:
        raise ValueError(f"No {title} available")
    print(f"\nChoose {title}:")
    for number, item in enumerate(items, 1):
        print(f"  {number}: {label(item)}")
    while True:
        value = input("Number: ").strip()
        if value.isdigit() and 1 <= int(value) <= len(items):
            return items[int(value) - 1]
        print(f"Enter a number between 1 and {len(items)}.")


def choose_method(keys: list[tuple], title: str) -> tuple:
    """Narrow method names before asking for an explicit numbered choice."""
    if not keys:
        raise ValueError(f"No {title} available. Check the project and completed mapping.")
    while True:
        query = input(f"\nSearch {title} (e.g. 'climate' or 'fatal'): ").strip().casefold()
        found = [key for key in keys if query in " | ".join(key).casefold()]
        if not found:
            print("No matches. Try another phrase.")
        elif len(found) > 20:
            print(f"{len(found)} matches. Enter a more specific phrase.")
        else:
            return tuple(choose(found, lambda key: " | ".join(key), title))


def main() -> int:
    """Calculate environmental and social scores for the same demand."""
    try:
        if PROJECT not in bd.projects:
            raise ValueError(f"Project {PROJECT!r} is absent. Check apply_sense.py settings.")
        bd.projects.set_current(PROJECT)
        missing = set(DATABASES).difference(bd.databases)
        if missing:
            raise ValueError(f"Configured databases are absent: {sorted(missing)}")
        database = choose(DATABASES, str, "a mapped process database")
        while True:
            query = input("\nSearch activity name: ").strip()
            if not query:
                print("Enter an activity name or part of one.")
                continue
            activities = bd.Database(database).search(query, limit=20)
            if activities:
                break
            print("No matches. Try another phrase.")
        activity = choose(
            activities,
            lambda act: (
                f"{act.get('name')} | {act.get('reference product')} | "
                f"{act.get('location')} | {act.get('unit')}"
            ),
            "an activity (up to 20 search results)",
        )
        amount = float(input(f"Reference-product amount in {activity['unit']} [1]: ") or "1")
        if not math.isfinite(amount) or amount <= 0:
            raise ValueError("This example requires a finite positive reference-product amount")
        environmental = choose_method(
            sorted(key for key in bd.methods if key[0] != "SENSE"), "an environmental method"
        )
        social = choose_method(
            sorted(key for key in bd.methods if key[:2] == ("SENSE", SOCIOSPHERE_DB)),
            "a SENSE method",
        )
        lca = LCA({activity.key: amount}, environmental)
        lca.lci()
        for method in (environmental, social):
            lca.switch_method(method)
            lca.lcia()
            print(" | ".join(method), ":", lca.score, bd.methods[method].get("unit", ""))
        return 0
    except (KeyboardInterrupt, EOFError):
        print("\nCalculation cancelled.", file=sys.stderr)
        return 1
    except Exception as error:
        print(f"\nCalculation stopped: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
