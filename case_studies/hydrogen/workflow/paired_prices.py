"""Establish common foreground prices from baseline inventories before SENSE runs."""

from __future__ import annotations

from typing import Any

import pandas as pd

from . import case_config, study_protocol
from .study_checks import check_pairs


def synchronize_prices(cfg, spec):
    import bw2data as bd

    from sense.artifacts import write_json_atomic
    from sense.data.ecoinvent import fetch_ecoinvent_activities
    from sense.utils.helpers import norm_code, slug

    bd.projects.set_current(cfg["brightway"]["project"])
    pairs = check_pairs(
        list(study_protocol.foregrounds("baseline").values()),
        list(study_protocol.foregrounds("regionalized").values()),
        spec["countries"],
    )
    config = study_protocol.sense_config(spec)
    output = case_config.path("data_dir", cfg) / "paired_hydrogen/prices"
    output.mkdir(parents=True, exist_ok=True)
    seeds = case_config.path("data_dir", cfg) / "Output/SeedPrices"
    seeds.mkdir(parents=True, exist_ok=True)
    mapping = pd.DataFrame(
        {
            "Shortname": spec["countries"],
            "PSILCA_mapping": spec["countries"],
            "simplified_ISO2": spec["countries"],
        }
    )
    prices = {}
    for database in study_protocol.foregrounds("baseline").values():
        database_config = study_protocol.sense_config(spec, database=database)
        data = fetch_ecoinvent_activities(
            mapping,
            str(output / f"{slug(database)}.json"),
            database,
            exclude_terms=config.exclude_terms,
            exclude_terms_starting_with=config.exclude_terms_starting_with,
            exclude_exact=database_config.exclude_exact,
            exclude_empty_inventories=config.exclude_empty_inventories,
            price_floor_percentile=config.price_floor_percentile,
            zero_price_policy=config.zero_price_policy,
            auto_seed_prices_path=str(seeds / f"seed_prices_{slug(database)}.json"),
            price_currency=config.ecoinvent_price_currency,
            price_year=config.ecoinvent_price_year,
            price_source_database=case_config.ecoinvent_database_name(cfg),
        )
        prices.update(
            {
                (database, norm_code(row.code)): None if pd.isna(row.price) else float(row.price)
                for row in data.itertuples()
            }
        )
    audit: list[dict[str, Any]] = []
    pending: list[tuple[Any, float]] = []
    seed_cache = {}
    for baseline, regionalized in pairs:
        key = (baseline.key[0], norm_code(baseline.key[1]))
        if key not in prices:
            continue  # Excluded activities have no paired direct inventory.
        price = prices[key]
        for activity in (baseline, regionalized):
            seed_file = seeds / f"seed_prices_{slug(activity.key[0])}.json"
            if seed_file.exists():
                from sense.data.prices import prod_key
                from sense.data.seed_prices import load_seed_template

                if seed_file not in seed_cache:
                    seed_cache[seed_file] = load_seed_template(
                        str(seed_file),
                        expected_currency=config.ecoinvent_price_currency,
                        expected_year=config.ecoinvent_price_year,
                    )
                manual = seed_cache[seed_file]
                override = manual.get(
                    prod_key(activity.get("unit", ""), activity.get("reference product", ""))
                )
                expected_price = 0.0 if price is None else price
                if override is not None and abs(override - expected_price) > 1e-10 * max(
                    1, expected_price
                ):
                    raise ValueError(f"Existing seed conflicts with paired price: {activity.key}")
        if price is None:
            # Physical treatment hours do not provide a monetary price to copy.
            # Validate both recipients instead of writing NaN activity metadata.
            from sense.brightway.repository import reference_production_amounts
            from sense.eligibility import is_treatment

            reference_amounts = reference_production_amounts(
                {activity.key: activity for activity in (baseline, regionalized)},
                preserve_sign=True,
            )
            for activity in (baseline, regionalized):
                if not is_treatment(activity, reference_amounts[activity.key]):
                    raise ValueError(f"Unpriced paired activity is not a treatment: {activity.key}")
                own = activity.get("price")
                if isinstance(own, dict):
                    own = own.get("amount")
                stale_generated = (
                    activity.get("sense_paired_price_basis") is not None
                    and activity.get("sense_paired_price_policy") != config.zero_price_policy
                )
                if own is not None and float(own) != 0 and not stale_generated:
                    raise ValueError(
                        f"Paired treatment has a conflicting supplied price: {activity.key}"
                    )
            audit.append(
                {
                    "baseline": list(baseline.key),
                    "regionalized": list(regionalized.key),
                    "price": None,
                    "basis": "shared physical treatment working-hour reference",
                }
            )
            continue
        pending.extend((activity, price) for activity in (baseline, regionalized))
        audit.append(
            {
                "baseline": list(baseline.key),
                "regionalized": list(regionalized.key),
                "price": price,
                "currency": config.ecoinvent_price_currency,
                "year": config.ecoinvent_price_year,
                "basis": "baseline SENSE price resolution",
            }
        )
    # Validate all proposed prices/seeds before writing any activity metadata.
    from bw2data.backends.peewee import sqlite3_lci_db

    with sqlite3_lci_db.db.atomic():
        for activity, price in pending:
            if (
                activity.get("price") != price
                or activity.get("sense_paired_price_policy") != config.zero_price_policy
            ):
                activity["price"] = price
                activity["sense_paired_price_basis"] = "baseline"
                activity["sense_paired_price_policy"] = config.zero_price_policy
                activity.save()
    write_json_atomic(
        output / "paired_prices.json", {"study": study_protocol.fingerprint(spec), "pairs": audit}
    )
    return audit
