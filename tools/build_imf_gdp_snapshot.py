"""Build the bundled IMF fallback from saved official DataMapper responses.

No network requests are made. Archive the API responses before running this
tool; their hashes, indicator metadata and transformations are recorded.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from decimal import Decimal
from pathlib import Path

import pycountry

INDICATOR = "NGDPD"
ALIASES = {"UVK": "XKX", "WBG": "PSE"}
NAME = "imf_weo_gdp_current_usd_2000_2024"


def build(data_path: Path, indicators_path: Path, countries_path: Path, output: Path) -> dict:
    indicator = json.loads(indicators_path.read_text(encoding="utf-8"))["indicators"][INDICATOR]
    if indicator["unit"] != "Billions of U.S. dollars" or indicator["dataset"] != "WEO":
        raise ValueError("Expected WEO NGDPD in billions of current US dollars")
    series = json.loads(data_path.read_text(encoding="utf-8"), parse_float=Decimal)["values"][
        INDICATOR
    ]
    countries = json.loads(countries_path.read_text(encoding="utf-8"))["countries"]
    rows = []
    for code, observations in sorted(series.items()):
        if code not in countries:
            continue  # Regional and analytical totals are not country GDP.
        iso3 = ALIASES.get(code, code)
        if iso3 != "XKX" and pycountry.countries.get(alpha_3=iso3) is None:
            raise ValueError(f"Unmapped IMF country code: {code}")
        for raw_year, value in sorted(observations.items()):
            year = int(raw_year)
            if not 2000 <= year <= 2024 or value is None:
                continue
            value = Decimal(str(value))
            if not value.is_finite() or value <= 0:
                raise ValueError(f"Invalid GDP for {code}/{year}")
            usd = value * Decimal(10**9)
            if not math.isfinite(float(usd)):
                raise ValueError(f"GDP outside numeric range: {code}/{year}")
            rows.append(
                {
                    "country_code": iso3,
                    "country_name": countries[code]["label"],
                    "year": year,
                    "gdp_current_usd": format(usd, "f"),
                    "imf_code": code,
                }
            )
    if not rows or len({(r["country_code"], r["year"]) for r in rows}) != len(rows):
        raise ValueError("Empty or duplicate country/year observations")
    rows.sort(key=lambda r: (r["country_code"], r["year"]))
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{NAME}.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "source_id": "imf_weo",
        "source": indicator["source"],
        "indicator": INDICATOR,
        "indicator_metadata": indicator,
        "original_unit": indicator["unit"],
        "unit": "current US dollars",
        "conversion_factor": 10**9,
        "year_range": [2000, 2024],
        "countries": len({r["country_code"] for r in rows}),
        "rows": len(rows),
        "country_code_aliases": ALIASES,
        "source_url": "https://www.imf.org/external/datamapper/NGDPD@WEO",
        "api_urls": {
            "data": "https://www.imf.org/external/datamapper/api/v1/NGDPD",
            "indicators": "https://www.imf.org/external/datamapper/api/v1/indicators",
            "countries": "https://www.imf.org/external/datamapper/api/v1/countries",
        },
        "input_sha256": {
            kind: hashlib.sha256(p.read_bytes()).hexdigest()
            for kind, p in [
                ("data", data_path),
                ("indicators", indicators_path),
                ("countries", countries_path),
            ]
        },
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "license": "IMF terms for use of statistical data; separate from SENSE MIT license",
        "terms_url": "https://www.imf.org/en/about/copyright-and-terms",
        "note": "Historical WEO estimates; not necessarily directly measured observations. "
        "Years after 2024 and regional totals excluded. SENSE converts billions to dollars "
        "and derives weights; these transformations are not IMF products.",
    }
    (output / f"{NAME}.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("indicators", type=Path)
    parser.add_argument("countries", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.data, args.indicators, args.countries, args.output), indent=2))
