# SENSE

<p align="center">
  <img src="docs/assets/SENSE_logo_text.svg" alt="SENSE logo" width="300">
</p>

<p align="center">
  <a href="https://www.python.org/downloads/">
    <img src="https://img.shields.io/badge/Python-3.11-blue.svg" alt="Python 3.11">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/Code%20licence-MIT-yellow.svg" alt="Code licence: MIT">
  </a>
</p>

SENSE adds PSILCA-derived working hours, social risk and social opportunity indicators
to your existing Brightway inventories. You can then calculate environmental and
social results for the same product system using your usual LCA workflow.

Use it with ecoinvent, custom backgrounds generated with [regioinvent](https://github.com/CIRAIG/Regioinvent), and your own
foregrounds. Sector matching is automatic. Social scores represent risk- or
opportunity-weighted working hours, not measured social impacts.

**For your own study, follow this guide.** To reproduce the manuscript figures,
database comparisons and Supporting Information, use the separate
[case-study guide](case_studies/hydrogen/README.md).

- [Install](#1-install)
- [Prepare your inputs](#2-prepare-your-inputs)
- [Map your project](#3-map-your-project)
- [Custom databases and foregrounds](#4-custom-databases-and-foregrounds)
- [Change the main parameters](#5-change-the-main-parameters)
- [Calculate results](#6-calculate-results)
- [How SENSE works](#how-sense-works)

## 1. Install

SENSE 1.1.0 uses **Python 3.11 and Brightway 2 (`bw2data` 3.6.6)**. Brightway 2.5
projects are not supported. Install Git and Conda. On macOS, also install Xcode
command line tools (`xcode-select --install` if absent).

```bash
git clone --branch main https://github.com/nicoschr/SENSE.git sense
cd sense
```

Create the environment for your platform:

| Platform | Command |
| --- | --- |
| Windows / Linux x86-64 | `conda env create --file environment.yml` |
| macOS, Intel / Apple Silicon | `conda env create --file environment-macos.yml` |

```bash
conda activate sense
python -m pip check
sense --version
```

These environments include SENSE, the ecoinvent importer, Java 21 and a fast sparse
solver: PARDISO on Windows/Linux and UMFPACK on macOS.
Run the following terminal commands from this repository root.
[Other installation options](docs/package-reference.md#other-installation-options)
are documented separately.

## 2. Prepare your inputs

| You need | What to provide |
| --- | --- |
| Database licences | PSILCA and the inventory databases you use. SENSE does not include them. |
| Your Brightway project | An existing project containing the process databases to assess. SENSE does not import or regionalize them. |
| PSILCA database | **PSILCA** (licensed, not included). Use the openLCA database **PSILCA v4 - Professional**, as an openLCA backup file (`.zolca`). Download it from [Nexus](https://nexus.openlca.org/) and leave it unextracted. |
| Geography workbook | The complete **Database Overview** workbook for your ecoinvent release, including its `Geographies` sheet. For v3.12, download it from the [ecoinvent release page](https://support.ecoinvent.org/ecoinvent-version-3.12), under **Relevant Documents and Files → Database Overview File**. For other versions, use the corresponding release page. |
| Output directory | A writable folder for SENSE caches and diagnostics. The example uses `Data`. |

The input files can stay wherever you downloaded them. Enter their actual paths
in the example below. You do not need openLCA or a manual XML export, and no
separate ecoinvent `.spold` directory is required.

Custom activities need meaningful activity and reference-product names, a location,
a physical reference unit and valid reference production. SENSE can resolve many
missing prices automatically. See [custom databases](#4-custom-databases-and-foregrounds)
for the requirements and corrective steps. You do not need to assign PSILCA sectors
or ISIC codes manually for default mapping.

If you have not imported an inventory yet, follow the
[optional ecoinvent import instructions](docs/package-reference.md#import-ecoinvent-if-needed).
An existing project can be used from the new environment provided it points to the
same Brightway data directory. A new environment does not create a new project.

## 3. Map your project

Open [examples/apply_sense.py](examples/apply_sense.py). Edit the settings at the
top to match your project, exact database names and downloaded files:

```python
PROJECT = "my-sense-study"
DATABASES = ["ecoinvent-3.12-cutoff"]
PRICE_SOURCE_DB = "ecoinvent-3.12-cutoff"
PSILCA_PATH = Path("Data/psilca/psilca.zolca")
GEOGRAPHIES = Path("Data/Database-Overview-for-ecoinvent-v3.12.xlsx")
ECOINVENT_VERSION = "3.12"
PSILCA_VERSION = "4.0"
DATA_DIR = Path("Data")
SOCIOSPHERE_DB = "sociosphere"
CONFIG = SenseConfig()
```

Use your own filenames. Python paths can use forward slashes on Windows, for
example `Path("C:/Data/my-database.zolca")`. Start with `SenseConfig()` defaults.
SENSE creates `sociosphere` and its methods automatically.

Run:

```bash
python examples/apply_sense.py
```

The script calls the public API for each database in your list:

```python
results = run_sense(
    project=PROJECT,
    technosphere_db=database,
    sociosphere_db=SOCIOSPHERE_DB,
    price_source_db=PRICE_SOURCE_DB,
    psilca_path=PSILCA_PATH,
    ecoinvent_version=ECOINVENT_VERSION,
    psilca_version=PSILCA_VERSION,
    ecoinvent_geographies_excel=GEOGRAPHIES,
    data_dir=DATA_DIR,
    config=CONFIG,
)
```

SENSE adds or updates its own social exchanges in the selected databases. Existing
technosphere and environmental biosphere exchanges are preserved. The first run
reads PSILCA and downloads the embedding model if it is not cached. Reruns reuse
valid caches. Keep mapping and other project edits sequential.

### Which databases should I map?

Map every process database supplying the product system you will calculate.
SENSE processes **one database per call**. It does not map linked databases or
change their links automatically. Do not add biosphere databases to `DATABASES`.

For a foreground linked to ordinary ecoinvent:

```python
DATABASES = ["ecoinvent-3.12-cutoff", "my-foreground"]
```

For a foreground linked to a regioinvent supply chain:

```python
DATABASES = [
    "ecoinvent-3.12-cutoff-regionalized",
    "regioinvent",
    "my-regionalized-foreground",
]
```

Replace these illustrative names with those in your project. Regioinvent normally
uses both its own database and a linked regionalized ecoinvent database, so map
both. The original ecoinvent can remain `PRICE_SOURCE_DB`, even if it is not in
this supply chain. Add it to `DATABASES` if another product system uses it.

Use the same configuration and `SOCIOSPHERE_DB` for all contributing databases.
Only the databases in your list are mapped. Check that foreground exchanges already
link to the background scenario you intend to assess.

## 4. Custom databases and foregrounds

Use the same mapping script. Set `PRICE_SOURCE_DB` to the imported background
providing reference prices, normally your original ecoinvent database. This also
sets the working-hour reference for unpriced treatments. Custom databases can use
`None` to supply their own reference data. Merely selecting a price source does
not map that database.

Before mapping custom activities, check:

| Field | Practical requirement |
| --- | --- |
| `name`, `reference product` | Describe the actual process and product. Labels such as `process 1` give little information for sector matching. |
| `location` | Use an ecoinvent location or country code such as `DE`. A custom code needs the country assignment shown below. Blank locations must be corrected in Brightway. |
| `unit`, production exchanges | Use consistent reference units and valid production. Unallocated multiple outputs and zero reference production require correction. |
| Prices | Supplied prices must use the configured currency and year, normally **EUR 2005 per reference-product unit**. Explicit zero means zero for ordinary activities. |

For missing prices, SENSE first uses available reference
prices and purchased-input cost inference. That inference does not add value added
or profit. Review inferred prices where they influence your conclusions.

### Activities without physical inputs or environmental exchanges

By default, SENSE excludes activities with neither nonzero technosphere inputs nor
nonzero environmental exchanges from direct social mapping. Production exchanges
and social exchanges added by SENSE do not count. Their supply-chain links remain.

If your foreground intentionally contains such activities, for example labour-only
services, disable this rule in the configuration passed to that mapping call:

```python
CONFIG = SenseConfig(exclude_empty_inventories=False)
```

The CLI equivalent is `sense run --include-empty-inventories`. Other eligibility
rules still apply. Disabling the rule also allows empty activities into the
working-hour reference pool if their prices are positive. Keep the chosen reference
configuration consistent across compared systems.

### Supply your own prices

To use your own prices instead of automatic estimates, save them on the relevant
Brightway activities before running SENSE:

```python
import bw2data as bd

bd.projects.set_current("my-sense-study")
activity = bd.get_activity(("my-foreground", "activity-code"))
activity["price"] = 2.0  # Illustrative EUR 2005 per reference-product unit.
activity.save()
```

Replace the project, database and activity code with your own identifiers. Use the
configured currency and year, normally EUR 2005, and the activity's reference-product
unit. These prices take precedence over reference prices and input-cost estimates.
An existing [product-price entry](#if-sense-requests-a-price) takes precedence over
the activity price. Rerun SENSE after adding or changing prices.

### If SENSE requests a price

It stops and prints the exact JSON worklist path under `Data/Output/SeedPrices/`.
In that existing file, fill `price` for every entry marked `"required": true`.
Use the currency and year declared in its metadata. For example, a filled entry
for an **invented** product might be:

```json
{
  "unit": "kilogram",
  "reference_product": "Example catalyst",
  "price": 2.0,
  "currency": "EUR",
  "year": 2005,
  "source": "Reviewed example estimate",
  "required": true
}
```

Edit the generated entries rather than replacing the whole file with this example.
Keep their product names and units unchanged. Then rerun
`python examples/apply_sense.py`. The supplied seed overrides automatic price
estimation for that product/unit.

### If SENSE requests a location assignment

It prints a JSON worklist path under `Data/Output/LocationConcordances/`.
Fill `countries` for each required entry with the full list of constituent ISO2
country codes, separated by semicolons. For an invented site in Germany:

```json
{
  "location_code": "MY_SITE",
  "countries": "DE",
  "comment": "Site is in Germany",
  "required": true
}
```

For a location comprising Germany and France, include both `DE` and `FR`.
Edit the generated entry and rerun the same script. Price and location
input requests stop before social exchanges are written for the current database.
Earlier databases in the list may already be complete.

### Other common situations

| What you see | What to do |
| --- | --- |
| Project or database absent | Check exact names. The script lists available names. If your project is missing, [point the environment to its Brightway root](docs/package-reference.md#find-an-existing-project). |
| Low-confidence sector matches | Inspect the reported activities and their descriptions. These matches are flagged, not rejected. Review them before interpreting results. |
| Invalid price | Correct negative, malformed or non-finite prices. A current-year price must be converted to the configured basis before use. |
| Treatment without a price | The default estimates working hours directly from comparable units in the reference database. Review this assumption, or change the fallback setting below. |
| Missing treatment reference for a unit | Supply suitable reference data or a justified treatment price, or explicitly disable the fallback. See the [treatment reference](docs/package-reference.md#treatment-activities-and-zero-prices). |
| Interrupted or failed mapping | Resolve the reported cause and rerun. A whole run is not one transaction. Do not calculate from an unfinished mapping. |

Markets and other bookkeeping nodes receive no direct social inventory. Their
upstream contributions remain. Treatment activities remain eligible. Details of
[price handling and exclusions](docs/package-reference.md#prices-and-eligible-activities) are in the reference.

## 5. Change the main parameters

In `apply_sense.py`, replace `CONFIG = SenseConfig()` with, for example:

```python
CONFIG = SenseConfig(
    social_flow_mean="arithmetic",
    country_weighting="equal",
    similarity_delta=0.028,
    treatment_working_hours_percentile=20,
)
```

This is an alternative configuration, not a recommendation to change the defaults.

| Parameter | Default | What changing it does |
| --- | --- | --- |
| `social_flow_mean` | `"geometric"` | `"arithmetic"` changes how social intensities are averaged across retained sectors and countries. Working hours always use arithmetic averaging. |
| `country_weighting` | `"gdp"` | `"equal"` gives mapped countries equal weight when an inventory location covers several countries. |
| `similarity_delta` | `0.028` | Increasing it retains more sector candidates near the best match. It does not necessarily improve matching. |
| `level_weights` | PSILCA-derived factors with documented SENSE choices | Changes the numerical factors assigned to risk/opportunity levels, including missing assessments. |
| `treatment_working_hours_percentile` | `20` | Changes estimated direct hours for unpriced treatments. `0` disables those direct fallback contributions, leaving their upstream inputs in the supply chain. |

To change one factor while retaining the others:

```python
CONFIG.level_weights["no data"] = 1.0
```

Rerunning replaces that social layer's owned exchanges. To retain both versions,
change `SOCIOSPHERE_DB`, for example to `"sociosphere_arithmetic"`, and map **all**
contributing databases again with the alternative settings. Select methods from the
matching namespace when calculating.

[Advanced settings](docs/package-reference.md#advanced-settings) include monetary
alignment, diagnostic thresholds, external labour-productivity data, custom
encoders and partial runs. Keep the default full geographic scope for complete
supply-chain calculations.

## 6. Calculate results

After mapping succeeds, SENSE methods are available in your Brightway project.
Continue using your usual Brightway calculation workflow or a compatible Activity
Browser installation. Environmental and social methods use the same functional
unit and inventory model. Regioinvent environmental calculations require methods
compatible with its spatialized biosphere.

For a guided Python calculation using the settings already entered in
`apply_sense.py`, run:

```bash
python examples/calculate_scores.py
```

Choose a mapped database, search for your activity, select it from the labelled
results and enter the reference-product amount. Then search and select an
appropriate environmental method and a SENSE method. The example prints both
scores with their units. It does not choose an activity or indicator for you.
There is no intermediate results ZIP required for package use.

Your existing Python workflow can access the created methods with:

```python
import bw2data as bd

bd.projects.set_current("my-sense-study")
for method in sorted(bd.methods):
    if method[:2] == ("SENSE", "sociosphere"):
        print(method)
```

The mapping script prints the path to `diagnostics.json`. Check low-confidence
matches, source-data omissions and economic assumptions before interpreting scores.
Completed run records and caches are under `DATA_DIR/Output/`.
Keep environmental and social scores in their own units. See the
[interpretation guidance](docs/package-reference.md#interpreting-results) for modelling limitations and checks.

## How SENSE works

SENSE matches each process activity to country-specific PSILCA sectors. Product
prices and sector labour intensities estimate direct working hours. Risk or
opportunity factors turn these into indicator-specific weighted hours, which are
aggregated and added as Brightway biosphere exchanges and methods. Supply-chain
calculations follow your existing technosphere.

The [technical reference](docs/package-reference.md) explains geography, semantic
matching, the price hierarchy, monetary alignment, treatment fallback, aggregation,
external productivity inputs, cache behaviour and run evidence. Existing EcoSpold01
XML inputs and the `sense run` CLI are also documented there. The optional
[notebook tutorial](docs/package-reference.md#optional-notebook) uses the same API.

## License and citation

SENSE's code is licensed under the [MIT License](LICENSE). External databases,
model weights and bundled resources retain their own terms. The bundled IMF GDP
fallback is governed by [IMF's data-use terms](https://www.imf.org/en/about/copyright-and-terms),
which direct users to request permission for potential commercial reuse.
See the [third-party notices](src/sense/resources/THIRD_PARTY_NOTICES.txt) for sources and conditions.
Use [CITATION.cff](CITATION.cff) and cite the exact version or commit used.

## Contact

For questions about SENSE or the hydrogen case study, including requests for the
prepared inventory workbooks, contact either Nicolas Schröder
(<nicoschr@ethz.ch>) or Fabian Lechtenberg (<fabian.lechtenberg@upc.edu>).
Report bugs through [GitHub Issues](https://github.com/nicoschr/SENSE/issues).
