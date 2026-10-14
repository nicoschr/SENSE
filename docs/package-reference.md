# SENSE reference

Start with the [package guide](../README.md) to map your own project. This reference
covers alternative setup, advanced settings, model assumptions and saved outputs.
The [configuration class](../src/sense/config.py) documents every Python option.

## Other installation options

In an existing **Python 3.11 / Brightway 2** environment, install from the repository
root:

```bash
python -m pip install -e ".[ecoinvent-import]"
conda install -c conda-forge openjdk=21
```

Reactivate after adding Java. Pip does not install Java or a fast sparse solver and
is not equivalent to the platform Conda environments. Brightway 2.5 is unsupported.

### Find an existing project

From an environment that sees your project, print its registry and storage root:

```bash
python -c "from pathlib import Path; import bw2data as bd; print(bd.projects); print(Path(bd.projects.dir).parent)"
```

Use the same storage root in the SENSE environment. Set `BRIGHTWAY2_DIR` before
starting Python if needed:

```bash
export BRIGHTWAY2_DIR="/path/to/your/brightway-root"
```

Windows PowerShell:

```powershell
$env:BRIGHTWAY2_DIR = "C:/path/to/your/brightway-root"
```

This is the root containing the project registry, not an individual project folder.
Verify the project list again. The mapping example refuses absent projects.

### Optional notebook

The scripts in the main guide do not require Jupyter. To use
[getting_started.ipynb](../examples/getting_started.ipynb) instead:

```bash
python -m pip install -e ".[tutorial]"
python -m ipykernel install --user --name sense --display-name "SENSE"
python -m jupyterlab examples/getting_started.ipynb
```

Select the **SENSE** kernel and supply your project, paths, activity and method keys.
The notebook maps the selected database through the same public API.

## Import ecoinvent if needed

Skip this if the inventory already exists. Set `ECOINVENT_USERNAME` and
`ECOINVENT_PASSWORD` in your terminal, keeping credentials out of source files.
Use `export NAME="value"` on macOS/Linux or `$env:NAME = "value"` in PowerShell.
Save this as `import_inventory.py` and run `python import_inventory.py`:

```python
import os
import bw2data as bd
import bw2io as bi

bd.projects.set_current("my-sense-study")
if "ecoinvent-3.12-cutoff" not in bd.databases:
    bi.import_ecoinvent_release(
        version="3.12",
        system_model="cutoff",
        username=os.environ["ECOINVENT_USERNAME"],
        password=os.environ["ECOINVENT_PASSWORD"],
        use_mp=False,
    )
```

Choose your intended project and inventory release. Brightway stores inventories
outside the checkout. SENSE does not import ecoinvent or construct regioinvent.

## How mapping works

1. **Geography:** inventory locations are resolved to PSILCA countries. Composite
   locations use GDP or equal weights. Custom codes require an explicit assignment.
2. **Matching:** activity names, reference products and available ISIC labels are
   compared with country-specific sector labels using text embeddings. The best
   match and candidates within `similarity_delta` are retained.
3. **Direct quantities:** aligned product prices and sector labour intensities
   estimate working hours. Assessment-level factors weight these hours for each
   indicator. Unpriced treatments use the separate fallback below.
4. **Calculation:** retained sector/country values are averaged and added as
   Brightway exchanges and methods. Supply-chain propagation follows the existing
   technosphere. PSILCA's Eora supply-chain matrix is not imported.

Map every contributing process database with consistent settings. Mapping does not
change providers or automatically map linked databases. Social scores are
risk- or opportunity-weighted modelled hours, not measured harm or benefits.
Environmental and social scores remain separate.

## Command line

The CLI maps a single database through the same pipeline:

```bash
sense run --help
sense run --project my-sense-study --technosphere-db ecoinvent-3.12-cutoff --psilca-path Data/psilca/psilca.zolca --ecoinvent-version 3.12 --psilca-version 4.0 --ecoinvent-geographies-excel Data/Database-Overview-for-ecoinvent-v3.12.xlsx --data-dir Data
```

It exposes price-source, treatment, diagnostic and working-hours-only options.
Use the Python API to change aggregation, weighting, assessment factors or matching.

## PSILCA input

**PSILCA** (licensed, not included). Use the openLCA database **PSILCA v4 - Professional**,
as an openLCA backup file (`.zolca`). The case study was run with
`psilca_professional_work_hours_cutoff_1.0E-7_20250624-clean.zolca`. Library ZIPs and
variants in other units are rejected. A variant with a different cutoff or release date
runs, but its results will differ from the paper.

Supply the backup through `psilca_path`. Java 21+ is required. Import reads the backup locally without openLCA, manual
export or full extraction. A Library ZIP lacks the original working-hour scale
and is not accepted. Unsupported schemas stop import rather than producing a
partial replacement cache.

EcoSpold01 XML directories are also accepted through `psilca_path`. The legacy
`psilca_dir` keyword remains supported, but cannot be combined with `psilca_path`.
XML and working-hours-only runs require no Java.

Use trusted original backups and trusted model providers. The bundled Derby engine
is retired, and compatibility dependencies retain known advisories. Native reading
is read-only and resource-limited, but is not a sandbox for arbitrary archives.
Do not expose the importer as an upload service. An abrupt termination can leave
`sense-zolca-*` folders in the system temporary directory for manual cleanup.

## Advanced settings

Set these through `SenseConfig` in your mapping script. The main guide explains
aggregation, country weighting, matching tolerance, risk factors and treatment hours.

| Setting | Use |
| --- | --- |
| `iso2_selection` | Restrict mapping to specified regions. Use full scope for a complete supply-chain assessment. |
| `iso2_start_at` | Resume from a region in activity-list order. Cannot be combined with `iso2_selection`. |
| `minimum_similarity` | Flag low-confidence best matches without discarding them. Default: `0.5`. This is not a probability of correct matching. |
| `embedding_model`, `embedding_revision` | Choose an encoder and immutable provider commit. Built-in BGE is pinned. Custom models require a 40-character commit and SafeTensors weights. Local paths and repository Python modules are unsupported. |
| `country_gdp_path` | Replace bundled GDP data with your own source. It is not supplemented automatically. |
| `enable_social_flows`, `enable_working_hours_flows` | Select which quantities to write. |
| `enable_isic_working_hours_methods` | Add working-hour methods by native two-digit ISIC division. Requires working-hour flows. |
| `rebuild_cache` | Rebuild derived caches after out-of-band input edits. |

GDP weighting uses World Bank data first, then IMF WEO when no World Bank value is
available within the nearest-year window (five years by default). Both bundled
sources cover 2000–2024. A year tie prefers the later observation. Countries without
usable GDP are omitted from multi-country weights and recorded in the audit.
GDP is a geographical proxy, not activity-specific production. Use `"equal"` to
avoid GDP weighting. RoW/RoE membership excludes countries represented by other
locations of the same activity/product, then intersects with PSILCA coverage.
Subnational locations represent their parent country for these exclusions.

Distinct PSILCA industry and commodity processes can share a country-sector label.
All qualifying processes enter aggregation separately, without a default preference.
To select a reviewed process explicitly:

```python
from sense import SenseConfig

config = SenseConfig(
    psilca_process_overrides={"CH | Example sector": "reviewed-process.xml"},
)
```

The filename must match the source identity. Native imports retain logical
`process_<UUID>.xml` identities without opening or creating XML files.

## Prices and eligible activities

Bookkeeping nodes receive no direct social inventory, but their upstream inputs
remain. Exclusions cover markets, regioinvent production/generic markets,
production/consumption/technology mixes, commodity import nodes and documented
renaming activities. Operational import processes such as LNG evaporation remain
eligible. By default, activities with neither nonzero technosphere inputs nor
nonzero environmental exchanges are excluded in every database. The check ignores
production exchanges, zero amounts and existing SENSE social exchanges. Excluded
activities remain in the supply chain and available as price sources, but do not
enter the treatment working-hour reference pool. For intentionally empty custom
activities, such as labour-only services, set `exclude_empty_inventories=False`
in `SenseConfig`, or use `sense run --include-empty-inventories` on the CLI.
This disables only the empty-inventory rule for that run.
Treatment activities with nonempty physical inventories remain eligible.

Set `price_source_db` to the original background when mapping transformed or
foreground databases. Otherwise, reference data come from the target database.
Price resolution follows this order: reviewed user seed, activity price,
reference-exchange price, exact source activity, source-product average, then
purchased-input cost inference. The last option adds no value added or profit.
Explicit zeros do not trigger cost inference. Negative, malformed or non-finite
prices require correction.

### Treatment activities and zero prices

| Activity | Direct quantities |
| --- | --- |
| Eligible activity with a positive price | Aligned price × monetary labour intensity, then indicator factors. |
| Eligible treatment with a zero or missing price | Physical working-hour fallback, then indicator factors. |
| Other eligible activity with an explicit zero price | Zero. |
| Other eligible activity with a missing price | Price resolution and purchased-input cost inference. |
| Bookkeeping node | None. Upstream contributions remain. |

A negative reference production identifies treatment after eligibility filtering.
Custom positive-output treatment services can set
`activity["sense_activity_role"] = "treatment"` and save the activity. A name
containing “waste” alone does not establish treatment.

The default `zero_price_policy="treatment_working_hours"` uses the 20th percentile
of positive ordinary direct hours per physical unit in the complete reference
database. There is one observation per eligible priced activity, linear
interpolation, and no fallback recipients among the donors. `price_source_db`
selects this reference even when target mapping is geographically restricted.
Use the same reference across compared backgrounds and foregrounds.

These hours bypass price and monetary labour intensity. They are a heuristic, not
observed treatment labour. Missing unit-specific donors stop processing. Change
`treatment_working_hours_percentile` to test this assumption. **`0` disables the
direct fallback**, rather than selecting the minimum. Treatment upstream inputs
and positively priced treatment calculations remain. Rerunning synchronizes the
selected regions' owned exchanges without manual cache deletion.

`zero_price_policy="legacy_price_floor"` instead uses the former price heuristic
with `price_floor_percentile`. `zero_price_policy="zero"` leaves explicit zero
prices without direct quantities. Neither reproduces every assumption of an older
release. CLI options expose the same policy and percentile settings.

## Monetary basis

ecoinvent prices default to EUR 2005. PSILCA labour intensities default to USD with
a representative year selected from the declared PSILCA release. That year is an
approximation, not a verified common year for every observation. Other versions
require an explicit year. Declare different bases through:

```python
config = SenseConfig(
    ecoinvent_price_currency="EUR",
    ecoinvent_price_year=2005,
    psilca_price_currency="USD",
    psilca_price_year=2015,
)
```

Use `ecoinvent_to_psilca_price_factor` for a justified conversion not covered by the
built-in data. Supplied prices must match the configured currency, year and
reference-product unit. Keep their valuation basis consistent with the social data.
The run's monetary audit records the bases and resolved or supplied multiplier.

## Optional productivity baseline

An external Parquet file can replace PSILCA monetary labour intensities while
retaining PSILCA risk/opportunity levels. Required columns are:

| Column | Format |
| --- | --- |
| `country` | Uppercase ISO3 code. |
| `year` | Integer. |
| `isic4_code` | Two-digit ISIC Rev.4 division string, including leading zeros. |
| `productivity_hrs_per_usd` | Working hours per USD, numeric and finite. |

Country/division/year keys must be unique. Missing or nonpositive observations do
not supply an intensity. Lookups use the nearest year within five years, preferring
the earlier year on a tie. Declare the monetary conversion explicitly:

```python
from pathlib import Path

config = SenseConfig(
    productivity_baseline_path=Path("productivity.parquet"),
    productivity_year=2020,
    productivity_eur_per_usd=0.84,  # Illustrative: justify for your data.
)
```

Unavailable or incomplete productivity coverage retains PSILCA hours and is reported
in the audit. With `use_psilca=False`, complete coverage is required instead.
Uncovered locations can use explicit ISO3 donor countries through
`productivity_location_proxies`. No donor proxies are supplied by default.

The default `productivity_gdp_policy="strict"` requires complete constituent-country
coverage. `"available_countries"` explicitly permits regional averages over usable
GDP countries, renormalizing weights and recording omissions. It requires a
productivity baseline and GDP weighting. Missing productivity among retained
countries, no usable GDP or an unmapped location still stops preflight.

## Outputs and reuse

```text
Data/Output/
├── SeedPrices/             requested price inputs
├── LocationConcordances/   requested geographic assignments
├── Ecoinvent/              activity, location and price caches
├── PSILCA/                 assessment cache and sector list
├── EmbeddingStore/         cached vectors
├── Diagnostics/            latest quality report
├── Latest/                 pointers to completed runs
└── Runs/<run-id>/          saved inputs, settings and diagnostics
```

Resolve requested inputs as described in the main guide and rerun. A successful
rerun updates SENSE-owned exchanges without duplicates, including removal for
newly excluded activities or disabled outputs. It preserves physical exchanges.
A whole mapping run is not one transaction. Resolve failures before calculating.

Region summaries return `run_directory`. Keep this evidence, the configuration,
monetary audit and diagnostics with your results. Completed study runs also check
source/configuration identity and live inventory state before reusing calculations.
See the [study reference](../case_studies/hydrogen/REFERENCE.md#result-archive) for
its calculation archive. Changed settings require remapping, not just redrawing.

Caches normally detect source changes through archive or filesystem metadata.
After out-of-band edits, use `rebuild_cache=True`. Embedding caches also record the
encoder revision. Keep source-derived caches and snapshots private under the input
licences. Diagnostic details are for local review, not unrestricted publication.

Methods start with `("SENSE", sociosphere_db)`. Each risk/opportunity indicator has
its own method. Total working hours use:

```python
("SENSE", "sociosphere", "Social", "Working hours", "working hours")
```

Change the namespace to retain an alternative model, then remap every contributing
database with its corresponding settings. Methods and cleanup are scoped to that
namespace.

## Interpreting results

Amounts are per reference-product unit. Brightway exchange writing accounts for the
production amount. Single negative treatment outputs use absolute throughput with
their physical sign retained. Unallocated multiple outputs and zero/non-finite
reference production are rejected. Substitution credits inherit their sign from
the supply vector.

Risk factors follow the PSILCA scale, with a SENSE value of `0.001` for `no` so
geometric aggregation remains positive. The
[PSILCA v3 manual](https://psilca.net/wp-content/uploads/2020/06/PSILCA_documentation_v3.pdf)
reports zero for that class. The default `no data = 0.1` is an assumption, not evidence
of low risk. Geometric averaging is not an additive expected exposure.

Before interpreting differences, review influential matches, inferred prices,
treatment assumptions and missing assessments. Test aggregation, country weights,
productivity or missing-data factors when they influence a conclusion. SOCA
agreement compares models using related inputs and is not independent validation
of social outcomes. Software tests likewise do not validate the source observations.
