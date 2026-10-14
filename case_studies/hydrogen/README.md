# Reproduce the SENSE hydrogen study

This guide starts with **no SENSE environment and no Brightway study project**.
It builds and maps the original and regionalized backgrounds, constructs paired
hydrogen inventories, calculates the results, and exports the manuscript plots.
The expensive database-wide regionalization comparison comes last.

The fixed study uses ecoinvent 3.12 cutoff, PSILCA 4.0, Regioinvent 1.4.0.2,
five hydrogen routes, and Germany, China and the United States. The hydrogen figure
shows climate change, children in employment (total), and fatal-accident risk
scores. A second figure shows US fatal-accident contributions. The SOCA comparison
also requires SOCA v5 for ecoinvent 3.12.

Use this guide for the study. Use the [package README](../../README.md) to apply
SENSE to your own databases. Scientific choices are fixed in
[study_protocol.py](workflow/study_protocol.py) and explained in the
[study reference](REFERENCE.md#fixed-study).
Local configuration supplies paths, project names and credentials.

**Run terminal commands from the repository root, one at a time. Continue only
when the previous command succeeds.** These are Python scripts, not notebooks.

Use the same code revision throughout the run. If you update SENSE between
mapping and calculation or comparison stages, follow
[Refresh mapping after code or assumption changes](#refresh-mapping-after-code-or-assumption-changes)
before continuing. Saved runs from an older revision may no longer be compatible.

- [1. Install](#1-install)
- [2. Configure inputs](#2-configure-inputs)
- [3. Fetch trade data and check setup](#3-fetch-trade-data-and-check-setup)
- [4. Build backgrounds](#4-build-backgrounds)
- [5. Map backgrounds and build paired foregrounds](#5-map-backgrounds-and-build-paired-foregrounds)
- [6. Calculate and export both hydrogen figures](#6-calculate-and-export-both-hydrogen-figures)
- [7. Supporting Information figures and tables](#7-supporting-information-figures-and-tables)
- [8. SOCA benchmark](#8-soca-benchmark)
- [9. Database-wide regionalization comparison](#9-database-wide-regionalization-comparison)
- [Existing projects, groups and recovery](#existing-projects-groups-and-recovery)

## 1. Install

Install Git and Conda. On macOS, install Xcode command line tools before creating
the environment (`xcode-select --install` if absent). Clone the public SENSE
repository at the v1.1.0 release tag to reproduce this study:

```bash
git clone --branch v1.1.0 https://github.com/nicoschr/SENSE.git sense
cd sense
git rev-parse HEAD
```

Keep the printed revision with your results. Use only the environment command
for your platform:

| Platform | Command | Solver |
| --- | --- | --- |
| Windows / Linux x86-64 | `conda env create --file environment.yml` | PARDISO |
| macOS, Intel / Apple Silicon | `conda env create --file environment-macos.yml` | UMFPACK |

```bash
conda activate sense
python -m pip install -e ".[case-study]"
python -m pip check
```

The case-study extra is required in addition to the base environment. Keep this
terminal in the repository root. Both environment files also install Java 21 for
native PSILCA import.
Run Activity Browser from a separate terminal/environment.

Check the installed numerical environment using **one** command:

| Platform | Command |
| --- | --- |
| Windows / Linux x86-64 | `python tests/run_setup_smoke.py --solver pardiso --output case_studies/hydrogen/Data/setup-evidence` |
| macOS, Intel / Apple Silicon | `python tests/run_setup_smoke.py --solver umfpack --output case_studies/hydrogen/Data/setup-evidence` |

This uses a temporary Brightway project to check imports, the solver, known LCA
scores and PDF/SVG export. It does not build the study project.

## 2. Configure inputs

Create the local configuration. This command works in macOS/Linux terminals and
Windows PowerShell, and retains an existing configuration:

```bash
python -c "from pathlib import Path; p = Path('case_studies/hydrogen/config.local.yaml'); p.write_bytes(p.with_name('config.yaml.template').read_bytes()) if not p.exists() else print('Keeping existing configuration')"
```

Edit `case_studies/hydrogen/config.local.yaml`:

| Setting | Required value |
| --- | --- |
| `brightway.project` | A **new** project name for a fresh build, e.g. `sense_hydrogen_ei312` if absent. |
| `ecoinvent_credentials.username` / `.password` | ecoQuery credentials as quoted strings, or leave both null and set `ECOINVENT_USERNAME` / `ECOINVENT_PASSWORD` in the terminal. |
| `paths.ecoinvent_geographies_excel` | Licensed ecoinvent 3.12 Database Overview workbook with `Geographies` and, for the SOCA benchmark, `Cut-Off AO`. Use the complete workbook. |
| `paths.psilca_path` | PSILCA v4 - Professional, openLCA backup (`.zolca`). Study file: `psilca_professional_work_hours_cutoff_1.0E-7_20250624-clean.zolca`. Supply the file path without unzipping it. |
| `paths.hydrogen_inventory` / `paths.methanol_inventory` | The **prepared ecoinvent-3.12 foreground workbooks** described below. |
| `paths.data_dir` | Output/cache directory. Default: `Data` inside this case-study folder. |
| `paths.soca_dir` | Folder with the SOCA v5 (ecoinvent 3.12) processes exported from openLCA as EcoSpold, see [step 8](#8-soca-benchmark). Needed only for step 8. Set null if not using SOCA. |

Relative configuration paths resolve from `case_studies/hydrogen`, not the repository
root. Absolute paths also work. The trade database has a separate fixed download
location: `case_studies/hydrogen/Data/regioinvent/trade_data.db`.

Ecoinvent is downloaded and imported by step 4. **No manual ecoinvent `.spold`
export is needed.** PSILCA is read directly from its `.zolca` backup, so PSILCA
needs no openLCA installation or XML export. Existing XML exports remain
supported through `paths.psilca_path` (a directory) or the legacy `paths.psilca_dir`
key. Set one key only. Only the optional SOCA benchmark needs openLCA, to export
SOCA as described in [step 8](#8-soca-benchmark).
Credentials, licensed inputs and generated outputs are Git-ignored.

### Prepared inventory workbooks

The exact hydrogen and methanol workbooks are **external inputs, not included in
this checkout or supplied by an ecoinvent/PSILCA licence**. Request the prepared
copies from Nicolas Schröder or Fabian Lechtenberg (see [Contact](../../README.md#contact))
and set their paths before continuing. Legacy inventories
with different ecoinvent names or missing study sheets are not interchangeable.

The hydrogen workbook must contain `SENSE activities` and `SENSE settings` sheets,
with `skip` in A1, headers in row 2, and the five route selections and declared
ecoinvent version in subsequent rows. The full
[workbook specification](REFERENCE.md#prepared-inventory-workbooks)
explains the required fields. Adding these sheets alone does not migrate old inventory
exchanges to ecoinvent 3.12.

Brightway projects persist outside the clone and environment. Recloning does not
isolate a project or recover private input files. Choose a new project name for
an independent build.

## 3. Fetch trade data and check setup

```bash
python -m case_studies.hydrogen.fetch_open_data
python -m case_studies.hydrogen.check_setup
```

The first command downloads the pinned Regioinvent trade database (about 2.4 GB)
and checks its SHA-256. The second checks runtime imports and the solver, credentials,
input availability, native PSILCA archive structure/Java version, and the hydrogen
workbook's study sheets/version. Resolve every
`MISSING` item. Foreground construction later checks exchange links against the
actual databases. Passing preflight does **not** mean the project already exists.

## 4. Build backgrounds

```bash
python -m case_studies.hydrogen.run build-backgrounds
```

This selects/creates the configured project, imports ecoinvent, spatializes its
biosphere and methods, and regionalizes the supply chain using Regioinvent trade
data with a cutoff of 0.99. Construction can take hours.

With the template defaults, the same project holds:

- `ecoinvent-3.12-cutoff`, the original background.
- `ecoinvent-3.12-cutoff - regionalized` and `Regioinvent`, a connected regionalized pair.

The original technosphere stays disconnected from that pair. Construction refuses
to overwrite existing regionalized backgrounds. **`build-backgrounds` is excluded
from every command group**, including `hydrogen` and `all`.

## 5. Map backgrounds and build paired foregrounds

```bash
python -m case_studies.hydrogen.run check-backgrounds
python -m case_studies.hydrogen.run sense-backgrounds
python -m case_studies.hydrogen.run foregrounds
python -m case_studies.hydrogen.run sense-foregrounds
```

| Stage | What it does |
| --- | --- |
| `check-backgrounds` | Verify all three backgrounds and their actual technosphere links. |
| `sense-backgrounds` | Run geometric SENSE on each background and register completed runs. |
| `foregrounds` | Build/reuse paired hydrogen, methanol and supporting-subset databases, checking recipes, identities and scenario boundaries. |
| `sense-foregrounds` | Establish common baseline-derived prices and map the six foreground databases. |

The baseline foregrounds link to original ecoinvent. The regionalized foregrounds
link to the regionalized pair. Paired recipes and prices are held equal so the
comparison concerns supply-chain geography.

The fixed study uses **PSILCA labour intensities**, not external productivity data,
with BGE matching, tolerance 0.028, GDP country weights, geometric social aggregation
and the default risk factors. Eligible unpriced treatments use the 20th percentile
of direct working hours by unit from **original ecoinvent** in both scenarios.
Other explicit zero prices remain zero. These are model assumptions, documented in
[study reference](REFERENCE.md#fixed-study) and the
[package reference](../../docs/package-reference.md#treatment-activities-and-zero-prices).

The supporting methanol foregrounds exclude three biogas supply nodes from direct
social and working-hour assignment. These nodes account for biogenic carbon uptake
and contain no operating inputs. The exact names are listed in
[`study_protocol.py`](workflow/study_protocol.py). This study-specific exclusion
preserves their physical exchanges and applies to neither the background databases
nor other treatment activities.

Mapping can take substantial time and initially downloads embedding models.
Keep project writing sequential. If a stage stops for a price or location worklist,
fill the required entries in the printed JSON file and rerun that stage before
proceeding. The [package guide](../../README.md#4-custom-databases-and-foregrounds)
shows both worklist formats. Do not replace generated entries with a sample file.

## 6. Calculate and export both hydrogen figures

```bash
python -m case_studies.hydrogen.run calculate
python -m case_studies.hydrogen.run figure-hydrogen
python -m case_studies.hydrogen.run figure-contributions
```

Calculation covers **30 reference flows**: five routes × three production countries
× two scenarios, each for 1 kg hydrogen at its modelled production gate. The archive
captures all methods in the selected ReCiPe environmental family, every mapped
geometric SENSE risk/opportunity method and working hours. The method/score count
is determined from the project and printed at runtime, not fixed in this README.
Every signed nonzero direct activity contribution is retained.

Calculation prepares the ReCiPe-derived climate method with
[explicit biogenic-carbon accounting](REFERENCE.md#climate-convention). It records
the changes in `troubleshooting/explicit_biogenic_carbon.json`.

`calculate` creates `paired_hydrogen/results.zip`. The two figure commands then
render that archive without another LCA:

| Output under the configured `data_dir` | Contents |
| --- | --- |
| `figures/hydrogen_paired.{pdf,svg,png}` | Climate, children-in-employment and fatal-accident totals for all three countries. |
| `figures/hydrogen_us_contributions.{pdf,svg,png}` | US fatal-accident contributions, with one activity panel and one inventory-location panel. |
| `paired_hydrogen/results.zip` | All scores, method/activity identities, contributors, coverage and provenance. |
| `paired_hydrogen/scores.csv` | All calculated scores. |
| `paired_hydrogen/figure_scores.csv` | The three displayed indicators. |
| `paired_hydrogen/indicator_audit/` | Export-level worker-indicator coverage and domestic data gaps. |

The contribution figure has five route pairs per panel, grouped by activity or
inventory location. The [study reference](REFERENCE.md#contribution-figure) explains
normalization and display thresholds; its [archive example](REFERENCE.md#result-archive)
shows how to export or filter contributors yourself.

## 7. Supporting Information figures and tables

After `calculate`, run:

```bash
python -m case_studies.hydrogen.run si
```

This generates the three matching-evaluation figures, the productivity diagnostic
figure, the missing-assessment sensitivity table and the carbon-convention table.
The diagnostics use the hydrogen archive, PSILCA cache, saved embeddings and
immutable mapping-run snapshots from steps 5–6. The carbon comparison additionally
reads the corresponding Brightway project and methods. These stages do not change
the archive, remap inventories or solve the technosphere again.

The matching figures use bundled **aggregate results of the original manual audit**.
They need neither PSILCA inputs nor the labelled audit workbook. They redraw the
reported metrics and calibration curve; they do not independently reproduce the
manual judgements or rerun encoder evaluation. Full audit reconstruction requires
the original external workbook, which is not distributed.

| Output under the configured `data_dir` | Contents |
| --- | --- |
| `figures/Fig1_semantic_vs_baseline_TEST.{pdf,svg,png}` | Held-out semantic/lexical matching comparison. |
| `figures/Fig2_semantic_model_choice_DEV_TEST.{pdf,svg,png}` | Encoder comparison on development and test samples. |
| `figures/Fig3_delta_selection_utility_high_res.{pdf,svg,png}` | Matching-tolerance calibration. |
| `figures/hydrogen_productivity_diagnostic.{pdf,svg,png}` | PSILCA productivity distribution and US coal-hydrogen counterfactuals. |
| `si/missing_assessment_sensitivity.tex` | Fatal-accident coverage and no-data sensitivities. |
| `si/carbon_convention_comparison.tex` | Conventional and explicit-biogenic-carbon climate ranges. |
| `si/productivity/`, `si/missing_assessments/`, `si/carbon/`, `si/matching/` | Derived CSVs, diagnostic evidence and local caches. |

These diagnostics use PSILCA labour intensities and check reconstruction against
the archive. First reconstruction can take several minutes; reruns reuse verified
caches. See the [diagnostic reference](REFERENCE.md#evidence-and-supplementary-analyses).

To redraw these SI figures and tables after successful diagnostics:

```bash
python -m case_studies.hydrogen.run si-outputs
```

The stages write into `data_dir`, not the manuscript or Overleaf repositories.
Copy the desired PDF assets and LaTeX tables into the SI project. Keep private
source-derived coefficient tables under their input licences.

## 8. SOCA benchmark

Skip this step if you do not have SOCA. It is independent of step 9.

SOCA is distributed as an openLCA database, so export its processes once:

1. Download SOCA v5 for ecoinvent 3.12 from [Nexus](https://nexus.openlca.org/)
   as a `.zolca` database.
2. Import the database into openLCA and open it.
3. Go to **Processes** and select all sectors.
4. Right-click the selection and choose **Export**.
5. In the dropdown menu, select **EcoSpold/Process** and click **Next**.
6. Choose an empty folder for the exported files and click **Next**. The export
   may take a couple of minutes.
7. Set `paths.soca_dir` in `config.local.yaml` to that folder. SENSE reads the
   `.xml` files directly in it, not in subfolders.

With `paths.soca_dir` set to the SOCA export:

```bash
python -m case_studies.hydrogen.check_setup --reuse-backgrounds --include-validation
python -m case_studies.hydrogen.run sense-arithmetic
python -m case_studies.hydrogen.run benchmark
python -m case_studies.hydrogen.run figure-benchmark
python -m case_studies.hydrogen.run table-si-coverage
```

Arithmetic SENSE uses `sociosphere_arithmetic` on original ecoinvent. Geometric
runs remain in `sociosphere`. No second project or duplicate technosphere is needed.
Hydrogen calculation continues to select only the geometric layer, so no restoration
or removal of the arithmetic layer is needed afterwards.
The benchmark compares **direct intensities**, with typed indicator identity and
coverage checks. It does not calculate arithmetic supply-chain results on the
regionalized backgrounds or foregrounds.

Outputs include `figure_inputs/validation.zip`, aggregation-specific coverage
workbooks and summary JSON files, plus these vector figures:

- `figures/validation.{pdf,svg}`, geometric agreement with SOCA.
- `figures/validation_arithmetic.{pdf,svg}`, arithmetic sensitivity.
- `figures/validation_factor_neutral.{pdf,svg}`, SENSE's recorded monetary factor removed.
- `si/comparison_coverage.tex`, the SI table of shared and one-sided exchange coverage.

SOCA agreement is an implementation benchmark, not independent validation of
social outcomes. Removing SENSE's multiplier does not establish SOCA's monetary basis.

## 9. Database-wide regionalization comparison

Run this last. It can take much longer than the hydrogen calculations:

```bash
python -m case_studies.hydrogen.run regionalization
python -m case_studies.hydrogen.run regionalization-summary
python -m case_studies.hydrogen.run figure-regionalization
```

The calculation compares corresponding eligible original/regionalized activities,
using all six PSILCA stakeholder groups in the geometric layer. It reuses compatible
saved scores. It does not require SOCA or rerun geometric mapping.
Reuse requires checksummed score manifests, matching methods and functional units,
matching paired populations, and completed runs that still match the live Brightway
state. File presence alone is insufficient. Legacy scores without these receipts
are recalculated rather than declared valid.

Outputs are the two `*_total_scores.csv` files and `regionalization_runs.json`
under `case_study_results/hydrogen_case_study_results/`,
`figure_inputs/regionalization.zip`, and `figures/regionalization_effect.{pdf,svg}`.
The summary uses positive paired scores and retains locations with at least 30
valid activity pairs for the plot. `regionalization-summary` only reads saved
scores and receipts. It does not perform new inventory solves.

## Reproduction scope and evidence

The commands above generate the main computational figures and SI figures/tables,
including the SOCA sensitivities, missing-data analysis and productivity/carbon
diagnostics. No manuscript checkout or historical carbon-patch backup is required.
Matching figures reproduce the bundled original audit aggregates; reconstruction
of that manually labelled audit remains outside this workflow. The draw.io diagrams
and synthetic illustrative calculation are separate manuscript assets.

Retain `study_runs.json`, `Output/Runs/`, calculation archives and original inputs
together. Reuse verifies their evidence against current inputs and inventories.
Do not edit receipts to bypass checks. Source-derived outputs remain subject to
the database licences. See [saved evidence](REFERENCE.md#evidence-and-supplementary-analyses).

Assessment coverage and low-confidence matches need interpretation. In particular,
the default `no data` factor is an assumption, not evidence of low risk. The
[indicator rationale](REFERENCE.md#displayed-indicators) explains the selected worker indicators.
Comparisons concern paired supply chains within each route, not a delivered-service
technology ranking or observed numbers of affected workers.

## Existing projects, groups and recovery

These options replace completed steps above. They are not extra steps after a
successful fresh run.

### Reuse an installation or backgrounds

Activate the existing environment instead of cloning/creating it again. If it
predates native PSILCA import, first run `conda install -c conda-forge openjdk=21`
and reactivate it. To create
another environment, add `--name sense-clean` to the creation command and activate
that name. For an existing project with all three backgrounds, skip trade download
and construction and check:

```bash
python -m case_studies.hydrogen.check_setup --reuse-backgrounds
python -m case_studies.hydrogen.run check-backgrounds
```

`--reuse-backgrounds` skips credentials and construction-only trade inputs. It
**does not create or validate the project**. `check-backgrounds` validates its databases.
If the project is absent, follow the fresh-build steps.

### Run groups

After building backgrounds, `hydrogen` replaces steps 5–6, including both hydrogen
figures. After geometric background mapping, `comparisons` combines steps 8–9 and
requires SOCA. `si` runs step 7 from existing hydrogen results. `all` runs hydrogen,
SI and comparisons in that order. All leave construction explicit and put
the expensive regionalization calculation after the SOCA work.

Inspect the order without loading configuration, inputs or Brightway:

```bash
python -m case_studies.hydrogen.run --list
python -m case_studies.hydrogen.run all --dry-run
```

Each stage runs in a fresh Python process. Groups stop on the first failure.
Running a whole group again reruns its mapping/calculation stages. Resume using
the failed named stage followed by the remaining stages shown in the dry run.

### Refresh mapping after code or assumption changes

For existing, paired foregrounds and backgrounds:

```bash
python -m case_studies.hydrogen.run recalculate-hydrogen
```

This checks backgrounds, remaps all three backgrounds and six foreground databases,
recalculates hydrogen, and redraws both hydrogen figures. It does not rebuild
inventories or refresh database comparisons. Afterwards repeat step 7 for SI
diagnostics, then steps 8–9 if those comparisons are required. This keeps the
expensive regionalization calculation last. If physical foreground inventories
changed, a mapping-only rerun does not rebuild them. Use a new project or
inspect/rebuild them deliberately.

Older mappings or receipts require this refresh too. Do not edit them to match new
settings. Existing valid backgrounds need not be rebuilt solely for a receipt change.

### Rebuild comparisons or redraw saved figures

Use `--rebuild` only for a supported single comparison/preparation stage:

```bash
python -m case_studies.hydrogen.run benchmark --rebuild
python -m case_studies.hydrogen.run figure-benchmark
python -m case_studies.hydrogen.run regionalization --rebuild
python -m case_studies.hydrogen.run regionalization-summary --rebuild
python -m case_studies.hydrogen.run figure-regionalization
python -m case_studies.hydrogen.run si-productivity --rebuild
python -m case_studies.hydrogen.run si-missing-assessments --rebuild
```

`regionalization --rebuild` recalculates the expensive score CSVs. Summary/figure
refreshes reuse them. `benchmark --rebuild` recalculates agreement, not SENSE mapping.
Changed SENSE source/settings require new mapping records first.

Regionalization totals use one transposed-system solve per indicator, avoiding
separate inventory calculations for every activity. One matrix factorization per
scenario is still required and can take substantial time on a large database.

To redraw all six figure stages without LCA, after their preparation stages have
completed:

```bash
python -m case_studies.hydrogen.run figures
```

This group requires the saved hydrogen archive, regionalization and SOCA snapshots,
and productivity diagnostics. Matching inputs are bundled. It stops at missing or
incompatible inputs rather than skipping them. If you skipped SOCA or
regionalization, use the individual figure commands in steps
6–9 instead.

`si-outputs` redraws the matching/productivity figures and the two hydrogen SI
tables. The SOCA coverage table has its separate `table-si-coverage` command in
step 8. Style-only changes can reuse the hydrogen archive. Comparison snapshots fingerprint
analysis code and may require `benchmark` or `regionalization-summary` to refresh
their inputs. An incomplete foreground scenario or leftover staging databases
require inspection before rebuilding. The runner stops rather than overwriting them.
