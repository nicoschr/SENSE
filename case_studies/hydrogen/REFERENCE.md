# Hydrogen study reference

Follow the [case-study guide](README.md) to run the workflow. This reference records
its fixed scientific choices, workbook requirements and result formats. The
[package reference](../../docs/package-reference.md) explains general SENSE settings.

## Fixed study

Scientific settings are versioned in [study_protocol.py](workflow/study_protocol.py).
Local configuration supplies paths, project names and credentials, not alternative
study designs.

| Choice | Setting |
| --- | --- |
| Inputs | ecoinvent 3.12 cutoff, PSILCA 4.0, Regioinvent 1.4.0.2, trade data v5. SOCA benchmark: SOCA v5 for ecoinvent 3.12. |
| Production countries | Germany, China and the United States, chosen before inspecting revised results. |
| Routes | Biomass gasification, biomethane SMR, wind PEM electrolysis, coal gasification and natural-gas SMR. |
| Reference flow | 1 kg hydrogen at each route's modelled production gate. |
| Comparison | Original versus regionalized supply chains within each route. Recipes and inferred product prices are held equal; provider links and spatial biosphere counterparts change. |
| Matching | BGE base English v1.5, tolerance `0.028`. |
| Social aggregation | Geometric, with GDP country weights. Working hours use arithmetic averaging. |
| Labour data | PSILCA intensities. No external productivity baseline. |
| Empty physical inventories | Excluded from direct social mapping and the treatment-hour reference pool in all backgrounds and foregrounds. Production and SENSE social exchanges are ignored. |
| Foreground biogas bookkeeping | Three exact-name nodes in the supporting methanol foregrounds are excluded from direct social and working-hour assignment. Their only environmental exchange records biogenic carbon uptake. Physical exchanges remain unchanged. See `foreground_exclude_exact` in the study protocol. |
| Unpriced treatments | 20th percentile of direct positive working hours by unit in original ecoinvent, shared across scenarios. |
| Assessment factors | `no`: 0.001; very low: 0.01; low: 0.1; medium: 1; high: 10; very high: 100; no data: 0.1. |

Thirty reference flows demonstrate custom foregrounds and geographically differentiated
backgrounds on one inventory basis. This is a paired model comparison, not a ranking
of equivalent delivered hydrogen services or a complete sustainability assessment.
Regionalization can change supply-chain composition as well as country attribution.

### Displayed indicators

The totals figure displays climate change, children in employment (total), and
fatal workplace accidents. The two worker concerns were selected before calculating
the revised results. Source interpretation follows the
[PSILCA 4 manual](https://www.openlca.org/wp-content/uploads/2025/07/PSILCA_manual_for-v.4_04072025.pdf),
Sections 4.1.1 (pp. 34–36) and 4.1.6 (pp. 51–52).

Children in employment includes non-hazardous employment and is broader than harmful
child labour. Fatal-accident assessments use ILOSTAT data with sector mapping and,
where necessary, national substitutes. Populated sector records do not establish
independently observed sector rates. Both scores are risk-weighted modelled labour,
not child-worker hours, numbers of children or predicted fatalities.

`calculate` audits exported assessments and stores coverage with the results.
Explicit `no` counts as assessed; `no data`, unknown levels and absent records do
not. Industry/commodity processes sharing a label are not independent sector
observations. A label is assessed only when every represented process has a recognized
assessment. Chinese fatal-accident gaps include important energy sectors in the
inspected source. The `si-missing-assessments` stage separately measures supply-chain
labour associated with missing assessments and tests alternative no-data factors.
Export availability alone does not establish primary-data quality or robustness.

### Climate convention

The displayed climate method extends ReCiPe 2016 v1.03 midpoint (H), regionalized,
GWP100 with explicit biogenic carbon. Uptake and allocation correction receive −1,
non-fossil CO₂ release +1, and non-fossil methane the corresponding fossil-methane
factor. Existing stock-change factors remain. `calculate` creates this named method
and records its changes in `troubleshooting/explicit_biogenic_carbon.json`.
The `si-carbon` stage compares this convention with the original method.

### Contribution figure

The second figure shows US fatal-accident scores for five original/regionalized
route pairs. The US example was selected after inspecting the three-country results.
One panel groups activity contributions across inventory locations; the other
sums by inventory country/region across activities.

Shares use the sum of absolute direct contributions, retaining signs. Activities
reaching 10% in any bar are shown separately; the rest are pooled. Every location
remains a separate segment, with main colours/legend entries for locations reaching
5% in any bar. Subnational locations are grouped into their parent country;
composite regions retain their labels. This is inventory geography, not reconstructed
PSILCA-sector or proxy-country attribution. Display thresholds do not truncate the
saved contribution data.

## Prepared inventory workbooks

The exact hydrogen and methanol workbooks are external inputs available from
Nicolas Schröder or Fabian Lechtenberg (see [Contact](../../README.md#contact)),
subject to the source licences. A database licence or fresh clone does not
supply these study-specific files. Use the prepared ecoinvent-3.12 versions.
Changing names alone does not migrate older inventories.

The hydrogen workbook contains these two study sheets in addition to inventory
sheets. **A1 must be `skip`**, headers are in row 2 and data start in row 3:

| Sheet | Required columns and values |
| --- | --- |
| `SENSE activities` | `purpose`, `name`, `reference product`, `location`, `demand`. `subset` rows have an exact source location and blank demand. `functional unit` rows have blank location and positive demand. Worksheet order determines functional-unit order. |
| `SENSE settings` | `setting`, `value`. Numeric `schema` = `1`, text `ecoinvent_version` matching the configuration, and the settings below. |

Other required settings are `methanol.target_activity_name`,
`methanol.target_reference_product`, `methanol.old_input_names`,
`methanol.old_input_reference_product`, `methanol.new_input_name`,
`methanol.new_input_reference_product`, `co2.name`, `co2.categories` and `co2.keyword`.
Use text values, with one row per list item for `methanol.old_input_names` and
`co2.categories`, otherwise one row per setting.

`check_setup` validates study sheets/version before construction. Foreground stages
resolve identities against the actual databases and stop on ambiguous links.
They preserve recipes, uncertainty fields and reference production when cloning,
then relink providers and spatial biosphere counterparts. Replacements are new
modelling choices unless shown to reproduce the original inventories. Preserve
amounts, units, uncertainty, formulas and cached values when preparing new versions.

## Result archive

Paths are under the configured `data_dir`. `calculate` writes
`paired_hydrogen/results.zip`, containing:

| File | Contents |
| --- | --- |
| `scores.csv`, `methods.csv` | All captured scores, method identities and units. |
| `functional_units.csv` | Reference-flow identities, demands and order. |
| `activities.csv` | Database/code identities, products, units, locations and native ISIC classifications. |
| `contributions/NNN.npz` | One sparse contribution matrix per system. Rows follow `methods.method_index`; columns follow `activities.activity_index`. |
| `indicator_coverage.csv`, `manifest.json` | Assessment coverage, calculation settings and provenance. |

The archive captures the selected ReCiPe family, all geometric SENSE risk/opportunity
methods, and working hours. Every signed nonzero direct activity contribution is
saved, including small terms and credits. Implicit zeros are exact zeros. Column
sums reproduce each method's score; these are direct contributions, not overlapping
recursive subtree totals. One inventory solve per reference flow feeds all methods.
The archive is replaced only after calculation completes.

Copies of `scores.csv`, the displayed `figure_scores.csv` subset and the indicator
audit are also written alongside the archive. Redrawing selects stored methods
without another LCA. A method outside the archive, changed physical inventories or
changed scientific settings requires new calculation. Plot style alone does not.

To inspect contributions without Brightway:

```python
from pathlib import Path
from case_studies.hydrogen.workflow.result_archive import contributor_table

contributors = contributor_table(
    Path("case_studies/hydrogen/Data/paired_hydrogen/results.zip"),
    fu_index=0,
    method=("SENSE", "sociosphere", "Social", "Working hours", "working hours"),
)
contributors.to_csv("working_hours_by_activity.csv", index=False)
contributors.groupby(["isic_division", "location"], dropna=False)["contribution"].sum().to_csv(
    "working_hours_by_sector_and_location.csv"
)
```

Choose `fu_index` from `functional_units.csv` and exact method keys from `methods.csv`.
`contribution_unit` describes the score; `unit` describes the activity product.
Keep database/code identities when filtering activities with identical names.
Missing native ISIC classifications remain `Unclassified`. ISIC grouping does not
represent the PSILCA sectors selected by semantic matching. Retain unclassified and
signed terms when checking grouped totals.

## Evidence and supplementary analyses

`study_runs.json` identifies completed mapping runs for every database/layer.
Retain it with `Output/Runs/`, calculation archives and original input versions.
Reuse checks source/configuration identity, input/artifact hashes and the live
inventory/social layer. Arithmetic and geometric layers coexist in one project.
Old receipts cannot be updated by editing JSON; use the guide's mapping refresh.

The SI productivity and missing-assessment diagnostics reconstruct archived
contributions before accepting counterfactual results. They need the matching
PSILCA cache, embeddings and immutable mapping snapshots. Their coefficients are
cached locally and checked before reuse. Treatment fallback hours remain fixed
when equalizing monetary labour intensities. The carbon diagnostic reads Brightway
matrices/methods and checks archived contributions without another inventory solve.
Keep source-derived diagnostic tables private under their licences.

### Matching-audit plotting data

The bundled [matching_audit](resources/matching_audit) directory contains only
aggregates of the original author-labelled evaluation:

| File | Contents |
| --- | --- |
| `encoder_metrics.csv` | Eight summary rows for three encoders and a lexical baseline on development/test samples (150/100 activities). Metrics: first-choice plausibility, plausible sector among the top ten and mean reciprocal rank. |
| `threshold_curve.csv` | 51 tolerances from 0 to 0.05 for each high-resolution development/test subset (90/60 activities). Hit rate, mean candidate-set size and utility are retained. |
| `provenance.json` | Source/reconstruction hashes, calibration grid, selection rule and limitations. No source records or local paths. |

Tolerance `0.028` maximized development utility: hit rate minus 0.1 times the mean
number of additional candidates. The test subset did not select it. These are
sample-dependent plausibility judgments, not ground-truth sector assignments.
`figure-si-matching` redraws the aggregates without rerunning encoders or recreating
the audit. Full reconstruction needs the original external workbook, which is not
shipped. Draw.io diagrams and the synthetic illustrative calculation are separate
manuscript assets, not pipeline outputs.
