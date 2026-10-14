# Changelog

## 1.1.0

Changes relative to [Git tag v1.0.0](https://github.com/nicoschr/SENSE/tree/v1.0.0).

### Inputs and API

- Read original PSILCA `.zolca` backups locally using Java 21. Manual openLCA export
  is no longer required. EcoSpold01 directories remain supported.
- Read inventory metadata, ISIC classifications and reference-exchange prices from
  Brightway instead of requiring a separate ecoinvent EcoSpold download.
- Use one mapping interface for backgrounds and foregrounds, with an explicit
  `price_source_db`. Remove the separate foreground-mode arguments.
- Add editable mapping/calculation scripts and JSON worklists for unresolved
  prices or custom locations. User price seeds take precedence over inferred prices.
- Add optional external labour-productivity data, explicit geographic donors and
  a working-hours-only mode without PSILCA or embedding models.
- Add optional working-hour methods by native two-digit ISIC division.

- Reuse validated empty-inventory eligibility on unchanged calls. Stream physical
  exchanges and inspect only relevant biosphere flows when a new check is needed.
- Reuse verified inventory fingerprints within a process and avoid rewriting
  unchanged SENSE impact methods. Report final inventory verification separately.
- Calculate regionalization totals with one transposed-system solve per indicator
  instead of one inventory calculation per activity, retaining a single factorization.

### Modelling changes

- Exclude inventories with neither nonzero technosphere inputs nor nonzero
  environmental exchanges from direct social mapping and treatment-hour references
  in every database. Preserve prices and supply-chain links. Set
  `exclude_empty_inventories=False` or use `--include-empty-inventories` for
  intentionally empty custom activities.
- Align product prices with PSILCA's currency and representative monetary year.
  Record the conversion and permit explicit basis/factor overrides.
- Resolve prices by exact activity identity before product-and-unit averages.
  Follow linked suppliers across databases for input-cost estimates, normalize
  recipes to reference production and exclude negative input credits from costs.
- Change default country weighting from equal to GDP, using bundled World Bank
  data with IMF fallback. Equal weighting and custom GDP data remain available.
- Change the matching tolerance from `0.016` to `0.028`, aligning the pipeline and
  matching helper. Add configurable low-confidence flags without rejecting matches.
- Replace the default zero-price floor with direct working-hour estimates for
  eligible unpriced treatments: the 20th percentile by physical unit in the reference
  database. The percentile is configurable, and `0` disables the fallback.
  Other explicit zero prices remain zero. The former price heuristic is optional.
- Consolidate eligibility rules across pricing, donor selection and writing.
  Exclude regioinvent production markets and structural bookkeeping/renaming nodes,
  while retaining operational import processes and eligible treatments.
- Correct RoW/RoE exclusions to use constituent countries, including subnational
  and composite locations, rather than subtracting inventory labels directly.
- Retain distinct PSILCA processes sharing country-sector names instead of
  overwriting earlier records. Allow explicit process selection and reject duplicate
  process identities or inconsistent working-hour values. Estimate working hours
  once per source process, independently of
  how many indicator records it supplies.
- Separate risk and opportunity methods and namespace all methods by sociosphere.
  Alternative model layers can coexist without overwriting each other's methods.
- Scale social exchanges to reference production, including negative treatment
  references. Reject unsupported multiple outputs and zero/non-finite production.

Geometric social aggregation, arithmetic working-hour aggregation, the BGE encoder
family and the default assessment-level factors are retained from v1.0.0.

### Reliability and performance

- Validate configuration, locations, prices, productivity coverage and reference
  production before affected writes. Stop on regional failures by default, with
  continuation available explicitly. CLI failures return nonzero exit codes.
- Synchronize SENSE-owned exchanges and methods, removing duplicates and obsolete
  contributions while preserving user data and physical exchanges. Use transactional
  exchange writes on the supported SQLite backend and lock concurrent SENSE runs.
- Replace the large PSILCA JSON assessment cache with compressed Parquet. Reuse
  country-specific lookups, batch matching/aggregation and cache reference prices
  and treatment donors to reduce repeated work and peak memory use.
- Validate caches against input/configuration identity and artifact hashes. Record
  immutable run snapshots, monetary assumptions, country weights and diagnostics.
  Study reuse also checks live inventory/social state and saved result integrity.
- Pin encoder revisions, require SafeTensors weights and reject local model-path
  shadowing and repository Python code. Record native-reader build provenance.

### Study workflow and installation

- Replace the manuscript notebook sequence with sequential Python stages, local
  configuration, setup checks and a checksum-verified trade-data download.
- Update the fixed study to ecoinvent 3.12, PSILCA 4.0 and Regioinvent 1.4.0.2.
  Compare five hydrogen routes in Germany, China and the United States, retaining
  paired recipes/prices and explicit biogenic-carbon climate accounting.
- Save all selected environmental-family scores, social indicators, working hours
  and complete signed activity contributions. Generate totals, contribution,
  SOCA, regionalization and SI diagnostic outputs from the workflow.
- Supply Windows/Linux PARDISO and macOS UMFPACK Conda environments, including
  Java 21. Support Python 3.11 patch releases rather than only 3.11.8.
- Separate import, solver, tutorial and case-study dependencies into extras.
  Add regression tests, setup smoke checks and opt-in full platform CI.
- Replace the old documentation with package/study guides and references. Add
  citation metadata and explicit distribution allowlists. Remove bundled licensed
  geography workbooks; users supply their own source inputs.

### Upgrading

- Update `run_sense` calls: remove `biosphere_db`, `ecoinvent_dir`,
  `foreground_active`, `foreground_database` and `background_databases`. Select the
  target through `technosphere_db` and its price reference through `price_source_db`.
  Use `psilca_path`; the `psilca_dir` alias remains accepted. Old CLI flags for the
  removed arguments are also gone.
- Remove obsolete configuration arguments `tiny_threshold`, `max_activity_examples`,
  `max_global_examples` and `match_quality_buckets`. `enable_social_flows` now follows
  `use_psilca` when unspecified. `price_floor_percentile` applies only to the
  explicitly selected legacy policy.
- Update method selections to the new `("SENSE", sociosphere_db, ...)` keys and
  select risk or opportunity separately. Use a fresh project for migration from
  v1.0.0 so old unmanaged exchanges/methods cannot mix with the new layer.
- Recompute results after migration. Changed defaults and corrected calculations
  can change scores. Older 1.1 receipts require remapping, without rebuilding valid
  backgrounds. Follow the study guide's
  [mapping refresh](case_studies/hydrogen/README.md#refresh-mapping-after-code-or-assumption-changes).
  Plotting alone does not update modelling assumptions.

## 1.0.0

Initial tagged release with a Python API and CLI, automatic semantic matching,
custom background/foreground support, configurable arithmetic/geometric averaging
and notebook-based demonstrations. Its package metadata and runtime version were
`0.1.0`, despite the `v1.0.0` Git tag.
