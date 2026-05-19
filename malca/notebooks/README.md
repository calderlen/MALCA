# Notebook Organization

The notebooks are grouped by workflow purpose. Keep new notebooks in the shallowest folder that matches their primary use, and prefer repo-root path helpers from `malca.notebook_paths` over location-relative paths.

## Folders

- `exploration/`: broad EDA and catalog/source exploration.
- `pipeline/`: detection pipeline behavior, baseline work, runtime profiling, and event outputs.
- `evaluation/`: benchmarks, injection tests, trigger comparisons, and periodicity experiments.
- `review/`: review database analysis, vetting, triage, and label separability.
- `diagnostics/`: targeted debugging, threshold tuning, cut-point investigations, and LOO/logBF diagnostics.
- `science/`: domain-specific result analysis such as microlensing, contamination, and anomalous fields.
- `archive/`: stale one-off prototypes or superseded notebooks after they are confirmed inactive.

## Notebook Index

### Exploration

- `asassn_index_eda.ipynb`: extended ASAS-SN catalog EDA for filtering and transient/AGN context.
- `comprehensive_eda.ipynb`: multi-bin detection-result EDA across pipeline run products.
- `eda.ipynb`: scratch EDA for event result tables.
- `eda_events_results.ipynb`: focused EDA for light-curve event outputs.
- `population_stats.ipynb`: inventory and plots for persisted scalar `stats_*` columns.
- `skypatrol_explore.ipynb`: SkyPatrol light-curve summary metrics and plots.
- `skypatrol2_gaps.ipynb`: SkyPatrol2 cadence and gap inspection.

### Pipeline

- `baseline.ipynb`: batch GP baseline plots and baseline smoke tests.
- `detection_runs_analysis.ipynb`: analysis of saved detection runs.
- `events.ipynb`: event-result inspection and event-scoring experiments.
- `events_profiling.ipynb`: `malca.events` profiling on SkyPatrol light curves.
- `pipeline_runtime_profiling.ipynb`: runtime profiling for small candidate subsets.
- `post_filter_attrition.ipynb`: post-filter failure and retained-candidate attrition analysis.

### Evaluation

- `march18_flat_full_gp_trigger_mode_benchmark.ipynb`: March 18 flat-directory full-GP trigger-mode benchmark.
- `march18_periodicity_pregate.ipynb`: March 18 CE-only pre-periodicity gate benchmark.
- `march18_periodicity_pregate_gate_only.ipynb`: lightweight gate-only variant of the March 18 pregate benchmark.
- `periodic_branch_simulation_benchmark.ipynb`: synthetic periodic-branch simulation benchmark.
- `periodic_branch_trigger_mode_benchmark.ipynb`: trigger-mode comparison for periodic-branch simulations.
- `periodicity_gate_injection_benchmark.ipynb`: periodicity gate injection benchmark.

### Review

- `compare_db_to_brayden_candidates.ipynb`: compare a review database against Brayden candidates.
- `dipper_review_label_separability.ipynb`: separability analysis for dipper review labels.
- `ltv_candidate_label_separability.ipynb`: separability analysis for LTV candidate review labels.
- `ltv_review_inspection.ipynb`: LTV review database inspection.
- `march18_other_eb_triage.ipynb`: March 18 `other`-pool EB triage.
- `sydney_rejection_reasons.ipynb`: Sydney candidate rejection-reason analysis.
- `vetting_stats.ipynb`: distributions and completeness for retrieved vetting fields.

### Diagnostics

- `debug_*_cut_points.ipynb`: magnitude-bin cut-point investigations.
- `logbf_loo_diagnostics.ipynb`: local logBF, global proxy, and LOO posterior diagnostics.
- `morphology_rnd_padding_init.ipynb`: morphology R&D for padding, initialization, BIC thresholds, jump skew, and rejectors.
- `test_loo_march18.ipynb`: March 18 LOO test notebook.
- `test_new_baseline.ipynb`: two-pass GP baseline testing on March 18 bundles.
- `tune_periodicity_thresholds.ipynb`: canonical periodicity threshold tuning.

### Science

- `analyze_microlensing_results.ipynb`: comprehensive EDA for microlensing result products.
- `bayes_output_analysis.ipynb`: EDA for Bayesian light-curve excursion outputs.
- `contamination_analysis.ipynb`: GP residual analysis for a possible contaminating source.
- `march18_anomalous_fields.ipynb`: March 18 anomalous ASAS-SN fields audit.

## Generated Outputs

Notebook-generated artifacts should go under top-level `output/notebooks/` or the existing top-level `output/diagnostics/` tree, not under `malca/notebooks/`.
