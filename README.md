# MALCA: Multi-timescale ASAS-SN Light Curve Analysis

![Tests](https://github.com/calderlen/malca/actions/workflows/tests.yml/badge.svg)

MALCA is a Bayesian event-detection pipeline for finding dimming and dipping events in ASAS-SN photometric light curves. It fits per-camera Gaussian process baselines, scores candidate events via marginal log-likelihood grids and leave-one-out posterior probabilities, and applies multi-stage quality filters to produce a catalog of dipper candidates. Post-detection modules add multi-wavelength characterization (Gaia, WISE, dust maps) and astrophysical classification.

## Contents

- [Install](#install)
  - [Input Files](#input-files)
  - [Dependencies](#dependencies)
- [Quick Start](#quick-start)
- [Pipeline Architecture](#pipeline-architecture)
- [Usage Guide](#usage-guide)
  - [Detection Pipeline](#detection-pipeline)
  - [Individual Commands](#individual-commands)
  - [Candidate Review](#candidate-review)
- [Output Directory Structure](#output-directory-structure)
  - [Integrated Pipeline](#integrated-pipeline---detect-run)
  - [Standalone Module Outputs](#standalone-module-outputs)
- [Citation](#citation)
- [License](#license)

## Install

```bash
# Requires Python >= 3.10
git clone https://github.com/calderlen/malca.git && cd malca
pip install -e "."          # core pipeline
pip install -e ".[dev]"     # + pytest tooling
```

See `pyproject.toml` for optional extras: `[multiwavelength]`, `[visualization]`, `[gui]`, `[notebooks]`, `[all]`.

### Input Files
- Per-mag-bin directories: `<lcsv2_root>/<mag_bin>/`
  - Index CSVs: `index*.csv` with columns like `asas_sn_id, ra_deg, dec_deg, pm_ra, pm_dec, ...`
- Light curves: `lc<num>_cal/` folders containing `<asas_sn_id>.dat2`
- Optional catalogs:
  - VSX crossmatch: `input/vsx/asassn_x_vsx_matches_20250919_2252.csv` (pre-crossmatched with columns: asas_sn_id, sep_arcsec, class)
  - Raw VSX: `input/vsx/vsxcat.090525.csv` (used by `vsx/filter.py` to generate crossmatch)
  - Note: Bright nearby star (BNS) filtering is handled upstream by ASAS-SN during LC generation

### Dependencies
- Core pipeline: numpy, pandas, scipy, numba, astropy, celerite2, matplotlib, tqdm
- Required: pyarrow (imported by core pipeline; needed even if you only write CSV)
- Optional visualization: plotly (3D injection plots)
- Multi-wavelength characterization: astroquery (Gaia queries), dustmaps3d (3D dust extinction), pyvo (StarHorse TAP queries), banyan-sigma (young associations), requests (unWISE queries)
- Notebooks/EDA: jupyterlab, ipykernel, seaborn, scikit-learn, joblib

## Quick Start
```bash
# Build manifest (source_id → path index)
malca manifest --index-root /path/to/lcsv2 --lc-root /path/to/lcsv2 \
    --mag-bin 13_13.5 --out output/manifest.parquet --workers 10

# Run event detection pipeline
malca detect --mag-bin 13_13.5 --workers 10 \
    --lc-root /path/to/lcsv2 --index-root /path/to/lcsv2 \
    --output output/results.csv --min-mag-offset 0.1

# Validate results against known candidates (no raw data needed)
malca validate --results output/results.csv

# Plot light curves
malca plot --input /path/to/lc123.dat2 --out-dir output/plots

# Apply quality filters
malca filter --input output/results.csv --output output/filtered.csv

# Multi-wavelength characterization (post-detection)
malca characterize --input output/filtered.csv --output output/characterized.csv --dust --starhorse input/starhorse/starhorse.parquet

# Get help for any command
malca --help
malca detect --help
```

## Pipeline Architecture

```mermaid
graph TB
    subgraph "Data Sources"
        RAW[ASAS-SN Raw Data<br/>.dat2 files]
        SKY[SkyPatrol CSVs]
        VSX_CAT[VSX Catalog]
        GAIA[Gaia Catalog]
    end

    subgraph "Core Libraries"
        UTILS[utils.py<br/>LC I/O, cleaning, bad camera filtering]
        BASE[baseline.py<br/>GP/median baselines]
        STATS_LIB[stats.py<br/>Statistics]
        SCORE_LIB[score.py<br/>Event scoring]
    end

    subgraph "Data Management"
        MAN[manifest.py<br/>Build index]
        RAW --> MAN
        MAN --> MAN_OUT[(Manifest)]
    end

    subgraph "VSX Tools"
        VSX_FILT[vsx/filter.py]
        VSX_CROSS[vsx/crossmatch.py]
        VSX_CAT --> VSX_FILT
        VSX_CAT --> VSX_CROSS
        VSX_FILT --> VSX_CLEAN[(Cleaned VSX)]
        VSX_CROSS --> VSX_MATCH[(Crossmatch)]
    end

    subgraph "Production Pipeline"
        EV_FILT[detect.py<br/>Wrapper + Batching]
        PREFILT[pre_filter.py<br/>Quality filters]
        EVENTS[events.py<br/>Bayesian Detection]
        AMP_FILT[filter.py<br/>Signal amplitude filter]
        POSTFILT[post_filter.py<br/>Quality filters]

        MAN_OUT --> EV_FILT
        VSX_MATCH -.-> PREFILT
        EV_FILT --> PREFILT
        PREFILT --> EVENTS
        EVENTS --> POSTFILT
        EVENTS -.-> AMP_FILT
        AMP_FILT -.-> POSTFILT
        GAIA -.-> POSTFILT
        POSTFILT --> CAND[(Final Candidates)]
    end

    subgraph "Post-Detection"
        CHAR[characterize.py<br/>Multi-wavelength]
        CLASSIFY[classify.py<br/>Dipper classification]

        CAND -.-> CHAR
        CHAR --> CHAR_OUT[(Characterized)]
        CHAR_OUT -.-> CLASSIFY
        CLASSIFY --> CLASSIFY_OUT[(Classified)]
    end

    subgraph "Evaluation"
        REPRO[reproduce.py<br/>Known objects]
        VALID[evaluation/validation.py<br/>Results validation]
        INJ[evaluation/injection.py<br/>Synthetic dips]

        MAN_OUT -.-> REPRO
        CAND -.-> REPRO
        REPRO --> REPRO_OUT[(Validation)]

        CAND --> VALID
        VALID --> VALID_OUT[(Metrics)]

        MAN_OUT --> INJ
        INJ --> INJ_OUT[(Completeness)]
    end

    subgraph "Analysis"
        PLOT[plot.py<br/>Visualization]
        LTV[ltv/pipeline.py<br/>Long-term variability]
        FP[evaluation/attrition.py<br/>Filter attrition]

        CAND --> PLOT
        RAW -.-> PLOT
        SKY -.-> PLOT

        MAN_OUT --> LTV
        LTV --> LTV_OUT[(LTV Results)]

        CAND --> FP
        FP --> FP_OUT[(FP Report)]
    end

    subgraph "Command Line Interface"
        CLI[__main__.py]
        CLI -.-> MAN
        CLI -.-> EV_FILT
        CLI -.-> REPRO
        CLI -.-> VALID
        CLI -.-> PLOT
        CLI -.-> POSTFILT
    end

    %% Dependencies
    UTILS -.-> EVENTS
    UTILS -.-> REPRO
    BASE -.-> EVENTS
    BASE -.-> REPRO
    SCORE_LIB -.-> EVENTS
    STATS_LIB -.-> SCORE_LIB

    %% Styling
    style EVENTS fill:#9cf,stroke:#333,stroke-width:2px
    style REPRO fill:#ff9,stroke:#333,stroke-width:2px
    style CLI fill:#fcf,stroke:#333,stroke-width:2px
```

**Key Components:**
- **Production**: `manifest.py` → `pre_filter.py` → `events.py` → `post_filter.py`
- **Evaluation**: `reproduce.py` (re-runs detection), `evaluation/validation.py` (validates results), `evaluation/injection.py` (synthetic dips)
- **CLI**: Unified interface via `malca [command]`

See [docs/architecture.md](docs/architecture.md) for detailed documentation.

## Usage Guide

### Detection Pipeline

The full detection workflow has three steps: build a manifest, run detection with batching/resume, then post-filter.

1) Build a manifest (map IDs -> light-curve directories):
   ```bash
   malca manifest --index-root /path/to/lcsv2 --lc-root /path/to/lcsv2 \
       --mag-bin 13_13.5 --out output/lc_manifest_13_13.5.parquet --workers 10
   ```
2) Pre-filter and run events in batches with resume support:
   ```bash
   malca detect --mag-bin 13_13.5 --workers 10 \
       --min-time-span 100 --min-points-per-day 0.05 --min-cameras 2 \
       --vsx-crossmatch input/vsx/asassn_x_vsx_matches_20250919_2252.csv \
       --batch-size 2000 \
       --lc-root /path/to/lcsv2 \
       --index-root /path/to/lcsv2 \
       --output output/lc_events_results_13_13.5.csv \
       --trigger-mode posterior_prob --baseline-func gp --min-mag-offset 0.1
   ```
   - The wrapper builds/loads the manifest, runs pre-filters, then calls `events.py` in batches.
   - Resume: if interrupted, skips already-processed paths using the checkpoint file.
   - VSX tags are saved to `prefilter/vsx_tags/` and merged into results.
   - To disable VSX handling: `--skip-vsx`. To tag instead of filter: `--vsx-mode tag`.

3) Post-filter events:
   ```bash
   malca post_filter --input output/lc_events_results_13_13.5.csv \
       --output output/lc_events_results_13_13.5_filtered.csv

   # With custom thresholds
   malca post_filter --input results.csv --output filtered.csv \
       --min-bayes-factor 20 --min-event-prob 0.7 --apply-morphology
   ```
   - **Implemented filters**: posterior strength, event probability, run robustness, morphology
   - **Placeholder filters** (not yet implemented): periodicity (LSP), Gaia RUWE, periodic catalog crossmatch

**Detect options:**
```bash
# logBF triggering (faster)
malca detect --mag-bin 13_13.5 --workers 8 \
    --lc-root /path/to/lcsv2 --index-root /path/to/lcsv2 \
    --output output/events_logbf.csv --trigger-mode logbf \
    --baseline-func gp_masked --min-mag-offset 0.1

# Multiple mag bins (writes one output per bin)
malca detect --mag-bin 12_12.5 12.5_13 13_13.5 \
    --lc-root /path/to/lcsv2 --index-root /path/to/lcsv2 \
    --output output/lc_events_results.csv --trigger-mode logbf
```

### Individual Commands

#### malca manifest

```bash
malca manifest --index-root <index_dir> --lc-root <lc_dir> \
    --mag-bin 12_12.5 --out output/lc_manifest.parquet
```

#### malca events

Run event detection directly (without the detect wrapper):
```bash
malca events --input /path/to/lc*_cal/*.dat2 --output output/results.parquet --workers 10

# With signal amplitude filtering (requires |event_mag - baseline_mag| > 0.1)
malca events --input /path/to/lc*_cal/*.dat2 --output output/results.parquet \
    --workers 10 --min-mag-offset 0.1
```
- Default Bayesian grid is 12x12. Change p-grid with `--p-points`.

#### malca pre_filter

```bash
malca pre_filter --help
```
- Expects columns `asas_sn_id` and `path` pointing to lc_dir.
- VSX handling: default is filter (drops VSX matches but keeps `sep_arcsec` and `class` on survivors). Use `--vsx-mode tag` to keep all matches and only tag.

#### malca post_filter

```bash
malca post_filter --input output/results.parquet --output output/results_filtered.parquet
```

#### malca filter

```bash
malca filter --input output/results.csv --output output/filtered.csv
```

#### malca plot

```bash
# Single file
malca plot --input /path/to/lc123.dat2 --out-dir output/plots --format png

# Multiple files (glob patterns supported)
malca plot --input input/skypatrol2/*.csv --out-dir output/plots --skip-events

# All files from events.py results
malca plot --events output/lc_events_results_13_13.5_filtered.csv --out-dir output/plots
```

**Note:** Event scores are computed automatically during detection and included in the results CSV (dipper_score, dipper_n_dips, dipper_n_valid_dips columns).

Legacy batch plotting: `malca old.plot_results_bayes /path/to/*.csv --results-csv output/lc_events_results_13_13.5.csv --out-dir output/plots`

#### malca injection

```bash
# Full run
malca injection --workers 10

# Quick test with limited trials
malca injection --max-trials 1000 --workers 10

# Custom manifest and output directory
malca injection --manifest /path/to/manifest.parquet --out-dir output/injection
```

See [Injection Testing output](#injection-testing) for the directory layout.

- Injects synthetic dips with skew-normal profiles onto real observed light curves
- Preserves real cadence, systematics, and noise characteristics
- Supports resume for long-running parameter sweeps

Python API:
```python
from malca.evaluation.injection import (
    load_efficiency_cube,
    plot_efficiency_all,
    plot_efficiency_mag_slices,
    plot_efficiency_marginalized,
    plot_efficiency_threshold_contour,
    plot_efficiency_3d,
)

cube = load_efficiency_cube("output/injection/cubes/efficiency_cube.npz")
plot_efficiency_marginalized(cube, axis="mag", output_path="avg_over_mag.png")
plot_efficiency_threshold_contour(cube, threshold=0.5, output_path="depth_at_50pct.png")
```

#### malca reproduce

```bash
# Re-run detection on raw data (requires manifest and .dat2 files)
malca reproduce --method bayes --manifest output/lc_manifest.parquet \
    --candidates my_targets.csv --out-dir output/results_repro --workers 10
```
**Note**: Reproduction uses Bayesian detection.

#### malca validate

```bash
# Auto-discover and validate ALL results for LOO method
malca validate --method loo

# Auto-discover for Bayes Factor method
malca validate --method bf

# Filter to specific magnitude bin
malca validate --method loo --mag-bin 13_13.5

# Direct file specification
malca validate --results output/results.csv

# Validate latest detect run output (output/runs/<timestamp>/results)
malca validate --latest-run

# Validate a specific detect run directory
malca validate --run-dir output/runs/20250119_1349

# With custom candidates
malca validate --method loo --candidates my_targets.csv -v

# Reproduce on built-in candidates using local SkyPatrol CSVs
malca validate --candidates brayden_candidates --skypatrol-dir input/skypatrol2 \
    --method bayes --trigger-mode logbf --workers 4

# Reproduce using events.py output directly (uses the 'path' column)
malca validate --input output/events_logbf.csv --method bayes --trigger-mode logbf
```

#### malca characterize

After detecting dipper candidates, characterize them using multi-wavelength data:

```bash
malca characterize \
  --input output/filtered.csv \
  --output output/characterized.csv \
  --dust \
  --starhorse input/starhorse/starhorse2021.parquet
```

**Features:**
- **Gaia DR3 Queries**: Astrometry, astrophysics (Teff, logg, metallicity, distance), 2MASS/AllWISE photometry
- **3D Dust Extinction**: All-sky coverage via `dustmaps3d` (Wang et al. 2025, ~350MB)
- **YSO Classification**: Koenig & Leisawitz (2014) IR color-color diagram with dust correction
- **Galactic Population**: Thin/thick disk classification using metallicity or StarHorse ages
- **StarHorse** (optional): Stellar ages, masses, distances from local catalog join
- **Auxiliary Catalog Crossmatches** (Tzanidakis+2025):
  - BANYAN Σ: Young stellar association membership probabilities
  - IPHAS DR2: Hα emission detection for Galactic plane sources
  - Star-forming regions: Proximity check to known SFRs (Prisinzano+2022)
  - Open clusters: Cantat-Gaudin+2020 membership crossmatch
  - unWISE/unTimely: Mid-IR variability z-scores
- **Color Evolution Analysis**: (g-r) color differences and CMD slope fitting
- **Caching**: Gaia results cached locally to speed up repeated analyses

**Setup:**
```bash
# Install multiwavelength dependencies
pip install -e ".[multiwavelength]"

# Dust maps auto-download on first use (~350MB)
# For StarHorse, download catalog manually:
# https://cdsarc.cds.unistra.fr/viz-bin/cat/I/354
```

**Output columns:**
- `source_id`, `ra`, `dec`, `parallax`, `distance_gspphot`
- `tmass_j`, `tmass_h`, `tmass_k`, `unwise_w1`, `unwise_w2`
- `A_v_3d`, `ebv_3d` (3D dust extinction)
- `H_K`, `W1_W2`, `yso_class` (Class I/II/Transition Disk/Main Sequence)
- `population` (thin_disk/thick_disk from metallicity or age)
- `age50`, `mass50` (if StarHorse provided)
- Auxiliary crossmatches (Tzanidakis+2025):
  - `banyan_field_prob`, `banyan_best_assoc` (BANYAN Σ membership)
  - `iphas_r_ha`, `iphas_ha_excess` (IPHAS Hα)
  - `near_sfr`, `sfr_name` (star-forming region proximity)
  - `cluster_name`, `cluster_age_myr` (open cluster membership)
  - `unwise_w1_zscore`, `unwise_w1_var` (IR variability)
- Color evolution (if multi-band available):
  - `color_baseline`, `color_dip`, `color_diff`, `is_redder`
  - `cmd_slope`, `cmd_slope_angle`, `cmd_ism_consistent`

#### malca classify

```bash
malca classify --input output/characterized.csv --output output/classified.csv
```

#### malca stats

```bash
malca stats /path/to/lc123.dat2
```

#### malca attrition

```bash
malca attrition --pre output/pre.csv --post output/post.csv
```

### Candidate Review

- Install GUI dependency:
  `pip install -e ".[gui]"`
- Launch reviewer app:
  `streamlit run malca/review/app.py`
- Launch terminal triage tool:
  `malca review.tui --db output/review/review.db`
- In the app:
  - Set a SQLite path (default: `output/review/review.db`)
  - Import a filtered candidates file (`CSV`/`Parquet`)
  - Score candidates (`interest_score` integer 0-5, `interest_reason`, `review_pass`, `notes`, `status`)
  - Filter by quantitative periodicity metrics (`periodicity_score`, `lsp_bootstrap_sig`, `lsp_power`)
  - Export reviewed candidates to CSV/Parquet

## Output Directory Structure

### Integrated Pipeline (`--detect-run`)

When running the full detection pipeline with `--detect-run`, the following directory structure is created for complete provenance tracking:

```
output/runs/20250121_143052/          # Timestamp-based run directory
├── run_params.json                   # Detection parameters (detect.py)
├── run_summary.json                 # Detection results stats (detect.py)
├── filter_log.json                   # Filtering parameters & stats (post_filter.py)
├── plot_log.json                     # Plotting parameters (plot.py)
├── run.log                           # Simple text log with paths
│
├── manifests/                        # Manifest files
│   └── lc_manifest_{mag_bin}.parquet
│
├── prefilter/                        # Pre-filtering results
│   ├── lc_filtered_{mag_bin}.parquet
│   ├── lc_stats_checkpoint_{mag_bin}.parquet
│   ├── rejected_pre_filter_{mag_bin}.csv
│   └── vsx_tags/
│       └── vsx_tags_{mag_bin}.csv
│
├── paths/                            # Input paths
│   └── filtered_paths_{mag_bin}.txt
│
├── results/                          # Detection results
│   ├── lc_events_results.csv         # Raw detection output (includes dipper_score)
│   ├── lc_events_results_PROCESSED.txt  # Checkpoint log
│   ├── lc_events_results_filtered.csv   # After post_filter.py
│   └── rejected_post_filter.csv      # Post-filter rejections
│
└── plots/                            # Visualizations (plot.py)
    ├── {source_id}_dips.png
    ├── {source_id}_dips.png
    └── ...
```

**Key Features:**
- **JSON logs track full provenance**: Every parameter and result is logged for reproducibility
- **Self-contained runs**: Each timestamped directory contains everything needed to reproduce the analysis
- **Checkpoint support**: Detection runs can be interrupted and resumed using `*_PROCESSED.txt` files
- **Rejection tracking**: Both pre-filter and post-filter rejections are logged with reasons

**JSON Log Contents:**
- `run_params.json`: All pre-filter and detection parameters (thresholds, workers, baseline settings)
- `run_summary.json`: Manifest statistics, pre-filter rejection breakdown, detection results
- `filter_log.json`: Filter toggles, thresholds, input/output counts, rejection breakdown
- `plot_log.json`: Plotting parameters, GP settings, number of plots generated

**Note:** Event scores (dipper_score, dipper_n_dips, dipper_n_valid_dips) are automatically computed during detection for significant events and included in the results CSV.

### Standalone Module Outputs

#### Injection Testing

```
output/injection/                     # Default output directory
├── results/
│   ├── injection_results.csv         # Trial-by-trial injection results
│   └── injection_results_PROCESSED.txt  # Checkpoint for resume
│
├── cubes/
│   └── efficiency_cube.npz           # 3D efficiency cube (depth × duration × mag)
│
└── plots/
    ├── mag_slices/                   # Per-magnitude 2D heatmaps
    │   ├── mag_12.0_efficiency.png
    │   ├── mag_13.0_efficiency.png
    │   └── ...
    ├── efficiency_marginalized_*.png  # Averaged over one axis
    ├── depth_at_*pct_efficiency.png   # Threshold contour maps
    └── efficiency_3d_volume.html      # Interactive 3D (if plotly installed)
```

#### Detection Rate

```
output/detection_rate/                # Default base directory
├── 20250121_143052/                  # Timestamped run directory
│   ├── run_params.json                # Full parameter dump
│   ├── results/
│   │   ├── detection_rate_results.csv
│   │   ├── detection_rate_results_PROCESSED.txt  # Checkpoint
│   │   └── detection_summary.json     # Detection rate summary
│   └── plots/
│       ├── detection_rate_vs_mag.png
│       ├── detection_duration_dist.png
│       └── detection_depth_dist.png
│
├── 20250121_150318_custom_tag/       # Optional --run-tag appended
│   └── ...
│
└── latest -> 20250121_150318_custom_tag/  # Symlink to latest run
```

#### Multi-Wavelength Characterization

```
output/
├── characterized.csv                 # Single output file with added columns:
                                      #   - Gaia astrometry & photometry
                                      #   - 3D dust extinction (A_v_3d, ebv_3d)
                                      #   - YSO classification (yso_class)
                                      #   - Galactic population (thin_disk/thick_disk)
                                      #   - StarHorse ages/masses (if provided)
                                      #   - Auxiliary crossmatches (BANYAN Σ, IPHAS, etc.)
└── gaia_cache/                       # Gaia query cache (optional)
    └── gaia_results_{hash}.parquet
```

#### Dipper Classification

```
output/
└── classified.csv                    # Single output file with added columns:
                                      #   - P_eb, P_cv, P_starspot, P_disk
                                      #   - yso_class
                                      #   - a_circ_au, transit_prob
                                      #   - final_class (EB/CV/Starspot/Disk/YSO/Unknown)
```

#### Manifest Building

```
output/
└── lc_manifest_{mag_bin}.parquet     # Single parquet file with:
                                      #   - asas_sn_id
                                      #   - ra_deg, dec_deg
                                      #   - lc_dir (directory path)
                                      #   - dat_path (full .dat2 path)
                                      #   - dat_exists (bool)
```

## Citation

If you use MALCA or any part of its codebase in published research, please cite this repository:

```
Lenhart, C. (2025). MALCA: Multi-timescale ASAS-SN Light Curve Analysis [Software].
https://github.com/calderlen/malca
```

## License

This project is licensed under the GNU General Public License v3.0. See [LICENSE](LICENSE) for details.
