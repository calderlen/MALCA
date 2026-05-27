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
  - [Integrated Pipeline](#integrated-pipeline)
  - [Standalone Module Outputs](#standalone-module-outputs)
- [Citation](#citation)
- [License](#license)

## Install

```bash
# Requires Python >= 3.9
git clone https://github.com/calderlen/malca.git && cd malca
pip install -e "."          # installs all runtime + test dependencies
```

Conda option:

```bash
conda env create -f environment.yml
conda activate malca
```

### Input Files
- Per-mag-bin directories: `<lcsv2_root>/<mag_bin>/`
  - Index CSVs: `index*.csv` with columns like `asas_sn_id, ra_deg, dec_deg, pm_ra, pm_dec, ...`
- Light curves: `lc<num>_cal/` folders containing `<asas_sn_id>.dat2`
- Optional catalogs:
  - VSX crossmatch: `input/vsx/asassn_x_vsx_matches_20250919_2252.parquet` (pre-crossmatched with columns: asas_sn_id, sep_arcsec, class)
  - Raw VSX: `input/vsx/vsxcat.090525.csv` (used by `vsx/filter.py` to generate crossmatch)
  - Note: Bright nearby star (BNS) filtering is handled upstream by ASAS-SN during LC generation

### Dependencies
- Core + runtime modules: numpy, pandas, scipy, numba, astropy, celerite2, matplotlib, tqdm, pyarrow
- Review + plotting: dash, dash-bootstrap-components, plotly
- Characterization + catalog access: astroquery, dustmaps3d, pyvo, banyan-sigma, requests
- ML utilities: lightgbm

## Quick Start
```bash
# Build manifest (source_id → path index)
malca manifest --index-root /path/to/lcsv2 --lc-root /path/to/lcsv2 --mag-bin 13_13.5 --output output/manifest.parquet --workers 10

# Run event detection pipeline
malca pipeline --mag-bin 13_13.5 --workers 10 --lc-root /path/to/lcsv2 --index-root /path/to/lcsv2 --output-dir output/run_001

# Validate results against known candidates (no raw data needed)
malca validate --results output/results.parquet

# Plot light curves
malca plot --input /path/to/lc123.dat2 --output-dir output/plots

# Apply quality filters
malca filter --input output/results.parquet --output output/filtered.parquet

# Multi-wavelength characterization (post-detection)
malca characterize --input output/filtered.parquet --output output/characterized.parquet --enable-dust --starhorse input/starhorse/starhorse2021.parquet

# Get help for any command
malca --help
malca pipeline --help
```

Minimal split workflow (cluster -> home):

```bash
# On cluster: run upstream/raw-dependent steps and export transfer bundle
malca pipeline --stage cluster --mag-bin 13_13.5 --output-dir output/run_001 --config cluster.toml

# On home machine: import bundle and run downstream/catalog steps only
malca pipeline --stage home --output-dir output/run_001 --config home.toml
```

## Pipeline Architecture

```mermaid
flowchart TB

    %% ── Data Sources ─────────────────────────────────────────
    subgraph sources["Data Sources"]
        RAW["ASAS-SN .dat2 Light Curves"]
        IDX["Index CSVs<br/>(per mag bin)"]
        SKY["SkyPatrol CSVs"]
        VSX_RAW["VSX Catalog"]
        GAIA_SRC["Gaia DR3"]
        SH_SRC["StarHorse Catalog"]
        DUST_SRC["3D Dust Maps<br/>(Wang+ 2025)"]
    end

    %% ── Data Preparation ─────────────────────────────────────
    subgraph prep["Data Preparation"]
        MAN["manifest.py<br/>Build source_id-to-path index"]
        MAN_OUT[("Manifest .parquet")]
        MAN --> MAN_OUT

        subgraph vsxtools["VSX Preprocessing (vsx/)"]
            VFILT["filter.py<br/>Clean variable classes"]
            VCROSS["crossmatch.py<br/>PM-corrected positional match"]
            VFILT --> VCROSS
        end
        VCROSS --> VSX_MATCH[("VSX Crossmatch")]
    end

    RAW --> MAN
    IDX --> MAN
    VSX_RAW --> VFILT

    %% ── Discovery Pipeline ───────────────────────────────────
    subgraph discovery["Discovery Pipeline (detect.py orchestrator)"]
        TAG["tag.py<br/>Sparse-LC, multi-camera,<br/>VSX quality tags"]
        EVENTS["events.py<br/>Bayesian detection, morphology fits,<br/>recurrence analysis, Bayes factors"]
        FILT["filter.py<br/>Evidence strength, run robustness,<br/>morphology, periodicity,<br/>Gaia RUWE/PM, periodic catalogs"]
        TAG --> EVENTS --> FILT
    end

    MAN_OUT --> TAG
    VSX_MATCH -.-> TAG
    GAIA_SRC -.-> FILT
    FILT --> CAND[("Candidates .parquet")]

    %% ── Post-Detection Characterization ──────────────────────
    subgraph postdet["Post-Detection"]
        CHAR["characterize.py<br/>Gaia astrometry/photometry, 3D dust,<br/>YSO classes, galactic coords,<br/>BANYAN, IPHAS, SFR, clusters, unWISE"]
        VET["vetting.py<br/>SIMBAD, Gaia variability/EB,<br/>ASAS-SN Var, ZTF, TNS, ALeRCE,<br/>eROSITA, ATLAS, NEOWISE"]
        CLASS["classify.py<br/>EB/CV/starspot/disk/YSO"]

        subgraph enrichgrp["Enrichment (enrich/)"]
            NEIGH["neighbor.py<br/>Gaia, 2MASS, AllWISE, VSX"]
            SPECTRA["spectra.py<br/>SDSS, LAMOST, GALAH, RAVE"]
        end

        CHAR --> VET --> CLASS --> enrichgrp
    end

    CAND --> CHAR
    GAIA_SRC -.-> CHAR
    DUST_SRC -.-> CHAR
    SH_SRC -.-> CHAR
    enrichgrp --> ENRICHED[("Enriched .parquet")]

    %% ── Visualization ────────────────────────────────────────
    PLOT["plot.py<br/>Light curve + event visualization"]
    CAND --> PLOT
    RAW -.-> PLOT
    SKY -.-> PLOT

    %% ── Review App ───────────────────────────────────────────
    subgraph reviewgrp["Review App (review/)"]
        STORE["store.py<br/>SQLite candidate DB"]
        APP["app.py<br/>Dash app: scoring, event classes,<br/>vetting cards, diagnostic plots"]
        RPIPE["pipeline.py<br/>Run missing stages on demand"]
        RMERGE["merge.py<br/>Merge review DBs"]
        RDIAG["diagnostic_plots.py<br/>CMD, Kiel, NEOWISE, Gaia epoch"]
        STORE --> APP
        RPIPE -.-> APP
        RDIAG -.-> APP
    end

    CAND --> STORE
    ENRICHED -.-> STORE
    APP --> LABELS[("Labeled Reviews<br/>score + event_class")]

    %% ── Machine Learning ─────────────────────────────────────
    subgraph mlgrp["Machine Learning (meta_analysis/ml/)"]
        FEAT["features.py<br/>107 curated features"]
        TRAIN["train.py<br/>LightGBM classifier"]
        PRED["predict.py<br/>Score new candidates"]
        FEAT --> TRAIN --> MODEL[("Model + schema")]
        MODEL --> PRED
    end

    LABELS -.-> TRAIN
    ENRICHED -.-> FEAT

    %% ── LTV Pipeline ─────────────────────────────────────────
    subgraph ltvpipe["LTV Pipeline - Long-Term Variability (ltv/)"]
        LTV_PIPE["pipeline.py<br/>Orchestrator"]
        LTV_CORE["core.py<br/>Season medians, linear/quad fits,<br/>slopes, Lomb-Scargle"]
        LTV_FILT["filter.py<br/>Slope, max diff, dec, PM cuts"]
        LTV_CROSS["crossmatch.py<br/>Gaia, VSX, OGLE, ZTF,<br/>Gaia Alerts, MilliQuas, SIMBAD"]
        LTV_STOCH["stochastic.py<br/>Structure function, IAR,<br/>MHPS, DRW"]
        LTV_NEO["neowise.py<br/>IRSA TAP IR light curves"]
        LTV_DUST["dust.py<br/>Dust excess flags"]
        LTV_CMD["cmd.py<br/>MIST grid, Bailer-Jones distances"]
        LTV_BUNDLE["bundle.py<br/>Package .dat2 files"]
        LTV_INGEST["review.py<br/>Ingest into review DB"]
        LTV_PIPE --> LTV_CORE --> LTV_FILT
        LTV_FILT --> LTV_CROSS --> LTV_STOCH
        LTV_STOCH --> LTV_NEO --> LTV_DUST --> LTV_CMD
        LTV_CMD --> LTV_BUNDLE --> LTV_INGEST
    end

    RAW --> LTV_PIPE
    IDX --> LTV_PIPE
    GAIA_SRC -.-> LTV_CROSS
    LTV_INGEST --> STORE

    %% ── Evaluation ───────────────────────────────────────────
    subgraph evalgrp["Evaluation (evaluation/)"]
        INJ["injection.py<br/>Synthetic dip injection-recovery"]
        DET_RATE["detection_rate.py<br/>Baseline detection rate"]
        VALID["validation.py<br/>Precision/recall vs known targets"]
        REPRO["reproduce.py<br/>Re-run detection on known objects"]
        ATTR["attrition.py<br/>Filter attrition summary"]
        FP_EVAL["false_positive.py<br/>FP contaminant benchmark"]
    end

    MAN_OUT -.-> INJ
    MAN_OUT -.-> DET_RATE
    MAN_OUT -.-> REPRO
    CAND -.-> VALID
    CAND -.-> REPRO
    CAND -.-> ATTR

    %% ── Core Libraries ───────────────────────────────────────
    subgraph corelibs["Core Libraries"]
        UTILS["utils.py<br/>LC I/O, cleaning, kernels"]
        LCIO["lightcurve_io.py<br/>.dat2 / .csv readers"]
        BASE["baseline.py<br/>GP + median baselines"]
        TRIG["triggering.py<br/>logBF / posterior trigger resolution"]
        SCORE_LIB["score.py<br/>Dip/jump/microlensing scoring"]
        STATS_LIB["stats.py<br/>Stetson, von Neumann, RoMS, LS"]
        PERIOD_LIB["periodogram.py<br/>Lomb-Scargle, PDM,<br/>Conditional Entropy"]
        FETCH_LIB["fetch.py<br/>SkyPatrol V1/V2 download"]
        GAIA_FETCH["gaia_fetch.py<br/>Bulk Gaia DR3 via AIP TAP"]
    end

    UTILS -.-> EVENTS
    BASE -.-> EVENTS
    TRIG -.-> EVENTS
    SCORE_LIB -.-> EVENTS
    STATS_LIB -.-> SCORE_LIB
    PERIOD_LIB -.-> FILT
    UTILS -.-> REPRO
    BASE -.-> REPRO

    %% ── Configuration ────────────────────────────────────────
    subgraph configgrp["Configuration"]
        direction LR
        CONF["config.py<br/>centralized constants, paths, thresholds, and service strings"]
    end

    %% ── CLI Entry Point ──────────────────────────────────────
    CLI["__main__.py — malca CLI<br/>manifest, pipeline, filter, tag, events, plot, characterize, classify,<br/>vetting, review, injection, validate, reproduce,<br/>ltv-pipeline, attrition, dev, ..."]
    CLI -.-> discovery
    CLI -.-> postdet
    CLI -.-> reviewgrp
    CLI -.-> ltvpipe
    CLI -.-> evalgrp
    CLI -.-> PLOT
```

**Key Components:**
- **Discovery pipeline**: `manifest.py` &rarr; `tag.py` &rarr; `events.py` &rarr; `filter.py` (orchestrated by `detect.py`)
- **Post-detection**: `characterize.py` (Gaia, dust, YSO, galactic coords, auxiliary catalogs) &rarr; `vetting.py` (SIMBAD, ZTF, TNS, eROSITA, ALeRCE, ATLAS, NEOWISE, ...) &rarr; `classify.py` (EB/CV/starspot/disk/YSO) &rarr; `enrich/` (neighbor catalogs, spectra availability)
- **LTV pipeline**: `ltv/pipeline.py` &rarr; `core.py` &rarr; `filter.py` &rarr; `crossmatch.py` &rarr; `stochastic.py` &rarr; `neowise.py` &rarr; `dust.py` &rarr; `cmd.py` &rarr; `bundle.py` &rarr; `review.py` (ingest to review DB)
- **Review**: `review/app.py` (Dash app with scoring, event classes, diagnostic plots, vetting cards) &rarr; labeled training set
- **Machine learning**: `malca/meta_analysis/ml/lightgbm_classifier_prototype.ipynb` (draft LightGBM classifier workflow)
- **Notebooks**: `malca/notebooks/README.md` documents the purpose-based notebook folders.
- **Evaluation**: `injection.py` (synthetic dips), `detection_rate.py`, `validation.py`, `reproduce.py`, `attrition.py`, `false_positive.py`
- **Core libraries**: `utils.py`, `lightcurve_io.py`, `baseline.py`, `triggering.py`, `score.py`, `stats.py`, `periodogram.py`, `fetch.py`, `gaia_fetch.py`
- **Configuration**: `config.py` centralizes all pipeline parameters
- **CLI**: Unified interface via `malca [command]` (`__main__.py`)

See [docs/architecture.md](docs/architecture.md) for detailed documentation.

## Usage Guide

### Detection Pipeline

The full detection workflow has three steps: build a manifest, run detection with batching/resume, then filter.

1) Build a manifest (map IDs -> light-curve directories):
   ```bash
   malca manifest --index-root /path/to/lcsv2 --lc-root /path/to/lcsv2 --mag-bin 13_13.5 --output output/lc_manifest_13_13.5.parquet --workers 10
   ```
2) Tag and run events in batches with resume support:
   ```bash
   malca pipeline --mag-bin 13_13.5 --workers 10 --lc-root /path/to/lcsv2 --index-root /path/to/lcsv2 --output-dir output/run_13_13.5 --config pipeline.toml
   ```
   - The pipeline command builds/loads the manifest, runs tag checks, then calls `events.py` in batches.
   - Resume: if interrupted, skips already-processed paths using the checkpoint file.
   - VSX tags are saved to `tags/vsx_tags/` and merged into results.
   - Advanced tag, detection, filter, and catalog settings live in `--config` / `--profile`.

3) Filter events:
   ```bash
   malca filter --input output/lc_events_results_13_13.5.parquet --output output/lc_events_results_13_13.5_filtered.parquet

   # With custom thresholds
   malca filter --input results.parquet --output filtered.parquet --min-bayes-factor 20 --min-run-points 3 --apply-morphology
   ```
   - **Implemented filters**: posterior strength, run robustness, score, morphology, periodicity, Gaia RUWE, Gaia PM, multi-catalog periodic consensus

4) Optional: tune filter behavior directly from `malca pipeline` / `malca detect`.
   ```bash
   malca pipeline --mag-bin 13_13.5 --config pipeline.toml --profile strict
   ```
   - **Defaults in pipeline**: evidence strength, run robustness, score, Gaia RUWE, Gaia PM, and periodic-catalog consensus validation are on; morphology and periodicity-validation are off.
   - Advanced controls are config/profile keys rather than public `malca pipeline` flags.

**Detect options:**
```bash
# logBF triggering (faster)
malca pipeline --mag-bin 13_13.5 --workers 8 --lc-root /path/to/lcsv2 --index-root /path/to/lcsv2 --output-dir output/events_logbf --config logbf.toml

# Multiple mag bins (writes one output per bin)
malca pipeline --mag-bin 12_12.5 12.5_13 13_13.5 --lc-root /path/to/lcsv2 --index-root /path/to/lcsv2 --output-dir output/multi_bin --config logbf.toml
```

### Individual Commands

#### malca manifest

```bash
malca manifest --index-root <index_dir> --lc-root <lc_dir> --mag-bin 12_12.5 --output output/lc_manifest.parquet
```

#### malca events

Run event detection directly (without the pipeline orchestrator):
```bash
malca events --input /path/to/lc*_cal/*.dat2 --output output/results.parquet --workers 10

# Advanced detection settings are supplied through --config / --profile
malca events --input /path/to/lc*_cal/*.dat2 --output output/results.parquet --workers 10 --config events.toml
```
- Default Bayesian grid is 12x12. Change advanced detection settings through config/profile keys.
- Output includes per-event morphology fit parameters (`best_amp`, `best_t0`, `best_alpha`, `best_tau`, `best_morph`, `delta_bic`, `width_param`, `symmetry_score`) and recurrence statistics (`is_single_event`, `inter_event_spacing_median/std`, `amplitude_consistency`, `duration_consistency`) for both dips and jumps.

#### malca tag

```bash
malca tag --help
```
- Expects columns `asas_sn_id` and `path` pointing to lc_dir.
- VSX handling tags rows with `vsx_sep_arcsec` / `vsx_class` when enabled.

#### malca filter

```bash
malca filter --input output/results.parquet --output output/results_filtered.parquet
```

#### malca plot

```bash
# Single file
malca plot --input /path/to/lc123.dat2 --output-dir output/plots --format png

# Multiple files (glob patterns supported)
malca plot --input input/skypatrol2/*.csv --output-dir output/plots --skip-events

# All files from events.py results
malca plot --results output/lc_events_results_13_13.5_filtered.parquet --output-dir output/plots
```

**Note:** Event scores are computed automatically during detection and included in the results table (dipper_score, dipper_n_dips, dipper_n_valid_dips columns).

#### malca injection

```bash
# Full run
malca injection --workers 10

# Quick test with limited trials
malca injection --max-trials 1000 --workers 10

# Custom manifest and output directory
malca injection --manifest /path/to/manifest.parquet --output-dir output/injection
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
malca reproduce --manifest output/lc_manifest.parquet --candidates my_targets.parquet --output-dir output/results_repro --workers 10
```
**Note**: Reproduction uses Bayesian detection.

#### malca ltv-pipeline

LTV commands use the same run-bundle layout as detection runs. By default, new outputs go under `output/runs/ltv`.

```bash
# Full LTV workflow: source metrics, audit filtering, enrichment, and review ingest
malca ltv-pipeline --mag-bin 13_13.5

# Full LTV workflow plus external light curves and LTV multi-survey summaries
malca ltv-pipeline --stage full-extended --mag-bin 13_13.5

# Open the review app
malca review --review-db output/runs/ltv/latest/review/review.db

# Raw-dependent cluster stage only
malca ltv-pipeline --stage cluster --mag-bin 13_13.5 --export-bundle output/ltv_cluster_bundle.zip

# Home/catalog/review stage from an imported bundle
malca ltv-pipeline --stage home --import-bundle output/ltv_cluster_bundle.zip

# Open the migrated March 18 LTV bundle
malca review --review-db output/runs/ltv_march18/review/review.db
```

#### malca validate

```bash
# Auto-discover and validate ALL results for LOO method
malca validate --method loo

# Auto-discover for Bayes Factor method
malca validate --method bf

# Filter to specific magnitude bin
malca validate --method loo --mag-bin 13_13.5

# Direct file specification
malca validate --results output/results.parquet

# Validate latest detect run output (output/runs/<timestamp>/results)
malca validate --latest-run

# Validate a specific detect run directory
malca validate --run-dir output/runs/20250119_1349

# With custom candidates
malca validate --method loo --candidates my_targets.parquet -v

# Reproduce on built-in candidates using local SkyPatrol CSVs
malca validate --candidates brayden_candidates --skypatrol-dir input/skypatrol2 --method bf --workers 4

# Validate using a direct results file path
malca validate --results output/events_logbf.parquet
```

#### malca characterize

After detecting dipper candidates, characterize them using multi-wavelength data:

```bash
malca characterize --input output/filtered.parquet --output output/characterized.parquet --enable-dust --starhorse input/starhorse/starhorse2021.parquet
```

**Features:**
- **Gaia DR3 Queries**: Astrometry, astrophysics (Teff, logg, metallicity, distance), 2MASS/AllWISE photometry
- **3D Dust Extinction**: All-sky coverage via `dustmaps3d` (Wang et al. 2025, ~350MB)
- **YSO Classification**: Koenig & Leisawitz (2014) IR color-color diagram with dust correction
- **Galactic Coordinates**: Galactic longitude/latitude (l, b) from ra/dec
- **Galactic Population**: Thin/thick disk classification using metallicity or StarHorse ages
- **StarHorse** (if provided): Stellar ages, masses, distances from local catalog join
- **Auxiliary Catalog Crossmatches** (Tzanidakis+2025):
  - BANYAN Σ: Young stellar association membership probabilities
  - IPHAS DR2: Hα emission detection for Galactic plane sources
  - Star-forming regions: Proximity check to known SFRs (Prisinzano+2022)
  - Open clusters: Cantat-Gaudin+2020 membership crossmatch
  - unWISE/unTimely: Mid-IR variability z-scores
- **Caching**: Gaia results cached locally to speed up repeated analyses

**Setup:**
```bash
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
- `gal_l`, `gal_b` (Galactic coordinates)
- Auxiliary crossmatches (Tzanidakis+2025):
  - `banyan_field_prob`, `banyan_best_assoc` (BANYAN Σ membership)
  - `iphas_r_ha`, `iphas_ha_excess` (IPHAS Hα)
  - `near_sfr`, `sfr_name` (star-forming region proximity)
  - `cluster_name`, `cluster_age_myr` (open cluster membership)
  - `unwise_w1_zscore`, `unwise_w2_zscore`, `unwise_w1_var` (IR variability)

#### malca vetting

Run post-review vetting against external catalogs:

```bash
# Vet all candidates in a characterized parquet
malca vetting output/characterized.parquet -o output/vetted.parquet

# Skip slow modules
malca vetting output/characterized.parquet --no-simbad --no-alerce

# Only vet high-scoring candidates
malca vetting output/characterized.parquet --min-score 3.0

# With crash-resume checkpoint
malca vetting output/characterized.parquet --checkpoint output/vetting_checkpoint.parquet
```

**Modules** (all on by default, disable with `--no-*`):
- **SIMBAD**: Object type, bibliography, cross-IDs
- **Gaia DR3 variability**: Variable flag, classification, score
- **Gaia DR3 eclipsing binaries**: Period, morphology, global ranking
- **Gaia epoch photometry**: Availability, observation count, G-band range
- **ASAS-SN variables**: Variable star catalog crossmatch
- **ZTF variables**: Chen+ 2020 periodic variables (type, period, amplitude)
- **TNS**: Transient Name Server (name, type, redshift, discovery date)
- **ALeRCE**: ZTF broker classifications and stamp probabilities
- **eROSITA**: X-ray detection, flux, separation
- **PM consistency**: Proper motion agreement with host cluster
- **ATLAS** (opt-in, `--atlas-token`): Forced photometry light curves
- **NEOWISE** (opt-in, `--neowise-lc`): Full NEOWISE light curves

**Pipeline default:** vetting runs by default in `malca pipeline`; use `--no-run-vetting` to opt out.

**Vetting is also available during import** in the review app ("Vet on import" toggle). Results are cached per input file so re-imports skip already-vetted candidates.

#### malca classify

```bash
malca classify --input output/characterized.parquet --output output/classified.parquet
```

#### malca dev stats

```bash
malca dev stats /path/to/lc123.dat2
```

#### malca attrition

```bash
malca attrition --pre output/pre.parquet --post output/post.parquet
```

### Candidate Review

```bash
# Launch the Dash review app against an existing run bundle
malca review --plot-dir output/runs/YOUR_RUN/plots

# Standalone mode (no plot directory required)
malca review
```

**Dash review app features:**
- Native Plotly light-curve viewer with PNG fallback, camera filtering, and plot presets/overlays (raw points, dip/jump markers, residuals, phase-fold, diagnostics)
- Confidence scoring (`1-4`) via number keys or clickable buttons
- Event class labeling (single-select) with direct key shortcuts and clickable badges: `dipper`, `microlensing`, `flare`, `yso`, `unknown_interesting`, `instrumental`, `other` (toggle off to `unclassified`)
- Collapsible candidate panels with metadata health, vetting banner, external follow-up cards, diagnostic plots, and run-config provenance
- Sidebar queue controls: unreviewed/failed filters, grouped numeric/text/select filters, multi-column sort, open-existing jump, and native camera selection
- Import/fetch workflows: import tables or raw LC files (optional characterize + vet on import), or fetch by ASAS-SN ID, Gaia DR3 ID, or coordinates
- Per-candidate pipeline stage chips with "Run All Missing" / "Re-run Current", plus notes/followup/review-pass tracking and Parquet export

#### malca vsx-filter

Build the cleaned ASAS-SN index and filtered VSX catalog:
```bash
malca vsx-filter --help
malca vsx-filter --vsx-file input/vsx/vsxcat.090525.csv --masked-dir /path/to/lcsv2_masked --output-dir input/vsx
malca vsx-filter --stamp 20260213_120000   # timestamped output filenames
```
- Reads the raw fixed-width VSX catalog and filters out unwanted variability classes (eclipsing binaries, supernovae, AGN, etc.)
- Concatenates masked ASAS-SN index CSVs from all magnitude bins
- Outputs `asassn_catalog.parquet` and `vsx_cleaned.parquet` (or timestamped variants with `--stamp`)

#### malca vsx-crossmatch

Crossmatch ASAS-SN sources with VSX by position (with proper-motion correction):
```bash
malca vsx-crossmatch --help
malca vsx-crossmatch --asassn-table input/vsx/asassn_catalog.parquet --vsx-table input/vsx/vsx_cleaned.parquet
malca vsx-crossmatch --radius 5.0 --stamp 20260213_120000
```
- Propagates ASAS-SN coordinates from epoch 2016.0 to 2000.0 using proper motions
- Default match radius is 3 arcseconds
- Outputs `asassn_x_vsx_matches_{stamp}.parquet` to `input/vsx/`

## Output Directory Structure

### Integrated Pipeline

When running `malca pipeline`, the following directory structure is created for complete provenance tracking:

```
output/runs/20250121_143052/          # Timestamp-based run directory
├── run_params.json                   # Detection parameters (detect.py)
├── run_summary.json                 # Detection results stats (detect.py)
├── filter_log.json                   # Filtering parameters & stats (filter.py)
├── plot_log.json                     # Plotting parameters (plot.py)
├── run.log                           # Simple text log with paths
│
├── manifests/                        # Manifest files
│   └── lc_manifest_{mag_bin}.parquet
│
├── tags/                             # Tagging results
│   ├── lc_filtered_{mag_bin}.parquet
│   ├── lc_stats_checkpoint_{mag_bin}.parquet
│   └── rejected_tag_{mag_bin}.parquet
│
├── paths/                            # Input paths
│   └── filtered_paths_{mag_bin}.txt
│
├── results/                          # Detection results
│   ├── lc_events_results.parquet     # Raw detection output (includes dipper_score)
│   ├── lc_events_results_PROCESSED.txt  # Checkpoint log
│   ├── lc_events_results_filtered.parquet   # After filter.py
│   └── rejected_filter.parquet       # Filter rejections
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
- **Rejection tracking**: Both tagging and filter rejections are logged with reasons

**JSON Log Contents:**
- `run_params.json`: All tagging and detection parameters (thresholds, workers, baseline settings)
- `run_summary.json`: Manifest statistics, tag rejection breakdown, detection results
- `filter_log.json`: Filter toggles, thresholds, input/output counts, rejection breakdown
- `plot_log.json`: Plotting parameters, GP settings, number of plots generated

**Note:** Event scores (dipper_score, dipper_n_dips, dipper_n_valid_dips) are automatically computed during detection for significant events and included in the results table.

### LTV Run Bundle

LTV run artifacts are stored under `output/runs/<ltv_run>/`, with March 18 migrated to `output/runs/ltv_march18/`.

```
output/runs/ltv/
├── latest -> 20260527_143052
├── 20260527_143052/
│   ├── run_params.json
│   ├── run_summary.json
│   ├── run.log
│   ├── results/
│   │   ├── LTvar13-13.5.parquet
│   │   ├── LTvar13-13.5_filtered.parquet
│   │   ├── LTvar13-13.5_pipeline.parquet
│   │   ├── LTvar13-13.5_external_lcs.parquet
│   │   ├── LTvar13-13.5_ltv_multi_survey.parquet
│   │   └── external_lcs/
│   ├── review/
│   │   └── review.db
│   └── bundle_assets/
│       └── lightcurves/
```

Legacy fixed run directories and migrated LTV bundles are still supported:

```
output/runs/ltv_march18/
├── results/
│   └── LTvar13-13.5_pipeline.parquet
├── review/
│   └── review.db
└── bundle_assets/
    └── lightcurves/
```

### Standalone Module Outputs

#### Injection Testing

```
output/injection/                     # Default output directory
├── results/
│   ├── injection_results.parquet     # Trial-by-trial injection results
│   └── injection_results_PROCESSED.txt  # Checkpoint for resume
│
├── cubes/
│   └── efficiency_cube.npz           # 3D efficiency cube (depth x duration x mag)
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
│   │   ├── detection_rate_results.parquet
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
├── characterized.parquet             # Single output file with added columns:
                                      #   - Gaia astrometry & photometry
                                      #   - 3D dust extinction (A_v_3d, ebv_3d)
                                      #   - YSO classification (yso_class)
                                      #   - Galactic population (thin_disk/thick_disk)
                                      #   - StarHorse ages/masses (if provided)
                                      #   - Auxiliary crossmatches (BANYAN Σ, IPHAS, etc.)
└── cache/                            # Reusable lookup/cache products
    ├── catalogs/
    │   ├── gaia/                     # Gaia DR3 local catalog + chunks
    │   ├── sed/                      # Per-source SED catalog caches
    │   ├── characterize/             # Per-module XMatch/dust caches
    │   └── vetting/                  # SIMBAD/Gaia/ALeRCE vetting caches
    ├── lightcurves/                  # External light-curve caches
    └── joblib/                       # Joblib-backed compute caches
```

#### Dipper Classification

```
output/
└── classified.parquet                # Single output file with added columns:
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
