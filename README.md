# ASAS-SN Variability Tools

Pipeline to search for peaks and dips in ASAS-SN light curves

## how to use
- `scripts/lc_manifest.py`  
  Build a csv/parquet file mapping ASAS-SN IDs to LC paths:  
  `python scripts/lc_manifest.py --index-root <path_to_index_root> --lc-root <path_to_lc_root> --out ./lc_manifest.csv [--mag-bin 12_12.5 ...]`

- `src/excursions_bayes.py`  
  Bayesian dip/peak search (writes CSV/Parquet), with optional baseline override from `src/baseline.py`:  
  `python src/excursions_bayes.py --mode dips|peaks --out-dir ./results_bayes --chunk-size 5000 [--mag-bin 12_12.5 ...]`  
  Example with a baseline: `--baseline-func baseline:per_camera_median_baseline`

- `scripts/reproduce_candidates.py`  
  Search for peaks/dips from a list of candidates (default: Brayden list):  
  `python scripts/reproduce_candidates.py --method biweight|naive --manifest ./lc_manifest.csv --out-dir ./results_repro --out-format csv [--candidates <file_or_builtin>] [--n-workers 8]`

- `src/filter.py`  
  Filter the list of peaks/dips:  
  `python src/filter.py <peaks_csv_or_dir> [--biweight] [--band g|v|both|either] [--latest-per-bin] [--output <csv>] [--output-dir <dir>]`

- `scripts/fp_analysis.py`  
  Summarize false-positive reduction (pre vs post filter retention):  
  `python scripts/fp_analysis.py --pre <pre_csv_or_dir> --post <post_csv_or_dir> [--id-col asas_sn_id]`

## dependencies
- numpy
- pandas
- scipy
- astropy
- tqdm
- celerite2
- pyarrow (optional; required for Parquet outputs)

## layout
```
.
├── docs
│   ├── ARCHITECTURE.md
│   ├── NOTES.md
│   └── TODO.md
├── environment.yml
├── .gitignore
├── pyproject.toml
├── README.md
├── scripts
│   ├── fp_analysis.py
│   ├── lc_manifest.py
│   ├── plot_results_bayes.py
│   ├── plot_results.py
│   └── reproduce_candidates.py
├── src
│   ├── baseline.py
│   ├── df_utils.py
│   ├── excursions_bayes.py
│   ├── filter.py
│   ├── lc_utils.py
│   ├── old
│   │   ├── __init__.py
│   │   ├── lc_excursions_naive.py
│   │   ├── lc_excursions.py
│   │   └── lc_metrics.py
│   ├── plot.py
│   ├── stats.py
│   ├── test
│   │   ├── test.py
│   │   └── test_skypatrol.py
│   └── vsx
│       ├── crossmatch.py
│       ├── filter.py
│       └── reproducibility.py
└── .vscode
    └── extensions.json
```
