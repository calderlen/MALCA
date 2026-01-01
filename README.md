# ASAS-SN Variability Tools

Pipeline to search for peaks and dips in ASAS-SN light curves

## How to use
- `calder/lc_manifest.py`  
  Build a csv/parquet file mapping ASAS-SN IDs to LC paths:  
  `python calder/lc_manifest.py --index-root <path_to_index_root> --lc-root <path_to_lc_root> --out ./lc_manifest.csv [--mag-bin 12_12.5 ...]`

- `calder/lc_excursions.py`  
  Run dip/peak search (writes CSV/Parquet), with optional baseline override from `calder/lc_baseline.py`:  
  `python calder/lc_excursions.py --mode dips|peaks --out-dir ./results_dips --format csv --n-workers 10 --chunk-size 250000 [--mag-bin 12_12.5 ...]`  
  Example with a baseline: `--baseline-func lc_baseline:per_camera_median_baseline`

- `calder/reproduce_candidates.py`  
  Search for peaks/dips from a list of candidates (default: brayden list):  
  `python calder/reproduce_candidates.py --method biweight|naive --manifest ./lc_manifest.csv --out-dir ./results_repro --out-format csv [--candidates <file_or_builtin>] [--n-workers 8]`

- `calder/df_filter.py`  
  Filter the list of peaks/dips
  `python calder/df_filter.py <peaks_csv_or_dir> [--biweight] [--band g|v|both|either] [--latest-per-bin] [--output <csv>] [--output-dir <dir>]`

- `calder/fp_analysis.py`  
  Summarize false-positive reduction (pre vs post filter retention):  
  `python calder/fp_analysis.py --pre <pre_csv_or_dir> --post <post_csv_or_dir> [--id-col asas_sn_id]`

## Dependencies
- numpy
- pandas
- scipy
- astropy
- tqdm
- celerite2
- pyarrow (optional; required for Parquet outputs)
