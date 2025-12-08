# ASAS-SN Variability Tools

Pipeline to search for peaks and dips in ASAS-SN light curves


## Modules
- `calder/`
  - `lc_excursions` - searches for dips and microlensing peaks
    - `clean_lc`
      - removes saturated light curves
      - mask any NaN's in JD and mag, may be unnecessary?
      - sort light curves by JD, may be unnecessary ?
    - `empty_metrics`
      - creates empty dict for dip metrics for a single band
    - `lc_proc_naive`
      - optionally computes baseline of a single light curve
      - finds peaks in baseline residuals for a single light curve
      - computes dip metrics for a single light curve
    - `score_dips_gaussian`
    - `compute_signifiance_dips` 
    - `compute_signifigance_peaks`
    - `lc_proc` (modeled dips or peaks via `mode`)
    - `naive_dip_finder`
    - `dip_finder`
    - `microlensing_peak_finder`

  - `lc_peaks` (thin wrapper that re-exports microlensing runner from `lc_excursions`)


## Dependencies
  - numpy
  - pandas
  - scipy
  - astropy
  - tqdm
  - pyarrow (optional; required for Parquet outputs)


