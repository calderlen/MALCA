# Changelog


## 19-11-2025
-  switched from Boolean masks to binary search w/ `np.searchsorted` in `lc_baseline.rolling_time_median` and `lc_baseline.rolling_time_mad`

## 17-11-2025
- Added a CLI entry point directly to `df_filter_naive`, eliminating the need for the separate `script_filter_naive`.
- Removed `script_filter_naive` to avoid duplicated logic.
- Added `build_manifest` to search for asas-sn ID's in `lc_cal` subdirs BEFORE doing reproduction check, so `reproduce_candidates` only needs to check against a csv file instead of searching through all `mag_bins` each time.
- Edited `reproduce_candidates` to match the `brayden_candidates` with `lc_manifest.csv` instead of iterating through all `mag_bins` each time. 
