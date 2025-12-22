todo
- obtain "recently calibrated" light curves for 15-15.5, 15.5-16 mag. dom's newly calibrated light curves only go down to 15 mag
- if you continue to fail to recover Brayden's candidates, ensure that they aren't affected by bright nearby stars and thus filtered out 
- need to collect camera# for each peak detected
- vsx_class_extract doesn't make sense to be placed in df_filter_naive
- crossmatch w/ VSX earlier inthe pipeline so you can filter out known types earlier -- can also grab data such as period (when available) to reduce LSP calls; make sure code is aware when period is detected and when it isn't
- need to update read_lc_dat to accept the new dat2 files
- parallelize the lc_excursions_bayes across light curves
    - use processes, not threads, for scaling
    - lock BLAS/OMP threads to 1
    - environment vars: OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1, NUMEXPR_NUM_THREADS=1



- advanced filtration steps
    - remove box filter
    - dynamic dispersion threshold for peak fitting, likely informed by local uncertainties
    - camera consensus during a dip, there must be a parameterized concoardance between cameras during a dip (ML or bayes stuff)
    - adapt camfilter and camfilter2 from more_utils OR ensure per_camera_baseline is robust


