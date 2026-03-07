"""Compatibility event-detection settings for review modules.

This project now stores the canonical event/detection knobs in
``malca.config.config_pipeline``. Some review modules still import
``config_events``. Keep this shim so those imports continue to work.
"""

from malca.config.config_pipeline import (
    BASELINE_FUNC,
    LOGBF_THRESHOLD_DIP,
    LOGBF_THRESHOLD_JUMP,
    MAG_POINTS,
    P_POINTS,
    RUN_MAX_GAP_POINTS,
    RUN_MIN_POINTS,
    SIGNIFICANCE_THRESHOLD,
    TRIGGER_MODE,
)


# Optional Bayesian p-grid bounds; None defers to score_lightcurve defaults.
P_MIN_DIP = None
P_MAX_DIP = None
P_MIN_JUMP = None
P_MAX_JUMP = None

# Legacy naming used in review modules.
MAX_GAP_POINTS = RUN_MAX_GAP_POINTS
RUN_MAX_GAP_DAYS = None
RUN_MIN_DURATION_DAYS = None
BASELINE_TAG = BASELINE_FUNC
