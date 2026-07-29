"""Train and score the July 1 microlensing-like Review LightGBM model.

Run directly from the repository root:

    conda run -n malca python scripts/train_july1_microlensing_ml.py
"""

from malca.meta_analysis.ml.july1_review_training import script_main


if __name__ == "__main__":
    raise SystemExit(script_main("microlensing"))
