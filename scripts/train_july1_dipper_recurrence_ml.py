"""Train and score the human-labeled recurrent/non-recurrent dipper model.

Run directly from the repository root:

    conda run -n malca python scripts/train_july1_dipper_recurrence_ml.py
"""

from malca.meta_analysis.ml.july1_review_training import script_main


if __name__ == "__main__":
    raise SystemExit(script_main("dipper_recurrence"))
