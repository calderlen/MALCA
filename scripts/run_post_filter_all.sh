#!/bin/bash
# Run filter on all detect run directories in output_migrated_camera_field_20260606/runs/

set -e

RUNS_DIR="${1:-output_migrated_camera_field_20260606/runs}"

if [ ! -d "$RUNS_DIR" ]; then
    echo "Error: $RUNS_DIR does not exist"
    exit 1
fi

echo "Processing all runs in $RUNS_DIR"
echo "================================"

for run_dir in "$RUNS_DIR"/*/; do
    run_name=$(basename "$run_dir")

    # Check if results directory exists
    if [ ! -d "$run_dir/results" ]; then
        echo "[$run_name] Skipping - no results/ directory"
        continue
    fi

    # Check if there's an events results file
    if ! ls "$run_dir/results/"*events_results*.parquet 2>/dev/null | head -1 > /dev/null; then
        echo "[$run_name] Skipping - no events results file"
        continue
    fi

    echo ""
    echo "[$run_name] Running filter..."
    python -m malca.filter --detect-run "$run_dir" -v || {
        echo "[$run_name] Failed!"
        continue
    }
    echo "[$run_name] Done"
done

echo ""
echo "================================"
echo "All runs processed"
