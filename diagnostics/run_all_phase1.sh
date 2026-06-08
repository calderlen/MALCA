#!/bin/bash
# Run all Phase 1 diagnostic tests
# Total: 9 injection runs

set -e
cd "$(dirname "$0")"

echo "==================================================================="
echo "Running ALL Phase 1 Diagnostic Tests"
echo "==================================================================="
echo ""
echo "This will run 9 injection tests to diagnose:"
echo "  - Shallow dip bottleneck (3 tests)"
echo "  - Short duration bottleneck (4 tests)"
echo "  - Deep-short combination bottleneck (3 tests)"
echo ""
echo "Estimated runtime: 6-12 hours (depending on your system)"
echo ""
read -p "Press Enter to continue or Ctrl+C to cancel..."

# Phase 1A: Shallow dips
echo ""
echo "Starting Phase 1A: Shallow Dip Diagnostic..."
bash phase1_shallow_dips.sh

# Phase 1B: Short duration
echo ""
echo "Starting Phase 1B: Short Duration Diagnostic..."
bash phase1_short_duration.sh

# Phase 1C: Deep-short
echo ""
echo "Starting Phase 1C: Deep-Short Combination Diagnostic..."
bash phase1_deep_short.sh

echo ""
echo "==================================================================="
echo "ALL PHASE 1 TESTS COMPLETE!"
echo "==================================================================="
echo ""
echo "Results are in output_migrated_camera_field_20260606/injection/ with run tags:"
echo ""
echo "Shallow dips:"
echo "  - 1a_baseline, 1a_no_mag_offset, 1a_low_logbf"
echo ""
echo "Short duration:"
echo "  - 1b_baseline, 1b_short_gp, 1b_median_baseline, 1b_masked_gp"
echo ""
echo "Deep-short:"
echo "  - 1c_baseline, 1c_extended_mag_grid, 1c_deep_mag_grid"
echo ""
echo "Next step: Run the analysis script to compare efficiency cubes"
echo "==================================================================="
