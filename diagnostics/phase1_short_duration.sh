#!/bin/bash
# Phase 1B: Diagnose Short Duration Bottleneck
# Tests: baseline, short_gp, median_baseline, masked_gp
# Also runs detection_rate to measure false positive rates

set -e
cd "$(dirname "$0")/.."

echo "==================================================================="
echo "Phase 1B: Short Duration Bottleneck Diagnostic"
echo "==================================================================="

# Test 1B-1: Baseline (current configuration)
echo ""
echo "[1/4] Running baseline (current config)..."
echo "  - Injection test (100x100 grid, 100 inj/cell = 1M trials)..."
malca injection --run-tag "1b_baseline" \
  --amp-min 0.05 --amp-max 5.0 \
  --dur-min 1 --dur-max 300 \
  --total-trials 1000000 \
  --mag-points 25 \
  --workers 40 \
  --measure-pre-injection

echo "  - Detection rate (false positive measurement)..."
malca detection-rate --run-tag "1b_baseline" \
  --control-sample-size 10000 \
  --mag-points 25 \
  --workers 40

# Test 1B-2: Shorter GP timescale (~100 days instead of ~2000)
echo ""
echo "[2/4] Running with shorter GP timescale (~100 days)..."
echo "  - Injection test (100x100 grid, 100 inj/cell = 1M trials)..."
malca injection --run-tag "1b_short_gp" \
  --amp-min 0.05 --amp-max 5.0 \
  --dur-min 1 --dur-max 300 \
  --total-trials 1000000 \
  --mag-points 25 \
  --workers 40 \
  --baseline-s0 0.002 --baseline-w0 0.0314 \
  --measure-pre-injection

echo "  - Detection rate (false positive measurement)..."
malca detection-rate --run-tag "1b_short_gp" \
  --control-sample-size 10000 \
  --mag-points 25 \
  --workers 40 \
  --baseline-s0 0.002 --baseline-w0 0.0314

# Test 1B-3: Use per-camera median baseline (no GP absorption)
echo ""
echo "[3/4] Running with per-camera median baseline (no GP)..."
echo "  - Injection test (100x100 grid, 100 inj/cell = 1M trials)..."
malca injection --run-tag "1b_median_baseline" \
  --amp-min 0.05 --amp-max 5.0 \
  --dur-min 1 --dur-max 300 \
  --total-trials 1000000 \
  --mag-points 25 \
  --workers 40 \
  --baseline-func per_camera_median \
  --measure-pre-injection

echo "  - Detection rate (false positive measurement)..."
malca detection-rate --run-tag "1b_median_baseline" \
  --control-sample-size 10000 \
  --mag-points 25 \
  --workers 40 \
  --baseline-func per_camera_median

# Test 1B-4: Use masked GP
echo ""
echo "[4/4] Running with masked GP..."
echo "  - Injection test (100x100 grid, 100 inj/cell = 1M trials)..."
malca injection --run-tag "1b_masked_gp" \
  --amp-min 0.05 --amp-max 5.0 \
  --dur-min 1 --dur-max 300 \
  --total-trials 1000000 \
  --mag-points 25 \
  --workers 40 \
  --baseline-func gp_masked \
  --measure-pre-injection

echo "  - Detection rate (false positive measurement)..."
malca detection-rate --run-tag "1b_masked_gp" \
  --control-sample-size 10000 \
  --mag-points 25 \
  --workers 40 \
  --baseline-func gp_masked

echo ""
echo "==================================================================="
echo "Phase 1B Complete!"
echo "Results in:"
echo "  Injection (completeness):"
echo "    - output_migrated_camera_field_20260606/injection/1b_baseline/"
echo "    - output_migrated_camera_field_20260606/injection/1b_short_gp/"
echo "    - output_migrated_camera_field_20260606/injection/1b_median_baseline/"
echo "    - output_migrated_camera_field_20260606/injection/1b_masked_gp/"
echo ""
echo "  Detection rate (contamination):"
echo "    - output_migrated_camera_field_20260606/detection_rate/1b_baseline/"
echo "    - output_migrated_camera_field_20260606/detection_rate/1b_short_gp/"
echo "    - output_migrated_camera_field_20260606/detection_rate/1b_median_baseline/"
echo "    - output_migrated_camera_field_20260606/detection_rate/1b_masked_gp/"
echo ""
echo "Next: Compare completeness vs contamination trade-offs"
echo "==================================================================="
