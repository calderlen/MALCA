#!/bin/bash
# Phase 1A: Diagnose Shallow Dip Bottleneck
# Tests: baseline, no_mag_offset, low_logbf
# Also runs detection_rate to measure false positive rates

set -e
cd "$(dirname "$0")/.."

echo "==================================================================="
echo "Phase 1A: Shallow Dip Bottleneck Diagnostic"
echo "==================================================================="

# Test 1A-1: Baseline (current configuration)
echo ""
echo "[1/3] Running baseline (current config)..."
echo "  - Injection test (100x100 grid, 100 inj/cell = 1M trials)..."
malca dip-injection --run-tag "1a_baseline" \
  --amp-min 0.05 --amp-max 5.0 \
  --dur-min 1 --dur-max 300 \
  --total-trials 1000000 \
  --mag-points 25 \
  --workers 40 \
  --measure-pre-injection

echo "  - Detection rate (false positive measurement)..."
malca detection-rate --run-tag "1a_baseline" \
  --control-sample-size 10000 \
  --mag-points 25 \
  --workers 40

# Test 1A-2: Remove min-mag-offset filter
echo ""
echo "[2/3] Running with min-mag-offset=0.0 (remove filter)..."
echo "  - Injection test (100x100 grid, 100 inj/cell = 1M trials)..."
malca dip-injection --run-tag "1a_no_mag_offset" \
  --amp-min 0.05 --amp-max 5.0 \
  --dur-min 1 --dur-max 300 \
  --total-trials 1000000 \
  --mag-points 25 \
  --workers 40 \
  --min-mag-offset 0.0 \
  --measure-pre-injection

echo "  - Detection rate (false positive measurement)..."
malca detection-rate --run-tag "1a_no_mag_offset" \
  --control-sample-size 10000 \
  --mag-points 25 \
  --workers 40 \
  --min-mag-offset 0.0

# Test 1A-3: Lower logbf threshold
echo ""
echo "[3/3] Running with lower LogBF threshold (3.0)..."
echo "  - Injection test (100x100 grid, 100 inj/cell = 1M trials)..."
malca dip-injection --run-tag "1a_low_logbf" \
  --amp-min 0.05 --amp-max 5.0 \
  --dur-min 1 --dur-max 300 \
  --total-trials 1000000 \
  --mag-points 25 \
  --workers 40 \
  --logbf-threshold-dip 3.0 \
  --measure-pre-injection

echo "  - Detection rate (false positive measurement)..."
malca detection-rate --run-tag "1a_low_logbf" \
  --control-sample-size 10000 \
  --mag-points 25 \
  --workers 40 \
  --logbf-threshold-dip 3.0

echo ""
echo "==================================================================="
echo "Phase 1A Complete!"
echo "Results in:"
echo "  Injection (completeness):"
echo "    - output/injection/1a_baseline/"
echo "    - output/injection/1a_no_mag_offset/"
echo "    - output/injection/1a_low_logbf/"
echo ""
echo "  Detection rate (contamination):"
echo "    - output/detection_rate/1a_baseline/"
echo "    - output/detection_rate/1a_no_mag_offset/"
echo "    - output/detection_rate/1a_low_logbf/"
echo ""
echo "Next: Compare completeness vs contamination trade-offs"
echo "==================================================================="
