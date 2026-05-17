#!/bin/bash
# Phase 2: Min-Mag-Offset Parameter Sweep
#
# Tests different min-mag-offset thresholds to find optimal balance:
# - 0.00: Maximum completeness (no amplitude filter)
# - 0.01: Very light filter
# - 0.02: Light filter
# - 0.05: Moderate filter
# - 0.10: Standard filter

set -e
cd "$(dirname "$0")/.."

echo "==================================================================="
echo "Phase 2: Min-Mag-Offset Parameter Sweep"
echo "==================================================================="
echo ""
echo "Parameter space:"
echo "  Amplitude: 0.05 - 5.0 mag (fractional depth 0.045 - 0.99)"
echo "  Duration:  1 - 300 days (log-uniform sampling)"
echo "  Mag grid:  25 points (0.2 mag resolution)"
echo "  Grid:      200 x 200 = 40,000 cells"
echo "  Trials:    200 x 200 x 100 = 4M injections per config"
echo ""
echo "Testing min-mag-offset values: 0.0, 0.01, 0.02, 0.05, 0.1"
echo ""

# Test 2-1: min-mag-offset=0.0
echo "[1/5] Running min-mag-offset=0.00 (no amplitude filter)..."
echo "  - Injection test..."
malca injection --run-tag "2_offset_0.00" \
  --amp-min 0.05 --amp-max 5.0 --amp-steps 200 \
  --dur-min 1 --dur-max 300 --dur-steps 200 \
  --n-injections-per-grid 100 \
  --mag-points 25 \
  --workers 50 \
  --min-mag-offset 0.0

echo "  - Detection rate (false positive measurement)..."
malca detection-rate --run-tag "2_offset_0.00" \
  --control-sample-size 10000 \
  --mag-points 25 \
  --workers 50 \
  --min-mag-offset 0.0

# Test 2-2: min-mag-offset=0.01
echo ""
echo "[2/5] Running min-mag-offset=0.01..."
echo "  - Injection test..."
malca injection --run-tag "2_offset_0.01" \
  --amp-min 0.05 --amp-max 5.0 --amp-steps 200 \
  --dur-min 1 --dur-max 300 --dur-steps 200 \
  --n-injections-per-grid 100 \
  --mag-points 25 \
  --workers 50 \
  --min-mag-offset 0.01

echo "  - Detection rate (false positive measurement)..."
malca detection-rate --run-tag "2_offset_0.01" \
  --control-sample-size 10000 \
  --mag-points 25 \
  --workers 50 \
  --min-mag-offset 0.01

# Test 2-3: min-mag-offset=0.02
echo ""
echo "[3/5] Running min-mag-offset=0.02..."
echo "  - Injection test..."
malca injection --run-tag "2_offset_0.02" \
  --amp-min 0.05 --amp-max 5.0 --amp-steps 200 \
  --dur-min 1 --dur-max 300 --dur-steps 200 \
  --n-injections-per-grid 100 \
  --mag-points 25 \
  --workers 50 \
  --min-mag-offset 0.02

echo "  - Detection rate (false positive measurement)..."
malca detection-rate --run-tag "2_offset_0.02" \
  --control-sample-size 10000 \
  --mag-points 25 \
  --workers 50 \
  --min-mag-offset 0.02

# Test 2-4: min-mag-offset=0.05
echo ""
echo "[4/5] Running min-mag-offset=0.05..."
echo "  - Injection test..."
malca injection --run-tag "2_offset_0.05" \
  --amp-min 0.05 --amp-max 5.0 --amp-steps 200 \
  --dur-min 1 --dur-max 300 --dur-steps 200 \
  --n-injections-per-grid 100 \
  --mag-points 25 \
  --workers 50 \
  --min-mag-offset 0.05

echo "  - Detection rate (false positive measurement)..."
malca detection-rate --run-tag "2_offset_0.05" \
  --control-sample-size 10000 \
  --mag-points 25 \
  --workers 50 \
  --min-mag-offset 0.05

# Test 2-5: min-mag-offset=0.1
echo ""
echo "[5/5] Running min-mag-offset=0.10..."
echo "  - Injection test..."
malca injection --run-tag "2_offset_0.10" \
  --amp-min 0.05 --amp-max 5.0 --amp-steps 200 \
  --dur-min 1 --dur-max 300 --dur-steps 200 \
  --n-injections-per-grid 100 \
  --mag-points 25 \
  --workers 50 \
  --min-mag-offset 0.1

echo "  - Detection rate (false positive measurement)..."
malca detection-rate --run-tag "2_offset_0.10" \
  --control-sample-size 10000 \
  --mag-points 25 \
  --workers 50 \
  --min-mag-offset 0.1

echo ""
echo "==================================================================="
echo "Phase 2 Complete!"
echo "==================================================================="
echo ""
echo "Results:"
echo "  Injection (completeness):"
echo "    - output/injection/*_2_offset_0.00/"
echo "    - output/injection/*_2_offset_0.01/"
echo "    - output/injection/*_2_offset_0.02/"
echo "    - output/injection/*_2_offset_0.05/"
echo "    - output/injection/*_2_offset_0.10/"
echo ""
echo "  Detection rate (contamination):"
echo "    - output/detection_rate/*_2_offset_0.00/"
echo "    - output/detection_rate/*_2_offset_0.01/"
echo "    - output/detection_rate/*_2_offset_0.02/"
echo "    - output/detection_rate/*_2_offset_0.05/"
echo "    - output/detection_rate/*_2_offset_0.10/"
echo ""
echo "Next: Compare completeness vs contamination across thresholds"
echo "==================================================================="
