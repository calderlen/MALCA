# Phase 1 Injection Diagnostic Tests

This directory contains scripts to systematically diagnose injection testing bottlenecks by measuring **completeness AND contamination** for each configuration.

## Quick Start

### Option 1: Run all tests at once (recommended, 12-24 hours)
```bash
cd /Volumes/drive-2tb/code/malca
bash diagnostics/run_all_phase1.sh
```

This runs **18 total tests** (9 injection + 9 detection_rate) across 9 configurations.

### Option 2: Run tests individually

```bash
cd /Volumes/drive-2tb/code/malca

# Phase 1A: Shallow dip diagnostic (6 tests: 3 injection + 3 detection_rate)
bash diagnostics/phase1_shallow_dips.sh

# Phase 1B: Short duration diagnostic (8 tests: 4 injection + 4 detection_rate)  
bash diagnostics/phase1_short_duration.sh

# Phase 1C: Deep-short combination diagnostic (6 tests: 3 injection + 3 detection_rate)
bash diagnostics/phase1_deep_short.sh
```

### Analyze Results

After tests complete:

```bash
python diagnostics/analyze_phase1.py
```

This generates:
- **Efficiency delta plots**: Where each parameter change improved completeness
- **Side-by-side comparisons**: Baseline vs modified efficiency maps
- **False positive rate bar chart**: Contamination comparison across configs
- **Numeric summaries**: Quantitative trade-off metrics

Results saved to `diagnostics/results/`

## What Gets Measured

For each configuration, we measure:

1. **Completeness (Injection)**: What % of injected dips are recovered?
   - Output: Efficiency cube (depth x duration x magnitude)
   
2. **Contamination (Detection Rate)**: What % of clean light curves trigger false positives?
   - Output: Detection rate percentage on 10,000 clean samples

**The optimal configuration balances high completeness with low contamination.**

## What Each Phase Tests

### Phase 1A: Shallow Dip Bottleneck
Tests: depth 0.05-0.5, duration 50-200 days

1. **Baseline**: Current configuration
2. **No min-mag-offset**: Remove `--min-mag-offset` filter
3. **Low LogBF**: Reduce `--logbf-threshold-dip` to 3.0

**Hypothesis**: `--min-mag-offset 0.2` is killing shallow dips

### Phase 1B: Short Duration Bottleneck  
Tests: depth 0.2-0.8, duration 1-100 days

1. **Baseline**: Current configuration (GP ~2000 day timescale)
2. **Short GP**: GP ~100 day timescale
3. **Trend baseline**: No GP (linear trend only)
4. **Masked GP**: GP with event masking

**Hypothesis**: GP baseline absorption is killing short events

### Phase 1C: Deep-Short Combination
Tests: depth 1.0-3.0 mag, duration 1-50 days

1. **Baseline**: Current configuration (auto mag grid)
2. **Extended grid**: `--mag-max-dip 18.0`
3. **Deep grid**: `--mag-max-dip 20.0`

**Hypothesis**: Mag grid truncation is killing deep dips

## Expected Findings

After analysis, you should identify:

1. **Primary bottleneck** for each failure mode
2. **Quantitative improvement** from each parameter change
3. **Optimal configuration** combining successful changes

## Next Steps

Based on Phase 1 results, create optimized configuration for Phase 2:
- Full parameter grid (depth 0.05-3.0, duration 0.5-500 days)
- Higher resolution (150x150 grid)
- More injections per grid point (200+)
- Combine all successful parameter changes
