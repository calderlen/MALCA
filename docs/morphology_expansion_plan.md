# Morphology Expansion Plan

This file defines when MALCA is ready for production morphology expansion
(EB/transit/pulsator/SN families) and what stays notebook-only until then.

## Current Policy

- Pipeline morphology remains intentionally narrow and stable.
- Exploratory morphology tuning/model competition belongs in notebooks until
  readiness gates are met.

## Readiness Gates

All gates must pass before adding new production morphology families:

1. **Trigger semantics stable**
   - `events` and `reproduce` agree on trigger interpretation.
   - `docs/trigger_semantics.md` reflects current behavior.
2. **Run gating audited**
   - Clear evidence for run-building/filtering effects on morphology outcomes.
   - Attrition notebook includes morphology-impact views.
3. **Baseline interaction documented**
   - Demonstrated behavior across GP/median baseline modes for candidate classes.
4. **Benchmark dataset established**
   - Curated positive/negative examples for each candidate family.
   - Measurable acceptance metrics (precision/recall or equivalent).

## Planned Sequence

1. Notebook prototypes (model forms + initialization + BIC diagnostics)
2. Offline benchmark evaluation and error analysis
3. Minimal production integration behind strict defaults
4. Post-deployment validation and threshold tuning

## Non-Goals (for now)

- No broad morphology CLI knob expansion in `events`/`detect`.
- No rejector-model hard-coding in the main pipeline until benchmarked.
