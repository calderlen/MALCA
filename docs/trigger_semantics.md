# Trigger Semantics

This document is the source of truth for how MALCA computes and consumes
`logBF`/LOO-trigger quantities across stages.

## Core Definitions

- `log_bf_local`: Per-point local log Bayes factor from the branch likelihood ratio.
- `event_probability`: Per-point leave-one-out posterior event probability.
- `bayes_factor`: Branch-level global evidence contrast
  (`log_evidence_mixture - log_evidence_baseline`).
- `trigger_mode`:
  - `logbf`: per-point trigger on `log_bf_local >= logbf_threshold_*`
  - `posterior_prob`: per-point trigger on
    `event_probability >= significance_threshold`
    (`significance_threshold > 1` interpreted as percent)

## Stage-by-Stage Usage

| Stage | Metric(s) consumed | Purpose | Trigger/Filter behavior |
|---|---|---|---|
| `events` | `log_bf_local`, `event_probability` | Build point triggers and run candidates | Triggering uses `trigger_mode`; run gating then applies min points/duration/threshold |
| `detect` | forwards trigger args to `events` | Configure upstream trigger behavior | No independent trigger logic; argument pass-through only |
| `post_filter` | `dip_bayes_factor`/`jump_bayes_factor`, `dip_max_log_bf_local`/`jump_max_log_bf_local`, `dip_significant`/`jump_significant` | Candidate quality filtering | Evidence filter uses global BF (+ optional finite local BF gate); significant-detection filter uses run/peak/significant flags |
| `review` | `trigger_mode`, `*_trigger_threshold`, `*_max_event_prob`, `*_max_log_bf_local`, `failed_*` flags | Operator context + interactive filtering | Display/queue filtering only; no trigger recomputation |
| `reproduce` | `score_lightcurve` output with trigger metadata; post-filter fields | Cross-check known candidates under pipeline settings | Uses same trigger logic via shared helper path; then optional post-filter evaluation |

## Current Defaults

- Default trigger mode: `posterior_prob`
- Default significance threshold: `SIGNIFICANCE_THRESHOLD`
- `logbf` mode remains available for backward compatibility and controlled comparisons.

## Guardrails

- Keep trigger decision code in one helper path (`malca.triggering`) and call it
  from both `events` and `reproduce`-normalization logic.
- Avoid introducing new per-stage trigger variants without updating this table and tests.
