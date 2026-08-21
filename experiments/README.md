# MPPI mechanics experiments

This workspace turns the eight studies in *MPPI Experiments: Understanding the
Core Mechanics* into small, reproducible investigations. The goal is to
understand one mechanism at a time, not to tune the best controller.

## Organization

```text
experiments/
|-- README.md
|-- _templates/
|   |-- report.md
|   `-- experiment.yaml
|-- 01_horizon_foresight/
|-- 02_sample_count_reliability/
|-- 03_temperature_weights/
|-- 04_warm_starting/
|-- 05_correlated_noise_structure/
|-- 06_correlated_noise_efficiency/
|-- 07_sparse_vs_dense_cost/
`-- 08_cost_scale_temperature/
```

Every numbered folder has the same contract:

- `README.md`: the living one-to-two-page report;
- `experiment.yaml`: hypothesis, controlled variables, sweep, metrics, and
  references to the existing environment configuration;
- `results/`: machine-readable CSV/JSON outputs;
- `plots/`: static figures used by the report;
- `media/`: GIF/MP4 outputs when motion materially helps interpretation.

A GIF is not mandatory. It is useful for Horizon, Warm Starting, Correlated
Noise, and Sparse-vs-Dense behavior, but the Temperature and Cost-Scale studies
are more accurately communicated with plots.

## Reuse instead of duplication

Experiment configurations point at the existing files in `configs/`. Runners
should construct the existing dynamics and environment implementations from
`src/mppi_control/`; they must not maintain experiment-local copies.

| Study | Existing environment reused | Small shared extension needed |
| --- | --- | --- |
| 01 Horizon | Mountain Car dynamics | terminal-foresight cost and candidate-trajectory diagnostics |
| 02 Sample count | Pendulum dynamics and cost | fixed-state repeated-planning runner |
| 03 Temperature | Pendulum cost batch | frozen-cost weight analysis |
| 04 Warm start | Reacher MuJoCo dynamics and cost | paired nominal evaluation and cold-start policy |
| 05 Noise structure | Mountain Car dynamics and energy cost | configurable AR(1) perturbation sampling |
| 06 Noise efficiency | Same as 05 | success and useful-trajectory aggregation |
| 07 Sparse vs dense | Mountain Car dynamics and energy cost | sparse success-only cost mode |
| 08 Cost scale | Frozen batch from 03 | offline cost/temperature scaling analysis |

These extensions belong in shared package code or a shared experiment harness,
not inside individual experiment directories.

## Recommended implementation order

The folders follow the document order, but implementation should follow
dependency and risk:

1. **03 then 08** - frozen-cost calculations are cheap and provide exact
   numerical sanity checks for weighting, ESS, and scale invariance.
2. **02** - exercises repeated one-step planning with the existing Pendulum
   implementation and establishes the results/plotting pipeline.
3. **01 then 07** - add the two Mountain Car cost variants once and reuse the
   same trajectory diagnostics.
4. **05 then 06** - add temporal correlation once, validate its statistics,
   then measure control efficiency.
5. **04** - finish with paired warm/cold evaluation in the more expensive
   MuJoCo Reacher setup.

This order gives fast checks first and postpones changes to controller state
handling until the reporting pipeline is stable.

## Reproducibility rules

- Change only the declared independent variable within a comparison.
- Keep environment seeds and MPPI sampling seeds separate in output data.
- Use the same initial state, nominal plan, and perturbation seed for paired
  comparisons whenever possible.
- Run a small pilot only to catch clearly unsuitable fixed values. Record any
  adjustment before the final sweep and apply it to every comparison arm.
- Store one row per trial in `results/runs.csv`; put aggregate values and the
  exact command/configuration in `results/summary.json`.
- Report distributions or uncertainty across sampling seeds, not only a
  visually appealing representative run.
- Generate plots and media from saved results so the report never depends on
  an unrecorded interactive session.

## Report standard

Each report answers four questions compactly:

1. What did we expect, and why?
2. What was varied, and what was held fixed?
3. What did the measurements show, including variability and runtime?
4. Did the result support the mechanism, and what alternative explanation
   remains?

Use the files in `_templates/` when adding a ninth study.
