# Experiment 03: Temperature and weight concentration

**Status:** planned  
**Environment:** `Pendulum-v1`  
**Source section:** 1.3

## Question and hypothesis

If candidate trajectories and their costs are frozen, how does temperature
alone change MPPI selection? Lower `lambda` should concentrate mass on a few
low-cost samples (`max weight` up, ESS down); higher `lambda` should spread
weight more evenly and approach ESS `K`.

## Design

Generate one Pendulum candidate batch with `K=1024` and save the resulting cost
vector once. Reuse that exact vector for
`lambda = [0.05, 0.1, 0.3, 1, 3, 10, 30]`; do not resample trajectories. Save
the cost-batch identifier in every derived row so the frozen-population claim
is auditable.

## Planned evidence

Plot maximum weight and normalized ESS (`ESS/K`) against temperature. A second
figure can show sorted weight curves for a few temperatures. Exact checks
should confirm nonnegative normalized weights and the approach to uniform
weighting at high temperature. A GIF would add little here.

## Implementation decision

Reuse the existing Pendulum rollout to obtain costs. Put a public, stateless
weight/ESS helper in shared code rather than changing a controller's private
temperature repeatedly.

## Results and interpretation

Pending implementation and data collection.
