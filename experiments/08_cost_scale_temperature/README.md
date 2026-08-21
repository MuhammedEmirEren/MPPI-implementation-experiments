# Experiment 08: Cost scale and temperature

**Status:** planned  
**Environment:** `Pendulum-v1`  
**Source section:** 3.2

## Question and hypothesis

Does multiplying every cost by a positive constant leave MPPI behavior
unchanged when temperature is fixed? Although trajectory ordering is
unchanged, weights depend on `S/lambda`. Scaling costs by ten alone should
concentrate weights; scaling both costs and temperature by ten should recover
the original weights to numerical precision.

## Design

Reuse one exact frozen cost vector from Experiment 03. Compare `(S, lambda)`,
`(10*S, lambda)`, and `(10*S, 10*lambda)` without regenerating trajectories.
Record maximum weight, ESS, and the maximum absolute difference from the
original weight vector.

## Planned evidence

A three-case weight-distribution plot and a compact metrics table are enough.
The invariant case should have a strict numerical check, initially
`max_abs_weight_difference <= 1e-6` for float32. No animation is needed.

## Implementation decision

Reuse Experiment 03's frozen-batch format and stateless weighting helper. This
experiment should contain no environment or controller modifications.

## Results and interpretation

Pending implementation and data collection.
