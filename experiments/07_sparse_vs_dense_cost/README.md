# Experiment 07: Sparse versus dense planning cost

**Status:** planned  
**Environment:** `MountainCarContinuous-v0`  
**Source section:** 3.1

## Question and hypothesis

Does MPPI receive enough information to rank unsuccessful candidates? If no
sparse-cost sample reaches the goal, costs should differ mostly through a small
action penalty and provide little guidance about momentum building. The
existing dense energy-shaped cost should separate candidates by useful
intermediate progress and produce a more informative update.

## Design

From the same initial state, compare a sparse objective
`0.05*sum(u^2) - 100*goal_reached` against the existing energy-shaped cost.
Hold `H`, `K`, `sigma`, `lambda`, nominal plan, and paired perturbation seeds
fixed. Evaluate both the one-step candidate population and closed-loop
behavior; the former isolates ranking quality while the latter tests practical
consequences.

## Planned evidence

Compare cost spread, fraction of nearly tied costs, maximum weight, ESS, first
action/update magnitude, momentum built, and success. Show aligned cost
histograms and state trajectories. A paired Mountain Car GIF can communicate
the behavioral difference after the quantitative comparison is fixed.

## Implementation decision

Reuse `MountainCarContinuousDynamics` and the existing energy cost. Add a
shared sparse cost mode/class with the same `running`, `terminal`, and
`terminated` contract; do not branch the dynamics or duplicate the runner.

## Results and interpretation

Pending implementation and data collection.
