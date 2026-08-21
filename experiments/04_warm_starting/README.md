# Experiment 04: Warm-starting across MPC cycles

**Status:** planned  
**Environment:** `Reacher-v5`  
**Source section:** 1.4

## Question and hypothesis

Does shifting the unused portion of the previous optimized sequence preserve
useful information at the next MPC state? After the controller has begun to
form a useful plan, the shifted nominal is expected to have lower pre-update
cost than an all-zero nominal more often than not, especially at modest `K`.

## Design

During a normal paired episode, optimize, execute only the first action, and
observe the new state. Before another update, evaluate both the shifted nominal
and zero nominal from that exact same state. Then compare full warm-start and
cold-start controllers at `K = [64, 128, 256]` using the same environment
initialization and sampling-seed policy.

Primary evidence is the paired difference
`J_warm_initial - J_cold_initial` at every MPC step, followed by task metrics
such as distance, first reach, and return. Planning time and ESS are secondary.

## Planned evidence

Use a paired-difference distribution and time-series plot. A side-by-side
Reacher GIF for one preselected representative seed can make trajectory
continuity visible, but it must not replace aggregate results.

## Implementation decision

Reuse `ReacherMujocoDynamics` and `ReacherCost`. Add safe public operations to
evaluate and initialize nominal plans, or a shared experiment adapter; do not
reach into `_nominal_actions` from an experiment script.

## Results and interpretation

Pending implementation and data collection.
