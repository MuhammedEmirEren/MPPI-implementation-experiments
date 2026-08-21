# Experiment 01: Planning horizon and foresight

**Status:** planned  
**Environment:** `MountainCarContinuous-v0`  
**Source section:** 1.1

## Question and hypothesis

Can a longer planning horizon change the action selected now when the useful
strategy begins by moving away from the goal? Starting from approximately
`position=-0.5, velocity=0`, the prediction is that short horizons favor local
progress, while a sufficiently long horizon can value the left-build-reverse
maneuver and select an initially leftward action.

## Design

Sweep `H = [10, 25, 50, 75, 100, 150, 200]` while fixing `K=8192`,
`lambda=1`, `sigma=0.5`, one MPPI iteration, the initial state, and all random
seed sets. Use the document's terminal-foresight objective, with no energy or
momentum shaping. Repeat across sampling seeds so a sign change in the mean
first action is not confused with one lucky candidate population.

Record the first optimized action, whether the best predicted trajectory first
moves left and later reverses, whether any candidate reaches the goal within
the horizon, the best trajectory's state/action trace, and planning time.

## Planned evidence

The primary plot will show first-action distributions versus horizon and mark
the fraction of trials containing a goal-reaching sample. A second plot will
overlay the best predicted position trajectories. A representative Mountain
Car GIF is useful only after selecting horizons from the quantitative result.

If longer horizons still fail to produce the maneuver and no candidate reaches
the goal, the result indicates inadequate sampling coverage rather than
evidence against foresight.

## Implementation decision

Reuse `MountainCarContinuousDynamics`. Add a shared terminal-foresight cost
mode and a diagnostic rollout path that can retain candidate trajectories; do
not copy the environment into this folder.

## Results and interpretation

Pending implementation and data collection.
