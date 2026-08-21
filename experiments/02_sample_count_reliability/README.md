# Experiment 02: Sample count and update reliability

**Status:** planned  
**Environment:** `Pendulum-v1`  
**Source section:** 1.2

## Question and hypothesis

When the planning problem is held fixed, does increasing `K` make the first
MPPI action more repeatable across sampling seeds? The prediction is that the
standard deviation of `u0` and its error relative to a higher-sample reference
will generally fall as `K` increases, while planning time rises and benefits
eventually diminish.

## Design

Use the fixed state `theta=pi/2, theta_dot=0`, represented by the existing
Pendulum observation convention as `[0, 1, 0]`. Reset the nominal action
sequence to zero before every trial. Sweep
`K = [16, 32, 64, 128, 256, 512, 1024, 2048]` with `H=40`, `lambda=1`, and
`sigma=1`. Use separate MPPI sampling seeds with the identical state and
nominal plan. Compute a paired `K_ref=3000` action for each seed where useful.

## Planned evidence

Show the distribution of `u0` at each sample count, `Std(u0)` on a log sample
axis, absolute reference error, and planning time. The main claim is about
Monte-Carlo reliability, so a control animation is not required.

## Implementation decision

Reuse `PendulumDynamics`, `PendulumCost`, and `MPPIController.reset`. A small
fixed-state runner is sufficient; no controller algorithm change is expected.

## Results and interpretation

Pending implementation and data collection.
