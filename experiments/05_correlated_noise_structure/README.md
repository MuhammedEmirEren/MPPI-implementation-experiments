# Experiment 05: IID versus temporally correlated noise

**Status:** planned  
**Environment:** `MountainCarContinuous-v0`  
**Source section:** 2.1

## Question and hypothesis

Does positive correlation between neighboring action perturbations create more
temporally coherent candidate trajectories? As `rho` rises, sampled actions
should change direction less often and apply force consistently for longer,
allowing Mountain Car trajectories to build more momentum. Very high `rho`
should also make rapid reversals harder to represent.

## Design

Use the existing energy-shaped Mountain Car cost and hold `H`, `K`, `sigma`,
`lambda`, initial state, and candidate count fixed. Sweep
`rho = [0, 0.5, 0.8, 0.95]`. Generate AR(1) perturbations as
`epsilon_t = rho*epsilon_(t-1) + sqrt(1-rho^2)*sigma*z_t`, with a stationary
initial sample. Reuse common innovation seeds across rho values for paired
comparisons.

## Planned evidence

Plot example action sequences, lag-one correlation, direction changes per
sequence, run length, maximum absolute velocity, and peak energy. A compact
animation of sampled force/state trajectories may help, but statistical
validation of the sampler comes first.

## Implementation decision

Add a backward-compatible `noise_rho` parameter to shared MPPI sampling with
default `0`. Validate marginal standard deviation as well as correlation so a
change in exploration scale is not mistaken for temporal structure.

## Results and interpretation

Pending implementation and data collection.
