# Experiment 06: Sampling efficiency from correlated noise

**Status:** planned  
**Environment:** `MountainCarContinuous-v0`  
**Source section:** 2.2

## Question and hypothesis

For a fixed rollout budget, can a proposal with appropriate temporal structure
place more candidates in the useful part of action-sequence space? Moderate
correlation (`rho=0.8`) is expected to yield a larger useful-trajectory
fraction and reach comparable closed-loop success with fewer samples than IID
noise.

## Design

Compare `rho = [0, 0.8]` across `K = [32, 64, 128, 256, 512]`. Keep the
existing energy-shaped cost, horizon, `sigma`, `lambda`, environment initial
conditions, and paired seed sets fixed. Before running the final sweep, define
a useful sampled trajectory as one reaching `max(abs(velocity)) >= 0.04`
within the planning horizon. Do not tune this threshold after seeing results.

Measure the useful-candidate fraction at the planning level and success rate at
the closed-loop task level. Also save time, episode length, return, and peak
velocity so an apparent sample-efficiency improvement can be interpreted.

## Planned evidence

Plot useful-trajectory fraction and success rate against `K`, with uncertainty
over seeds. Explicitly compare the smallest correlated `K` that matches each
IID operating point. A representative GIF is optional and secondary.

## Implementation decision

Reuse the sampler and trajectory diagnostics introduced by Experiment 05.
This folder should add aggregation only, not another sampler implementation.

## Results and interpretation

Pending implementation and data collection.
