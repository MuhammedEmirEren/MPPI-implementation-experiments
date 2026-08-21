# Experiment NN: Title

**Status:** planned  
**Environment:** `Environment-vN`  
**Source section:** N.N

## Question and hypothesis

State one mechanism-level question and one falsifiable prediction. Do not frame
the experiment as hyperparameter tuning.

## Design

- **Independent variable:** one declared sweep.
- **Controlled variables:** environment/model, initial state, nominal plan,
  cost, seed policy, and all remaining MPPI parameters.
- **Repetitions:** pilot and final sampling-seed counts.
- **Primary metrics:** quantities that directly test the hypothesis.
- **Secondary metrics:** runtime and diagnostics that help explain failures.

## Results

Summarize the saved data and embed only the most informative plots or media.
Include uncertainty or the distribution across seeds.

## Interpretation

Say whether the observation supports the hypothesis. Separate lack of
foresight, lack of sampling coverage, weak cost information, and implementation
failure when they are plausible alternatives.

## Reproduce

Record the exact command after the shared experiment runner exists.
