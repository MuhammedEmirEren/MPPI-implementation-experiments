import torch


def rollout_costs(
    initial_state,
    action_sequences,
    dynamics,
    cost,
):
    num_samples, horizon, _ = action_sequences.shape

    trajectory_rollout = getattr(dynamics, "rollout", None)
    if callable(trajectory_rollout):
        predicted_states = trajectory_rollout(
            initial_state,
            action_sequences,
        )
        expected_shape = (
            num_samples,
            horizon,
            initial_state.shape[-1],
        )
        if predicted_states.shape != expected_shape:
            raise ValueError(
                "dynamics.rollout must return shape "
                f"{expected_shape}, but returned {tuple(predicted_states.shape)}"
            )

        # ``predicted_states[:, t]`` is the state after action t. Running
        # costs use the state before each action, matching the step-wise path.
        initial_batch = initial_state.reshape(1, 1, -1).expand(
            num_samples,
            1,
            -1,
        )
        running_states = torch.cat(
            (initial_batch, predicted_states[:, :-1, :]),
            dim=1,
        )
        running_costs = cost.running(running_states, action_sequences)
        terminal_costs = cost.terminal(predicted_states[:, -1, :])
        return running_costs.sum(dim=1) + terminal_costs

    state = initial_state.unsqueeze(0).expand(num_samples, -1)

    total_cost = torch.zeros(
        num_samples,
        device=initial_state.device,
        dtype=initial_state.dtype,
    )

    termination = getattr(cost, "terminated", None)
    active = None if termination is None else ~termination(state)

    for time_step in range(horizon):
        action = action_sequences[:, time_step, :]

        running_cost = cost.running(state, action)
        if active is not None:
            running_cost = torch.where(
                active,
                running_cost,
                torch.zeros_like(running_cost),
            )
        total_cost = total_cost + running_cost

        next_state = dynamics.step(state, action)
        if active is None:
            state = next_state
        else:
            state = torch.where(active.unsqueeze(-1), next_state, state)
            active = active & ~termination(state)

    total_cost = total_cost + cost.terminal(state)

    return total_cost
