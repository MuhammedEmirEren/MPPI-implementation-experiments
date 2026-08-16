def rollout_costs(
    initial_state,
    action_sequences,
    dynamics,
    cost,
):
    num_samples, horizon, action_dim = action_sequences.shape

    state = initial_state.unsqueeze(0).expand(num_samples, -1)

    total_cost = torch.zeros(
        num_samples,
        device=initial_state.device,
        dtype=initial_state.dtype,
    )

    for time_step in range(horizon):
        action = action_sequences[:, time_step, :]

        total_cost = total_cost + cost.running(state, action)

        state = dynamics.step(state, action)

    total_cost = total_cost + cost.terminal(state)

    return total_cost