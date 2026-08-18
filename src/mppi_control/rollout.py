import torch


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
