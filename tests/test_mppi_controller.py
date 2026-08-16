import numpy as np
import pytest
import torch

from mppi_control.controllers.mppi_controller import MPPIController
from mppi_control.costs.pendulum import PendulumCost
from mppi_control.dynamics.pendulum import PendulumDynamics


def make_controller(*, seed: int = 7) -> MPPIController:
    return MPPIController(
        PendulumDynamics(),
        PendulumCost(terminal_weight=2.0),
        horizon=8,
        num_samples=64,
        temperature=1.0,
        noise_sigma=0.8,
        action_low=-2.0,
        action_high=2.0,
        seed=seed,
    )


def test_act_returns_bounded_float32_action() -> None:
    controller = make_controller()
    observation = np.array([-1.0, 0.0, 0.0], dtype=np.float32)

    action = controller.act(observation)

    assert action.shape == (1,)
    assert action.dtype == np.float32
    assert -2.0 <= action[0] <= 2.0
    assert controller.last_costs is not None
    assert controller.last_weights is not None
    assert controller.last_costs.shape == (64,)
    assert controller.last_weights.shape == (64,)
    assert torch.isclose(controller.last_weights.sum(), torch.tensor(1.0))


def test_fixed_seed_produces_repeatable_action() -> None:
    first = make_controller(seed=11)
    second = make_controller(seed=11)
    observation = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    np.testing.assert_allclose(first.act(observation), second.act(observation))
    torch.testing.assert_close(first.nominal_actions, second.nominal_actions)


def test_reset_clears_warm_start_and_diagnostics() -> None:
    controller = make_controller()
    controller.act(np.array([-1.0, 0.0, 0.0], dtype=np.float32))

    controller.reset(seed=7)

    torch.testing.assert_close(
        controller.nominal_actions,
        torch.zeros_like(controller.nominal_actions),
    )
    assert controller.last_costs is None
    assert controller.last_weights is None
    assert controller.effective_sample_size is None


def test_lower_cost_receives_larger_weight() -> None:
    controller = make_controller()
    costs = torch.arange(64, dtype=torch.float32)

    weights = controller._compute_weights(costs)

    assert weights[0] > weights[1]
    assert weights[1] > weights[-1]
    assert torch.isclose(weights.sum(), torch.tensor(1.0))


def test_invalid_hyperparameters_are_rejected() -> None:
    with pytest.raises(ValueError, match="temperature"):
        MPPIController(
            PendulumDynamics(),
            PendulumCost(),
            horizon=8,
            num_samples=64,
            temperature=0.0,
            noise_sigma=1.0,
            action_low=-2.0,
            action_high=2.0,
        )
