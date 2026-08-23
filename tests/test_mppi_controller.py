import numpy as np
import pytest
import torch

from mppi_control.controllers.mppi_controller import MPPIController
from mppi_control.costs.pendulum import PendulumCost
from mppi_control.dynamics.pendulum import PendulumDynamics


def make_controller(
    *,
    seed: int = 7,
    noise_rho: float = 0.0,
) -> MPPIController:
    return MPPIController(
        PendulumDynamics(),
        PendulumCost(terminal_weight=2.0),
        horizon=8,
        num_samples=64,
        temperature=1.0,
        noise_sigma=0.8,
        noise_rho=noise_rho,
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


def test_correlated_noise_has_stationary_scale_and_requested_correlation() -> None:
    controller = MPPIController(
        PendulumDynamics(),
        PendulumCost(),
        horizon=64,
        num_samples=1024,
        temperature=1.0,
        noise_sigma=0.8,
        noise_rho=0.7,
        action_low=-2.0,
        action_high=2.0,
        seed=13,
    )

    perturbations = controller._sample_perturbations()[..., 0]
    previous = perturbations[:, :-1].reshape(-1)
    following = perturbations[:, 1:].reshape(-1)
    observed_correlation = torch.corrcoef(
        torch.stack((previous, following))
    )[0, 1]

    assert float(perturbations.std()) == pytest.approx(0.8, abs=0.02)
    assert float(observed_correlation) == pytest.approx(0.7, abs=0.02)


def test_correlated_sampling_cross_term_uses_ar1_precision() -> None:
    controller = MPPIController(
        PendulumDynamics(),
        PendulumCost(),
        horizon=4,
        num_samples=2,
        temperature=1.3,
        noise_sigma=0.8,
        noise_rho=0.6,
        action_low=-2.0,
        action_high=2.0,
    )
    controller._nominal_actions.copy_(
        torch.tensor([[0.2], [-0.1], [0.4], [0.3]])
    )
    perturbations = torch.tensor(
        [
            [[0.3], [-0.2], [0.1], [0.5]],
            [[-0.4], [0.2], [0.6], [-0.1]],
        ]
    )

    indices = torch.arange(controller.horizon)
    correlation = controller.noise_rho ** torch.abs(
        indices[:, None] - indices[None, :]
    )
    covariance = controller.noise_sigma**2 * correlation
    expected = controller.temperature * torch.einsum(
        "ha,ht,kta->k",
        controller.nominal_actions,
        torch.linalg.inv(covariance),
        perturbations,
    )

    torch.testing.assert_close(
        controller._sampling_cross_term(perturbations),
        expected,
    )


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

    with pytest.raises(ValueError, match="noise_rho"):
        MPPIController(
            PendulumDynamics(),
            PendulumCost(),
            horizon=8,
            num_samples=64,
            temperature=1.0,
            noise_sigma=1.0,
            noise_rho=1.0,
            action_low=-2.0,
            action_high=2.0,
        )
