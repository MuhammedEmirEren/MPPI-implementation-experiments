from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


EXPERIMENT_DIR = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "01_horizon_foresight"
)
sys.path.insert(0, str(EXPERIMENT_DIR))

from cost import MountainCarHorizonCost  # noqa: E402


def test_horizon_running_cost_contains_only_action_effort() -> None:
    cost = MountainCarHorizonCost(action_weight=0.05)
    state = torch.tensor([[-0.5, 0.0], [-0.2, 0.03]])
    action = torch.tensor([[0.5], [-1.0]])

    actual = cost.running(state, action)

    expected = torch.tensor([0.0125, 0.05])
    torch.testing.assert_close(actual, expected)


def test_horizon_terminal_cost_matches_the_experiment_equation() -> None:
    cost = MountainCarHorizonCost()
    state = torch.tensor(
        [
            [-0.5, -0.02],
            [-0.5, 0.02],
            [0.45, 0.0],
        ]
    )

    actual = cost.terminal(state)

    position_cost = 20.0 * (-0.5 - 0.45) ** 2
    expected = torch.tensor(
        [
            position_cost + 5.0 * (-0.02) ** 2,
            position_cost,
            -100.0,
        ]
    )
    torch.testing.assert_close(actual, expected)


def test_horizon_goal_requires_position_and_velocity_thresholds() -> None:
    cost = MountainCarHorizonCost()
    state = torch.tensor(
        [
            [0.45, 0.0],
            [0.46, -0.01],
            [0.44, 0.01],
        ]
    )

    actual = cost.terminated(state)

    expected = torch.tensor([True, False, False])
    torch.testing.assert_close(actual, expected)


def test_horizon_cost_rejects_negative_weights() -> None:
    with pytest.raises(ValueError, match="action_weight"):
        MountainCarHorizonCost(action_weight=-0.01)
