from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


EXPERIMENT_DIR = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "07_sparse_vs_dense_cost"
)
sys.path.insert(0, str(EXPERIMENT_DIR))

from sparse_cost import SparseMountainCarCost  # noqa: E402


def test_sparse_cost_only_charges_action_effort_before_success() -> None:
    cost = SparseMountainCarCost(action_weight=0.05)
    states = torch.tensor([[-0.5, 0.0], [0.2, 0.04]])
    actions = torch.tensor([[0.5], [-1.0]])

    actual = cost.running(states, actions)

    torch.testing.assert_close(actual, torch.tensor([0.0125, 0.05]))


def test_sparse_cost_only_rewards_goal_reaching_states() -> None:
    cost = SparseMountainCarCost(success_bonus=100.0)
    states = torch.tensor(
        [
            [0.44, 0.02],
            [0.45, -0.01],
            [0.45, 0.0],
            [0.50, 0.01],
        ]
    )

    actual = cost.terminal(states)

    torch.testing.assert_close(actual, torch.tensor([0.0, 0.0, -100.0, -100.0]))


def test_sparse_cost_rejects_negative_weights() -> None:
    with pytest.raises(ValueError, match="action_weight"):
        SparseMountainCarCost(action_weight=-0.01)
