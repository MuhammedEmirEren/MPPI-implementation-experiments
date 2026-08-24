from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class SparseMountainCarCost:
    """Success-only Mountain Car objective used in Experiment 07."""

    action_weight: float = 0.05
    success_bonus: float = 100.0
    goal_position: float = 0.45
    goal_velocity: float = 0.0

    def __post_init__(self) -> None:
        parameters = {
            "action_weight": self.action_weight,
            "success_bonus": self.success_bonus,
            "goal_position": self.goal_position,
            "goal_velocity": self.goal_velocity,
        }
        for name, value in parameters.items():
            if not isfinite(value):
                raise ValueError(f"{name} must be finite")
        for name in ("action_weight", "success_bonus"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")

    def running(self, state: Tensor, action: Tensor) -> Tensor:
        """Penalize action effort without rewarding intermediate progress."""

        self._validate_state_shape(state)
        self._validate_action_shape(action)
        return self.action_weight * action[..., 0].square()

    def terminal(self, state: Tensor) -> Tensor:
        """Give a bonus only if the trajectory reached the goal."""

        self._validate_state_shape(state)
        return -self.success_bonus * self.terminated(state).to(state.dtype)

    def terminated(self, state: Tensor) -> Tensor:
        """Return Gymnasium's Mountain Car goal condition."""

        self._validate_state_shape(state)
        return (
            (state[..., 0] >= self.goal_position)
            & (state[..., 1] >= self.goal_velocity)
        )

    @staticmethod
    def _validate_state_shape(state: Tensor) -> None:
        if state.ndim == 0 or state.shape[-1] != 2:
            raise ValueError(
                "state must have shape (..., 2), "
                f"but received {tuple(state.shape)}"
            )

    @staticmethod
    def _validate_action_shape(action: Tensor) -> None:
        if action.ndim == 0 or action.shape[-1] != 1:
            raise ValueError(
                "action must have shape (..., 1), "
                f"but received {tuple(action.shape)}"
            )
