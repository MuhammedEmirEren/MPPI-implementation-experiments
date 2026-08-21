"""Planning cost used only by the horizon-foresight experiment."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class MountainCarHorizonCost:
    """Terminal-foresight cost without energy or momentum shaping.

    The running cost contains only action effort. Position error, wrong-way
    terminal velocity, and the success bonus are evaluated at the end of the
    predicted trajectory.

    State and action shapes are:

        state:  (..., 2) containing [position, velocity]
        action: (..., 1) containing force
        cost:   (...)    containing one scalar per batch element
    """

    terminal_position_weight: float = 20.0
    terminal_wrong_direction_weight: float = 5.0
    action_weight: float = 0.05
    success_bonus: float = 100.0
    goal_position: float = 0.45
    goal_velocity: float = 0.0

    def __post_init__(self) -> None:
        """Require finite, nonnegative weights and a finite goal state."""

        weights = {
            "terminal_position_weight": self.terminal_position_weight,
            "terminal_wrong_direction_weight": (
                self.terminal_wrong_direction_weight
            ),
            "action_weight": self.action_weight,
            "success_bonus": self.success_bonus,
        }
        for name, value in weights.items():
            if not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")

        goal = {
            "goal_position": self.goal_position,
            "goal_velocity": self.goal_velocity,
        }
        for name, value in goal.items():
            if not isfinite(value):
                raise ValueError(f"{name} must be finite")

    def running(self, state: Tensor, action: Tensor) -> Tensor:
        """Return the quadratic action-effort cost for each rollout step."""

        self._validate_state_shape(state)
        self._validate_action_shape(action)

        force = action[..., 0]
        return self.action_weight * force.square()

    def terminal(self, state: Tensor) -> Tensor:
        """Return terminal position, direction, and success terms."""

        self._validate_state_shape(state)
        position = state[..., 0]
        velocity = state[..., 1]

        position_cost = self.terminal_position_weight * (
            position - self.goal_position
        ).square()

        negative_velocity = torch.minimum(
            velocity,
            torch.zeros_like(velocity),
        )
        is_before_goal = position < self.goal_position
        wrong_direction_cost = (
            self.terminal_wrong_direction_weight
            * is_before_goal.to(dtype=state.dtype)
            * negative_velocity.square()
        )

        success_reward_as_cost = (
            self.success_bonus
            * self.terminated(state).to(dtype=state.dtype)
        )

        return position_cost + wrong_direction_cost - success_reward_as_cost

    def terminated(self, state: Tensor) -> Tensor:
        """Return Gymnasium's Mountain Car goal condition."""

        self._validate_state_shape(state)
        position = state[..., 0]
        velocity = state[..., 1]
        return (
            (position >= self.goal_position)
            & (velocity >= self.goal_velocity)
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
