from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class ReacherCost:
    """Dense MPPI planning cost for Gymnasium's Reacher-v5.

    The state representation is:

        state[..., 0:2] = cosine of the two joint angles
        state[..., 2:4] = sine of the two joint angles
        state[..., 4:6] = target x and y coordinates
        state[..., 6:8] = angular velocity of the two joints
        state[..., 8:10] = fingertip position minus target position

    Gymnasium rewards small Euclidean fingertip distance and low squared
    control effort. This class expresses those rewards as positive costs and
    optionally adds a squared joint-velocity penalty to discourage aggressive
    movement. Reacher does not terminate when the fingertip reaches the target,
    so no success or failure mask is needed.
    """

    distance_weight: float = 1.0
    action_weight: float = 1.0
    joint_velocity_weight: float = 0.0
    terminal_distance_weight: float = 10.0

    def __post_init__(self) -> None:
        """Require every cost weight to be finite and nonnegative."""

        weights = {
            "distance_weight": self.distance_weight,
            "action_weight": self.action_weight,
            "joint_velocity_weight": self.joint_velocity_weight,
            "terminal_distance_weight": self.terminal_distance_weight,
        }
        for name, value in weights.items():
            if not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be nonnegative and finite")

    def running(self, state: Tensor, action: Tensor) -> Tensor:
        """Return fingertip-distance, action, and joint-speed costs.

        Args:
            state: Reacher observation or batch with shape ``(..., 10)``.
            action: Two joint torques with shape ``(..., 2)``.

        Returns:
            One scalar cost per broadcast state/action batch element.
        """

        self._validate_state(state)
        self._validate_action(action)

        distance_cost = self.distance_weight * self.distance(state)
        action_cost = self.action_weight * action.square().sum(dim=-1)
        joint_velocity_cost = (
            self.joint_velocity_weight
            * state[..., 6:8].square().sum(dim=-1)
        )

        return distance_cost + action_cost + joint_velocity_cost

    def terminal(self, state: Tensor) -> Tensor:
        """Return a stronger final fingertip-distance cost."""

        self._validate_state(state)
        return self.terminal_distance_weight * self.distance(state)

    def distance(self, state: Tensor) -> Tensor:
        """Return Euclidean fingertip-to-target distance for each state."""

        self._validate_state(state)
        fingertip_error = state[..., 8:10]
        return torch.linalg.vector_norm(fingertip_error, dim=-1)

    @staticmethod
    def _validate_state(state: Tensor) -> None:
        """Validate a Reacher observation tensor."""

        if not isinstance(state, Tensor):
            raise TypeError("state must be a torch.Tensor")
        if state.ndim == 0 or state.shape[-1] != 10:
            raise ValueError(
                "state must have shape (..., 10), "
                f"but received {tuple(state.shape)}"
            )
        if not state.dtype.is_floating_point:
            raise ValueError("state must use a floating-point dtype")
        if not torch.all(torch.isfinite(state)):
            raise ValueError("state must contain only finite values")

    @staticmethod
    def _validate_action(action: Tensor) -> None:
        """Validate a two-torque action tensor."""

        if not isinstance(action, Tensor):
            raise TypeError("action must be a torch.Tensor")
        if action.ndim == 0 or action.shape[-1] != 2:
            raise ValueError(
                "action must have shape (..., 2), "
                f"but received {tuple(action.shape)}"
            )
        if not action.dtype.is_floating_point:
            raise ValueError("action must use a floating-point dtype")
        if not torch.all(torch.isfinite(action)):
            raise ValueError("action must contain only finite values")
