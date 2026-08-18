from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class InvertedPendulumCost:
    """Dense stabilization cost for Gymnasium's InvertedPendulum-v5.

    The state representation is:

        state[..., 0] = cart position
        state[..., 1] = pole angle
        state[..., 2] = cart velocity
        state[..., 3] = pole angular velocity

    Every physical quantity is divided by a scale before it is squared. This
    keeps quantities with different units comparable and lets the weights
    express their relative importance.
    """

    pole_angle_weight: float = 5.0
    pole_angular_velocity_weight: float = 1.0
    cart_position_weight: float = 0.25
    cart_velocity_weight: float = 0.1
    action_weight: float = 0.02

    pole_angle_scale: float = 0.2
    pole_angular_velocity_scale: float = 2.0
    cart_position_scale: float = 1.0
    cart_velocity_scale: float = 2.0
    action_scale: float = 3.0

    healthy_angle_limit: float = 0.2
    failure_penalty: float = 100.0
    terminal_weight: float = 10.0

    def __post_init__(self) -> None:
        """Validate weights, normalization scales, and the failure limit."""

        weights = {
            "pole_angle_weight": self.pole_angle_weight,
            "pole_angular_velocity_weight": (
                self.pole_angular_velocity_weight
            ),
            "cart_position_weight": self.cart_position_weight,
            "cart_velocity_weight": self.cart_velocity_weight,
            "action_weight": self.action_weight,
            "failure_penalty": self.failure_penalty,
            "terminal_weight": self.terminal_weight,
        }
        for name, value in weights.items():
            if not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be nonnegative and finite")

        positive_parameters = {
            "pole_angle_scale": self.pole_angle_scale,
            "pole_angular_velocity_scale": (
                self.pole_angular_velocity_scale
            ),
            "cart_position_scale": self.cart_position_scale,
            "cart_velocity_scale": self.cart_velocity_scale,
            "action_scale": self.action_scale,
            "healthy_angle_limit": self.healthy_angle_limit,
        }
        for name, value in positive_parameters.items():
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")

    def running(self, state: Tensor, action: Tensor) -> Tensor:
        """Return dense state, action-effort, and failure costs."""

        self._validate_state_shape(state)
        self._validate_action_shape(action)
        if not torch.all(torch.isfinite(action)):
            raise ValueError("action must contain only finite values")

        force = action[..., 0] / self.action_scale
        action_cost = self.action_weight * force.square()
        failure_cost = self.failure_penalty * self.failed(state).to(
            dtype=state.dtype
        )

        return self.state_cost(state) + action_cost + failure_cost

    def terminal(self, state: Tensor) -> Tensor:
        """Return a stronger final-state cost and a final failure penalty."""

        self._validate_state_shape(state)
        failure_cost = self.failure_penalty * self.failed(state).to(
            dtype=state.dtype
        )
        return self.terminal_weight * self.state_cost(state) + failure_cost

    def state_cost(self, state: Tensor) -> Tensor:
        """Return the normalized dense stabilization cost for each state."""

        self._validate_state_shape(state)

        # Non-finite observations are failures in Gymnasium. Replace their
        # invalid components only while computing the shaped terms so the
        # returned cost remains finite; ``failed`` adds the failure penalty.
        safe_state = torch.where(
            torch.isfinite(state),
            state,
            torch.zeros_like(state),
        )

        cart_position = safe_state[..., 0] / self.cart_position_scale
        pole_angle = safe_state[..., 1] / self.pole_angle_scale
        cart_velocity = safe_state[..., 2] / self.cart_velocity_scale
        pole_angular_velocity = (
            safe_state[..., 3] / self.pole_angular_velocity_scale
        )

        return (
            self.pole_angle_weight * pole_angle.square()
            + self.pole_angular_velocity_weight
            * pole_angular_velocity.square()
            + self.cart_position_weight * cart_position.square()
            + self.cart_velocity_weight * cart_velocity.square()
        )

    def failed(self, state: Tensor) -> Tensor:
        """Return Gymnasium's non-finite-or-fallen failure mask."""

        self._validate_state_shape(state)
        nonfinite = ~torch.isfinite(state).all(dim=-1)
        pole_fallen = torch.abs(state[..., 1]) > self.healthy_angle_limit
        return nonfinite | pole_fallen

    @staticmethod
    def _validate_state_shape(state: Tensor) -> None:
        if state.ndim == 0 or state.shape[-1] != 4:
            raise ValueError(
                "state must have shape (..., 4), "
                f"but received {tuple(state.shape)}"
            )

    @staticmethod
    def _validate_action_shape(action: Tensor) -> None:
        if action.ndim == 0 or action.shape[-1] != 1:
            raise ValueError(
                "action must have shape (..., 1), "
                f"but received {tuple(action.shape)}"
            )
