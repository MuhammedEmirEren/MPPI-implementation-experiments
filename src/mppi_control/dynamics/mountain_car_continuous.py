from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class MountainCarContinuousDynamics:
    """Predict the next MountainCarContinuous-v0 observation.

    The state representation matches Gymnasium:

        state[..., 0] = position
        state[..., 1] = velocity

    The final dimensions must be:

        state:  (..., 2)
        action: (..., 1)
        result: (..., 2)

    The transition order and boundary behavior match Gymnasium 1.1.1.
    """

    min_position: float = -1.2
    max_position: float = 0.6
    max_speed: float = 0.07
    min_action: float = -1.0
    max_action: float = 1.0
    power: float = 0.0015
    gravity: float = 0.0025

    def __post_init__(self) -> None:
        """Validate the physical limits and transition coefficients."""

        parameters = {
            "min_position": self.min_position,
            "max_position": self.max_position,
            "max_speed": self.max_speed,
            "min_action": self.min_action,
            "max_action": self.max_action,
            "power": self.power,
            "gravity": self.gravity,
        }
        for name, value in parameters.items():
            if not isfinite(value):
                raise ValueError(f"{name} must be finite")

        if self.min_position >= self.max_position:
            raise ValueError("min_position must be below max_position")
        if self.max_speed <= 0:
            raise ValueError("max_speed must be positive")
        if self.min_action >= self.max_action:
            raise ValueError("min_action must be below max_action")
        if self.power <= 0:
            raise ValueError("power must be positive")
        if self.gravity <= 0:
            raise ValueError("gravity must be positive")

    def step(self, state: Tensor, action: Tensor) -> Tensor:
        """Predict one transition for a state or batch of states.

        Args:
            state:
                Tensor whose final dimension is ``[position, velocity]``.
            action:
                Tensor whose final dimension contains one force value. Its
                leading dimensions must be broadcast-compatible with those of
                ``state``.

        Returns:
            Tensor containing ``[next_position, next_velocity]``.

        Raises:
            ValueError:
                If the final state or action dimensions are incorrect.
        """

        self._validate_shapes(state, action)

        position = state[..., 0]
        velocity = state[..., 1]
        force = torch.clamp(
            action[..., 0],
            min=self.min_action,
            max=self.max_action,
        )

        # Gymnasium updates velocity before position.
        next_velocity = (
            velocity
            + force * self.power
            - self.gravity * torch.cos(3.0 * position)
        )
        next_velocity = torch.clamp(
            next_velocity,
            min=-self.max_speed,
            max=self.max_speed,
        )

        next_position = position + next_velocity
        next_position = torch.clamp(
            next_position,
            min=self.min_position,
            max=self.max_position,
        )

        # The left boundary is an inelastic wall. Gymnasium removes velocity
        # only when the car is at that boundary and still moving left.
        hit_left_wall = (
            (next_position == self.min_position) & (next_velocity < 0)
        )
        next_velocity = torch.where(
            hit_left_wall,
            torch.zeros_like(next_velocity),
            next_velocity,
        )

        return torch.stack((next_position, next_velocity), dim=-1)

    @staticmethod
    def _validate_shapes(state: Tensor, action: Tensor) -> None:
        """Check the required final tensor dimensions."""

        if state.ndim == 0 or state.shape[-1] != 2:
            raise ValueError(
                "state must have shape (..., 2), "
                f"but received {tuple(state.shape)}"
            )
        if action.ndim == 0 or action.shape[-1] != 1:
            raise ValueError(
                "action must have shape (..., 1), "
                f"but received {tuple(action.shape)}"
            )
