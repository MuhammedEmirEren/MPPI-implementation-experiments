from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

@dataclass(frozen=True, slots=True)
class PendulumDynamics:
    """Predict the next Pendulum-v1 observation.

    The state representation matches Gymnasium:

        state[..., 0] = cos(theta)
        state[..., 1] = sin(theta)
        state[..., 2] = angular velocity

    The final dimensions must be:

        state:  (..., 3)
        action: (..., 1)
        result: (..., 3)

    Leading dimensions may contain MPPI samples or other batch dimensions.
    """

    # Same values with gymnassium environment
    gravity: float = 10.0
    mass: float = 1.0
    length: float = 1.0
    dt: float = 0.05
    max_torque: float = 2.0
    max_speed: float = 8.0

    def __post_init__(self) -> None:
        """Validate the physical parameters."""

        if self.gravity <= 0:
            raise ValueError("gravity must be positive")

        if self.mass <= 0:
            raise ValueError("mass must be positive")

        if self.length <= 0:
            raise ValueError("length must be positive")

        if self.dt <= 0:
            raise ValueError("dt must be positive")

        if self.max_torque <= 0:
            raise ValueError("max_torque must be positive")

        if self.max_speed <= 0:
            raise ValueError("max_speed must be positive")

    def step(self, state: Tensor, action: Tensor) -> Tensor:
        """Predict the next state after applying an action.

        Args:
            state:
                Tensor whose final dimension is
                ``[cos(theta), sin(theta), theta_dot]``.

            action:
                Tensor whose final dimension contains one torque value.
                Its leading dimensions must be broadcast-compatible with
                those of ``state``.

        Returns:
            A tensor containing
            ``[cos(next_theta), sin(next_theta), next_theta_dot]``.

        Raises:
            ValueError:
                If the final state or action dimensions are incorrect.
        """

        self._validate_shapes(state, action)

        cos_theta = state[..., 0]
        sin_theta = state[..., 1]
        theta_dot = state[..., 2]

        # Recover the signed angle in approximately [-pi, pi].
        theta = torch.atan2(sin_theta, cos_theta)

        # Remove the one-dimensional action axis and enforce the actuator limit.
        torque = torch.clamp(
            action[..., 0],
            min=-self.max_torque,
            max=self.max_torque,
        )

        # Gymnasium Pendulum-v1 equation of motion.
        theta_acceleration = (
            (3.0 * self.gravity / (2.0 * self.length))
            * torch.sin(theta)
            + (3.0 / (self.mass * self.length**2))
            * torque
        )

        # Semi-implicit Euler integration:
        # update velocity first, then update angle with the new velocity.
        next_theta_dot = theta_dot + theta_acceleration * self.dt

        next_theta_dot = torch.clamp(
            next_theta_dot,
            min=-self.max_speed,
            max=self.max_speed,
        )

        next_theta = theta + next_theta_dot * self.dt

        # Return the same observation representation used by Gymnasium.
        return torch.stack(
            (
                torch.cos(next_theta),
                torch.sin(next_theta),
                next_theta_dot,
            ),
            dim=-1,
        )

    @staticmethod
    def _validate_shapes(state: Tensor, action: Tensor) -> None:
        """Check the required final tensor dimensions."""

        if state.ndim == 0 or state.shape[-1] != 3:
            raise ValueError(
                "state must have shape (..., 3), "
                f"but received {tuple(state.shape)}"
            )

        if action.ndim == 0 or action.shape[-1] != 1:
            raise ValueError(
                "action must have shape (..., 1), "
                f"but received {tuple(action.shape)}"
            )