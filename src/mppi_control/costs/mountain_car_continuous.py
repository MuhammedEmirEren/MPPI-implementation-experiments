from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sin
from typing import Literal

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class MountainCarContinuousCost:
    """Energy-shaped MPPI planning cost for MountainCarContinuous-v0.

    This is intentionally denser than Gymnasium's sparse evaluation reward.
    It encourages the car to build enough mechanical energy to reach the goal,
    while retaining Gymnasium's quadratic action-effort penalty.

    State and action shapes are:

        state:  (..., 2) containing [position, velocity]
        action: (..., 1) containing force
        cost:   (...)    containing one scalar per state or batch element
    """

    mode: Literal["energy_shaped"] = "energy_shaped"
    energy_weight: float = 1.0
    action_weight: float = 0.1
    terminal_position_weight: float = 20.0
    terminal_wrong_direction_weight: float = 5.0
    success_bonus: float = 100.0

    min_position: float = -1.2
    max_position: float = 0.6
    max_speed: float = 0.07
    goal_position: float = 0.45
    goal_velocity: float = 0.0
    gravity: float = 0.0025

    def __post_init__(self) -> None:
        """Validate weights, physical scales, and goal parameters."""

        if self.mode != "energy_shaped":
            raise ValueError("mode must be 'energy_shaped'")

        parameters = {
            "energy_weight": self.energy_weight,
            "action_weight": self.action_weight,
            "terminal_position_weight": self.terminal_position_weight,
            "terminal_wrong_direction_weight": (
                self.terminal_wrong_direction_weight
            ),
            "success_bonus": self.success_bonus,
            "min_position": self.min_position,
            "max_position": self.max_position,
            "max_speed": self.max_speed,
            "goal_position": self.goal_position,
            "goal_velocity": self.goal_velocity,
            "gravity": self.gravity,
        }
        for name, value in parameters.items():
            if not isfinite(value):
                raise ValueError(f"{name} must be finite")

        weights = {
            "energy_weight": self.energy_weight,
            "action_weight": self.action_weight,
            "terminal_position_weight": self.terminal_position_weight,
            "terminal_wrong_direction_weight": (
                self.terminal_wrong_direction_weight
            ),
            "success_bonus": self.success_bonus,
        }
        for name, value in weights.items():
            if value < 0:
                raise ValueError(f"{name} must be nonnegative")

        if self.min_position >= self.max_position:
            raise ValueError("min_position must be below max_position")
        if not self.min_position < self.goal_position <= self.max_position:
            raise ValueError("goal_position must lie within the track")
        if self.max_speed <= 0:
            raise ValueError("max_speed must be positive")
        if not -self.max_speed <= self.goal_velocity <= self.max_speed:
            raise ValueError("goal_velocity must lie within the speed limits")
        if self.gravity <= 0:
            raise ValueError("gravity must be positive")
        if self.energy_scale <= 0:
            raise ValueError("goal parameters must define a positive energy scale")

    @property
    def goal_energy(self) -> float:
        """Approximate mechanical energy required at the goal state."""

        kinetic_energy = 0.5 * self.goal_velocity**2
        potential_energy = (
            self.gravity / 3.0 * sin(3.0 * self.goal_position)
        )
        return kinetic_energy + potential_energy

    @property
    def energy_scale(self) -> float:
        """Energy gap from the lowest point of the valley to the goal."""

        minimum_potential_energy = -self.gravity / 3.0
        return self.goal_energy - minimum_potential_energy

    def energy(self, state: Tensor) -> Tensor:
        """Return approximate mechanical energy for each state."""

        self._validate_state_shape(state)
        position = state[..., 0]
        velocity = state[..., 1]
        kinetic_energy = 0.5 * velocity.square()
        potential_energy = (
            self.gravity / 3.0 * torch.sin(3.0 * position)
        )
        return kinetic_energy + potential_energy

    def running(self, state: Tensor, action: Tensor) -> Tensor:
        """Return normalized energy-deficit and action-effort cost."""

        self._validate_state_shape(state)
        self._validate_action_shape(action)

        energy_deficit = torch.clamp(
            (self.goal_energy - self.energy(state)) / self.energy_scale,
            min=0.0,
        )
        force = action[..., 0]

        energy_cost = self.energy_weight * energy_deficit.square()
        action_cost = self.action_weight * force.square()
        return energy_cost + action_cost

    def terminal(self, state: Tensor) -> Tensor:
        """Return goal-distance, direction, and success terminal cost."""

        self._validate_state_shape(state)
        position = state[..., 0]
        velocity = state[..., 1]

        position_scale = self.goal_position - self.min_position
        normalized_position_gap = torch.clamp(
            (self.goal_position - position) / position_scale,
            min=0.0,
        )
        normalized_wrong_direction = torch.clamp(
            (self.goal_velocity - velocity) / self.max_speed,
            min=0.0,
        )

        position_cost = (
            self.terminal_position_weight
            * normalized_position_gap.square()
        )
        direction_cost = (
            self.terminal_wrong_direction_weight
            * normalized_wrong_direction.square()
        )
        success_reward_as_cost = self.success_bonus * self.terminated(state).to(
            dtype=state.dtype
        )

        return position_cost + direction_cost - success_reward_as_cost

    def terminated(self, state: Tensor) -> Tensor:
        """Return a Boolean mask for Gymnasium's goal condition."""

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
