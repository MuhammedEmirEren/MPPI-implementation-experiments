from dataclasses import dataclass

import torch

@dataclass(frozen=True)
class PendulumCost:
    angle_weight: float = 1.0
    velocity_weight: float = 0.1
    action_weight: float = 0.001
    terminal_weight: float = 0.0

    def running(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        theta = torch.atan2(state[..., 1], state[..., 0])
        theta_dot = state[..., 2]
        torque = action[..., 0]

        angle_cost = self.angle_weight * theta.square()
        velocity_cost = self.velocity_weight * theta_dot.square()
        action_cost = self.action_weight * torque.square()

        return angle_cost + velocity_cost + action_cost

    def terminal(self, state: torch.Tensor) -> torch.Tensor:
        theta = torch.atan2(state[..., 1], state[..., 0])
        theta_dot = state[..., 2]

        state_cost = (self.angle_weight * theta.square() + self.velocity_weight * theta_dot.square())

        return self.terminal_weight * state_cost