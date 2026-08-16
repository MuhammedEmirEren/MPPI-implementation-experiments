from __future__ import annotations

from typing import Protocol

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from mppi_control.rollout import rollout_costs


class DynamicsModel(Protocol):
    """Interface required by the MPPI rollout."""

    def step(self, state: Tensor, action: Tensor) -> Tensor:
        """Predict the state reached after applying ``action``."""


class CostModel(Protocol):
    """Interface required by the MPPI rollout."""

    def running(self, state: Tensor, action: Tensor) -> Tensor:
        """Return one running cost per state-action pair."""

    def terminal(self, state: Tensor) -> Tensor:
        """Return one terminal cost per state."""


class MPPIController:
    """Model Predictive Path Integral controller for continuous actions.

    The controller maintains a nominal action sequence, samples Gaussian
    perturbations around it, evaluates all candidate sequences with the
    supplied model and cost, and updates the nominal sequence with an
    exponentially cost-weighted average of the perturbations.

    This implementation uses a scalar standard deviation shared by all
    action dimensions. Pendulum-v1 has one action dimension, but the tensor
    operations also support vector actions when vector bounds are supplied.
    """

    def __init__(
        self,
        dynamics: DynamicsModel,
        cost: CostModel,
        *,
        horizon: int,
        num_samples: int,
        temperature: float,
        noise_sigma: float,
        action_low: float | NDArray[np.floating] | Tensor,
        action_high: float | NDArray[np.floating] | Tensor,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        num_iterations: int = 1,
        sampling_correction: bool = True,
        seed: int | None = None,
    ) -> None:
        """Initialize the controller and its nominal action sequence.

        Args:
            dynamics: Batched dynamics model with a ``step`` method.
            cost: Cost model with ``running`` and ``terminal`` methods.
            horizon: Number of future control steps in each rollout.
            num_samples: Number of candidate trajectories per iteration.
            temperature: Positive MPPI temperature lambda. Smaller values
                concentrate weight on fewer low-cost samples.
            noise_sigma: Positive standard deviation of Gaussian action noise.
            action_low: Scalar or vector lower action bound.
            action_high: Scalar or vector upper action bound.
            device: PyTorch device used for planning.
            dtype: Floating-point dtype used for planning.
            num_iterations: MPPI updates performed during each ``act`` call.
            sampling_correction: Whether to include the fixed-covariance
                importance-sampling cross term in trajectory scores.
            seed: Optional seed for the controller's private random generator.
        """

        self._validate_scalar_parameters(
            horizon=horizon,
            num_samples=num_samples,
            temperature=temperature,
            noise_sigma=noise_sigma,
            num_iterations=num_iterations,
        )

        if not dtype.is_floating_point:
            raise ValueError("dtype must be a floating-point torch dtype")

        self.dynamics = dynamics
        self.cost = cost
        self.horizon = horizon
        self.num_samples = num_samples
        self.temperature = float(temperature)
        self.noise_sigma = float(noise_sigma)
        self.num_iterations = num_iterations
        self.sampling_correction = sampling_correction
        self.device = torch.device(device)
        self.dtype = dtype

        self.action_low = self._as_action_bound(action_low, "action_low")
        self.action_high = self._as_action_bound(action_high, "action_high")

        if self.action_low.shape != self.action_high.shape:
            raise ValueError(
                "action_low and action_high must have the same shape, but "
                f"received {tuple(self.action_low.shape)} and "
                f"{tuple(self.action_high.shape)}"
            )

        if torch.any(self.action_low >= self.action_high):
            raise ValueError("every action_low value must be below action_high")

        self.action_dim = self.action_low.numel()
        self._nominal_actions = torch.zeros(
            (self.horizon, self.action_dim),
            device=self.device,
            dtype=self.dtype,
        )

        self._generator = torch.Generator(device=self.device)
        if seed is None:
            self._generator.seed()
        else:
            self._generator.manual_seed(seed)

        self.last_costs: Tensor | None = None
        self.last_weights: Tensor | None = None

    @torch.no_grad()
    def act(self, observation: NDArray[np.floating] | Tensor) -> NDArray[np.float32]:
        """Optimize the plan and return the first action for the environment.

        ``observation`` must be a single state vector. The returned NumPy array
        always has shape ``(action_dim,)`` and dtype ``float32``.
        """

        state = torch.as_tensor(
            observation,
            device=self.device,
            dtype=self.dtype,
        )
        self._validate_observation(state)

        for _ in range(self.num_iterations):
            perturbations = self._sample_perturbations()
            candidates, effective_perturbations = self._build_candidates(
                perturbations
            )

            costs = rollout_costs(
                initial_state=state,
                action_sequences=candidates,
                dynamics=self.dynamics,
                cost=self.cost,
            )

            if costs.shape != (self.num_samples,):
                raise ValueError(
                    "rollout_costs must return shape "
                    f"({self.num_samples},), but returned {tuple(costs.shape)}"
                )

            if self.sampling_correction:
                costs = costs + self._sampling_cross_term(
                    effective_perturbations
                )

            weights = self._compute_weights(costs)
            self._update_nominal_actions(effective_perturbations, weights)

            self.last_costs = costs.detach().clone()
            self.last_weights = weights.detach().clone()

        action = self._nominal_actions[0].detach().clone()
        self._shift_nominal_actions()

        return action.cpu().numpy().astype(np.float32, copy=False)

    @torch.no_grad()
    def reset(self, *, seed: int | None = None) -> None:
        """Clear the warm-start plan and optionally reseed sampling."""

        self._nominal_actions.zero_()
        self.last_costs = None
        self.last_weights = None

        if seed is not None:
            self._generator.manual_seed(seed)

    @property
    def nominal_actions(self) -> Tensor:
        """Return a defensive copy of the current warm-start plan."""

        return self._nominal_actions.detach().clone()

    @property
    def effective_sample_size(self) -> float | None:
        """Return ``1 / sum(weights**2)`` for the most recent update."""

        if self.last_weights is None:
            return None

        return float(self.last_weights.square().sum().reciprocal().item())

    def _sample_perturbations(self) -> Tensor:
        """Sample independent Gaussian perturbation sequences."""

        return self.noise_sigma * torch.randn(
            (self.num_samples, self.horizon, self.action_dim),
            device=self.device,
            dtype=self.dtype,
            generator=self._generator,
        )

    def _build_candidates(self, perturbations: Tensor) -> tuple[Tensor, Tensor]:
        """Add perturbations to the plan and enforce action limits."""

        expected_shape = (self.num_samples, self.horizon, self.action_dim)
        if perturbations.shape != expected_shape:
            raise ValueError(
                f"perturbations must have shape {expected_shape}, but received "
                f"{tuple(perturbations.shape)}"
            )

        candidates = self._nominal_actions.unsqueeze(0) + perturbations
        candidates = torch.maximum(candidates, self.action_low)
        candidates = torch.minimum(candidates, self.action_high)

        # Clipping changes the perturbation near a control bound. Use the
        # perturbation that the rollout actually receives for the update.
        effective_perturbations = (
            candidates - self._nominal_actions.unsqueeze(0)
        )

        return candidates, effective_perturbations

    def _sampling_cross_term(self, perturbations: Tensor) -> Tensor:
        """Return the fixed-covariance importance-sampling correction.

        For covariance Sigma = noise_sigma**2 * I, the sample-dependent term
        is lambda * sum_t(u_t^T Sigma^-1 epsilon_t).
        """

        inverse_variance = 1.0 / (self.noise_sigma**2)
        return self.temperature * torch.sum(
            self._nominal_actions.unsqueeze(0)
            * perturbations
            * inverse_variance,
            dim=(1, 2),
        )

    def _compute_weights(self, costs: Tensor) -> Tensor:
        """Convert trajectory costs into normalized exponential weights."""

        if costs.shape != (self.num_samples,):
            raise ValueError(
                f"costs must have shape ({self.num_samples},), but received "
                f"{tuple(costs.shape)}"
            )

        if not torch.all(torch.isfinite(costs)):
            raise ValueError("trajectory costs must all be finite")

        shifted_costs = costs - torch.min(costs)
        weights = torch.softmax(-shifted_costs / self.temperature, dim=0)

        if not torch.all(torch.isfinite(weights)):
            raise ValueError("MPPI importance weights became non-finite")

        return weights

    def _update_nominal_actions(
        self,
        perturbations: Tensor,
        weights: Tensor,
    ) -> None:
        """Move the nominal sequence toward low-cost perturbations."""

        update = torch.einsum("k,kha->ha", weights, perturbations)
        self._nominal_actions.add_(update)
        self._nominal_actions.copy_(
            torch.maximum(self._nominal_actions, self.action_low)
        )
        self._nominal_actions.copy_(
            torch.minimum(self._nominal_actions, self.action_high)
        )

    def _shift_nominal_actions(self) -> None:
        """Discard the executed action and warm-start the next plan."""

        self._nominal_actions[:-1].copy_(self._nominal_actions[1:].clone())
        self._nominal_actions[-1].zero_()

    def _as_action_bound(
        self,
        bound: float | NDArray[np.floating] | Tensor,
        name: str,
    ) -> Tensor:
        tensor = torch.as_tensor(bound, device=self.device, dtype=self.dtype)

        if tensor.ndim == 0:
            tensor = tensor.reshape(1)
        elif tensor.ndim != 1:
            raise ValueError(
                f"{name} must be a scalar or one-dimensional vector, but "
                f"received shape {tuple(tensor.shape)}"
            )

        if not torch.all(torch.isfinite(tensor)):
            raise ValueError(f"{name} must contain only finite values")

        return tensor

    @staticmethod
    def _validate_observation(observation: Tensor) -> None:
        if observation.ndim != 1:
            raise ValueError(
                "observation must be one-dimensional, but received shape "
                f"{tuple(observation.shape)}"
            )

        if not torch.all(torch.isfinite(observation)):
            raise ValueError("observation must contain only finite values")

    @staticmethod
    def _validate_scalar_parameters(
        *,
        horizon: int,
        num_samples: int,
        temperature: float,
        noise_sigma: float,
        num_iterations: int,
    ) -> None:
        integer_parameters = {
            "horizon": horizon,
            "num_samples": num_samples,
            "num_iterations": num_iterations,
        }

        for name, value in integer_parameters.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be positive and finite")

        if not np.isfinite(noise_sigma) or noise_sigma <= 0:
            raise ValueError("noise_sigma must be positive and finite")
