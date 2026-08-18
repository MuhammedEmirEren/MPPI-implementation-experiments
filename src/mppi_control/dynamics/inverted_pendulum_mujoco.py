from __future__ import annotations

from numbers import Integral

import mujoco
import numpy as np
import torch
from mujoco import rollout as mujoco_rollout
from torch import Tensor


class InvertedPendulumMujocoDynamics:
    """Predict batched InvertedPendulum-v5 trajectories with MuJoCo.

    The observation representation matches Gymnasium:

        state[..., 0] = cart position
        state[..., 1] = pole angle
        state[..., 2] = cart velocity
        state[..., 3] = pole angular velocity

    ``rollout`` receives one current observation and a batch of open-loop
    action sequences. Each action is repeated for ``frame_skip`` MuJoCo steps,
    exactly as Gymnasium's ``do_simulation`` does. Returned observations are
    sampled at the Gymnasium control frequency rather than every MuJoCo step.

    This class owns independent ``MjData`` instances for its workers. It shares
    the immutable ``MjModel`` with the environment but never modifies the
    environment's live simulation state.
    """

    state_dim = 4
    action_dim = 1

    def __init__(
        self,
        model: mujoco.MjModel,
        *,
        frame_skip: int = 2,
        rollout_threads: int = 4,
    ) -> None:
        """Create a persistent MuJoCo batch-rollout backend.

        Args:
            model: MuJoCo model owned by an InvertedPendulum-v5 environment.
            frame_skip: Number of MuJoCo integration steps per Gymnasium action.
            rollout_threads: Worker threads used by MuJoCo. Zero runs rollouts
                on the calling thread.

        Raises:
            TypeError: If ``model`` is not an ``MjModel``.
            ValueError: If the model or rollout settings are incompatible with
                InvertedPendulum-v5.
        """

        if not isinstance(model, mujoco.MjModel):
            raise TypeError("model must be a mujoco.MjModel")
        if not isinstance(frame_skip, Integral) or isinstance(frame_skip, bool):
            raise ValueError("frame_skip must be a positive integer")
        if frame_skip <= 0:
            raise ValueError("frame_skip must be a positive integer")
        if not isinstance(rollout_threads, Integral) or isinstance(
            rollout_threads, bool
        ):
            raise ValueError("rollout_threads must be a non-negative integer")
        if rollout_threads < 0:
            raise ValueError("rollout_threads must be a non-negative integer")

        self._validate_model(model)

        self.model = model
        self.frame_skip = int(frame_skip)
        self.rollout_threads = int(rollout_threads)
        self.control_timestep = float(model.opt.timestep) * self.frame_skip

        self._full_state_size = mujoco.mj_stateSize(
            model,
            mujoco.mjtState.mjSTATE_FULLPHYSICS,
        )
        worker_count = max(1, self.rollout_threads)
        self._worker_data = [mujoco.MjData(model) for _ in range(worker_count)]
        self._rollout = mujoco_rollout.Rollout(nthread=self.rollout_threads)

    def rollout(
        self,
        initial_state: Tensor,
        action_sequences: Tensor,
    ) -> Tensor:
        """Simulate a batch of open-loop action sequences.

        Args:
            initial_state: Tensor with shape ``(4,)`` containing
                ``[cart_position, pole_angle, cart_velocity,
                pole_angular_velocity]``.
            action_sequences: Tensor with shape ``(K, H, 1)``, where ``K`` is
                the number of MPPI samples and ``H`` is the planning horizon.

        Returns:
            Tensor with shape ``(K, H, 4)`` on the same device and with the
            same dtype as ``initial_state``.

        Raises:
            RuntimeError: If called after ``close``.
            ValueError: If an input shape, dtype, device, or value is invalid.
        """

        if self._rollout is None:
            raise RuntimeError("rollout requested after dynamics were closed")

        self._validate_inputs(initial_state, action_sequences)

        # MuJoCo's FULLPHYSICS layout for this model is:
        # [time, qpos(2), qvel(2)]. Time is zero because the model is
        # autonomous and Gymnasium's four-value observation does not expose it.
        full_initial_state = np.zeros(
            (1, self._full_state_size),
            dtype=np.float64,
        )
        observation = initial_state.detach().cpu().numpy().astype(
            np.float64,
            copy=False,
        )
        full_initial_state[0, 1 : 1 + self.model.nq] = observation[:2]
        velocity_start = 1 + self.model.nq
        full_initial_state[
            0,
            velocity_start : velocity_start + self.model.nv,
        ] = observation[2:]

        controls = action_sequences.detach().cpu().numpy().astype(
            np.float64,
            copy=False,
        )
        controls = np.repeat(controls, self.frame_skip, axis=1)
        controls = np.ascontiguousarray(controls)

        physics_states, _ = self._rollout.rollout(
            self.model,
            self._worker_data,
            initial_state=full_initial_state,
            control=controls,
            control_spec=mujoco.mjtState.mjSTATE_CTRL,
        )

        # Each output is the state after one MuJoCo step. Select the state after
        # every complete frame-skip group to recover Gymnasium-frequency states.
        environment_states = physics_states[
            :,
            self.frame_skip - 1 :: self.frame_skip,
            :,
        ]
        observations = np.concatenate(
            (
                environment_states[..., 1 : 1 + self.model.nq],
                environment_states[
                    ...,
                    velocity_start : velocity_start + self.model.nv,
                ],
            ),
            axis=-1,
        )

        return torch.from_numpy(np.ascontiguousarray(observations)).to(
            device=initial_state.device,
            dtype=initial_state.dtype,
        )

    def close(self) -> None:
        """Shut down the persistent MuJoCo rollout thread pool."""

        if self._rollout is not None:
            self._rollout.close()
            self._rollout = None
        self._worker_data.clear()

    def __enter__(self) -> InvertedPendulumMujocoDynamics:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    @staticmethod
    def _validate_model(model: mujoco.MjModel) -> None:
        """Check the state and actuator sizes required by this environment."""

        if model.nq != 2 or model.nv != 2 or model.nu != 1:
            raise ValueError(
                "InvertedPendulum-v5 requires a MuJoCo model with "
                "nq=2, nv=2, and nu=1; received "
                f"nq={model.nq}, nv={model.nv}, nu={model.nu}"
            )
        if model.na != 0:
            raise ValueError(
                "InvertedPendulum-v5 is expected to have no actuator "
                f"activation state; received na={model.na}"
            )
        if not np.isfinite(model.opt.timestep) or model.opt.timestep <= 0:
            raise ValueError("the MuJoCo model timestep must be positive")

    @classmethod
    def _validate_inputs(
        cls,
        initial_state: Tensor,
        action_sequences: Tensor,
    ) -> None:
        """Validate the tensor contract used by batched MPPI rollouts."""

        if not isinstance(initial_state, Tensor):
            raise TypeError("initial_state must be a torch.Tensor")
        if not isinstance(action_sequences, Tensor):
            raise TypeError("action_sequences must be a torch.Tensor")
        if initial_state.shape != (cls.state_dim,):
            raise ValueError(
                f"initial_state must have shape ({cls.state_dim},), "
                f"but received {tuple(initial_state.shape)}"
            )
        if action_sequences.ndim != 3 or action_sequences.shape[-1] != cls.action_dim:
            raise ValueError(
                "action_sequences must have shape (K, H, 1), "
                f"but received {tuple(action_sequences.shape)}"
            )
        if action_sequences.shape[0] <= 0 or action_sequences.shape[1] <= 0:
            raise ValueError("action_sequences requires positive K and H dimensions")
        if not initial_state.dtype.is_floating_point:
            raise ValueError("initial_state must use a floating-point dtype")
        if not action_sequences.dtype.is_floating_point:
            raise ValueError("action_sequences must use a floating-point dtype")
        if initial_state.device != action_sequences.device:
            raise ValueError("initial_state and action_sequences must share a device")
        if not torch.all(torch.isfinite(initial_state)):
            raise ValueError("initial_state must contain only finite values")
        if not torch.all(torch.isfinite(action_sequences)):
            raise ValueError("action_sequences must contain only finite values")
