from __future__ import annotations

from numbers import Integral

import mujoco
import numpy as np
import torch
from mujoco import rollout as mujoco_rollout
from torch import Tensor


class ReacherMujocoDynamics:
    """Predict batched Reacher-v5 trajectories with MuJoCo.

    The observation representation matches Gymnasium:

        state[..., 0:2] = cosine of the two joint angles
        state[..., 2:4] = sine of the two joint angles
        state[..., 4:6] = target x and y coordinates
        state[..., 6:8] = angular velocity of the two joints
        state[..., 8:10] = fingertip position minus target position

    Reacher's observation is not MuJoCo's raw physics state. Before a rollout,
    the sine and cosine pairs are converted back into joint angles and the two
    unobserved target velocities are set to zero. After simulation, the raw
    states are converted back into Gymnasium observations. The fingertip error
    is reconstructed from the two-link planar arm geometry.

    Each action is repeated for ``frame_skip`` MuJoCo integration steps, just
    as Gymnasium's ``do_simulation`` does. Returned states are sampled at the
    Gymnasium control frequency.

    This class owns independent ``MjData`` instances for its rollout workers.
    It shares the immutable ``MjModel`` with the environment but never changes
    the environment's live ``MjData`` simulation state.
    """

    state_dim = 10
    action_dim = 2

    def __init__(
        self,
        model: mujoco.MjModel,
        *,
        frame_skip: int = 2,
        rollout_threads: int = 4,
    ) -> None:
        """Create a persistent MuJoCo batch-rollout backend.

        Args:
            model: MuJoCo model owned by a Reacher-v5 environment.
            frame_skip: Number of MuJoCo integration steps per Gymnasium action.
            rollout_threads: Worker threads used by MuJoCo. Zero runs rollouts
                on the calling thread.

        Raises:
            TypeError: If ``model`` is not an ``MjModel``.
            ValueError: If the model or rollout settings are incompatible with
                Reacher-v5.
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

        first_link_length, second_link_length = self._validate_model(model)

        self.model = model
        self.frame_skip = int(frame_skip)
        self.rollout_threads = int(rollout_threads)
        self.control_timestep = float(model.opt.timestep) * self.frame_skip
        self.first_link_length = first_link_length
        self.second_link_length = second_link_length

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
            initial_state: Gymnasium observation with shape ``(10,)``.
            action_sequences: Tensor with shape ``(K, H, 2)``, where ``K`` is
                the number of MPPI samples and ``H`` is the planning horizon.

        Returns:
            Tensor with shape ``(K, H, 10)`` on the same device and with the
            same dtype as ``initial_state``. Entry ``[:, t]`` is the state
            after applying action ``[:, t]``.

        Raises:
            RuntimeError: If called after ``close``.
            ValueError: If an input shape, dtype, device, or value is invalid.
        """

        if self._rollout is None:
            raise RuntimeError("rollout requested after dynamics were closed")

        self._validate_inputs(initial_state, action_sequences)

        observation = initial_state.detach().cpu().numpy().astype(
            np.float64,
            copy=False,
        )

        # FULLPHYSICS is [time, qpos(4), qvel(4)] for Reacher-v5. The two
        # target slide joints are stationary, so their unobserved velocities
        # are zero. atan2 recovers each angle from Gymnasium's sine/cosine pair.
        full_initial_state = np.zeros(
            (1, self._full_state_size),
            dtype=np.float64,
        )
        joint_angles = np.arctan2(observation[2:4], observation[0:2])
        full_initial_state[0, 1:3] = joint_angles
        full_initial_state[0, 3:5] = observation[4:6]

        velocity_start = 1 + self.model.nq
        full_initial_state[0, velocity_start : velocity_start + 2] = (
            observation[6:8]
        )

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

        # MuJoCo returns the state after every integration step. Keep the state
        # after each complete frame-skip group to match Gymnasium's frequency.
        environment_states = physics_states[
            :,
            self.frame_skip - 1 :: self.frame_skip,
            :,
        ]
        observations = self._states_to_observations(
            environment_states,
            velocity_start=velocity_start,
        )

        return torch.from_numpy(np.ascontiguousarray(observations)).to(
            device=initial_state.device,
            dtype=initial_state.dtype,
        )

    def _states_to_observations(
        self,
        physics_states: np.ndarray,
        *,
        velocity_start: int,
    ) -> np.ndarray:
        """Convert batched MuJoCo FULLPHYSICS states to Reacher observations."""

        qpos = physics_states[..., 1 : 1 + self.model.nq]
        joint_angles = qpos[..., :2]
        target_position = qpos[..., 2:4]
        joint_velocities = physics_states[
            ...,
            velocity_start : velocity_start + 2,
        ]

        first_angle = joint_angles[..., 0]
        second_absolute_angle = first_angle + joint_angles[..., 1]
        fingertip_x = (
            self.first_link_length * np.cos(first_angle)
            + self.second_link_length * np.cos(second_absolute_angle)
        )
        fingertip_y = (
            self.first_link_length * np.sin(first_angle)
            + self.second_link_length * np.sin(second_absolute_angle)
        )
        fingertip_position = np.stack((fingertip_x, fingertip_y), axis=-1)
        fingertip_error = fingertip_position - target_position

        return np.concatenate(
            (
                np.cos(joint_angles),
                np.sin(joint_angles),
                target_position,
                joint_velocities,
                fingertip_error,
            ),
            axis=-1,
        )

    def close(self) -> None:
        """Shut down the persistent MuJoCo rollout thread pool."""

        if self._rollout is not None:
            self._rollout.close()
            self._rollout = None
        self._worker_data.clear()

    def __enter__(self) -> ReacherMujocoDynamics:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    @staticmethod
    def _validate_model(model: mujoco.MjModel) -> tuple[float, float]:
        """Check Reacher's state layout and extract its two link lengths."""

        if model.nq != 4 or model.nv != 4 or model.nu != 2:
            raise ValueError(
                "Reacher-v5 requires a MuJoCo model with nq=4, nv=4, and "
                f"nu=2; received nq={model.nq}, nv={model.nv}, nu={model.nu}"
            )
        if model.na != 0:
            raise ValueError(
                "Reacher-v5 is expected to have no actuator activation state; "
                f"received na={model.na}"
            )
        if model.njnt != 4:
            raise ValueError(
                "Reacher-v5 requires exactly four one-degree-of-freedom joints"
            )
        if not np.array_equal(model.jnt_qposadr, np.arange(4)):
            raise ValueError("Reacher-v5 joint-position layout is incompatible")
        if not np.array_equal(model.jnt_dofadr, np.arange(4)):
            raise ValueError("Reacher-v5 joint-velocity layout is incompatible")

        expected_joint_types = np.array(
            [
                mujoco.mjtJoint.mjJNT_HINGE,
                mujoco.mjtJoint.mjJNT_HINGE,
                mujoco.mjtJoint.mjJNT_SLIDE,
                mujoco.mjtJoint.mjJNT_SLIDE,
            ],
            dtype=model.jnt_type.dtype,
        )
        if not np.array_equal(model.jnt_type, expected_joint_types):
            raise ValueError("Reacher-v5 requires two hinge and two slide joints")
        if not np.allclose(
            model.jnt_axis,
            np.array(
                [
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                ]
            ),
        ):
            raise ValueError("Reacher-v5 joint axes are incompatible")

        body1_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            "body1",
        )
        fingertip_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            "fingertip",
        )
        target_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            "target",
        )
        if min(body1_id, fingertip_id, target_id) < 0:
            raise ValueError(
                "Reacher-v5 model must define body1, fingertip, and target"
            )

        first_link_offset = model.body_pos[body1_id, :2]
        second_link_offset = model.body_pos[fingertip_id, :2]
        if not np.allclose(first_link_offset[1], 0.0) or not np.allclose(
            second_link_offset[1], 0.0
        ):
            raise ValueError("Reacher-v5 links must extend along their local x axes")

        first_link_length = float(first_link_offset[0])
        second_link_length = float(second_link_offset[0])
        if first_link_length <= 0 or second_link_length <= 0:
            raise ValueError("Reacher-v5 link lengths must be positive")

        # In Reacher-v5, the target body's base position equals the reference
        # values of its x/y slide joints. Therefore its world x/y coordinates
        # equal qpos[2:4], as assumed by Gymnasium's observation.
        if not np.allclose(model.body_pos[target_id, :2], model.qpos0[2:4]):
            raise ValueError("Reacher-v5 target position layout is incompatible")
        if not np.isfinite(model.opt.timestep) or model.opt.timestep <= 0:
            raise ValueError("the MuJoCo model timestep must be positive")

        return first_link_length, second_link_length

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
                "action_sequences must have shape (K, H, 2), "
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
