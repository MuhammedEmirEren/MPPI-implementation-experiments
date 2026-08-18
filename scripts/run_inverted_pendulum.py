from __future__ import annotations

import argparse
import random
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import yaml

from mppi_control.controllers.mppi_controller import MPPIController
from mppi_control.costs.inverted_pendulum import InvertedPendulumCost
from mppi_control.dynamics.inverted_pendulum_mujoco import (
    InvertedPendulumMujocoDynamics,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "inverted_pendulum.yaml"


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    """Performance and planning diagnostics for one episode."""

    survived: bool
    episode_return: float
    episode_length: int
    final_cart_position: float
    final_pole_angle: float
    final_cart_velocity: float
    final_pole_angular_velocity: float
    rms_cart_position: float
    rms_pole_angle: float
    max_abs_pole_angle: float
    rms_action: float
    mean_planning_time_ms: float
    mean_ess: float | None
    final_ess: float | None


def parse_args() -> argparse.Namespace:
    """Parse command-line options that override the YAML configuration."""

    parser = argparse.ArgumentParser(
        description=(
            "Run the MPPI controller on Gymnasium InvertedPendulum-v5."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the InvertedPendulum YAML configuration.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Number of episodes; overrides environment.episodes.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base random seed; overrides the top-level seed.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Planning device such as cpu, cuda, or auto.",
    )

    render_group = parser.add_mutually_exclusive_group()
    render_group.add_argument(
        "--render",
        dest="render",
        action="store_true",
        help="Render the environment in a window.",
    )
    render_group.add_argument(
        "--no-render",
        dest="render",
        action="store_false",
        help="Disable rendering, regardless of the YAML setting.",
    )
    parser.set_defaults(render=None)

    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    """Load a YAML file and require a mapping at its top level."""

    config_path = path.expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as config_file:
        loaded = yaml.safe_load(config_file)

    if not isinstance(loaded, dict):
        raise ValueError("The configuration must contain a top-level mapping")
    return loaded


def require_section(config: Mapping[str, Any], name: str) -> dict[str, Any]:
    """Return a required mapping section with a useful error message."""

    section = config.get(name)
    if not isinstance(section, dict):
        raise ValueError(f"Configuration section '{name}' must be a mapping")
    return section


def seed_everything(seed: int) -> None:
    """Seed random generators used outside the controller."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    """Resolve an explicit device or choose CUDA automatically."""

    normalized = requested.strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(normalized)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "A CUDA device was requested, but torch.cuda.is_available() is false"
        )
    return device


def validate_environment_model(
    env: gym.Env,
    model_config: Mapping[str, Any],
) -> None:
    """Ensure configuration constants agree with the live MuJoCo model."""

    if not isinstance(env.action_space, gym.spaces.Box):
        raise TypeError("MPPI requires a continuous Gymnasium Box action space")

    unwrapped = env.unwrapped
    if not hasattr(unwrapped, "model") or not hasattr(unwrapped, "frame_skip"):
        raise TypeError("The environment must expose a MuJoCo model and frame_skip")

    configured_timestep = float(model_config["simulation_timestep"])
    actual_timestep = float(unwrapped.model.opt.timestep)
    if not np.isclose(configured_timestep, actual_timestep):
        raise ValueError(
            "Configured simulation_timestep does not match MuJoCo: "
            f"{configured_timestep} != {actual_timestep}"
        )

    configured_frame_skip = int(model_config["frame_skip"])
    actual_frame_skip = int(unwrapped.frame_skip)
    if configured_frame_skip != actual_frame_skip:
        raise ValueError(
            "Configured frame_skip does not match Gymnasium: "
            f"{configured_frame_skip} != {actual_frame_skip}"
        )

    configured_low = float(model_config["min_action"])
    configured_high = float(model_config["max_action"])
    if not np.allclose(env.action_space.low, configured_low):
        raise ValueError("Configured min_action does not match the action space")
    if not np.allclose(env.action_space.high, configured_high):
        raise ValueError("Configured max_action does not match the action space")


def build_controller(
    config: Mapping[str, Any],
    env: gym.Env,
    *,
    device: torch.device,
    seed: int,
) -> tuple[MPPIController, InvertedPendulumMujocoDynamics]:
    """Construct the MuJoCo trajectory model, cost, and MPPI controller."""

    model_config = require_section(config, "model")
    cost_config = require_section(config, "cost")
    mppi_config = require_section(config, "mppi")
    validate_environment_model(env, model_config)

    cost = InvertedPendulumCost(
        pole_angle_weight=float(cost_config["pole_angle_weight"]),
        pole_angular_velocity_weight=float(
            cost_config["pole_angular_velocity_weight"]
        ),
        cart_position_weight=float(cost_config["cart_position_weight"]),
        cart_velocity_weight=float(cost_config["cart_velocity_weight"]),
        action_weight=float(cost_config["action_weight"]),
        pole_angle_scale=float(cost_config["pole_angle_scale"]),
        pole_angular_velocity_scale=float(
            cost_config["pole_angular_velocity_scale"]
        ),
        cart_position_scale=float(cost_config["cart_position_scale"]),
        cart_velocity_scale=float(cost_config["cart_velocity_scale"]),
        action_scale=float(cost_config["action_scale"]),
        healthy_angle_limit=float(model_config["healthy_angle_limit"]),
        failure_penalty=float(cost_config["failure_penalty"]),
        terminal_weight=float(cost_config["terminal_weight"]),
    )
    dynamics = InvertedPendulumMujocoDynamics(
        env.unwrapped.model,
        frame_skip=int(model_config["frame_skip"]),
        rollout_threads=int(model_config["rollout_threads"]),
    )

    try:
        controller = MPPIController(
            dynamics=dynamics,
            cost=cost,
            horizon=int(mppi_config["horizon"]),
            num_samples=int(mppi_config["num_samples"]),
            temperature=float(mppi_config["temperature"]),
            noise_sigma=float(mppi_config["noise_sigma"]),
            action_low=env.action_space.low,
            action_high=env.action_space.high,
            device=device,
            num_iterations=int(mppi_config.get("num_iterations", 1)),
            sampling_correction=bool(
                mppi_config.get("sampling_correction", True)
            ),
            seed=seed,
        )
    except Exception:
        dynamics.close()
        raise

    return controller, dynamics


def synchronize_device(device: torch.device) -> None:
    """Wait for queued CUDA work so timing measurements are accurate."""

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_episode(
    env: gym.Env,
    controller: MPPIController,
    *,
    seed: int,
) -> EpisodeResult:
    """Run until the pole falls or the environment time limit is reached."""

    observation, _ = env.reset(seed=seed)
    env.action_space.seed(seed)
    controller.reset(seed=seed)

    terminated = False
    truncated = False
    episode_return = 0.0
    episode_length = 0
    planning_times: list[float] = []
    effective_sample_sizes: list[float] = []
    cart_positions = [float(observation[0])]
    pole_angles = [float(observation[1])]
    squared_actions: list[float] = []

    while not (terminated or truncated):
        synchronize_device(controller.device)
        planning_start = perf_counter()
        action = controller.act(observation)
        synchronize_device(controller.device)
        planning_times.append(perf_counter() - planning_start)

        effective_sample_size = controller.effective_sample_size
        if effective_sample_size is not None:
            effective_sample_sizes.append(effective_sample_size)

        squared_actions.append(float(np.mean(np.square(action))))
        observation, reward, terminated, truncated, _ = env.step(action)
        episode_return += float(reward)
        episode_length += 1
        cart_positions.append(float(observation[0]))
        pole_angles.append(float(observation[1]))

    mean_ess = (
        float(np.mean(effective_sample_sizes))
        if effective_sample_sizes
        else None
    )
    final_ess = effective_sample_sizes[-1] if effective_sample_sizes else None

    return EpisodeResult(
        survived=bool(truncated and not terminated),
        episode_return=episode_return,
        episode_length=episode_length,
        final_cart_position=float(observation[0]),
        final_pole_angle=float(observation[1]),
        final_cart_velocity=float(observation[2]),
        final_pole_angular_velocity=float(observation[3]),
        rms_cart_position=float(np.sqrt(np.mean(np.square(cart_positions)))),
        rms_pole_angle=float(np.sqrt(np.mean(np.square(pole_angles)))),
        max_abs_pole_angle=float(np.max(np.abs(pole_angles))),
        rms_action=float(np.sqrt(np.mean(squared_actions))),
        mean_planning_time_ms=1000.0 * float(np.mean(planning_times)),
        mean_ess=mean_ess,
        final_ess=final_ess,
    )


def format_optional(value: float | None, precision: int = 1) -> str:
    """Format a diagnostic value that may be unavailable."""

    return "n/a" if value is None else f"{value:.{precision}f}"


def main() -> None:
    """Load configuration, create the environment, and run all episodes."""

    args = parse_args()
    config = load_config(args.config)
    environment_config = require_section(config, "environment")

    seed = int(config.get("seed", 0) if args.seed is None else args.seed)
    episodes = int(
        environment_config.get("episodes", 1)
        if args.episodes is None
        else args.episodes
    )
    if episodes <= 0:
        raise ValueError("The number of episodes must be positive")

    max_episode_steps = int(
        environment_config.get("max_episode_steps", 1000)
    )
    if max_episode_steps <= 0:
        raise ValueError("environment.max_episode_steps must be positive")

    requested_device = str(
        config.get("device", "cpu") if args.device is None else args.device
    )
    device = resolve_device(requested_device)

    configured_render_mode = environment_config.get("render_mode")
    if args.render is True:
        render_mode: str | None = "human"
    elif args.render is False:
        render_mode = None
    elif configured_render_mode is None:
        render_mode = None
    else:
        render_mode = str(configured_render_mode)

    seed_everything(seed)

    environment_id = str(
        environment_config.get("id", "InvertedPendulum-v5")
    )
    env = gym.make(
        environment_id,
        render_mode=render_mode,
        max_episode_steps=max_episode_steps,
    )
    dynamics: InvertedPendulumMujocoDynamics | None = None

    try:
        controller, dynamics = build_controller(
            config,
            env,
            device=device,
            seed=seed,
        )

        print(
            f"Environment: {environment_id} | Device: {device} | "
            f"Episodes: {episodes} | Render: {render_mode or 'off'}"
        )

        results: list[EpisodeResult] = []
        for episode_index in range(episodes):
            episode_seed = seed + episode_index
            result = run_episode(
                env,
                controller,
                seed=episode_seed,
            )
            results.append(result)

            outcome = "survived" if result.survived else "failed"
            print(
                f"Episode {episode_index + 1:>3}/{episodes} | "
                f"seed={episode_seed} | "
                f"outcome={outcome} | "
                f"steps={result.episode_length} | "
                f"return={result.episode_return:.1f} | "
                f"final_angle={result.final_pole_angle:.4f} rad | "
                f"max_angle={result.max_abs_pole_angle:.4f} rad | "
                f"cart_rms={result.rms_cart_position:.4f} | "
                f"action_rms={result.rms_action:.3f} | "
                f"plan={result.mean_planning_time_ms:.2f} ms | "
                f"ESS(mean/final)={format_optional(result.mean_ess)}/"
                f"{format_optional(result.final_ess)}"
            )

        returns = np.asarray(
            [result.episode_return for result in results],
            dtype=np.float64,
        )
        survived = [result for result in results if result.survived]
        survival_rate = 100.0 * len(survived) / len(results)
        mean_episode_length = float(
            np.mean([result.episode_length for result in results])
        )
        mean_planning_time_ms = float(
            np.mean([result.mean_planning_time_ms for result in results])
        )
        mean_rms_pole_angle = float(
            np.mean([result.rms_pole_angle for result in results])
        )

        print(
            f"Survived: {len(survived)}/{len(results)} "
            f"({survival_rate:.1f}%) | "
            f"Mean return: {returns.mean():.1f} | "
            f"Std: {returns.std():.1f}"
        )
        print(
            f"Mean episode length: {mean_episode_length:.1f} | "
            f"Mean pole RMS: {mean_rms_pole_angle:.4f} rad | "
            f"Mean planning time: {mean_planning_time_ms:.2f} ms/step"
        )
    finally:
        if dynamics is not None:
            dynamics.close()
        env.close()


if __name__ == "__main__":
    main()
