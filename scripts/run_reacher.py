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
from mppi_control.costs.reacher import ReacherCost
from mppi_control.dynamics.reacher_mujoco import ReacherMujocoDynamics


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "reacher.yaml"


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    """Performance and planning diagnostics for one Reacher episode."""

    success: bool
    episode_return: float
    episode_length: int
    first_success_step: int | None
    final_distance: float
    minimum_distance: float
    mean_distance: float
    rms_action: float
    mean_planning_time_ms: float
    mean_ess: float | None
    final_ess: float | None


def parse_args() -> argparse.Namespace:
    """Parse command-line options that override the YAML configuration."""

    parser = argparse.ArgumentParser(
        description="Run the MPPI controller on Gymnasium Reacher-v5."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the Reacher YAML configuration.",
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
    """Ensure configuration constants agree with Reacher's live model."""

    if not isinstance(env.action_space, gym.spaces.Box):
        raise TypeError("MPPI requires a continuous Gymnasium Box action space")
    if env.action_space.shape != (2,):
        raise ValueError(
            "Reacher-v5 must have action shape (2,), but received "
            f"{env.action_space.shape}"
        )
    if env.observation_space.shape != (10,):
        raise ValueError(
            "Reacher-v5 must have observation shape (10,), but received "
            f"{env.observation_space.shape}"
        )

    unwrapped = env.unwrapped
    if not hasattr(unwrapped, "model") or not hasattr(unwrapped, "frame_skip"):
        raise TypeError("The environment must expose a MuJoCo model and frame_skip")

    model = unwrapped.model
    configured_timestep = float(model_config["simulation_timestep"])
    actual_timestep = float(model.opt.timestep)
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

    first_link_length = float(model.body("body1").pos[0])
    second_link_length = float(model.body("fingertip").pos[0])
    configured_first_link = float(model_config["first_link_length"])
    configured_second_link = float(model_config["second_link_length"])
    if not np.isclose(configured_first_link, first_link_length):
        raise ValueError(
            "Configured first_link_length does not match MuJoCo: "
            f"{configured_first_link} != {first_link_length}"
        )
    if not np.isclose(configured_second_link, second_link_length):
        raise ValueError(
            "Configured second_link_length does not match MuJoCo: "
            f"{configured_second_link} != {second_link_length}"
        )


def build_controller(
    config: Mapping[str, Any],
    env: gym.Env,
    *,
    device: torch.device,
    seed: int,
) -> tuple[MPPIController, ReacherMujocoDynamics]:
    """Construct the Reacher MuJoCo dynamics, cost, and MPPI controller."""

    model_config = require_section(config, "model")
    cost_config = require_section(config, "cost")
    mppi_config = require_section(config, "mppi")
    validate_environment_model(env, model_config)

    cost = ReacherCost(
        distance_weight=float(cost_config["distance_weight"]),
        action_weight=float(cost_config["action_weight"]),
        joint_velocity_weight=float(cost_config["joint_velocity_weight"]),
        terminal_distance_weight=float(
            cost_config["terminal_distance_weight"]
        ),
    )
    dynamics = ReacherMujocoDynamics(
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


def fingertip_distance(observation: np.ndarray) -> float:
    """Return Euclidean fingertip-to-target distance from an observation."""

    return float(np.linalg.norm(observation[8:10]))


def run_episode(
    env: gym.Env,
    controller: MPPIController,
    *,
    seed: int,
    success_distance: float,
) -> EpisodeResult:
    """Run one complete Reacher episode and collect diagnostics."""

    observation, _ = env.reset(seed=seed)
    env.action_space.seed(seed)
    controller.reset(seed=seed)

    terminated = False
    truncated = False
    episode_return = 0.0
    episode_length = 0
    planning_times: list[float] = []
    effective_sample_sizes: list[float] = []
    distances = [fingertip_distance(observation)]
    squared_actions: list[float] = []
    first_success_step = 0 if distances[0] <= success_distance else None

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

        distance = fingertip_distance(observation)
        distances.append(distance)
        if first_success_step is None and distance <= success_distance:
            first_success_step = episode_length

    mean_ess = (
        float(np.mean(effective_sample_sizes))
        if effective_sample_sizes
        else None
    )
    final_ess = effective_sample_sizes[-1] if effective_sample_sizes else None

    return EpisodeResult(
        success=first_success_step is not None,
        episode_return=episode_return,
        episode_length=episode_length,
        first_success_step=first_success_step,
        final_distance=distances[-1],
        minimum_distance=float(np.min(distances)),
        mean_distance=float(np.mean(distances)),
        rms_action=float(np.sqrt(np.mean(squared_actions))),
        mean_planning_time_ms=1000.0 * float(np.mean(planning_times)),
        mean_ess=mean_ess,
        final_ess=final_ess,
    )


def format_optional(value: float | int | None, precision: int = 1) -> str:
    """Format a diagnostic value that may be unavailable."""

    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{precision}f}"


def main() -> None:
    """Load configuration, create Reacher, and run all episodes."""

    args = parse_args()
    config = load_config(args.config)
    environment_config = require_section(config, "environment")
    evaluation_config = require_section(config, "evaluation")

    seed = int(config.get("seed", 0) if args.seed is None else args.seed)
    episodes = int(
        environment_config.get("episodes", 1)
        if args.episodes is None
        else args.episodes
    )
    if episodes <= 0:
        raise ValueError("The number of episodes must be positive")

    max_episode_steps = int(environment_config.get("max_episode_steps", 50))
    if max_episode_steps <= 0:
        raise ValueError("environment.max_episode_steps must be positive")

    success_distance = float(evaluation_config["success_distance"])
    if not np.isfinite(success_distance) or success_distance <= 0:
        raise ValueError("evaluation.success_distance must be positive and finite")

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

    environment_id = str(environment_config.get("id", "Reacher-v5"))
    env = gym.make(
        environment_id,
        render_mode=render_mode,
        max_episode_steps=max_episode_steps,
    )
    dynamics: ReacherMujocoDynamics | None = None

    try:
        controller, dynamics = build_controller(
            config,
            env,
            device=device,
            seed=seed,
        )

        print(
            f"Environment: {environment_id} | Device: {device} | "
            f"Episodes: {episodes} | Render: {render_mode or 'off'} | "
            f"Success radius: {success_distance:.3f}"
        )

        results: list[EpisodeResult] = []
        for episode_index in range(episodes):
            episode_seed = seed + episode_index
            result = run_episode(
                env,
                controller,
                seed=episode_seed,
                success_distance=success_distance,
            )
            results.append(result)

            outcome = "reached" if result.success else "not_reached"
            print(
                f"Episode {episode_index + 1:>3}/{episodes} | "
                f"seed={episode_seed} | "
                f"outcome={outcome} | "
                f"steps={result.episode_length} | "
                f"first_reach={format_optional(result.first_success_step)} | "
                f"return={result.episode_return:.3f} | "
                f"distance(final/min)={result.final_distance:.4f}/"
                f"{result.minimum_distance:.4f} | "
                f"action_rms={result.rms_action:.3f} | "
                f"plan={result.mean_planning_time_ms:.2f} ms | "
                f"ESS(mean/final)={format_optional(result.mean_ess)}/"
                f"{format_optional(result.final_ess)}"
            )

        returns = np.asarray(
            [result.episode_return for result in results],
            dtype=np.float64,
        )
        successes = [result for result in results if result.success]
        success_rate = 100.0 * len(successes) / len(results)
        mean_minimum_distance = float(
            np.mean([result.minimum_distance for result in results])
        )
        mean_final_distance = float(
            np.mean([result.final_distance for result in results])
        )
        mean_planning_time_ms = float(
            np.mean([result.mean_planning_time_ms for result in results])
        )

        print(
            f"Reached: {len(successes)}/{len(results)} "
            f"({success_rate:.1f}%) | "
            f"Mean return: {returns.mean():.3f} | "
            f"Std: {returns.std():.3f}"
        )
        print(
            f"Mean minimum distance: {mean_minimum_distance:.4f} | "
            f"Mean final distance: {mean_final_distance:.4f} | "
            f"Mean planning time: {mean_planning_time_ms:.2f} ms/step"
        )
    finally:
        if dynamics is not None:
            dynamics.close()
        env.close()


if __name__ == "__main__":
    main()
