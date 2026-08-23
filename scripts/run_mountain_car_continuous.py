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
from mppi_control.costs.mountain_car_continuous import (
    MountainCarContinuousCost,
)
from mppi_control.dynamics.mountain_car_continuous import (
    MountainCarContinuousDynamics,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "mountain_car_continuous.yaml"
)


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    """Performance and planning diagnostics for one episode."""

    success: bool
    episode_return: float
    episode_length: int
    final_position: float
    final_velocity: float
    mean_planning_time_ms: float
    mean_ess: float | None
    final_ess: float | None


def parse_args() -> argparse.Namespace:
    """Parse command-line options that override the YAML configuration."""

    parser = argparse.ArgumentParser(
        description=(
            "Run the MPPI controller on Gymnasium "
            "MountainCarContinuous-v0."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the MountainCarContinuous YAML configuration.",
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


def build_controller(
    config: Mapping[str, Any],
    env: gym.Env,
    *,
    device: torch.device,
    seed: int,
) -> MPPIController:
    """Construct MountainCar dynamics, planning cost, and MPPI."""

    model_config = require_section(config, "model")
    cost_config = require_section(config, "cost")
    mppi_config = require_section(config, "mppi")

    dynamics = MountainCarContinuousDynamics(
        min_position=float(model_config["min_position"]),
        max_position=float(model_config["max_position"]),
        max_speed=float(model_config["max_speed"]),
        min_action=float(model_config["min_action"]),
        max_action=float(model_config["max_action"]),
        power=float(model_config["power"]),
        gravity=float(model_config["gravity"]),
    )
    cost = MountainCarContinuousCost(
        mode=str(cost_config["mode"]),
        energy_weight=float(cost_config["energy_weight"]),
        action_weight=float(cost_config["action_weight"]),
        terminal_position_weight=float(
            cost_config["terminal_position_weight"]
        ),
        terminal_wrong_direction_weight=float(
            cost_config["terminal_wrong_direction_weight"]
        ),
        success_bonus=float(cost_config["success_bonus"]),
        min_position=float(model_config["min_position"]),
        max_position=float(model_config["max_position"]),
        max_speed=float(model_config["max_speed"]),
        goal_position=float(model_config["goal_position"]),
        goal_velocity=float(model_config["goal_velocity"]),
        gravity=float(model_config["gravity"]),
    )

    if not isinstance(env.action_space, gym.spaces.Box):
        raise TypeError("MPPI requires a continuous Gymnasium Box action space")

    return MPPIController(
        dynamics=dynamics,
        cost=cost,
        horizon=int(mppi_config["horizon"]),
        num_samples=int(mppi_config["num_samples"]),
        temperature=float(mppi_config["temperature"]),
        noise_sigma=float(mppi_config["noise_sigma"]),
        noise_rho=float(mppi_config.get("noise_rho", 0.0)),
        action_low=env.action_space.low,
        action_high=env.action_space.high,
        device=device,
        num_iterations=int(mppi_config.get("num_iterations", 1)),
        sampling_correction=bool(mppi_config.get("sampling_correction", True)),
        seed=seed,
    )


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
    """Run one episode until goal success or Gymnasium truncation."""

    observation, _ = env.reset(seed=seed)
    env.action_space.seed(seed)
    controller.reset(seed=seed)

    terminated = False
    truncated = False
    episode_return = 0.0
    episode_length = 0
    planning_times: list[float] = []
    effective_sample_sizes: list[float] = []

    while not (terminated or truncated):
        synchronize_device(controller.device)
        planning_start = perf_counter()
        action = controller.act(observation)
        synchronize_device(controller.device)
        planning_times.append(perf_counter() - planning_start)

        effective_sample_size = controller.effective_sample_size
        if effective_sample_size is not None:
            effective_sample_sizes.append(effective_sample_size)

        observation, reward, terminated, truncated, _ = env.step(action)
        episode_return += float(reward)
        episode_length += 1

    mean_planning_time_ms = 1000.0 * float(np.mean(planning_times))
    mean_ess = (
        float(np.mean(effective_sample_sizes))
        if effective_sample_sizes
        else None
    )
    final_ess = (
        effective_sample_sizes[-1] if effective_sample_sizes else None
    )

    return EpisodeResult(
        success=terminated,
        episode_return=episode_return,
        episode_length=episode_length,
        final_position=float(observation[0]),
        final_velocity=float(observation[1]),
        mean_planning_time_ms=mean_planning_time_ms,
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
        environment_config.get("id", "MountainCarContinuous-v0")
    )
    env = gym.make(environment_id, render_mode=render_mode)

    try:
        controller = build_controller(
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

            outcome = "success" if result.success else "timeout"
            print(
                f"Episode {episode_index + 1:>3}/{episodes} | "
                f"seed={episode_seed} | "
                f"outcome={outcome} | "
                f"steps={result.episode_length} | "
                f"return={result.episode_return:.3f} | "
                f"position={result.final_position:.4f} | "
                f"velocity={result.final_velocity:.4f} | "
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
        mean_planning_time_ms = float(
            np.mean([result.mean_planning_time_ms for result in results])
        )

        print(
            f"Success: {len(successes)}/{len(results)} "
            f"({success_rate:.1f}%) | "
            f"Mean return: {returns.mean():.3f} | "
            f"Std: {returns.std():.3f}"
        )
        if successes:
            mean_success_steps = float(
                np.mean([result.episode_length for result in successes])
            )
            print(f"Mean steps to goal: {mean_success_steps:.1f}")
        else:
            print("Mean steps to goal: n/a")
        print(f"Mean planning time: {mean_planning_time_ms:.2f} ms/step")
    finally:
        env.close()


if __name__ == "__main__":
    main()
