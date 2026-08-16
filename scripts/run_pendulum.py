from __future__ import annotations

import argparse
import random
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import yaml

from mppi_control.controllers.mppi_controller import MPPIController
from mppi_control.costs.pendulum import PendulumCost
from mppi_control.dynamics.pendulum import PendulumDynamics


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "pendulum.yaml"


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    """Summary of one completed Pendulum episode."""

    episode_return: float
    episode_length: int
    final_angle: float
    final_angular_velocity: float


def parse_args() -> argparse.Namespace:
    """Parse command-line options that override the YAML configuration."""

    parser = argparse.ArgumentParser(
        description="Run the MPPI controller on Gymnasium Pendulum-v1."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the Pendulum YAML configuration.",
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
    """Seed the random number generators used outside the controller."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    """Resolve an explicit device or choose CUDA automatically when requested."""

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
    """Construct the analytical model, cost, and MPPI controller."""

    model_config = require_section(config, "model")
    cost_config = require_section(config, "cost")
    mppi_config = require_section(config, "mppi")

    dynamics = PendulumDynamics(
        gravity=float(model_config["gravity"]),
        mass=float(model_config["mass"]),
        length=float(model_config["length"]),
        dt=float(model_config["dt"]),
        max_torque=float(model_config["max_torque"]),
        max_speed=float(model_config["max_speed"]),
    )
    cost = PendulumCost(
        angle_weight=float(cost_config["angle_weight"]),
        velocity_weight=float(cost_config["velocity_weight"]),
        action_weight=float(cost_config["action_weight"]),
        terminal_weight=float(cost_config["terminal_weight"]),
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
        action_low=env.action_space.low,
        action_high=env.action_space.high,
        device=device,
        num_iterations=int(mppi_config.get("num_iterations", 1)),
        sampling_correction=bool(mppi_config.get("sampling_correction", True)),
        seed=seed,
    )


def run_episode(
    env: gym.Env,
    controller: MPPIController,
    *,
    seed: int,
) -> EpisodeResult:
    """Run one episode and return its basic performance measurements."""

    observation, _ = env.reset(seed=seed)
    env.action_space.seed(seed)
    controller.reset(seed=seed)

    terminated = False
    truncated = False
    episode_return = 0.0
    episode_length = 0

    while not (terminated or truncated):
        action = controller.act(observation)
        observation, reward, terminated, truncated, _ = env.step(action)

        episode_return += float(reward)
        episode_length += 1

    final_angle = float(np.arctan2(observation[1], observation[0]))
    final_angular_velocity = float(observation[2])

    return EpisodeResult(
        episode_return=episode_return,
        episode_length=episode_length,
        final_angle=final_angle,
        final_angular_velocity=final_angular_velocity,
    )


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

    environment_id = str(environment_config.get("id", "Pendulum-v1"))
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

        returns: list[float] = []
        for episode_index in range(episodes):
            episode_seed = seed + episode_index
            result = run_episode(
                env,
                controller,
                seed=episode_seed,
            )
            returns.append(result.episode_return)

            effective_sample_size = controller.effective_sample_size
            ess_text = (
                "n/a"
                if effective_sample_size is None
                else f"{effective_sample_size:.1f}"
            )
            print(
                f"Episode {episode_index + 1:>3}/{episodes} | "
                f"seed={episode_seed} | "
                f"steps={result.episode_length} | "
                f"return={result.episode_return:.3f} | "
                f"final_angle={result.final_angle:.3f} rad | "
                f"final_velocity={result.final_angular_velocity:.3f} rad/s | "
                f"ESS={ess_text}"
            )

        print(
            f"Mean return: {np.mean(returns):.3f} | "
            f"Std: {np.std(returns):.3f}"
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
