from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
from PIL import Image

from mppi_control.dynamics.reacher_mujoco import ReacherMujocoDynamics
from run_reacher import (
    DEFAULT_CONFIG_PATH,
    build_controller,
    fingertip_distance,
    load_config,
    require_section,
    resolve_device,
    seed_everything,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "assets"
DEFAULT_SHOWCASE_SEED = 7


@dataclass(frozen=True, slots=True)
class TargetScenario:
    """One deterministic target used by the Reacher showcase."""

    name: str
    label: str
    target: tuple[float, float]


SCENARIOS = (
    TargetScenario("close", "Close target", (0.18, 0.04)),
    TargetScenario("far", "Far target", (-0.17, -0.08)),
    TargetScenario("upper", "Upper target", (0.03, 0.16)),
)
SCENARIOS_BY_NAME = {scenario.name: scenario for scenario in SCENARIOS}


@dataclass(frozen=True, slots=True)
class RecordingResult:
    """Control and media diagnostics for one recorded scenario."""

    scenario: TargetScenario
    success: bool
    first_success_step: int | None
    episode_return: float
    episode_length: int
    final_distance: float
    minimum_distance: float
    video_path: Path
    preview_path: Path
    playback_seconds: float


def parse_args() -> argparse.Namespace:
    """Parse recording and output options."""

    parser = argparse.ArgumentParser(
        description=(
            "Record close, far, and upper-target MPPI Reacher showcases."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
    )
    parser.add_argument(
        "--scenario",
        choices=("all", *SCENARIOS_BY_NAME),
        default="all",
        help="Record all showcase targets or one named target.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SHOWCASE_SEED,
        help=f"Initial arm and controller seed (default: {DEFAULT_SHOWCASE_SEED}).",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Playback FPS; defaults to Gymnasium's render FPS.",
    )
    parser.add_argument(
        "--start-hold-seconds",
        type=float,
        default=1.0,
        help="Seconds to show the initial arm and target before control.",
    )
    parser.add_argument(
        "--end-hold-seconds",
        type=float,
        default=1.5,
        help="Seconds to hold the final state.",
    )
    parser.add_argument(
        "--preview-width",
        type=int,
        default=260,
        help="GIF preview width; aspect ratio is preserved.",
    )
    parser.add_argument(
        "--preview-stride",
        type=int,
        default=4,
        help="Keep every Nth MP4 frame in each GIF preview.",
    )
    return parser.parse_args()


def resize_frame(
    frame: np.ndarray[Any, np.dtype[np.uint8]],
    width: int,
) -> Image.Image:
    """Resize an RGB frame while preserving its aspect ratio."""

    image = Image.fromarray(frame)
    height = round(width * image.height / image.width)
    return image.resize((width, height), Image.Resampling.LANCZOS)


def set_target(env: gym.Env, target: tuple[float, float]) -> np.ndarray:
    """Replace Reacher's randomly sampled target after a normal reset."""

    unwrapped = env.unwrapped
    target_array = np.asarray(target, dtype=np.float64)
    if target_array.shape != (2,) or not np.all(np.isfinite(target_array)):
        raise ValueError("target must contain two finite coordinates")
    if np.linalg.norm(target_array) >= 0.2:
        raise ValueError("Reacher showcase targets must lie inside radius 0.2")

    qpos = unwrapped.data.qpos.copy()
    qvel = unwrapped.data.qvel.copy()
    qpos[-2:] = target_array
    qvel[-2:] = 0.0
    unwrapped.goal = target_array.copy()
    unwrapped.set_state(qpos, qvel)
    return np.asarray(unwrapped._get_obs(), dtype=np.float64)


def record_scenario(
    *,
    env: gym.Env,
    controller: Any,
    scenario: TargetScenario,
    output_directory: Path,
    seed: int,
    success_distance: float,
    fps: int,
    start_hold_seconds: float,
    end_hold_seconds: float,
    preview_width: int,
    preview_stride: int,
) -> RecordingResult:
    """Run and encode one deterministic Reacher target scenario."""

    video_path = output_directory / f"reacher_{scenario.name}_target.mp4"
    preview_path = output_directory / f"reacher_{scenario.name}_target.gif"
    writer = imageio.get_writer(
        video_path,
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=None,
    )

    preview_frames: list[Image.Image] = []
    video_frame_index = 0
    last_preview_frame_index = -1

    def append_frame(frame: np.ndarray[Any, np.dtype[np.uint8]]) -> None:
        """Write an MP4 frame and periodically retain a GIF frame."""

        nonlocal video_frame_index, last_preview_frame_index
        writer.append_data(frame)
        if video_frame_index % preview_stride == 0:
            preview_frames.append(resize_frame(frame, preview_width))
            last_preview_frame_index = video_frame_index
        video_frame_index += 1

    episode_return = 0.0
    episode_length = 0
    terminated = False
    truncated = False
    first_success_step: int | None = None
    distances: list[float] = []

    try:
        env.reset(seed=seed)
        observation = set_target(env, scenario.target)
        env.action_space.seed(seed)
        controller.reset(seed=seed)

        initial_distance = fingertip_distance(observation)
        distances.append(initial_distance)
        if initial_distance <= success_distance:
            first_success_step = 0

        frame = env.render()
        start_hold_frames = max(1, round(start_hold_seconds * fps))
        for _ in range(start_hold_frames):
            append_frame(frame)

        while not (terminated or truncated):
            action = controller.act(observation)
            observation, reward, terminated, truncated, _ = env.step(action)
            episode_return += float(reward)
            episode_length += 1

            distance = fingertip_distance(observation)
            distances.append(distance)
            if first_success_step is None and distance <= success_distance:
                first_success_step = episode_length

            frame = env.render()
            append_frame(frame)

        end_hold_frames = round(end_hold_seconds * fps)
        for _ in range(end_hold_frames):
            append_frame(frame)

        if last_preview_frame_index != video_frame_index - 1:
            preview_frames.append(resize_frame(frame, preview_width))
    finally:
        writer.close()

    frame_duration_ms = round(1000 * preview_stride / fps)
    preview_frames[0].save(
        preview_path,
        save_all=True,
        append_images=preview_frames[1:],
        duration=frame_duration_ms,
        loop=0,
        optimize=True,
    )

    return RecordingResult(
        scenario=scenario,
        success=first_success_step is not None,
        first_success_step=first_success_step,
        episode_return=episode_return,
        episode_length=episode_length,
        final_distance=distances[-1],
        minimum_distance=float(np.min(distances)),
        video_path=video_path,
        preview_path=preview_path,
        playback_seconds=video_frame_index / fps,
    )


def main() -> None:
    """Create the selected Reacher showcase recordings."""

    args = parse_args()
    if args.preview_width <= 0:
        raise ValueError("--preview-width must be positive")
    if args.preview_stride <= 0:
        raise ValueError("--preview-stride must be positive")
    if args.fps is not None and args.fps <= 0:
        raise ValueError("--fps must be positive")
    if args.start_hold_seconds < 0 or args.end_hold_seconds < 0:
        raise ValueError("video hold durations cannot be negative")

    config = load_config(args.config)
    environment_config = require_section(config, "environment")
    evaluation_config = require_section(config, "evaluation")
    success_distance = float(evaluation_config["success_distance"])
    seed = int(args.seed)
    requested_device = str(
        config.get("device", "cpu") if args.device is None else args.device
    )
    device = resolve_device(requested_device)
    seed_everything(seed)

    output_directory = args.output_directory.expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    environment_id = str(environment_config.get("id", "Reacher-v5"))
    max_episode_steps = int(environment_config.get("max_episode_steps", 100))
    env = gym.make(
        environment_id,
        render_mode="rgb_array",
        max_episode_steps=max_episode_steps,
    )
    fps = (
        int(args.fps)
        if args.fps is not None
        else int(env.metadata["render_fps"])
    )
    scenarios = (
        SCENARIOS
        if args.scenario == "all"
        else (SCENARIOS_BY_NAME[args.scenario],)
    )
    dynamics: ReacherMujocoDynamics | None = None

    try:
        controller, dynamics = build_controller(
            config,
            env,
            device=device,
            seed=seed,
        )
        results = [
            record_scenario(
                env=env,
                controller=controller,
                scenario=scenario,
                output_directory=output_directory,
                seed=seed,
                success_distance=success_distance,
                fps=fps,
                start_hold_seconds=args.start_hold_seconds,
                end_hold_seconds=args.end_hold_seconds,
                preview_width=args.preview_width,
                preview_stride=args.preview_stride,
            )
            for scenario in scenarios
        ]
    finally:
        if dynamics is not None:
            dynamics.close()
        env.close()

    for result in results:
        outcome = "reached" if result.success else "not_reached"
        first_reach = (
            str(result.first_success_step)
            if result.first_success_step is not None
            else "n/a"
        )
        x, y = result.scenario.target
        print(
            f"{result.scenario.label}: target=({x:.2f}, {y:.2f}), "
            f"outcome={outcome}, first_reach={first_reach}, "
            f"return={result.episode_return:.3f}, "
            f"distance(final/min)={result.final_distance:.4f}/"
            f"{result.minimum_distance:.4f}"
        )
        print(
            f"  {result.playback_seconds:.1f}s | "
            f"{result.video_path.name} | {result.preview_path.name}"
        )

    failed = [result.scenario.name for result in results if not result.success]
    if failed:
        raise RuntimeError(
            "showcase target was not reached: " + ", ".join(failed)
        )


if __name__ == "__main__":
    main()
