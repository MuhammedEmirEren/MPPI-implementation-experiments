from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
from PIL import Image

from run_pendulum import (
    DEFAULT_CONFIG_PATH,
    build_controller,
    load_config,
    require_section,
    resolve_device,
    seed_everything,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SHOWCASE_SEED = 20
DEFAULT_VIDEO_PATH = PROJECT_ROOT / "assets" / "pendulum_mppi_showcase.mp4"
DEFAULT_PREVIEW_PATH = PROJECT_ROOT / "assets" / "pendulum_mppi_showcase.gif"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record one MPPI-controlled Pendulum-v1 episode."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_VIDEO_PATH)
    parser.add_argument("--preview", type=Path, default=DEFAULT_PREVIEW_PATH)
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SHOWCASE_SEED,
        help=f"Environment and controller seed (default: {DEFAULT_SHOWCASE_SEED}).",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Playback FPS; defaults to round(1 / model.dt).",
    )
    parser.add_argument(
        "--start-hold-seconds",
        type=float,
        default=1.5,
        help="Seconds to hold the initial state before control begins.",
    )
    parser.add_argument(
        "--end-hold-seconds",
        type=float,
        default=2.5,
        help="Seconds to hold the final stabilized state.",
    )
    parser.add_argument(
        "--preview-size",
        type=int,
        default=320,
        help="Width and height of the animated GIF preview.",
    )
    parser.add_argument(
        "--preview-stride",
        type=int,
        default=4,
        help="Keep every Nth video frame in the GIF preview.",
    )
    return parser.parse_args()


def resize_frame(frame: np.ndarray[Any, np.dtype[np.uint8]], size: int) -> Image.Image:
    """Convert a rendered RGB frame into a square GIF preview frame."""

    image = Image.fromarray(frame)
    return image.resize((size, size), Image.Resampling.LANCZOS)


def main() -> None:
    args = parse_args()
    if args.preview_size <= 0:
        raise ValueError("--preview-size must be positive")
    if args.preview_stride <= 0:
        raise ValueError("--preview-stride must be positive")
    if args.fps is not None and args.fps <= 0:
        raise ValueError("--fps must be positive")
    if args.start_hold_seconds < 0 or args.end_hold_seconds < 0:
        raise ValueError("video hold durations cannot be negative")

    config = load_config(args.config)
    environment_config = require_section(config, "environment")
    model_config = require_section(config, "model")
    seed = int(args.seed)
    requested_device = str(
        config.get("device", "cpu") if args.device is None else args.device
    )
    device = resolve_device(requested_device)
    seed_everything(seed)

    video_path = args.output.expanduser().resolve()
    preview_path = args.preview.expanduser().resolve()
    video_path.parent.mkdir(parents=True, exist_ok=True)
    preview_path.parent.mkdir(parents=True, exist_ok=True)

    environment_id = str(environment_config.get("id", "Pendulum-v1"))
    env = gym.make(environment_id, render_mode="rgb_array")
    fps = (
        int(args.fps)
        if args.fps is not None
        else round(1.0 / float(model_config["dt"]))
    )
    writer = imageio.get_writer(
        video_path,
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=None,
    )

    preview_frames: list[Image.Image] = []
    episode_return = 0.0
    episode_length = 0
    video_frame_index = 0
    last_preview_frame_index = -1

    def append_frame(frame: np.ndarray[Any, np.dtype[np.uint8]]) -> None:
        """Write one MP4 frame and periodically retain a GIF frame."""

        nonlocal video_frame_index, last_preview_frame_index
        writer.append_data(frame)
        if video_frame_index % args.preview_stride == 0:
            preview_frames.append(resize_frame(frame, args.preview_size))
            last_preview_frame_index = video_frame_index
        video_frame_index += 1

    try:
        controller = build_controller(
            config,
            env,
            device=device,
            seed=seed,
        )
        observation, _ = env.reset(seed=seed)
        env.action_space.seed(seed)
        controller.reset(seed=seed)

        initial_frame = env.render()
        start_hold_frames = max(1, round(args.start_hold_seconds * fps))
        for _ in range(start_hold_frames):
            append_frame(initial_frame)

        terminated = False
        truncated = False
        while not (terminated or truncated):
            action = controller.act(observation)
            observation, reward, terminated, truncated, _ = env.step(action)
            episode_return += float(reward)
            episode_length += 1

            frame = env.render()
            append_frame(frame)

        end_hold_frames = round(args.end_hold_seconds * fps)
        for _ in range(end_hold_frames):
            append_frame(frame)

        if last_preview_frame_index != video_frame_index - 1:
            preview_frames.append(resize_frame(frame, args.preview_size))
    finally:
        writer.close()
        env.close()

    frame_duration_ms = round(1000 * args.preview_stride / fps)
    preview_frames[0].save(
        preview_path,
        save_all=True,
        append_images=preview_frames[1:],
        duration=frame_duration_ms,
        loop=0,
        optimize=True,
    )

    print(f"Recorded {episode_length} steps with return {episode_return:.3f}")
    print(f"Playback: {video_frame_index / fps:.1f} seconds at {fps} FPS")
    print(f"MP4: {video_path}")
    print(f"README preview: {preview_path}")


if __name__ == "__main__":
    main()
