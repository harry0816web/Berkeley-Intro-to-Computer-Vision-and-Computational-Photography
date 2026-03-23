"""Generate frames 0–45 and an animated GIF for the face morph."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from morph import morph, read_image, write_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the complete morph sequence.")
    parser.add_argument("--image-a", default="eleven_aligned.jpg")
    parser.add_argument("--image-b", default="george_small.jpg")
    parser.add_argument("--points", default="face_points.npz")
    parser.add_argument("--output-dir", default="results/morph_frames")
    parser.add_argument("--gif", default="results/eleven_to_george.gif")
    parser.add_argument("--frame-count", type=int, default=46)
    parser.add_argument("--fps", type=float, default=30.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.frame_count < 2:
        raise ValueError("frame-count must be at least 2.")
    if args.fps <= 0:
        raise ValueError("fps must be positive.")

    image_a = read_image(args.image_a)
    image_b = read_image(args.image_b)
    if image_a.shape != image_b.shape:
        raise ValueError(
            f"Images must have identical shapes; got {image_a.shape} and {image_b.shape}."
        )

    data = np.load(args.points, allow_pickle=False)
    points_a = data["points_a"]
    points_b = data["points_b"]
    triangles = data["triangles"]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gif_frames: list[np.ndarray] = []

    for frame_index, fraction in enumerate(np.linspace(0, 1, args.frame_count)):
        frame, _, _ = morph(
            image_a,
            image_b,
            points_a,
            points_b,
            triangles,
            warp_fraction=float(fraction),
            dissolve_fraction=float(fraction),
        )
        frame_path = output_dir / f"frame_{frame_index:02d}.png"
        write_image(frame_path, frame)
        gif_frames.append(np.round(frame * 255).astype(np.uint8))
        print(
            f"Rendered frame {frame_index:02d}/{args.frame_count - 1:02d} "
            f"(fraction={fraction:.3f})"
        )

    gif_path = Path(args.gif)
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    pil_frames = [Image.fromarray(frame) for frame in gif_frames]
    # GIF timing is stored in 10 ms units. Rounding cumulative frame boundaries
    # distributes the small timing error and closely matches the requested fps.
    exact_boundaries = np.arange(len(pil_frames) + 1) * (1000 / args.fps)
    gif_boundaries = np.round(exact_boundaries / 10).astype(int) * 10
    frame_durations = np.maximum(10, np.diff(gif_boundaries)).tolist()
    pil_frames[0].save(
        gif_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=frame_durations,
        loop=0,
        disposal=2,
    )
    print(f"Saved animation to {gif_path.resolve()}")


if __name__ == "__main__":
    main()
