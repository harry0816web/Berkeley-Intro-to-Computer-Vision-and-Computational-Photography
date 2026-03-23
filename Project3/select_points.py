"""Interactively select corresponding face landmarks and build a triangulation."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Delaunay


# Left and right mean the viewer's left and right sides of the image.
LANDMARK_LABELS = [
    "left temple",
    "left ear top",
    "left upper cheek",
    "left middle cheek",
    "left upper jaw",
    "left lower jaw",
    "left chin edge",
    "chin center",
    "right chin edge",
    "right lower jaw",
    "right upper jaw",
    "right middle cheek",
    "right upper cheek",
    "right ear top",
    "right temple",
    "left eyebrow outer",
    "left eyebrow center",
    "left eyebrow inner",
    "right eyebrow inner",
    "right eyebrow center",
    "right eyebrow outer",
    "left eye outer corner",
    "left eye top",
    "left eye inner corner",
    "left eye bottom",
    "right eye inner corner",
    "right eye top",
    "right eye outer corner",
    "right eye bottom",
    "nose bridge top",
    "nose bridge middle",
    "nose tip",
    "left nostril outer",
    "nose base center",
    "right nostril outer",
    "mouth left corner",
    "upper lip left peak",
    "upper lip center",
    "upper lip right peak",
    "mouth right corner",
    "lower lip right",
    "lower lip center",
    "lower lip left",
    "inner mouth left",
    "inner mouth top",
    "inner mouth right",
    "inner mouth bottom",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Click matching landmarks on image A and image B."
    )
    parser.add_argument("--image-a", default="eleven_aligned.jpg")
    parser.add_argument("--image-b", default="george_small.jpg")
    parser.add_argument("--output", default="face_points.npz")
    parser.add_argument(
        "--preview", default="results/landmarks_and_triangulation.png"
    )
    return parser.parse_args()


def border_points(width: int, height: int) -> np.ndarray:
    """Return image-border anchors so the triangulation covers the whole frame."""
    x_max = width - 1
    y_max = height - 1
    return np.array(
        [
            [0, 0],
            [x_max / 2, 0],
            [x_max, 0],
            [0, y_max / 2],
            [x_max, y_max / 2],
            [0, y_max],
            [x_max / 2, y_max],
            [x_max, y_max],
        ],
        dtype=float,
    )


def click_landmark(
    ax: plt.Axes,
    image: np.ndarray,
    completed_points: list[tuple[float, float]],
    image_name: str,
    label: str,
    index: int,
) -> tuple[float, float]:
    ax.clear()
    ax.imshow(image)
    if completed_points:
        previous = np.asarray(completed_points)
        ax.scatter(previous[:, 0], previous[:, 1], c="yellow", s=16)
        for point_index, (x, y) in enumerate(previous, start=1):
            ax.text(x + 3, y - 3, str(point_index), color="yellow", fontsize=7)
    ax.set_title(
        f"{image_name}: point {index + 1}/{len(LANDMARK_LABELS)} — {label}\n"
        "Click once. Left/right refer to the image as you see it."
    )
    ax.axis("off")
    ax.figure.canvas.draw_idle()

    clicked = plt.ginput(1, timeout=-1, show_clicks=True)
    if not clicked:
        raise RuntimeError("Point selection was cancelled before completion.")
    return clicked[0]


def collect_correspondences(
    image_a: np.ndarray, image_b: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    points_a: list[tuple[float, float]] = []
    points_b: list[tuple[float, float]] = []
    fig, ax = plt.subplots(figsize=(8, 9))
    fig.canvas.manager.set_window_title("CS180 Project 3 — Landmark Selection")

    try:
        for index, label in enumerate(LANDMARK_LABELS):
            points_a.append(
                click_landmark(ax, image_a, points_a, "Image A (Eleven)", label, index)
            )
            points_b.append(
                click_landmark(ax, image_b, points_b, "Image B (George)", label, index)
            )
    finally:
        plt.close(fig)

    return np.asarray(points_a), np.asarray(points_b)


def save_preview(
    image_a: np.ndarray,
    image_b: np.ndarray,
    points_a: np.ndarray,
    points_b: np.ndarray,
    triangles: np.ndarray,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))

    for ax, image, points, title in zip(
        axes,
        (image_a, image_b),
        (points_a, points_b),
        ("Eleven landmarks", "George landmarks"),
    ):
        ax.imshow(image)
        ax.triplot(points[:, 0], points[:, 1], triangles, color="cyan", linewidth=0.6)
        ax.scatter(points[:, 0], points[:, 1], c="yellow", s=9)
        ax.set_title(title)
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    image_a = mpimg.imread(args.image_a)
    image_b = mpimg.imread(args.image_b)

    if image_a.shape[:2] != image_b.shape[:2]:
        raise ValueError(
            "Both images must have the same height and width; got "
            f"{image_a.shape[:2]} and {image_b.shape[:2]}."
        )

    height, width = image_a.shape[:2]
    selected_a, selected_b = collect_correspondences(image_a, image_b)
    anchors = border_points(width, height)
    points_a = np.vstack([selected_a, anchors])
    points_b = np.vstack([selected_b, anchors])

    average_points = (points_a + points_b) / 2
    triangles = Delaunay(average_points).simplices

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        points_a=points_a,
        points_b=points_b,
        triangles=triangles,
        labels=np.asarray(LANDMARK_LABELS + [f"border {i}" for i in range(8)]),
    )
    save_preview(
        image_a,
        image_b,
        points_a,
        points_b,
        triangles,
        Path(args.preview),
    )
    print(f"Saved {len(points_a)} point pairs to {output_path}")
    print(f"Saved triangulation preview to {args.preview}")


if __name__ == "__main__":
    main()
