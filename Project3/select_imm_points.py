"""Select IMM-compatible landmarks on Eleven for the caricature experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


LANDMARK_COUNT = 58


def landmark_label(index: int) -> str:
    if index <= 12:
        return f"jawline {index}"
    if index <= 20:
        return f"eye group A {index - 13}"
    if index <= 28:
        return f"eye group B {index - 21}"
    if index <= 33:
        return f"eyebrow group A {index - 29}"
    if index <= 38:
        return f"eyebrow group B {index - 34}"
    if index <= 46:
        return f"mouth {index - 39}"
    return f"nose {index - 47}"


def wait_for_left_click(fig: plt.Figure, left_ax: plt.Axes) -> tuple[float, float]:
    clicked: list[tuple[float, float]] = []

    def on_click(event) -> None:
        if event.inaxes is left_ax and event.button == 1:
            if event.xdata is not None and event.ydata is not None:
                clicked.append((event.xdata, event.ydata))

    connection = fig.canvas.mpl_connect("button_press_event", on_click)
    while not clicked and plt.fignum_exists(fig.number):
        plt.pause(0.05)
    fig.canvas.mpl_disconnect(connection)
    if not clicked:
        raise RuntimeError("Landmark selection was cancelled.")
    return clicked[0]


def draw_step(
    axes: np.ndarray,
    image: np.ndarray,
    reference: np.ndarray,
    reference_points: np.ndarray,
    selected: list[tuple[float, float]],
    index: int,
) -> None:
    left_ax, right_ax = axes
    left_ax.clear()
    left_ax.imshow(image)
    if selected:
        previous = np.asarray(selected)
        left_ax.scatter(previous[:, 0], previous[:, 1], color="yellow", s=14)
        for point_index, (x, y) in enumerate(previous):
            left_ax.text(x + 3, y - 3, str(point_index), color="yellow", fontsize=7)
    left_ax.set_title(
        f"Eleven: click point {index}/{LANDMARK_COUNT - 1}\n"
        f"{landmark_label(index)}"
    )
    left_ax.axis("off")

    right_ax.clear()
    right_ax.imshow(reference)
    right_ax.scatter(
        reference_points[:index, 0],
        reference_points[:index, 1],
        color="cyan",
        s=10,
    )
    right_ax.scatter(
        [reference_points[index, 0]],
        [reference_points[index, 1]],
        color="red",
        s=32,
    )
    right_ax.text(
        reference_points[index, 0] + 4,
        reference_points[index, 1] - 4,
        str(index),
        color="red",
        fontsize=9,
    )
    right_ax.set_title("IMM mean reference\nred = current point")
    right_ax.axis("off")
    plt.tight_layout()
    plt.pause(0.05)


def main() -> None:
    parser = argparse.ArgumentParser(description="Select 58 IMM landmarks on Eleven.")
    parser.add_argument("--image", default="eleven_aligned.jpg")
    parser.add_argument("--geometry", default="results/imm_mean/imm_mean_geometry.npz")
    parser.add_argument("--reference", default="results/imm_mean/imm_mean_face.jpg")
    parser.add_argument("--output", default="eleven_imm_points.npz")
    parser.add_argument(
        "--preview", default="results/caricature/eleven_imm_landmarks.png"
    )
    args = parser.parse_args()

    image = mpimg.imread(args.image)
    reference = mpimg.imread(args.reference)
    geometry = np.load(args.geometry, allow_pickle=False)
    reference_points = geometry["mean_points"][:LANDMARK_COUNT]
    normalized_mean = geometry["mean_normalized_points"]
    triangles = geometry["triangles"]

    height, width = image.shape[:2]
    target_points = normalized_mean * np.array([width - 1, height - 1])
    fig, axes = plt.subplots(1, 2, figsize=(13, 7))
    try:
        manager = fig.canvas.manager
        manager.set_window_title("IMM landmarks for Eleven caricature")
    except AttributeError:
        pass

    selected: list[tuple[float, float]] = []
    try:
        for index in range(LANDMARK_COUNT):
            draw_step(axes, image, reference, reference_points, selected, index)
            selected.append(wait_for_left_click(fig, axes[0]))
    finally:
        plt.close(fig)

    border_points = target_points[LANDMARK_COUNT:]
    source_points = np.vstack([np.asarray(selected), border_points])
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        source_points=source_points,
        target_mean_points=target_points,
        triangles=triangles,
        image_shape=np.asarray(image.shape),
    )

    preview_path = Path(args.preview)
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 7))
    axes[0].imshow(image)
    axes[0].triplot(
        source_points[:, 0], source_points[:, 1], triangles, color="cyan", linewidth=0.5
    )
    axes[0].scatter(source_points[:, 0], source_points[:, 1], color="yellow", s=6)
    axes[0].set_title("Eleven IMM-compatible landmarks")
    axes[1].imshow(reference)
    axes[1].triplot(
        reference_points[:, 0],
        reference_points[:, 1],
        triangles,
        color="cyan",
        linewidth=0.5,
    )
    axes[1].scatter(reference_points[:, 0], reference_points[:, 1], color="yellow", s=6)
    axes[1].set_title("IMM mean reference landmarks")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(preview_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {len(source_points)} points to {output_path}")
    print(f"Saved preview to {preview_path}")


if __name__ == "__main__":
    main()
