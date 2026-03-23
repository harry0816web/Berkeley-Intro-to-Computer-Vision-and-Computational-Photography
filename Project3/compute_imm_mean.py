"""Compute the IMM population mean face from JPG/ASF pairs."""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v3 as iio
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Delaunay

from morph import read_image, warp_image, write_image


IMM_LANDMARK_COUNT = 58
BORDER_POINTS = np.array(
    [
        [0.0, 0.0],
        [0.5, 0.0],
        [1.0, 0.0],
        [0.0, 0.5],
        [1.0, 0.5],
        [0.0, 1.0],
        [0.5, 1.0],
        [1.0, 1.0],
    ],
    dtype=float,
)


def parse_asf(path: Path) -> np.ndarray:
    """Read the 58 normalized (x, y) landmarks from an IMM ASF file."""
    points: dict[int, tuple[float, float]] = {}
    for line in path.read_text(errors="replace").splitlines():
        fields = line.split()
        if len(fields) < 5:
            continue
        try:
            point_index = int(fields[4])
            x = float(fields[2])
            y = float(fields[3])
        except ValueError:
            continue
        if 0 <= point_index < IMM_LANDMARK_COUNT:
            points[point_index] = (x, y)

    if len(points) != IMM_LANDMARK_COUNT:
        raise ValueError(
            f"{path} contains {len(points)} landmarks; "
            f"expected {IMM_LANDMARK_COUNT}."
        )
    return np.asarray([points[index] for index in range(IMM_LANDMARK_COUNT)])


def normalized_to_pixels(points: np.ndarray, height: int, width: int) -> np.ndarray:
    return points * np.array([width - 1, height - 1], dtype=float)


def collect_pairs(dataset_dir: Path) -> tuple[list[tuple[Path, Path]], np.ndarray]:
    pairs: list[tuple[Path, Path]] = []
    shapes: list[tuple[int, int, int]] = []
    for image_path in sorted(dataset_dir.glob("*.jpg")):
        asf_path = image_path.with_suffix(".asf")
        if not asf_path.exists():
            raise FileNotFoundError(f"Missing annotation for {image_path.name}")
        image = iio.imread(image_path)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected an RGB image: {image_path}")
        pairs.append((image_path, asf_path))
        shapes.append(image.shape)

    if not pairs:
        raise ValueError(f"No JPG images found directly inside {dataset_dir}")
    if len(set(shapes)) != 1:
        raise ValueError(f"Dataset images do not share one shape: {set(shapes)}")
    return pairs, np.asarray(shapes[0])


def save_shape_preview(
    mean_image: np.ndarray,
    mean_points: np.ndarray,
    triangles: np.ndarray,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(mean_image)
    ax.triplot(mean_points[:, 0], mean_points[:, 1], triangles, color="cyan", linewidth=0.45)
    ax.scatter(mean_points[:, 0], mean_points[:, 1], color="yellow", s=4)
    ax.set_title("IMM population mean shape (58 landmarks + 8 border anchors)")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_montage(images: list[np.ndarray], titles: list[str], output_path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for ax, image, title in zip(axes.flat, images, titles):
        ax.imshow(image)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute the IMM mean face.")
    parser.add_argument("--dataset-dir", default="IMM-Face")
    parser.add_argument("--output-dir", default="results/imm_mean")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs, image_shape = collect_pairs(dataset_dir)
    height, width, _ = image_shape
    normalized_points = np.vstack(
        [parse_asf(asf_path) for _, asf_path in pairs]
    )
    normalized_points = normalized_points.reshape(len(pairs), IMM_LANDMARK_COUNT, 2)
    normalized_points = np.concatenate(
        [normalized_points, np.broadcast_to(BORDER_POINTS, (len(pairs), 8, 2))],
        axis=1,
    )
    mean_normalized_points = normalized_points.mean(axis=0)
    mean_points = normalized_to_pixels(mean_normalized_points, height, width)
    triangles = Delaunay(mean_points).simplices

    accumulator = np.zeros((height, width, 3), dtype=np.float64)
    example_indices = np.linspace(0, len(pairs) - 1, 6, dtype=int)
    examples: dict[int, np.ndarray] = {}

    for index, (image_path, _) in enumerate(pairs):
        image = read_image(image_path)
        source_points = normalized_to_pixels(normalized_points[index], height, width)
        warped = warp_image(image, source_points, mean_points, triangles, (height, width))
        accumulator += warped
        if index in example_indices:
            examples[index] = warped
        if (index + 1) % 20 == 0 or index == len(pairs) - 1:
            print(f"Warped {index + 1}/{len(pairs)} images")

    mean_face = np.clip(accumulator / len(pairs), 0, 1)
    write_image(output_dir / "imm_mean_face.jpg", mean_face)
    save_shape_preview(
        mean_face,
        mean_points,
        triangles,
        output_dir / "imm_mean_shape.png",
    )

    example_images = [examples[index] for index in example_indices]
    example_titles = [pairs[index][0].stem for index in example_indices]
    save_montage(
        example_images,
        example_titles,
        output_dir / "imm_warped_examples.png",
    )
    np.savez(
        output_dir / "imm_mean_geometry.npz",
        mean_normalized_points=mean_normalized_points,
        mean_points=mean_points,
        triangles=triangles,
        image_shape=np.asarray(image_shape),
        image_count=np.asarray(len(pairs)),
    )
    print(f"Computed mean face from {len(pairs)} images")
    print(f"Saved results to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
