"""Piecewise-affine face morphing for CS180 Project 3."""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v3 as iio
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import map_coordinates
from skimage.draw import polygon


def compute_affine(
    source_triangle: np.ndarray, destination_triangle: np.ndarray
) -> np.ndarray:
    """Return the 3x3 affine matrix mapping destination coordinates to source."""
    source_h = np.vstack([source_triangle.T, np.ones(3)])
    destination_h = np.vstack([destination_triangle.T, np.ones(3)])
    return np.linalg.solve(destination_h.T, source_h.T).T


def warp_image(
    image: np.ndarray,
    source_points: np.ndarray,
    destination_points: np.ndarray,
    triangles: np.ndarray,
    output_shape: tuple[int, int],
) -> np.ndarray:
    """Warp an image into destination_points using one inverse warp per triangle."""
    height, width = output_shape
    channels = 1 if image.ndim == 2 else image.shape[2]
    source = image[..., None] if image.ndim == 2 else image
    warped = np.zeros((height, width, channels), dtype=np.float64)

    for triangle_indices in triangles:
        source_triangle = source_points[triangle_indices]
        destination_triangle = destination_points[triangle_indices]

        rows, columns = polygon(
            destination_triangle[:, 1],
            destination_triangle[:, 0],
            shape=(height, width),
        )
        destination_pixels = np.vstack(
            [columns, rows, np.ones_like(columns, dtype=float)]
        )
        destination_to_source = compute_affine(
            source_triangle, destination_triangle
        )
        source_pixels = destination_to_source @ destination_pixels
        source_x = source_pixels[0]
        source_y = source_pixels[1]

        for channel in range(channels):
            warped[rows, columns, channel] = map_coordinates(
                source[..., channel],
                [source_y, source_x],
                order=1,
                mode="nearest",
            )

    return warped[..., 0] if image.ndim == 2 else warped


def morph(
    image_a: np.ndarray,
    image_b: np.ndarray,
    points_a: np.ndarray,
    points_b: np.ndarray,
    triangles: np.ndarray,
    warp_fraction: float,
    dissolve_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Warp both images to an intermediate shape, then cross-dissolve them."""
    if not 0 <= warp_fraction <= 1 or not 0 <= dissolve_fraction <= 1:
        raise ValueError("Morph fractions must be between 0 and 1.")

    intermediate_points = (
        (1 - warp_fraction) * points_a + warp_fraction * points_b
    )
    output_shape = image_a.shape[:2]
    warped_a = warp_image(
        image_a, points_a, intermediate_points, triangles, output_shape
    )
    warped_b = warp_image(
        image_b, points_b, intermediate_points, triangles, output_shape
    )
    result = (1 - dissolve_fraction) * warped_a + dissolve_fraction * warped_b
    return np.clip(result, 0, 1), warped_a, warped_b


def read_image(path: str | Path) -> np.ndarray:
    image = iio.imread(path).astype(np.float64)
    if image.max() > 1:
        image /= 255.0
    return image


def write_image(path: str | Path, image: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(path, np.round(np.clip(image, 0, 1) * 255).astype(np.uint8))


def save_comparison(
    image_a: np.ndarray,
    midway: np.ndarray,
    image_b: np.ndarray,
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    for ax, image, title in zip(
        axes,
        (image_a, midway, image_b),
        ("Image A — Eleven", "Midway face", "Image B — George"),
    ):
        ax.imshow(image)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create the Project 3 midway face.")
    parser.add_argument("--image-a", default="eleven_aligned.jpg")
    parser.add_argument("--image-b", default="george_small.jpg")
    parser.add_argument("--points", default="face_points.npz")
    parser.add_argument("--output-dir", default="results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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

    midway, warped_a, warped_b = morph(
        image_a,
        image_b,
        points_a,
        points_b,
        triangles,
        warp_fraction=0.5,
        dissolve_fraction=0.5,
    )

    output_dir = Path(args.output_dir)
    write_image(output_dir / "eleven_warped_to_midway.jpg", warped_a)
    write_image(output_dir / "george_warped_to_midway.jpg", warped_b)
    write_image(output_dir / "midway_face.jpg", midway)
    save_comparison(
        image_a,
        midway,
        image_b,
        output_dir / "midway_comparison.png",
    )
    print(f"Saved midway results to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
