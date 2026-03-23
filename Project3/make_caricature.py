"""Create shape caricatures of Eleven using the IMM population mean geometry."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from morph import read_image, warp_image, write_image


ROOT = Path(__file__).resolve().parent
ALPHAS = [1.0, 1.05, 1.1, 1.15]


def main() -> None:
    image = read_image(ROOT / "eleven_aligned.jpg")
    fixed_points = ROOT / "eleven_imm_points_fixed.npz"
    point_path = fixed_points if fixed_points.exists() else ROOT / "eleven_imm_points.npz"
    data = np.load(point_path, allow_pickle=False)
    source_points = data["source_points"]
    mean_points = data["target_mean_points"]
    triangles = data["triangles"]
    output_shape = image.shape[:2]

    output_dir = ROOT / "results/caricature"
    output_dir.mkdir(parents=True, exist_ok=True)
    images: list[np.ndarray] = []

    for alpha in ALPHAS:
        caricature_points = mean_points + alpha * (source_points - mean_points)
        caricature = warp_image(
            image,
            source_points,
            caricature_points,
            triangles,
            output_shape,
        )
        caricature = np.clip(caricature, 0, 1)
        images.append(caricature)
        output_name = f"eleven_caricature_alpha_{alpha:g}.jpg"
        write_image(output_dir / output_name, caricature)

    fig, axes = plt.subplots(1, len(ALPHAS), figsize=(16, 5))
    for ax, alpha, image_result in zip(axes, ALPHAS, images):
        ax.imshow(image_result)
        ax.set_title(f"alpha = {alpha:g}")
        ax.axis("off")
    fig.suptitle("Eleven caricature by extrapolating away from the IMM mean shape")
    fig.tight_layout()
    fig.savefig(output_dir / "caricature_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved caricatures to {output_dir}")


if __name__ == "__main__":
    main()
