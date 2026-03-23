"""Repair two outlier clicks detected by the triangulation sanity check."""

from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent


def main() -> None:
    input_path = ROOT / "eleven_imm_points.npz"
    output_path = ROOT / "eleven_imm_points_fixed.npz"
    data = np.load(input_path, allow_pickle=False)
    source_points = data["source_points"].copy()

    # Point 47 is the left upper nose point in the IMM ordering. The original
    # click was below the nose, so place it above point 48 on the nose bridge.
    source_points[47] = [source_points[48, 0], source_points[57, 1]]
    # Point 38 is the outer end of the viewer-left eyebrow. Move it onto the
    # eyebrow contour so the mean-shape triangulation remains orientation-safe.
    source_points[38, 1] = max(source_points[38, 1], source_points[37, 1] + 27)

    np.savez(
        output_path,
        source_points=source_points,
        target_mean_points=data["target_mean_points"],
        triangles=data["triangles"],
        image_shape=data["image_shape"],
    )
    print(f"Saved repaired points to {output_path}")


if __name__ == "__main__":
    main()
