from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from nerf_project.rays import RaysData, make_intrinsics
from nerf_project.rendering import sample_along_rays


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/lego_200x200.npz"))
    parser.add_argument("--output", type=Path, default=Path("outputs/part2/rays_and_samples.png"))
    parser.add_argument("--num-rays", type=int, default=100)
    args = parser.parse_args()

    data = np.load(args.data)
    images = torch.from_numpy(data["images_train"].astype(np.float32) / 255.0)
    c2ws = torch.from_numpy(data["c2ws_train"].astype(np.float32))
    height, width = images.shape[1:3]
    intrinsics = make_intrinsics(height, width, float(data["focal"]))
    dataset = RaysData(images, intrinsics, c2ws)
    generator = torch.Generator().manual_seed(42)
    origins, directions, _ = dataset.sample_rays(args.num_rays, generator=generator)
    points, _ = sample_along_rays(origins, directions, num_samples=16, perturb=False)

    fig = plt.figure(figsize=(9, 8))
    axis = fig.add_subplot(111, projection="3d")
    camera_centers = c2ws[:, :3, 3]
    camera_forward = c2ws[:, :3, 2]
    axis.scatter(*camera_centers.T, s=10, c="black", label="training cameras")
    axis.quiver(
        *camera_centers.T,
        *camera_forward.T,
        length=0.25,
        normalize=True,
        color="tab:blue",
        alpha=0.5,
    )
    for origin, direction in zip(origins, directions):
        segment = torch.stack([origin, origin + direction * 6.0]).numpy()
        axis.plot(segment[:, 0], segment[:, 1], segment[:, 2], color="tab:red", alpha=0.2)
    flat_points = points.reshape(-1, 3).numpy()
    axis.scatter(*flat_points.T, s=2, c="tab:orange", alpha=0.4, label="ray samples")
    axis.set_title("Training Cameras, Sampled Rays, and 3D Points")
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_zlabel("z")
    axis.legend()
    axis.set_box_aspect((1, 1, 1))
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    plt.close(fig)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
