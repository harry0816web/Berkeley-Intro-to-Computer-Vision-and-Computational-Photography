from __future__ import annotations

import torch
import torch.nn.functional as F


def transform(c2w: torch.Tensor, points_camera: torch.Tensor) -> torch.Tensor:
    """Transform camera-space points to world space, with batch broadcasting."""
    points_h = torch.cat([points_camera, torch.ones_like(points_camera[..., :1])], dim=-1)
    return torch.matmul(c2w, points_h.unsqueeze(-1)).squeeze(-1)[..., :3]


def pixel_to_camera(
    intrinsics: torch.Tensor,
    uv: torch.Tensor,
    depth: torch.Tensor | float,
) -> torch.Tensor:
    """Unproject pixel coordinates at optical-axis depth ``depth``."""
    uv_h = torch.cat([uv, torch.ones_like(uv[..., :1])], dim=-1)
    camera_direction = torch.linalg.solve(intrinsics, uv_h.unsqueeze(-1)).squeeze(-1)
    depth_tensor = torch.as_tensor(depth, dtype=uv.dtype, device=uv.device)
    if depth_tensor.ndim == uv.ndim - 1:
        depth_tensor = depth_tensor.unsqueeze(-1)
    return camera_direction * depth_tensor


def pixel_to_ray(
    intrinsics: torch.Tensor,
    c2w: torch.Tensor,
    uv: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert pixel coordinates to world-space ray origins and unit directions."""
    camera_points = pixel_to_camera(intrinsics, uv, 1.0)
    world_points = transform(c2w, camera_points)
    origins = c2w[..., :3, 3].expand_as(world_points)
    directions = F.normalize(world_points - origins, dim=-1)
    return origins, directions


def make_intrinsics(height: int, width: int, focal: float, device: torch.device | None = None) -> torch.Tensor:
    return torch.tensor(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
        device=device,
    )


def pixel_grid(height: int, width: int, device: torch.device) -> torch.Tensor:
    """Return flattened (u, v) pixel centers in row-major image order."""
    v, u = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    return torch.stack([u + 0.5, v + 0.5], dim=-1).reshape(-1, 2)


def get_rays_for_camera(
    intrinsics: torch.Tensor,
    c2w: torch.Tensor,
    height: int,
    width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    uv = pixel_grid(height, width, c2w.device)
    return pixel_to_ray(intrinsics, c2w, uv)


class RaysData:
    """On-demand ray sampler for a set of posed RGB images."""

    def __init__(self, images: torch.Tensor, intrinsics: torch.Tensor, c2ws: torch.Tensor):
        if images.ndim != 4 or images.shape[-1] != 3:
            raise ValueError("images must have shape [N, H, W, 3]")
        self.images = images
        self.intrinsics = intrinsics
        self.c2ws = c2ws
        self.num_images, self.height, self.width, _ = images.shape

    def sample_rays(
        self,
        batch_size: int,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = self.images.device
        pixels_per_image = self.height * self.width
        indices = torch.randint(
            self.num_images * pixels_per_image,
            (batch_size,),
            device=device,
            generator=generator,
        )
        image_indices = torch.div(indices, pixels_per_image, rounding_mode="floor")
        local_indices = indices % pixels_per_image
        v = torch.div(local_indices, self.width, rounding_mode="floor")
        u = local_indices % self.width
        uv = torch.stack([u, v], dim=-1).to(torch.float32) + 0.5
        colors = self.images[image_indices, v, u]
        origins, directions = pixel_to_ray(
            self.intrinsics,
            self.c2ws[image_indices],
            uv,
        )
        return origins, directions, colors
