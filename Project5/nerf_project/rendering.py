from __future__ import annotations

import torch


def sample_along_rays(
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    near: float = 2.0,
    far: float = 6.0,
    num_samples: int = 64,
    perturb: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stratified samples between near and far bounds for every ray."""
    device = ray_origins.device
    dtype = ray_origins.dtype
    edges = torch.linspace(near, far, num_samples + 1, device=device, dtype=dtype)
    lower, upper = edges[:-1], edges[1:]
    if perturb:
        random_offsets = torch.rand(
            (*ray_origins.shape[:-1], num_samples), device=device, dtype=dtype
        )
        t_values = lower + (upper - lower) * random_offsets
    else:
        t_values = (lower + upper) * 0.5
        t_values = t_values.expand(*ray_origins.shape[:-1], num_samples)
    points = ray_origins[..., None, :] + ray_directions[..., None, :] * t_values[..., :, None]
    return points, t_values


def volume_render(
    sigmas: torch.Tensor,
    rgbs: torch.Tensor,
    step_size: torch.Tensor | float,
    background_color: torch.Tensor | None = None,
) -> torch.Tensor:
    """Differentiable discrete volume rendering for batches of rays."""
    if sigmas.shape[-1] == 1:
        sigmas = sigmas.squeeze(-1)
    delta = torch.as_tensor(step_size, dtype=sigmas.dtype, device=sigmas.device)
    alpha = 1.0 - torch.exp(-sigmas * delta)
    transmittance = torch.cumprod(
        torch.cat([torch.ones_like(alpha[..., :1]), 1.0 - alpha + 1e-10], dim=-1),
        dim=-1,
    )[..., :-1]
    weights = transmittance * alpha
    rendered = torch.sum(weights[..., None] * rgbs, dim=-2)
    if background_color is not None:
        background = background_color.to(device=rgbs.device, dtype=rgbs.dtype)
        rendered = rendered + (1.0 - weights.sum(dim=-1, keepdim=True)) * background
    return rendered


def render_rays(
    model: torch.nn.Module,
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    near: float = 2.0,
    far: float = 6.0,
    num_samples: int = 64,
    perturb: bool = False,
    point_chunk_size: int = 65536,
    background_color: torch.Tensor | None = None,
) -> torch.Tensor:
    points, _ = sample_along_rays(
        ray_origins,
        ray_directions,
        near=near,
        far=far,
        num_samples=num_samples,
        perturb=perturb,
    )
    expanded_directions = ray_directions[..., None, :].expand_as(points)
    flat_points = points.reshape(-1, 3)
    flat_directions = expanded_directions.reshape(-1, 3)

    sigma_chunks: list[torch.Tensor] = []
    rgb_chunks: list[torch.Tensor] = []
    for start in range(0, flat_points.shape[0], point_chunk_size):
        end = start + point_chunk_size
        sigma, rgb = model(flat_points[start:end], flat_directions[start:end])
        sigma_chunks.append(sigma)
        rgb_chunks.append(rgb)
    sigmas = torch.cat(sigma_chunks).reshape(*points.shape[:-1], 1)
    rgbs = torch.cat(rgb_chunks).reshape(*points.shape[:-1], 3)
    return volume_render(
        sigmas,
        rgbs,
        (far - near) / num_samples,
        background_color=background_color,
    )
