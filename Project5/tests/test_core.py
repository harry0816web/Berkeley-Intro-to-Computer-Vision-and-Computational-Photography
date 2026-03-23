import torch

from nerf_project.encoding import PositionalEncoding
from nerf_project.models import NeRF, NeuralField2D
from nerf_project.rays import pixel_to_camera, pixel_to_ray, transform
from nerf_project.rendering import sample_along_rays, volume_render


def test_positional_encoding_shape() -> None:
    encoding = PositionalEncoding(2, 10)
    output = encoding(torch.zeros(5, 2))
    assert output.shape == (5, 42)


def test_transform_round_trip() -> None:
    angle = torch.tensor(0.37)
    c, s = torch.cos(angle), torch.sin(angle)
    c2w = torch.tensor(
        [[c, -s, 0.0, 1.0], [s, c, 0.0, -2.0], [0.0, 0.0, 1.0, 0.5], [0.0, 0.0, 0.0, 1.0]]
    )
    points = torch.randn(16, 3)
    recovered = transform(torch.linalg.inv(c2w), transform(c2w, points))
    assert torch.allclose(points, recovered, atol=1e-5)


def test_pixel_camera_and_ray() -> None:
    intrinsics = torch.tensor([[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]])
    uv = torch.tensor([[50.0, 40.0], [60.0, 40.0]])
    points = pixel_to_camera(intrinsics, uv, 2.0)
    assert torch.allclose(points[0], torch.tensor([0.0, 0.0, 2.0]))
    origins, directions = pixel_to_ray(intrinsics, torch.eye(4), uv)
    assert torch.allclose(origins, torch.zeros_like(origins))
    assert torch.allclose(torch.linalg.norm(directions, dim=-1), torch.ones(2))


def test_sample_shapes() -> None:
    origins = torch.zeros(7, 3)
    directions = torch.tensor([[0.0, 0.0, 1.0]]).expand(7, 3)
    points, t_values = sample_along_rays(origins, directions, num_samples=32, perturb=False)
    assert points.shape == (7, 32, 3)
    assert t_values.shape == (7, 32)
    assert torch.all(points[..., 2] >= 2.0)
    assert torch.all(points[..., 2] <= 6.0)


def test_volume_render_matches_staff_assertion() -> None:
    torch.manual_seed(42)
    sigmas = torch.rand((10, 64, 1))
    rgbs = torch.rand((10, 64, 3))
    rendered = volume_render(sigmas, rgbs, (6.0 - 2.0) / 64)
    correct = torch.tensor(
        [
            [0.5006, 0.3728, 0.4728],
            [0.4322, 0.3559, 0.4134],
            [0.4027, 0.4394, 0.4610],
            [0.4514, 0.3829, 0.4196],
            [0.4002, 0.4599, 0.4103],
            [0.4471, 0.4044, 0.4069],
            [0.4285, 0.4072, 0.3777],
            [0.4152, 0.4190, 0.4361],
            [0.4051, 0.3651, 0.3969],
            [0.3253, 0.3587, 0.4215],
        ]
    )
    assert torch.allclose(rendered, correct, rtol=1e-4, atol=1e-4)


def test_network_shapes_and_ranges() -> None:
    field = NeuralField2D(hidden_dim=32, num_hidden_layers=2, num_frequencies=4)
    rgb_2d = field(torch.rand(9, 2))
    assert rgb_2d.shape == (9, 3)
    assert torch.all((rgb_2d >= 0.0) & (rgb_2d <= 1.0))

    nerf = NeRF(hidden_dim=32, num_layers=5, skip_layer=2, position_frequencies=3, direction_frequencies=2)
    sigma, rgb = nerf(torch.rand(11, 3), torch.rand(11, 3))
    assert sigma.shape == (11, 1)
    assert rgb.shape == (11, 3)
    assert torch.all(sigma >= 0.0)
    assert torch.all((rgb >= 0.0) & (rgb <= 1.0))
