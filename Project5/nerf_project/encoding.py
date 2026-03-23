from __future__ import annotations

import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding used by both project parts.

    The input is retained, matching the project specification:
    [x, sin(2^0 pi x), cos(2^0 pi x), ..., sin(2^(L-1) pi x), cos(...)].
    """

    def __init__(self, input_dim: int, num_frequencies: int, include_input: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.num_frequencies = num_frequencies
        self.include_input = include_input
        self.register_buffer(
            "frequency_bands",
            (2.0 ** torch.arange(num_frequencies, dtype=torch.float32)) * math.pi,
            persistent=False,
        )

    @property
    def output_dim(self) -> int:
        multiplier = 2 * self.num_frequencies + int(self.include_input)
        return self.input_dim * multiplier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"Expected last dimension {self.input_dim}, got {x.shape[-1]}")
        encoded = [x] if self.include_input else []
        angles = x.unsqueeze(-2) * self.frequency_bands.to(dtype=x.dtype).view(
            *((1,) * (x.ndim - 1)), -1, 1
        )
        encoded.extend([torch.sin(angles).flatten(-2), torch.cos(angles).flatten(-2)])
        return torch.cat(encoded, dim=-1)
