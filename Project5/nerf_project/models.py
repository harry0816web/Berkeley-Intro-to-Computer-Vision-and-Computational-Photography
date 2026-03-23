from __future__ import annotations

import torch
from torch import nn

from .encoding import PositionalEncoding


class NeuralField2D(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 256,
        num_hidden_layers: int = 3,
        num_frequencies: int = 10,
    ):
        super().__init__()
        self.encoding = PositionalEncoding(2, num_frequencies)
        layers: list[nn.Module] = []
        input_dim = self.encoding.output_dim
        for _ in range(num_hidden_layers):
            layers.extend([nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)])
            input_dim = hidden_dim
        layers.extend([nn.Linear(hidden_dim, 3), nn.Sigmoid()])
        self.network = nn.Sequential(*layers)

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        return self.network(self.encoding(coordinates))


class NeRF(nn.Module):
    """NeRF MLP with a middle skip connection and view-conditioned RGB head."""

    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 8,
        position_frequencies: int = 10,
        direction_frequencies: int = 4,
        skip_layer: int = 4,
    ):
        super().__init__()
        if not 0 < skip_layer < num_layers:
            raise ValueError("skip_layer must be inside the MLP")
        self.position_encoding = PositionalEncoding(3, position_frequencies)
        self.direction_encoding = PositionalEncoding(3, direction_frequencies)
        self.skip_layer = skip_layer

        position_dim = self.position_encoding.output_dim
        self.trunk = nn.ModuleList()
        for layer_index in range(num_layers):
            input_dim = position_dim if layer_index == 0 else hidden_dim
            if layer_index == skip_layer:
                input_dim += position_dim
            self.trunk.append(nn.Linear(input_dim, hidden_dim))

        self.density_head = nn.Linear(hidden_dim, 1)
        self.feature_head = nn.Linear(hidden_dim, hidden_dim)
        self.color_head = nn.Sequential(
            nn.Linear(hidden_dim + self.direction_encoding.output_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 3),
            nn.Sigmoid(),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(
        self,
        positions: torch.Tensor,
        directions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded_position = self.position_encoding(positions)
        hidden = encoded_position
        for layer_index, layer in enumerate(self.trunk):
            if layer_index == self.skip_layer:
                hidden = torch.cat([hidden, encoded_position], dim=-1)
            hidden = self.activation(layer(hidden))

        density = self.activation(self.density_head(hidden))
        features = self.feature_head(hidden)
        encoded_direction = self.direction_encoding(directions)
        color = self.color_head(torch.cat([features, encoded_direction], dim=-1))
        return density, color
