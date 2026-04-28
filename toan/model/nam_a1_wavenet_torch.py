# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

import json

import torch
from torch import nn

from toan.model.activation_torch import get_activation_module_torch
from toan.model.metadata import ModelMetadata
from toan.model.nam_a1_wavenet_config import (
    NamA1WaveNetConfig,
    NamA1WaveNetLayerGroupConfig,
)

# Based on code from Neural Amp Modeler
# https://github.com/sdatkinson/neural-amp-modeler/blob/e054002e48cd102b0993811d69e8172db4a91597/nam/models/wavenet.py


class _NamA1Conv1dLayerTorch(nn.Conv1d):
    def export_nam_linear_weights(self) -> list[float]:
        result = []
        if self.weight is not None:
            result.extend(self.weight.detach().flatten().tolist())
        if self.bias is not None:
            result.extend(self.bias.detach().flatten().tolist())
        return result

    def import_nam_linear_weights(self, weights: torch.Tensor, i: int) -> int:
        if self.weight is not None:
            size = self.weight.numel()
            my_slice = weights[i : i + size]
            with torch.no_grad():
                self.weight.copy_(my_slice.reshape(self.weight.shape))
            i = i + size
        if self.bias is not None:
            size = self.bias.numel()
            my_slice = weights[i : i + size]
            with torch.no_grad():
                self.bias.copy_(my_slice.reshape(self.bias.shape))
            i = i + size
        return i


class _NamA1WaveNetLayerTorch(nn.Module):
    def __init__(
        self,
        condition_size: int,
        channels: int,
        kernel_size: int,
        dilation: int,
        activation: str,
        gated: bool,
    ):
        super().__init__()
        mid_channels = 2 * channels if gated else channels
        self.channels = channels
        self.conv = _NamA1Conv1dLayerTorch(
            channels, mid_channels, kernel_size, dilation=dilation
        )
        self.input_mixer = _NamA1Conv1dLayerTorch(
            condition_size, mid_channels, 1, bias=False
        )
        # NOTE: get_activation_module returns an MLX module, not a torch module.
        # We use it as-is per the porting plan; it will be swapped out later.
        self.activation = get_activation_module_torch(activation)
        self.activation_name = activation
        self.conv1x1 = _NamA1Conv1dLayerTorch(channels, channels, 1)
        self.gated = gated

    def forward(
        self, x: torch.Tensor, h: torch.Tensor, out_length: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        zconv = self.conv(x)
        z1 = zconv + self.input_mixer(h)[:, :, -zconv.shape[2] :]
        post_activation = (
            self.activation(z1)
            if not self.gated
            else (
                self.activation(z1[:, : self.channels, :])
                * torch.sigmoid(z1[:, self.channels :, :])
            )
        )
        return (
            x[:, :, -post_activation.shape[2] :] + self.conv1x1(post_activation),
            post_activation[:, :, -out_length:],
        )

    def export_nam_linear_weights(self) -> list[float]:
        result = []
        result.extend(self.conv.export_nam_linear_weights())
        result.extend(self.input_mixer.export_nam_linear_weights())
        result.extend(self.conv1x1.export_nam_linear_weights())
        return result

    def import_nam_linear_weights(self, weights: torch.Tensor, i: int) -> int:
        i = self.conv.import_nam_linear_weights(weights, i)
        i = self.input_mixer.import_nam_linear_weights(weights, i)
        i = self.conv1x1.import_nam_linear_weights(weights, i)
        return i


class _NamA1WaveNetLayerGroupTorch(nn.Module):
    def __init__(self, config: NamA1WaveNetLayerGroupConfig):
        super().__init__()
        self.config = config
        self.rechannel = _NamA1Conv1dLayerTorch(
            config.input_size, config.channels, 1, bias=False
        )
        self.layers = nn.ModuleList(
            [
                _NamA1WaveNetLayerTorch(
                    config.condition_size,
                    config.channels,
                    config.kernel_size,
                    dilation,
                    config.activation,
                    config.gated,
                )
                for dilation in config.dilations
            ]
        )
        self.head_rechannel = _NamA1Conv1dLayerTorch(
            config.channels, config.head_size, 1, bias=config.head_bias
        )

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        head_input: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out_length = x.shape[2] - (self.receptive_field - 1)
        x = self.rechannel(x)
        for layer in self.layers:
            x, head_term = layer(x, c, out_length)
            head_input = (
                head_term
                if head_input is None
                else head_input[:, :, -out_length:] + head_term
            )
        return self.head_rechannel(head_input), x

    def export_nam_linear_weights(self) -> list[float]:
        result = []
        result.extend(self.rechannel.export_nam_linear_weights())
        for layer in self.layers:
            result.extend(layer.export_nam_linear_weights())
        result.extend(self.head_rechannel.export_nam_linear_weights())
        return result

    def import_nam_linear_weights(self, weights: torch.Tensor, i: int) -> int:
        i = self.rechannel.import_nam_linear_weights(weights, i)
        for layer in self.layers:
            i = layer.import_nam_linear_weights(weights, i)
        i = self.head_rechannel.import_nam_linear_weights(weights, i)
        return i

    @property
    def receptive_field(self) -> int:
        return self.config.receptive_field()


class NamA1WaveNetTorch(nn.Module):
    head: None = None

    def __init__(
        self,
        config: NamA1WaveNetConfig,
        metadata: ModelMetadata,
        sample_rate: int,
        rng_seed: int = 0x35,
    ):
        super().__init__()

        rng_state = torch.get_rng_state()
        torch.manual_seed(rng_seed)

        self.config = config
        self.metadata = metadata
        self.sample_rate = sample_rate
        self.layer_groups = nn.ModuleList(
            [
                _NamA1WaveNetLayerGroupTorch(layer_config)
                for layer_config in config.layers
            ]
        )
        assert config.head_config is None

        torch.set_rng_state(rng_state)

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def receptive_field(self) -> int:
        return 1 + sum([(group.receptive_field - 1) for group in self.layer_groups])

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        y, head_input = x, None
        for group in self.layer_groups:
            head_input, y = group(y, x, head_input)
        result = head_input if self.head is None else self.head(head_input)
        return result

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x[:, None, :]
        y = self._forward(x)
        assert y.shape[1] == 1
        return y[:, 0, :]

    def export_nam_linear_weights(self) -> list[float]:
        result = []
        for group in self.layer_groups:
            group_weights: list[float] = group.export_nam_linear_weights()
            result.extend(group_weights)
        assert self.head is None
        result.append(
            1.0
        )  # head_scale, not actually used but we need to export something
        return result

    def export_nam_json_str(self) -> str:
        root = {
            "version": "0.5.4",
            "metadata": self.metadata.export_dict(),
            "architecture": "WaveNet",
            "config": self.config.export_dict(),
            "weights": self.export_nam_linear_weights(),
            "sample_rate": self.sample_rate,
        }
        return json.dumps(root)

    def import_nam_linear_weights(self, weights: list[float]) -> int:
        i = 0
        weights_t = torch.tensor(weights)
        for group in self.layer_groups:
            i = group.import_nam_linear_weights(weights_t, i)
        return i
