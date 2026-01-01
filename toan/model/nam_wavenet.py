# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

# Based on code from Neural Amp Modeler
# https://github.com/sdatkinson/neural-amp-modeler/blob/e054002e48cd102b0993811d69e8172db4a91597/nam/models/wavenet.py

from dataclasses import dataclass

import mlx.core as mx
from mlx import nn


@dataclass
class NameWaveNetLayerGroupConfig:
    input_size: int
    condition_size: int
    head_size: int
    channels: int
    kernel_size: int
    dilations: list[int]
    activation: str
    gated: bool
    head_bias: bool


@dataclass
class NamWaveNetConfig:
    layers: list[NameWaveNetLayerGroupConfig]
    head_config: None = None
    head_scale: float = 0.02


def DefaultConfig() -> NamWaveNetConfig:
    return NamWaveNetConfig(
        layers=[
            NameWaveNetLayerGroupConfig(
                input_size=1,
                condition_size=1,
                head_size=8,
                channels=16,
                kernel_size=3,
                dilations=[
                    1,
                    2,
                    4,
                    8,
                    16,
                    32,
                    64,
                    128,
                    256,
                    512,
                ],
                activation="Tanh",
                gated=False,
                head_bias=False,
            ),
            NameWaveNetLayerGroupConfig(
                input_size=16,
                condition_size=1,
                head_size=1,
                channels=8,
                kernel_size=3,
                dilations=[
                    1,
                    2,
                    4,
                    8,
                    16,
                    32,
                    64,
                    128,
                    256,
                    512,
                ],
                activation="Tanh",
                gated=True,
                head_bias=True,
            ),
        ]
    )


def _get_activation(activation: str) -> nn.Module:
    if activation == "Tanh":
        return nn.Tanh()
    assert False


# Specialization to support exporting/importing to a single huge array
# Functionally the same as normal conv1d
class _NamConv1dLayer(nn.Conv1d):
    pass


class _NamWaveNetLayer(nn.Module):
    channels: int
    conv: _NamConv1dLayer
    input_mixer: _NamConv1dLayer
    activation: nn.Module
    activation_name: str
    gated: bool

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
        self.conv = _NamConv1dLayer(
            channels, mid_channels, kernel_size, dilation=dilation
        )
        self.input_mixer = _NamConv1dLayer(condition_size, mid_channels, 1, bias=False)
        self.activation = _get_activation(activation)
        self.activation_name = activation
        self.conv1x1 = _NamConv1dLayer(channels, channels, 1)
        self.gated = gated

    def __call__(
        self, x: mx.array, h: mx.array, out_length: int
    ) -> tuple[mx.array, mx.array]:
        zconv = self.conv(x)
        z1 = zconv + self.input_mixer(h)[:, :, -zconv.shape[2] :]
        post_activation = (
            self.activation(z1)
            if not self.gated
            else (
                self.activation(z1[:, : self.channels])
                * mx.sigmoid(z1[:, self.channels :])
            )
        )
        return (
            x[:, :, -post_activation.shape[2] :] + self.conv1x1(post_activation),
            post_activation[:, :, -out_length:],
        )


class _NamWaveNetLayerGroup(nn.Module):
    config: NameWaveNetLayerGroupConfig
    rechannel: _NamConv1dLayer
    layers: list[_NamWaveNetLayer]
    head_rechannel: _NamConv1dLayer

    def __init__(self, config: NameWaveNetLayerGroupConfig):
        super().__init__()
        self.config = config
        self.rechannel = _NamConv1dLayer(
            config.input_size, config.channels, 1, bias=False
        )
        self.layers = [
            _NamWaveNetLayer(
                config.condition_size,
                config.channels,
                config.kernel_size,
                dilation,
                config.activation,
                config.gated,
            )
            for dilation in config.dilations
        ]
        self.head_rechannel = _NamConv1dLayer(
            config.channels, config.head_size, 1, bias=config.head_bias
        )

    def __call__(
        self, x: mx.array, c: mx.array, head_input: mx.array | None = None
    ) -> tuple[mx.array, mx.array]:
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

    @property
    def receptive_field(self) -> int:
        return 1 + (self.config.kernel_size - 1) * sum(self.config.dilations)


class NamWaveNet(nn.Module):
    layer_groups: list[_NamWaveNetLayerGroup]
    head: None = None

    def __init__(self, config: NamWaveNetConfig):
        super().__init__()
        self.layer_groups = [
            _NamWaveNetLayerGroup(layer_config) for layer_config in config.layers
        ]
        assert config.head_config is None

    @property
    def receptive_field(self) -> int:
        return 1 + sum([(group.receptive_field - 1) for group in self.layer_groups])

    def _forward(self, x: mx.array) -> mx.array:
        head_input, y = None, x
        for group in self.layer_groups:
            head_input, y = group(x, y, head_input)
        return head_input if self.head is None else self.head(head_input)

    def __call__(self, x: mx.array) -> mx.array:
        if x.ndim == 2:
            x = x[:, None, :]
        y = self._forward(x)
        assert y.shape[1] == 1
        return y[:, 0, :]
