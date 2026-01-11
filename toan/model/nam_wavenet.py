# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

import json

import mlx.core as mx
from mlx import nn, utils

from toan.model.metadata import ModelMetadata
from toan.model.nam_wavenet_config import NameWaveNetLayerGroupConfig, NamWaveNetConfig

# Based on code from Neural Amp Modeler
# https://github.com/sdatkinson/neural-amp-modeler/blob/e054002e48cd102b0993811d69e8172db4a91597/nam/models/wavenet.py


def _get_activation(activation: str) -> nn.Module:
    if activation == "Tanh":
        return nn.Tanh()
    assert False


# Specialization to support exporting/importing to a single huge array
# And also to transpose inputs to play nice with MLX dimension ordering
# Functionally the same as normal conv1d
class _NamConv1dLayer(nn.Conv1d):
    def export_nam_linear_weights(self) -> list[float]:
        result = []
        if self.weight is not None:
            result.extend(self.weight.flatten().tolist())
        try:
            if self.bias is not None:
                result.extend(self.bias.flatten().tolist())
        except AttributeError:
            # No 'bias'
            pass
        return result

    def import_nam_linear_weights(self, weights: mx.array, i: int) -> int:
        if self.weight is not None:
            size = self.weight.size
            my_slice = weights[i : i + size]

            # MLX uses dimensions arranged as (X, Y, Z)
            # Torch instead uses (X, Z, Y)
            # So we create weights in the torch shape then transpose axes
            assert self.weight.ndim == 3
            x, y, z = self.weight.shape
            torch_weights = my_slice.reshape((x, z, y))
            self.weight = torch_weights.transpose(0, 2, 1)
            i = i + size
        try:
            if self.bias is not None:
                size = self.bias.size
                my_slice = weights[i : i + size]
                self.bias = my_slice.reshape(self.bias.shape)
                i = i + size
        except AttributeError:
            # No `bias`
            pass
        return i

    def __call__(self, x: mx.array) -> mx.array:
        # Input is in the format (Batch, Channel, Sequence)
        # MLX Wants channel last
        x = x.transpose(0, 2, 1)
        x = super().__call__(x)
        return x.transpose(0, 2, 1)


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

    def export_nam_linear_weights(self) -> list[float]:
        result = []
        result.extend(self.conv.export_nam_linear_weights())
        result.extend(self.input_mixer.export_nam_linear_weights())
        result.extend(self.conv1x1.export_nam_linear_weights())
        return result

    def import_nam_linear_weights(self, weights: mx.array, i: int) -> int:
        i = self.conv.import_nam_linear_weights(weights, i)
        i = self.input_mixer.import_nam_linear_weights(weights, i)
        i = self.conv1x1.import_nam_linear_weights(weights, i)
        return i


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

    def export_nam_linear_weights(self) -> list[float]:
        result = []
        result.extend(self.rechannel.export_nam_linear_weights())
        for layer in self.layers:
            result.extend(layer.export_nam_linear_weights())
        result.extend(self.head_rechannel.export_nam_linear_weights())
        return result

    def import_nam_linear_weights(self, weights: mx.array, i: int) -> int:
        i = self.rechannel.import_nam_linear_weights(weights, i)
        for layer in self.layers:
            i = layer.import_nam_linear_weights(weights, i)
        i = self.head_rechannel.import_nam_linear_weights(weights, i)
        return i

    @property
    def receptive_field(self) -> int:
        return 1 + (self.config.kernel_size - 1) * sum(self.config.dilations)


class NamWaveNet(nn.Module):
    layer_groups: list[_NamWaveNetLayerGroup]
    head: None = None

    config: NamWaveNetConfig
    metadata: ModelMetadata
    sample_rate: int = 0

    def loss_fn(self, inputs: mx.array, targets: mx.array) -> mx.array:
        outputs = self(inputs)
        delta = targets - outputs
        delta2 = delta**2
        ms = delta2.mean()
        return mx.sqrt(ms)

    def __init__(
        self, config: NamWaveNetConfig, metadata: ModelMetadata, sample_rate: int
    ):
        super().__init__()
        self.config = config
        self.metadata = metadata
        self.sample_rate = sample_rate
        self.layer_groups = [
            _NamWaveNetLayerGroup(layer_config) for layer_config in config.layers
        ]
        assert config.head_config is None

    @property
    def parameter_count(self) -> int:
        return sum(p.size for _, p in utils.tree_flatten(self.parameters()))

    @property
    def receptive_field(self) -> int:
        return 1 + sum([(group.receptive_field - 1) for group in self.layer_groups])

    def _forward(self, x: mx.array) -> mx.array:
        y, head_input = x, None
        for group in self.layer_groups:
            head_input, y = group(y, x, head_input)
        return head_input if self.head is None else self.head(head_input)

    def __call__(self, x: mx.array) -> mx.array:
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
        weights_mx = mx.array(weights)
        for group in self.layer_groups:
            i = group.import_nam_linear_weights(weights_mx, i)
        return i
