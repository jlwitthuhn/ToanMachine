# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

import mlx.core as mx
from mlx import nn, utils

from toan.model.activation import get_activation_module
from toan.model.metadata import ModelMetadata
from toan.model.nam_a1_wavenet import _NamA1Conv1dLayer
from toan.model.nam_a2_wavenet_config import (
    NamA2ActivationDetails,
    NamA2WaveNetConfig,
    NamA2WaveNetHead1x1Config,
    NamA2WaveNetLayer1x1Config,
    NamA2WaveNetLayerGroupConfig,
)
from toan.training import LossFunction

# Based on code from Neural Amp Modeler
# https://github.com/sdatkinson/neural-amp-modeler/tree/main/nam/models/wavenet


class _NamA2Conv1dLayer(_NamA1Conv1dLayer):
    pass


class _NamA2WrappedActivation(nn.Module):
    gate_type: str
    primary: nn.Module
    secondary: nn.Module | None = None

    def __init__(
        self,
        primary_config: NamA2ActivationDetails,
        secondary_name: str,
        gating_mode: str,
    ):
        super().__init__()
        self.gate_type = gating_mode

        self.primary = get_activation_module(primary_config.type)
        if gating_mode is None or gating_mode == "none":
            pass
        elif gating_mode == "gated":
            self.secondary = get_activation_module(secondary_name)
        else:
            raise ValueError(f"Unknown gating mode: {gating_mode}")

    def __call__(self, x: mx.array) -> mx.array:
        if self.gate_type is None or self.gate_type == "none":
            return self.primary(x)
        else:
            assert self.secondary is not None
            x1, x2 = x.split(2, axis=2)
            return self.primary(x1) * self.secondary(x2)


class _NamA2WaveNetLayer(nn.Module):
    bottleneck: int

    conv: _NamA2Conv1dLayer
    input_mixer: _NamA2Conv1dLayer
    activation: _NamA2WrappedActivation
    layer1x1: nn.Module | None = None
    head1x1: nn.Module | None = None

    def __init__(
        self,
        condition_size: int,
        channels: int,
        kernel_size: int,
        dilation: int,
        activation: NamA2ActivationDetails,
        secondary_activation: str,
        bottleneck: int,
        head_1x1_config: NamA2WaveNetHead1x1Config,
        layer_1x1_config: NamA2WaveNetLayer1x1Config,
        groups_input: int,
        groups_input_mixin: int,
        gating_mode: str,
    ):
        super().__init__()

        self.layer1x1 = None
        self.head1x1 = None

        self.bottleneck = bottleneck

        if gating_mode is None or gating_mode == "none":
            mid_channels = bottleneck
        elif gating_mode == "gated":
            mid_channels = 2 * bottleneck
        else:
            raise ValueError(f"Unknown gating mode: {gating_mode}")

        self.conv = _NamA2Conv1dLayer(
            channels, mid_channels, kernel_size, dilation=dilation, groups=groups_input
        )
        self.input_mixer = _NamA2Conv1dLayer(
            condition_size, mid_channels, 1, bias=False, groups=groups_input_mixin
        )
        self.activation = _NamA2WrappedActivation(
            activation, secondary_activation, gating_mode
        )

        if layer_1x1_config.active:
            self.layer1x1 = _NamA2Conv1dLayer(
                bottleneck, channels, 1, groups=layer_1x1_config.groups
            )
        else:
            assert bottleneck == channels

        if head_1x1_config.active:
            self.head1x1 = _NamA2Conv1dLayer(
                bottleneck,
                head_1x1_config.out_channels,
                1,
                groups=head_1x1_config.groups,
            )

    @property
    def parameter_count(self) -> int:
        return sum(p.size for _, p in utils.tree_flatten(self.parameters()))

    def __call__(
        self, x: mx.array, h: mx.array, out_length: int
    ) -> tuple[mx.array, mx.array]:
        conv_input = x
        zconv = self.conv(conv_input)

        mixin_input = h
        mix_out = self.input_mixer(mixin_input)[:, -zconv.shape[1] :, :]

        z1len = min(zconv.shape[1], mix_out.shape[1])
        z1 = zconv[:, -z1len:, :] + mix_out[:, -z1len:, :]
        post_activation = self.activation(z1)

        layer_output = post_activation
        if self.layer1x1 is not None:
            layer_output = self.layer1x1(layer_output)

        head_output = post_activation
        if self.head1x1 is not None:
            head_output = self.head1x1(head_output)[:, -out_length:, :]
        else:
            head_output = head_output[:, -out_length:, :]

        residual = x[:, -layer_output.shape[1] :, :] + layer_output
        return residual, head_output

    def debug_print_size(self):
        def count_module_params(module: nn.Module) -> int:
            return sum(p.size for _, p in utils.tree_flatten(module.parameters()))

        print(f">>> layer: {self.parameter_count}")
        print(f">>>> conv: {count_module_params(self.conv)}")
        print(f">>>> input_mixer: {count_module_params(self.input_mixer)}")
        print(f">>>> activation: {count_module_params(self.activation)}")
        if self.layer1x1 is not None:
            print(f">>>> layer1x1: {count_module_params(self.layer1x1)}")
        else:
            print(f">>>> layer1x1: 0")
        if self.head1x1 is not None:
            print(f">>>> head1x1: {count_module_params(self.head1x1)}")
        else:
            print(f">>>> head1x1: 0")


class _NamA2WaveNetLayerGroup(nn.Module):
    config: NamA2WaveNetLayerGroupConfig

    rechannel: _NamA2Conv1dLayer
    layers: list[_NamA2WaveNetLayer]
    head_rechannel: _NamA2Conv1dLayer

    def __init__(self, config: NamA2WaveNetLayerGroupConfig):
        super().__init__()

        self.config = config

        self.rechannel = _NamA2Conv1dLayer(
            config.input_size, config.channels, 1, bias=False
        )

        real_bottleneck = (
            config.channels if config.bottleneck is None else config.bottleneck
        )

        self.layers = [
            _NamA2WaveNetLayer(
                config.condition_size,
                config.channels,
                config.kernel_size,
                config.dilations[i],
                config.activation,
                config.secondary_activation,
                real_bottleneck,
                config.head1x1,
                config.layer1x1,
                config.groups_input,
                config.groups_input_mixin,
                config.gating_mode,
            )
            for i in range(len(config.dilations))
        ]

        head_rechannel_in = (
            config.head1x1.out_channels if config.head1x1.active else real_bottleneck
        )
        self.head_rechannel = _NamA2Conv1dLayer(
            head_rechannel_in,
            config.head_size,
            1,
            bias=config.head_bias,
        )

    @property
    def parameter_count(self) -> int:
        return sum(p.size for _, p in utils.tree_flatten(self.parameters()))

    @property
    def receptive_field(self) -> int:
        return 1 + (self.config.kernel_size - 1) * sum(self.config.dilations)

    def __call__(
        self, x: mx.array, c: mx.array, head_input: mx.array | None
    ) -> tuple[mx.array, mx.array]:
        # Index 2 in original script
        out_length = min(x.shape[1], c.shape[1]) - (self.receptive_field - 1)
        x = self.rechannel(x)
        for layer in self.layers:
            x, head_term = layer(x, c, out_length)
            head_input = (
                head_term
                if head_input is None
                # '-out_length' was originally indexed to third dimension
                else head_input[:, -out_length:, :] + head_term
            )
        return self.head_rechannel(head_input), x

    def debug_print_size(self):
        print(f">> layergroup: {self.parameter_count}")
        for layer in self.layers:
            layer.debug_print_size()


class NamA2WaveNet(nn.Module):
    config: NamA2WaveNetConfig
    metadata: ModelMetadata

    layer_groups: list[_NamA2WaveNetLayerGroup]
    head: None = None
    head_scale: mx.array

    def __init__(
        self,
        config: NamA2WaveNetConfig,
        metadata: ModelMetadata,
        sample_rate: int,
        rng_seed: int = 0x35,
    ) -> None:
        super().__init__()

        rng_state = mx.random.state
        mx.random.seed(rng_seed)

        self.config = config
        self.metadata = metadata
        self.sample_rate = sample_rate
        self.layer_groups = [
            _NamA2WaveNetLayerGroup(layer_config) for layer_config in config.layers
        ]
        self.head_scale = mx.array(config.head_scale)

        mx.random.state = rng_state

    @property
    def parameter_count(self) -> int:
        return sum(p.size for _, p in utils.tree_flatten(self.parameters()))

    @property
    def receptive_field(self) -> int:
        return 1 + sum(
            [(layer_group.receptive_field - 1) for layer_group in self.layer_groups]
        )

    def _forward(self, x: mx.array) -> mx.array:
        c = x
        y, head_input = x, None
        for layer_group in self.layer_groups:
            head_input, y = layer_group(y, c, head_input=head_input)
        head_input = self.config.head_scale * head_input
        return head_input

    def __call__(self, x: mx.array) -> mx.array:
        if x.ndim == 2:
            # In original NAM source this adds a middle dimension instead
            # We add a third dimension because that is what MLX conv modules want
            x = x[:, :, None]
        y = self._forward(x)
        assert y.shape[2] == 1
        return y[:, :, 0]

    def loss(self, inputs: mx.array, targets: mx.array, func: LossFunction) -> mx.array:
        match func:
            case LossFunction.MSE:
                return self.loss_mse(inputs, targets)
            case LossFunction.RMSE:
                return self.loss_rmse(inputs, targets)
            case LossFunction.ESR:
                return self.loss_esr(inputs, targets)
            case _:
                assert False

    def loss_mse(self, inputs: mx.array, targets: mx.array) -> mx.array:
        outputs = self(inputs)
        delta = targets - outputs
        delta2 = delta**2
        return delta2.mean()

    def loss_rmse(self, inputs: mx.array, targets: mx.array) -> mx.array:
        return mx.sqrt(self.loss_mse(inputs, targets))

    def debug_print_size(self):
        print("++ A2 Model Size ++")
        print(f"> root: {self.parameter_count}")
        for layer_group in self.layer_groups:
            layer_group.debug_print_size()
