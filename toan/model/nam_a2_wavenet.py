# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

import mlx.core as mx
from mlx import nn, utils

from toan.model.activation import get_activation_module
from toan.model.metadata import ModelMetadata
from toan.model.nam_a1_wavenet import _NamA1Conv1dLayer
from toan.model.nam_a2_wavenet_config import (
    NamA2WaveNetConfig,
    NamA2WaveNetHead1x1Config,
    NamA2WaveNetLayer1x1Config,
    NamA2WaveNetLayerGroupConfig,
)

# Based on code from Neural Amp Modeler
# https://github.com/sdatkinson/NeuralAmpModelerCore/blob/v0.4.0/NAM/wavenet.h


class _NamA2Conv1dLayer(_NamA1Conv1dLayer):
    pass


class _NamA2WaveNetLayer(nn.Module):
    bottleneck: int

    conv: _NamA2Conv1dLayer
    input_mixer: _NamA2Conv1dLayer
    activation: nn.Module
    layer1x1: nn.Module | None = None
    head1x1: nn.Module | None = None

    def __init__(
        self,
        condition_size: int,
        channels: int,
        kernel_size: int,
        dilation: int,
        activation: nn.Module,
        bottleneck: int,
        head_1x1_config: NamA2WaveNetHead1x1Config,
        layer_1x1_config: NamA2WaveNetLayer1x1Config,
        groups_input: int,
        groups_input_mixin: int,
    ):
        super().__init__()

        self.layer1x1 = None
        self.head1x1 = None

        self.bottleneck = bottleneck

        mid_channels = bottleneck
        self.conv = _NamA2Conv1dLayer(
            channels, mid_channels, kernel_size, dilation=dilation, groups=groups_input
        )
        self.input_mixer = _NamA2Conv1dLayer(
            condition_size, mid_channels, 1, bias=False, groups=groups_input_mixin
        )
        self.activation = activation

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

        activations = [
            get_activation_module(config.activation.type)
            for _ in range(len(config.dilations))
        ]

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
                activations[i],
                real_bottleneck,
                config.head1x1,
                config.layer1x1,
                config.groups_input,
                config.groups_input_mixin,
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

    def debug_print_size(self):
        print(f">> layergroup: {self.parameter_count}")
        for layer in self.layers:
            layer.debug_print_size()


class NamA2WaveNet(nn.Module):
    config: NamA2WaveNetConfig
    metadata: ModelMetadata

    layer_groups: list[_NamA2WaveNetLayerGroup]
    head: None = None
    head_scale: float = 1.0

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
        self.head_scale = config.head_scale

        mx.random.state = rng_state

    @property
    def parameter_count(self) -> int:
        return sum(p.size for _, p in utils.tree_flatten(self.parameters()))

    def debug_print_size(self):
        print("++ A2 Model Size ++")
        print(f"> root: {self.parameter_count}")
        for layer_group in self.layer_groups:
            layer_group.debug_print_size()
