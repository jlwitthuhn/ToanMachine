# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

from dataclasses import dataclass, field


@dataclass
class NamA2ActivationDetails:
    type: str


@dataclass
class NamA2FilmConfig:
    active: bool
    shift: bool


@dataclass
class NamA2WaveNetHead1x1Config:
    active: bool
    groups: int
    out_channels: int


@dataclass
class NamA2WaveNetLayer1x1Config:
    active: bool
    groups: int


@dataclass
class NamA2WaveNetLayerGroupConfig:
    input_size: int
    condition_size: int
    head_size: int
    channels: int
    bottleneck: int
    kernel_size: int
    dilations: list[int]
    activation: NamA2ActivationDetails
    gating_mode: str
    head_bias: bool
    groups_input: int
    groups_input_mixin: int
    layer1x1: NamA2WaveNetLayer1x1Config
    head1x1: NamA2WaveNetHead1x1Config
    secondary_activation: str
    conv_pre_film: NamA2FilmConfig
    conv_post_film: NamA2FilmConfig
    input_mixin_pre_film: NamA2FilmConfig
    input_mixin_post_film: NamA2FilmConfig
    activation_pre_film: NamA2FilmConfig
    activation_post_film: NamA2FilmConfig
    layer1x1_post_film: NamA2FilmConfig
    head1x1_post_film: NamA2FilmConfig


@dataclass
class NamA2WaveNetConfig:
    layers: list[NamA2WaveNetLayerGroupConfig] = field(default_factory=list)
    head: None = None
    head_scale: float = 0.02


def _get_typed_value(root: dict, key: str, type_in: type):
    if key not in root or not isinstance(root[key], type_in):
        raise TypeError(f"key '{key}' must be a {type_in.__name__}")
    return root[key]


def _get_activation_config(root: dict, key: str) -> NamA2ActivationDetails:
    if key not in root or not isinstance(root[key], dict):
        raise TypeError(f"key '{key}' must be a dict.")
    the_type: str = _get_typed_value(root[key], "type", str)
    return NamA2ActivationDetails(
        type=the_type,
    )


def _get_film_config(root: dict, key: str) -> NamA2FilmConfig:
    if key not in root or not isinstance(root[key], dict):
        raise TypeError(f"key '{key}' must be a dict.")
    active = _get_typed_value(root[key], "active", bool)
    shift = _get_typed_value(root[key], "shift", bool)
    return NamA2FilmConfig(
        active=active,
        shift=shift,
    )


def _get_head1x1_config(root: dict, key: str) -> NamA2WaveNetHead1x1Config:
    if key not in root or not isinstance(root[key], dict):
        raise TypeError(f"key '{key}' must be a dict.")
    active = _get_typed_value(root[key], "active", bool)
    groups = _get_typed_value(root[key], "groups", int)
    out_channels = _get_typed_value(root[key], "out_channels", int)
    return NamA2WaveNetHead1x1Config(
        active=active,
        groups=groups,
        out_channels=out_channels,
    )


def _get_layer1x1_config(root: dict, key: str) -> NamA2WaveNetLayer1x1Config:
    if key not in root or not isinstance(root[key], dict):
        raise TypeError(f"key '{key}' must be a dict.")
    active = _get_typed_value(root[key], "active", bool)
    groups = _get_typed_value(root[key], "groups", int)
    return NamA2WaveNetLayer1x1Config(
        active=active,
        groups=groups,
    )


def json_a2_wavenet_config(config: dict) -> NamA2WaveNetConfig:
    if "layers" not in config or not isinstance(config["layers"], list):
        raise TypeError("root key 'layers' must be a list")

    config_layers: list[NamA2WaveNetLayerGroupConfig] = []
    layers = config["layers"]
    for layer in layers:
        if not isinstance(layer, dict):
            raise TypeError("layer object must be a dict")

        input_size = _get_typed_value(layer, "input_size", int)
        condition_size = _get_typed_value(layer, "condition_size", int)
        head_size = _get_typed_value(layer, "head_size", int)
        channels = _get_typed_value(layer, "channels", int)
        bottleneck = _get_typed_value(layer, "bottleneck", int)
        kernel_size = _get_typed_value(layer, "kernel_size", int)

        dilations: list[int] = []
        if "dilations" not in layer or not isinstance(layer["dilations"], list):
            raise TypeError("layer key 'dilations' must be a list")
        for this_dilation in layer["dilations"]:
            if not isinstance(this_dilation, int):
                raise TypeError("key 'dilations' must be a list of ints")
            dilations.append(this_dilation)

        activation = _get_activation_config(layer, "activation")
        gating_mode = _get_typed_value(layer, "gating_mode", str)
        head_bias = _get_typed_value(layer, "head_bias", bool)
        groups_input = _get_typed_value(layer, "groups_input", int)
        groups_input_mixin = _get_typed_value(layer, "groups_input_mixin", int)
        layer1x1 = _get_layer1x1_config(layer, "layer1x1")
        head1x1 = _get_head1x1_config(layer, "head1x1")
        secondary_activation = _get_typed_value(layer, "secondary_activation", str)

        conv_pre_film = _get_film_config(layer, "conv_pre_film")
        conv_post_film = _get_film_config(layer, "conv_post_film")
        input_mixin_pre_film = _get_film_config(layer, "input_mixin_pre_film")
        input_mixin_post_film = _get_film_config(layer, "input_mixin_post_film")
        activation_pre_film = _get_film_config(layer, "activation_pre_film")
        activation_post_film = _get_film_config(layer, "activation_post_film")
        layer1x1_post_film = _get_film_config(layer, "layer1x1_post_film")
        head1x1_post_film = _get_film_config(layer, "head1x1_post_film")

        config_layers.append(
            NamA2WaveNetLayerGroupConfig(
                input_size=input_size,
                condition_size=condition_size,
                head_size=head_size,
                channels=channels,
                bottleneck=bottleneck,
                kernel_size=kernel_size,
                dilations=dilations,
                activation=activation,
                gating_mode=gating_mode,
                head_bias=head_bias,
                groups_input=groups_input,
                groups_input_mixin=groups_input_mixin,
                layer1x1=layer1x1,
                head1x1=head1x1,
                secondary_activation=secondary_activation,
                conv_pre_film=conv_pre_film,
                conv_post_film=conv_post_film,
                input_mixin_pre_film=input_mixin_pre_film,
                input_mixin_post_film=input_mixin_post_film,
                activation_pre_film=activation_pre_film,
                activation_post_film=activation_post_film,
                layer1x1_post_film=layer1x1_post_film,
                head1x1_post_film=head1x1_post_film,
            )
        )

    head = None
    head_scale = _get_typed_value(config, "head_scale", float)

    return NamA2WaveNetConfig(
        layers=config_layers,
        head=head,
        head_scale=head_scale,
    )
