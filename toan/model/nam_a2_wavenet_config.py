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
    head_config: None = None
    head_scale: float = 0.02


def json_a2_wavenet_config(config: dict) -> NamA2WaveNetConfig:
    raise NotImplementedError
