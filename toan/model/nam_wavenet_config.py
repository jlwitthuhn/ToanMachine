# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

from dataclasses import dataclass


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


def default_wavenet_config() -> NamWaveNetConfig:
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
