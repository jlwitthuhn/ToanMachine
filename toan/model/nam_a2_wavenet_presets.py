# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

from toan.model.nam_a2_wavenet_config import (
    NamA2WaveNetConfig,
    NamA2WaveNetContainerConfig,
    NamA2WaveNetLayerGroupConfig,
    NamA2WaveNetSubmodelConfig,
)
from toan.model.presets import ModelConfigPreset

_A2_NAM_DILATIONS = [
    1,
    3,
    7,
    17,
    41,
    101,
    239,
    1,
    3,
    7,
    17,
    41,
    101,
    239,
    1,
    13,
    1,
    3,
    7,
    17,
    41,
    101,
    239,
]
_A2_NAM_KERNEL_SIZES = [6] * 14 + [15, 15] + [6] * 7


def _a2_nam_submodel(max_value: float, channels: int) -> NamA2WaveNetSubmodelConfig:
    return NamA2WaveNetSubmodelConfig(
        max_value=max_value,
        config=NamA2WaveNetConfig(
            layers=[
                NamA2WaveNetLayerGroupConfig(
                    input_size=1,
                    condition_size=1,
                    head_size=1,
                    head_bias=True,
                    head_kernel_size=16,
                    channels=channels,
                    bottleneck=channels,
                    kernel_sizes=list(_A2_NAM_KERNEL_SIZES),
                    dilations=list(_A2_NAM_DILATIONS),
                    activation="LeakyReLU",
                    negative_slope=0.01,
                ),
            ],
            head_scale=0.01,
        ),
    )


def get_a2_wavenet_config(
    size_preset: ModelConfigPreset,
) -> NamA2WaveNetContainerConfig | None:
    match size_preset:
        case ModelConfigPreset.A2_NAM:
            return NamA2WaveNetContainerConfig(
                submodels=[
                    _a2_nam_submodel(max_value=0.5, channels=3),
                    _a2_nam_submodel(max_value=1.0, channels=8),
                ]
            )
        case _:
            return None
