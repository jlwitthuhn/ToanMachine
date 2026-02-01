# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

from toan.model.nam_wavenet_config import NameWaveNetLayerGroupConfig, NamWaveNetConfig
from toan.model.presets import ModelConfigPreset


def get_wavenet_config(size_preset: ModelConfigPreset) -> NamWaveNetConfig:
    match size_preset:
        case ModelConfigPreset.NAM_STANDARD:
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
                        gated=False,
                        head_bias=True,
                    ),
                ]
            )
        case ModelConfigPreset.NAM_LITE:
            return NamWaveNetConfig(
                layers=[
                    NameWaveNetLayerGroupConfig(
                        input_size=1,
                        condition_size=1,
                        head_size=6,
                        channels=12,
                        kernel_size=3,
                        dilations=[
                            1,
                            2,
                            4,
                            8,
                            16,
                            32,
                            64,
                        ],
                        activation="Tanh",
                        gated=False,
                        head_bias=False,
                    ),
                    NameWaveNetLayerGroupConfig(
                        input_size=12,
                        condition_size=1,
                        head_size=1,
                        channels=6,
                        kernel_size=3,
                        dilations=[
                            128,
                            256,
                            512,
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
                        head_bias=True,
                    ),
                ]
            )
        case ModelConfigPreset.NAM_FEATHER:
            return NamWaveNetConfig(
                layers=[
                    NameWaveNetLayerGroupConfig(
                        input_size=1,
                        condition_size=1,
                        head_size=4,
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
                        ],
                        activation="Tanh",
                        gated=False,
                        head_bias=False,
                    ),
                    NameWaveNetLayerGroupConfig(
                        input_size=8,
                        condition_size=1,
                        head_size=1,
                        channels=4,
                        kernel_size=3,
                        dilations=[
                            128,
                            256,
                            512,
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
                        head_bias=True,
                    ),
                ]
            )
        case ModelConfigPreset.TOAN_STANDARD_PLUS:
            return NamWaveNetConfig(
                layers=[
                    NameWaveNetLayerGroupConfig(
                        input_size=1,
                        condition_size=1,
                        head_size=12,
                        channels=24,
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
                        input_size=24,
                        condition_size=1,
                        head_size=1,
                        channels=12,
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
                        head_bias=True,
                    ),
                ]
            )
        case _:
            assert False
