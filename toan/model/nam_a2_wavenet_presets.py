# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

from toan.model.nam_a2_wavenet_config import (
    NamA2ActivationDetails,
    NamA2FilmConfig,
    NamA2WaveNetConfig,
    NamA2WaveNetHead1x1Config,
    NamA2WaveNetLayer1x1Config,
    NamA2WaveNetLayerGroupConfig,
)
from toan.model.presets import ModelConfigPreset


def get_a2_wavenet_config(size_preset: ModelConfigPreset) -> NamA2WaveNetConfig | None:
    if size_preset == ModelConfigPreset.TOAN_A2_TEST:
        return NamA2WaveNetConfig(
            layers=[
                NamA2WaveNetLayerGroupConfig(
                    input_size=1,
                    condition_size=1,
                    head_size=1,
                    channels=12,
                    bottleneck=None,
                    kernel_size=6,
                    dilations=[
                        1,
                        5,
                        29,
                        97,
                        227,
                        1,
                        5,
                        29,
                        97,
                        227,
                        1,
                        5,
                        29,
                        97,
                        227,
                    ],
                    activation=NamA2ActivationDetails(
                        type="LeakyReLU",
                    ),
                    gating_mode=None,
                    head_bias=True,
                    groups_input=1,
                    groups_input_mixin=1,
                    layer1x1=NamA2WaveNetLayer1x1Config(
                        active=True,
                        groups=1,
                    ),
                    head1x1=NamA2WaveNetHead1x1Config(
                        active=False,
                    ),
                    secondary_activation=None,
                    conv_pre_film=NamA2FilmConfig(active=False),
                    conv_post_film=NamA2FilmConfig(active=False),
                    input_mixin_pre_film=NamA2FilmConfig(active=False),
                    input_mixin_post_film=NamA2FilmConfig(active=False),
                    activation_pre_film=NamA2FilmConfig(active=False),
                    activation_post_film=NamA2FilmConfig(active=False),
                    layer1x1_post_film=NamA2FilmConfig(active=False),
                    head1x1_post_film=NamA2FilmConfig(active=False),
                ),
            ],
            head=None,
            head_scale=0.02,
        )
    return None
