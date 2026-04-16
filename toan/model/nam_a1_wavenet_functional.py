# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

import mlx.core as mx

from toan.model.nam_a1_wavenet_config import (
    NamA1WaveNetConfig,
    NamA1WaveNetLayerGroupConfig,
)


class NamA1WavenetFunctional:

    @staticmethod
    def forward_activation(type: str, weights: dict, x: mx.array) -> mx.array:
        if type == "Tanh":
            return mx.tanh(x)
        raise NotImplementedError

    @staticmethod
    def forward_layer(
        activation: str,
        channels: int,
        dilation: int,
        gated: bool,
        weights: dict,
        x: mx.array,
        h: mx.array,
        out_length: int,
    ) -> tuple[mx.array, mx.array]:
        assert "conv" in weights
        assert "weight" in weights["conv"]
        zconv = mx.conv1d(x, weights["conv"]["weight"], dilation=dilation)
        assert "bias" in weights["conv"]
        zconv += weights["conv"]["bias"]

        assert "input_mixer" in weights
        assert "weight" in weights["input_mixer"]
        z1 = (
            zconv
            + mx.conv1d(h, weights["input_mixer"]["weight"])[:, -zconv.shape[1] :, :]
        )

        assert "activation" in weights
        if gated:
            post_activation = NamA1WavenetFunctional.forward_activation(
                activation, weights["activation"], z1[:, :, :channels]
            ) * mx.sigmoid(z1[:, :, channels:])
        else:
            post_activation = NamA1WavenetFunctional.forward_activation(
                activation, weights["activation"], z1
            )

        assert "conv1x1" in weights
        assert "weight" in weights["conv1x1"]
        convolved_out = mx.conv1d(post_activation, weights["conv1x1"]["weight"])
        assert "bias" in weights["conv1x1"]
        convolved_out += weights["conv1x1"]["bias"]

        return (
            x[:, -post_activation.shape[1] :, :] + convolved_out,
            post_activation[:, -out_length:, :],
        )

    @staticmethod
    def forward_layer_group(
        config: NamA1WaveNetLayerGroupConfig,
        weights: dict,
        x: mx.array,
        c: mx.array,
        head_input: mx.array | None,
    ) -> tuple[mx.array, mx.array]:
        out_length = x.shape[1] - (config.receptive_field() - 1)
        assert "rechannel" in weights
        assert "weight" in weights["rechannel"]
        x = mx.conv1d(x, weights["rechannel"]["weight"])

        assert len(config.dilations) == len(weights["layers"])
        for i in range(len(config.dilations)):
            this_dilation = config.dilations[i]
            this_layer_weight = weights["layers"][i]
            x, head_term = NamA1WavenetFunctional.forward_layer(
                config.activation,
                config.channels,
                this_dilation,
                config.gated,
                this_layer_weight,
                x,
                c,
                out_length,
            )
            head_input = (
                head_term
                if head_input is None
                else head_input[:, -out_length:, :] + head_term
            )
        assert head_input is not None

        assert "head_rechannel" in weights
        assert "weight" in weights["head_rechannel"]
        out_rechanneled = mx.conv1d(head_input, weights["head_rechannel"]["weight"])
        if config.head_bias:
            assert "bias" in weights["head_rechannel"]
            out_rechanneled += weights["head_rechannel"]["bias"]
        return out_rechanneled, x

    @staticmethod
    def forward_model(
        config: NamA1WaveNetConfig, weights: dict, x: mx.array
    ) -> mx.array:
        if x.ndim == 2:
            x = x[:, :, None]

        assert "layer_groups" in weights
        y, head_in = x, None
        assert len(config.layers) == len(weights["layer_groups"])
        for i in range(len(config.layers)):
            this_config = config.layers[i]
            this_weights = weights["layer_groups"][i]
            head_in, y = NamA1WavenetFunctional.forward_layer_group(
                this_config, this_weights, y, x, head_in
            )
        assert head_in is not None
        return head_in[:, :, 0]
