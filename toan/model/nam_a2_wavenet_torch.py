# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from toan.model.metadata import ModelA2Metadata, SubmodelA2Metadata
from toan.model.nam_a2_wavenet_config import (
    NamA2WaveNetConfig,
    NamA2WaveNetContainerConfig,
    NamA2WaveNetLayerGroupConfig,
)
from toan.wav import load_and_resample_wav

# Based on code from Neural Amp Modeler
# https://github.com/sdatkinson/neural-amp-modeler/tree/2de335b3ee1138529286117978a54fc16aeb313c/nam/models/wavenet


def _load_loudness_probe_signal(sample_rate: int) -> np.ndarray:
    script_dir = Path(__file__).resolve().parent
    root_dir = script_dir.parent.parent
    probe_path = root_dir.joinpath("data").joinpath("nam_loudness.flac").resolve()
    return load_and_resample_wav(sample_rate, str(probe_path))


def _reset_conv_from_generator(conv: nn.Conv1d, generator: torch.Generator) -> None:
    nn.init.kaiming_uniform_(conv.weight, a=math.sqrt(5), generator=generator)
    if conv.bias is not None:
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(conv.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(conv.bias, -bound, bound, generator=generator)


def _build_activation_module_a2(activation: str, negative_slope: float) -> nn.Module:
    if activation == "LeakyReLU":
        return nn.LeakyReLU(negative_slope)
    raise NotImplementedError


class _NamA2Conv1dLayerTorch(nn.Conv1d):
    def export_nam_linear_weights(self) -> list[float]:
        result = []
        if self.weight is not None:
            result.extend(self.weight.detach().flatten().tolist())
        if self.bias is not None:
            result.extend(self.bias.detach().flatten().tolist())
        return result

    def import_nam_linear_weights(self, weights: torch.Tensor, i: int) -> int:
        if self.weight is not None:
            size = self.weight.numel()
            my_slice = weights[i : i + size]
            with torch.no_grad():
                self.weight.copy_(my_slice.reshape(self.weight.shape))
            i = i + size
        if self.bias is not None:
            size = self.bias.numel()
            my_slice = weights[i : i + size]
            with torch.no_grad():
                self.bias.copy_(my_slice.reshape(self.bias.shape))
            i = i + size
        return i


class _NamA2WaveNetLayerTorch(nn.Module):
    def __init__(
        self,
        condition_size: int,
        channels: int,
        bottleneck: int,
        kernel_size: int,
        dilation: int,
        activation: str,
        negative_slope: float,
    ):
        super().__init__()
        self.channels = channels
        self.bottleneck = bottleneck
        self.conv = _NamA2Conv1dLayerTorch(
            channels, bottleneck, kernel_size, dilation=dilation
        )
        self.input_mixer = _NamA2Conv1dLayerTorch(
            condition_size, bottleneck, 1, bias=False
        )
        self.activation = _build_activation_module_a2(activation, negative_slope)
        self.activation_name = activation
        self.layer1x1 = _NamA2Conv1dLayerTorch(bottleneck, channels, 1)

    def forward(
        self, x: torch.Tensor, h: torch.Tensor, out_length: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        zconv = self.conv(x)
        z1 = zconv + self.input_mixer(h)[:, :, -zconv.shape[2] :]
        post_activation = self.activation(z1)
        return (
            x[:, :, -post_activation.shape[2] :] + self.layer1x1(post_activation),
            post_activation[:, :, -out_length:],
        )

    def export_nam_linear_weights(self) -> list[float]:
        result = []
        result.extend(self.conv.export_nam_linear_weights())
        result.extend(self.input_mixer.export_nam_linear_weights())
        result.extend(self.layer1x1.export_nam_linear_weights())
        return result

    def import_nam_linear_weights(self, weights: torch.Tensor, i: int) -> int:
        i = self.conv.import_nam_linear_weights(weights, i)
        i = self.input_mixer.import_nam_linear_weights(weights, i)
        i = self.layer1x1.import_nam_linear_weights(weights, i)
        return i


class _NamA2WaveNetLayerGroupTorch(nn.Module):
    def __init__(self, config: NamA2WaveNetLayerGroupConfig):
        super().__init__()
        self.config = config
        self.rechannel = _NamA2Conv1dLayerTorch(
            config.input_size, config.channels, 1, bias=False
        )
        self.layers = nn.ModuleList(
            [
                _NamA2WaveNetLayerTorch(
                    config.condition_size,
                    config.channels,
                    config.bottleneck,
                    kernel_size,
                    dilation,
                    config.activation,
                    config.negative_slope,
                )
                for kernel_size, dilation in zip(config.kernel_sizes, config.dilations)
            ]
        )
        self.head_rechannel = _NamA2Conv1dLayerTorch(
            config.bottleneck,
            config.head_size,
            config.head_kernel_size,
            bias=config.head_bias,
        )

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        head_input: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out_length = x.shape[2] - (self.receptive_field - 1)
        out_length_no_head = x.shape[2] - (
            self.config.receptive_field_no_head_rechannel() - 1
        )
        x = self.rechannel(x)
        for layer in self.layers:
            x, head_term = layer(x, c, out_length_no_head)
            head_input = (
                head_term
                if head_input is None
                else head_input[:, :, -out_length_no_head:] + head_term
            )
        return self.head_rechannel(head_input), x[:, :, -out_length:]

    def export_nam_linear_weights(self) -> list[float]:
        result = []
        result.extend(self.rechannel.export_nam_linear_weights())
        for layer in self.layers:
            result.extend(layer.export_nam_linear_weights())
        result.extend(self.head_rechannel.export_nam_linear_weights())
        return result

    def import_nam_linear_weights(self, weights: torch.Tensor, i: int) -> int:
        i = self.rechannel.import_nam_linear_weights(weights, i)
        for layer in self.layers:
            i = layer.import_nam_linear_weights(weights, i)
        i = self.head_rechannel.import_nam_linear_weights(weights, i)
        return i

    @property
    def receptive_field(self) -> int:
        return self.config.receptive_field()


class _NamA2WaveNetSubmodelTorch(nn.Module):
    head: None = None

    def __init__(self, config: NamA2WaveNetConfig):
        super().__init__()
        self.config = config
        self.layer_groups = nn.ModuleList(
            [
                _NamA2WaveNetLayerGroupTorch(layer_config)
                for layer_config in config.layers
            ]
        )
        assert config.head_config is None

    @property
    def receptive_field(self) -> int:
        return 1 + sum([(group.receptive_field - 1) for group in self.layer_groups])

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        y, head_input = x, None
        for group in self.layer_groups:
            head_input, y = group(y, x, head_input)
        head_input = self.config.head_scale * head_input
        result = head_input if self.head is None else self.head(head_input)
        return result

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x[:, None, :]
        y = self._forward(x)
        assert y.shape[1] == 1
        return y[:, 0, :]

    def metadata_loudness(
        self, probe: torch.Tensor, gain: float = 1.0, db: bool = True
    ) -> float:
        with torch.no_grad():
            y = self(gain * probe)
        loudness = torch.sqrt(torch.mean(torch.square(y)))
        if db:
            loudness = 20.0 * torch.log10(loudness)
        return loudness.item()

    def metadata_gain(self, probe: torch.Tensor) -> float:
        x = np.linspace(0.0, 1.0, 11)
        y = np.array([self.metadata_loudness(probe, gain=g, db=False) for g in x])
        max_gain = y[-1] * len(x)  # "Square" (no compression)
        min_gain = 0.5 * max_gain  # "Triangle" (full compression)
        gain_range = max_gain - min_gain
        this_gain = y.sum()
        normalized_gain = (this_gain - min_gain) / gain_range
        return float(np.clip(normalized_gain, 0.0, 1.0))

    def export_nam_linear_weights(self) -> list[float]:
        result = []
        for group in self.layer_groups:
            group_weights: list[float] = group.export_nam_linear_weights()
            result.extend(group_weights)
        assert self.head is None
        result.append(self.config.head_scale)  # head_scale is the trailing scalar
        return result

    def import_nam_linear_weights(self, weights: list[float]) -> int:
        i = 0
        weights_t = torch.tensor(weights)
        for group in self.layer_groups:
            i = group.import_nam_linear_weights(weights_t, i)
        if i < weights_t.numel():
            self.config.head_scale = float(weights_t[i].item())
            i = i + 1
        return i


class NamA2WaveNetTorch(nn.Module):
    def __init__(
        self,
        config: NamA2WaveNetContainerConfig,
        metadata: ModelA2Metadata,
        sample_rate: float,
        rng_seed: int = 0x35,
    ):
        super().__init__()

        self.config = config
        self.metadata = metadata
        self.submodel_metadata = [
            SubmodelA2Metadata.from_a2(metadata) for _ in config.submodels
        ]
        self.sample_rate = int(sample_rate)
        self.max_values = [submodel.max_value for submodel in config.submodels]
        self.submodels = nn.ModuleList(
            [
                _NamA2WaveNetSubmodelTorch(submodel.config)
                for submodel in config.submodels
            ]
        )

        generator = torch.Generator(device="cpu").manual_seed(rng_seed)
        for module in self.modules():
            if isinstance(module, _NamA2Conv1dLayerTorch):
                _reset_conv_from_generator(module, generator)

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def receptive_field(self) -> int:
        fields = [submodel.receptive_field for submodel in self.submodels]
        assert all(field == fields[0] for field in fields)
        return fields[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Outputs are stacked like (num_submodels, batch, length)
        outputs = [submodel(x) for submodel in self.submodels]
        return torch.stack(outputs, dim=0)

    def best_submodel_index(self) -> int:
        best = 0
        for index in range(len(self.max_values)):
            if self.max_values[index] > self.max_values[best]:
                best = index
        return best

    def forward_best(self, x: torch.Tensor) -> torch.Tensor:
        submodel = self.submodels[self.best_submodel_index()]
        return submodel(x)

    def populate_loudness_and_gain_metadata(self) -> None:
        device = next(self.parameters()).device
        probe_np = _load_loudness_probe_signal(self.sample_rate)
        probe = torch.from_numpy(probe_np).float().reshape((1, -1)).to(device)

        pad = torch.zeros((1, self.receptive_field - 1), device=device)
        probe = torch.cat((pad, probe), dim=1)

        was_training = self.training
        self.eval()
        try:
            for submodel, submodel_metadata in zip(
                self.submodels, self.submodel_metadata
            ):
                submodel_metadata.loudness = submodel.metadata_loudness(probe)
                submodel_metadata.gain = submodel.metadata_gain(probe)
        finally:
            self.train(was_training)

        best_metadata = self.submodel_metadata[self.best_submodel_index()]
        self.metadata.loudness = best_metadata.loudness
        self.metadata.gain = best_metadata.gain

    def _export_submodel_dict(
        self,
        submodel: _NamA2WaveNetSubmodelTorch,
        metadata: SubmodelA2Metadata,
    ) -> dict:
        return {
            "version": "0.7.0",
            "metadata": metadata.export_dict(),
            "architecture": "WaveNet",
            "config": submodel.config.export_dict(),
            "weights": submodel.export_nam_linear_weights(),
            "sample_rate": self.sample_rate,
        }

    def export_nam_json_str(self) -> str:
        metadata_dict = self.metadata.export_dict()
        submodel_entries = []
        for max_value, submodel, submodel_metadata in zip(
            self.max_values, self.submodels, self.submodel_metadata
        ):
            submodel_entries.append(
                {
                    "max_value": max_value,
                    "model": self._export_submodel_dict(submodel, submodel_metadata),
                }
            )
        root = {
            "version": "0.7.0",
            "architecture": "SlimmableContainer",
            "metadata": metadata_dict,
            "config": {"submodels": submodel_entries},
            "weights": [],
            # This sample rate has to be a float, the others are ints
            "sample_rate": float(self.sample_rate),
        }
        return json.dumps(root)

    def import_nam_linear_weights(self, submodel_weights: list[list[float]]) -> None:
        assert len(submodel_weights) == len(self.submodels)
        for submodel, weights in zip(self.submodels, submodel_weights):
            submodel.import_nam_linear_weights(weights)
