# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

from dataclasses import dataclass, field


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
    layers: list[NameWaveNetLayerGroupConfig] = field(default_factory=list)
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


def json_wavenet_config(config: dict) -> NamWaveNetConfig:
    if "layers" not in config or not isinstance(config["layers"], list):
        raise TypeError("root key 'layers' must be a list")

    layers = config["layers"]
    for layer in layers:
        if not isinstance(layer, dict):
            raise TypeError("layer object must be a list")

        if "input_size" not in layer or not isinstance(layer["input_size"], int):
            raise TypeError("layer key 'input_size' must be a int")
        if "condition_size" not in layer or not isinstance(
            layer["condition_size"], int
        ):
            raise TypeError("layer key 'condition_size' must be a int")
        if "head_size" not in layer or not isinstance(layer["head_size"], int):
            raise TypeError("layer key 'head_size' must be a int")
        if "channels" not in layer or not isinstance(layer["channels"], int):
            raise TypeError("layer key 'channels' must be a int")
        if "kernel_size" not in layer or not isinstance(layer["kernel_size"], int):
            raise TypeError("layer key 'kernel_size' must be a int")

        if "dilations" not in layer or not isinstance(layer["dilations"], list):
            raise TypeError("layer key 'dilations' must be a list")

        if "activation" not in layer or not isinstance(layer["activation"], str):
            raise TypeError("layer key 'activation' must be a string")
        if "gated" not in layer or not isinstance(layer["gated"], bool):
            raise TypeError("layer key 'gated' must be a bool")
        if "head_bias" not in layer or not isinstance(layer["head_bias"], bool):
            raise TypeError("layer key 'head_bias' must be a bool")

    result = NamWaveNetConfig()
    for layer in layers:
        out_layer = NameWaveNetLayerGroupConfig(
            input_size=layer["input_size"],
            condition_size=layer["condition_size"],
            head_size=layer["head_size"],
            channels=layer["channels"],
            kernel_size=layer["kernel_size"],
            dilations=layer["dilations"],
            activation=layer["activation"],
            gated=layer["gated"],
            head_bias=layer["head_bias"],
        )
        result.layers.append(out_layer)

    if "head_scale" not in config or not isinstance(config["head_scale"], float):
        raise TypeError("root key 'head_scale' must be a float")
    result.head_scale = config["head_scale"]

    return result
