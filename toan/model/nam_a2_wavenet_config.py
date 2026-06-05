# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

from dataclasses import dataclass, field


def _film_inactive_config() -> dict:
    return {"active": False, "shift": True, "groups": 1}


@dataclass
class NamA2WaveNetLayerGroupConfig:
    input_size: int
    condition_size: int
    head_size: int
    head_bias: bool
    head_kernel_size: int
    channels: int
    bottleneck: int
    kernel_sizes: list[int]
    dilations: list[int]
    activation: str
    negative_slope: float

    def export_dict(self) -> dict:
        layer_count = len(self.dilations)
        return {
            "input_size": self.input_size,
            "condition_size": self.condition_size,
            "head": {
                "out_channels": self.head_size,
                "kernel_size": self.head_kernel_size,
                "bias": self.head_bias,
            },
            "channels": self.channels,
            "kernel_sizes": list(self.kernel_sizes),
            "dilations": list(self.dilations),
            "activation": [
                {"type": self.activation, "negative_slope": self.negative_slope}
                for _ in range(layer_count)
            ],
            "bottleneck": self.bottleneck,
            "head1x1": {"active": False, "out_channels": 1, "groups": 1},
            "layer1x1": {"active": True, "groups": 1},
            "groups_input": 1,
            "groups_input_mixin": 1,
            "conv_pre_film": _film_inactive_config(),
            "conv_post_film": _film_inactive_config(),
            "input_mixin_pre_film": _film_inactive_config(),
            "input_mixin_post_film": _film_inactive_config(),
            "activation_pre_film": _film_inactive_config(),
            "activation_post_film": _film_inactive_config(),
            "layer1x1_post_film": _film_inactive_config(),
            "head1x1_post_film": _film_inactive_config(),
            "gating_mode": ["none" for _ in range(layer_count)],
            "secondary_activation": [None for _ in range(layer_count)],
            "slimmable": None,
        }

    def receptive_field_no_head_rechannel(self) -> int:
        total = 1
        for kernel_size, dilation in zip(self.kernel_sizes, self.dilations):
            total += (kernel_size - 1) * dilation
        return total

    def receptive_field(self) -> int:
        return self.receptive_field_no_head_rechannel() + (self.head_kernel_size - 1)


@dataclass
class NamA2WaveNetConfig:
    layers: list[NamA2WaveNetLayerGroupConfig] = field(default_factory=list)
    head_config: None = None
    head_scale: float = 0.02

    def export_dict(self) -> dict:
        layer_list: list[dict] = []
        for layer in self.layers:
            layer_list.append(layer.export_dict())
        return {
            "layers": layer_list,
            "head": None,
            "head_scale": self.head_scale,
        }


@dataclass
class NamA2WaveNetSubmodelConfig:
    max_value: float
    config: NamA2WaveNetConfig


@dataclass
class NamA2WaveNetContainerConfig:
    submodels: list[NamA2WaveNetSubmodelConfig] = field(default_factory=list)


def _json_a2_kernel_sizes(layer: dict, dilations: list[int]) -> list[int]:
    if "kernel_sizes" in layer:
        kernel_sizes = layer["kernel_sizes"]
        if not isinstance(kernel_sizes, list) or len(kernel_sizes) == 0:
            raise TypeError("layer key 'kernel_sizes' must be a non-empty list")
    elif "kernel_size" in layer and isinstance(layer["kernel_size"], int):
        kernel_sizes = [layer["kernel_size"] for _ in dilations]
    else:
        raise TypeError("layer must have 'kernel_sizes' or 'kernel_size'")
    for kernel_size in kernel_sizes:
        if not isinstance(kernel_size, int):
            raise TypeError("layer kernel sizes must be ints")
    if len(kernel_sizes) != len(dilations):
        raise TypeError("layer 'kernel_sizes' must be the same length as 'dilations'")
    return kernel_sizes


def _json_a2_uniform_activation(layer: dict) -> tuple[str, float]:
    if "activation" not in layer or not isinstance(layer["activation"], list):
        raise TypeError("layer key 'activation' must be a list")
    activations = layer["activation"]
    if len(activations) == 0:
        raise TypeError("layer key 'activation' must be a non-empty list")
    first = activations[0]
    if not isinstance(first, dict) or first.get("type") != "LeakyReLU":
        raise NotImplementedError("only the LeakyReLU activation is supported")
    negative_slope = first["negative_slope"]
    if not isinstance(negative_slope, float):
        raise TypeError("activation key 'negative_slope' must be a float")
    for activation in activations:
        if activation != first:
            raise NotImplementedError("only a uniform activation is supported")
    return "LeakyReLU", negative_slope


def _json_a2_assert_unpacked(layer: dict) -> None:
    if layer.get("head1x1", {}).get("active", False):
        raise NotImplementedError("head1x1 is not supported")
    if not layer.get("layer1x1", {"active": True}).get("active", True):
        raise NotImplementedError("layer1x1 must be active")
    for key in (
        "conv_pre_film",
        "conv_post_film",
        "input_mixin_pre_film",
        "input_mixin_post_film",
        "activation_pre_film",
        "activation_post_film",
        "layer1x1_post_film",
        "head1x1_post_film",
    ):
        if layer.get(key, {}).get("active", False):
            raise NotImplementedError("FiLM is not supported")
    for gating in layer.get("gating_mode", []):
        if gating not in (None, "none"):
            raise NotImplementedError("gating is not supported")
    for secondary in layer.get("secondary_activation", []):
        if secondary is not None:
            raise NotImplementedError("secondary activations are not supported")
    if layer.get("slimmable") is not None:
        raise NotImplementedError("slimmable layers are not supported")


def _json_a2_layer_group(layer: dict) -> NamA2WaveNetLayerGroupConfig:
    if not isinstance(layer, dict):
        raise TypeError("layer object must be a dict")

    if "input_size" not in layer or not isinstance(layer["input_size"], int):
        raise TypeError("layer key 'input_size' must be a int")
    if "condition_size" not in layer or not isinstance(layer["condition_size"], int):
        raise TypeError("layer key 'condition_size' must be a int")
    if "head" not in layer or not isinstance(layer["head"], dict):
        raise TypeError("layer key 'head' must be a dict")
    head = layer["head"]
    if "out_channels" not in head or not isinstance(head["out_channels"], int):
        raise TypeError("head key 'out_channels' must be a int")
    head_kernel_size = head.get("kernel_size", 1)
    if not isinstance(head_kernel_size, int):
        raise TypeError("head key 'kernel_size' must be a int")
    if "bias" not in head or not isinstance(head["bias"], bool):
        raise TypeError("head key 'bias' must be a bool")
    if "channels" not in layer or not isinstance(layer["channels"], int):
        raise TypeError("layer key 'channels' must be a int")
    if "bottleneck" in layer and not isinstance(layer["bottleneck"], int):
        raise TypeError("layer key 'bottleneck' must be a int")
    if "dilations" not in layer or not isinstance(layer["dilations"], list):
        raise TypeError("layer key 'dilations' must be a list")

    _json_a2_assert_unpacked(layer)
    kernel_sizes = _json_a2_kernel_sizes(layer, layer["dilations"])
    activation, negative_slope = _json_a2_uniform_activation(layer)
    bottleneck = layer["bottleneck"] if "bottleneck" in layer else layer["channels"]

    return NamA2WaveNetLayerGroupConfig(
        input_size=layer["input_size"],
        condition_size=layer["condition_size"],
        head_size=head["out_channels"],
        head_bias=head["bias"],
        head_kernel_size=head_kernel_size,
        channels=layer["channels"],
        bottleneck=bottleneck,
        kernel_sizes=kernel_sizes,
        dilations=layer["dilations"],
        activation=activation,
        negative_slope=negative_slope,
    )


def _json_a2_wavenet_config(config: dict) -> NamA2WaveNetConfig:
    if "layers" not in config or not isinstance(config["layers"], list):
        raise TypeError("submodel key 'layers' must be a list")
    if config.get("head") is not None:
        raise NotImplementedError("a top-level head is not supported")
    if "head_scale" not in config or not isinstance(config["head_scale"], float):
        raise TypeError("submodel key 'head_scale' must be a float")

    result = NamA2WaveNetConfig()
    for layer in config["layers"]:
        result.layers.append(_json_a2_layer_group(layer))
    result.head_scale = config["head_scale"]
    return result


def json_a2_wavenet_container_config(config: dict) -> NamA2WaveNetContainerConfig:
    if "submodels" not in config or not isinstance(config["submodels"], list):
        raise TypeError("root key 'submodels' must be a list")

    result = NamA2WaveNetContainerConfig()
    for submodel in config["submodels"]:
        if not isinstance(submodel, dict):
            raise TypeError("submodel object must be a dict")
        if "max_value" not in submodel or not isinstance(submodel["max_value"], float):
            raise TypeError("submodel key 'max_value' must be a float")
        if "model" not in submodel or not isinstance(submodel["model"], dict):
            raise TypeError("submodel key 'model' must be a dict")
        model = submodel["model"]
        if "config" not in model or not isinstance(model["config"], dict):
            raise TypeError("submodel model key 'config' must be a dict")
        result.submodels.append(
            NamA2WaveNetSubmodelConfig(
                max_value=submodel["max_value"],
                config=_json_a2_wavenet_config(model["config"]),
            )
        )
    return result
