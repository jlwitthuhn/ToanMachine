from dataclasses import dataclass

import sounddevice as sd


@dataclass
class SdChannel:
    device_index: int
    channel_index: int


@dataclass
class SdDevice:
    index: int
    name: str
    channels_in: int
    channels_out: int


def _get_devices(kind: str | None) -> list[SdDevice]:
    devices = sd.query_devices(kind=kind)
    if isinstance(devices, dict):
        devices = [devices]
    result = []
    for device in devices:
        this_device = SdDevice(
            index=device["index"],
            name=device["name"],
            channels_in=device["max_input_channels"],
            channels_out=device["max_output_channels"],
        )
        result.append(this_device)
    return result


def get_input_devices() -> list[SdDevice]:
    return _get_devices("input")


def get_output_devices() -> list[SdDevice]:
    return _get_devices("output")
