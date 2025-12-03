# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identified: GPL-3.0-only

from dataclasses import dataclass
from typing import Callable

import numpy
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


class SdPlayrecController:
    stream_in: sd.InputStream | None
    stream_out: sd.OutputStream | None
    stream_io: sd.Stream | None

    def __init__(self, stream_in: sd.InputStream, stream_out: sd.OutputStream):
        self.stream_in = stream_in
        self.stream_out = stream_out
        self.stream_io = None

    def close(self):
        if self.stream_in is not None:
            self.stream_in.close()
        if self.stream_out is not None:
            self.stream_out.close()
        if self.stream_io is not None:
            self.stream_io.close()

    def start(self):
        if self.stream_in is not None:
            self.stream_in.start()
        if self.stream_out is not None:
            self.stream_out.start()
        if self.stream_io is not None:
            self.stream_io.start()


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


def _activate_play_record_2device(
    sample_rate: int,
    channel_in: SdChannel,
    channel_out: SdChannel,
    callback_in: Callable[[numpy.ndarray, int, any, sd.CallbackFlags], None],
    callback_out: Callable[[numpy.ndarray, int, any, sd.CallbackFlags], None],
) -> SdPlayrecController:
    input_stream = sd.InputStream(
        samplerate=sample_rate,
        blocksize=1024,
        device=channel_in.device_index,
        callback=callback_in,
    )
    output_stream = sd.OutputStream(
        samplerate=sample_rate,
        blocksize=1024,
        device=channel_out.device_index,
        callback=callback_out,
    )
    return SdPlayrecController(input_stream, output_stream)


def prepare_play_record(
    sample_rate: int,
    channel_in: SdChannel,
    channel_out: SdChannel,
    callback_in: Callable[[numpy.ndarray, int, any, sd.CallbackFlags], None],
    callback_out: Callable[[numpy.ndarray, int, any, sd.CallbackFlags], None],
) -> SdPlayrecController:
    if channel_in.device_index == channel_out.device_index:
        raise Exception("Input and output devices must be different")

    input_stream = sd.InputStream(
        samplerate=sample_rate,
        blocksize=1024,
        device=channel_in.device_index,
        callback=callback_in,
    )
    output_stream = sd.OutputStream(
        samplerate=sample_rate,
        blocksize=1024,
        device=channel_out.device_index,
        callback=callback_out,
    )
    return SdPlayrecController(input_stream, output_stream)


def get_input_devices() -> list[SdDevice]:
    return _get_devices("input")


def get_output_devices() -> list[SdDevice]:
    return _get_devices("output")
