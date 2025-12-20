# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

from dataclasses import dataclass
from typing import Callable

import numpy
import numpy as np
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

    def __init__(
        self,
        *,
        stream_in: sd.InputStream | None = None,
        stream_out: sd.OutputStream | None = None,
        stream_io: sd.Stream | None = None
    ):
        self.stream_in = stream_in
        self.stream_out = stream_out
        self.stream_io = stream_io

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


def _get_devices(
    include_input: bool = True, include_output: bool = True
) -> list[SdDevice]:
    devices = sd.query_devices()
    if isinstance(devices, dict):
        devices = [devices]
    result = []
    for device in devices:
        is_input = device["max_input_channels"] > 0
        is_output = device["max_output_channels"] > 0
        if (include_input and is_input) or (include_output and is_output):
            this_device = SdDevice(
                index=device["index"],
                name=device["name"],
                channels_in=device["max_input_channels"],
                channels_out=device["max_output_channels"],
            )
            result.append(this_device)
    return result


def prepare_play_record(
    sample_rate: int,
    channel_in: SdChannel,
    channel_out: SdChannel,
    callback_in: Callable[[numpy.ndarray, int, any, sd.CallbackFlags], None],
    callback_out: Callable[[numpy.ndarray, int, any, sd.CallbackFlags], None],
) -> SdPlayrecController:

    if channel_in.device_index == channel_out.device_index:

        def callback_io(
            indata: np.ndarray,
            outdata: np.ndarray,
            frames: int,
            time,
            status: sd.CallbackFlags,
        ) -> None:
            callback_in(indata, frames, time, status)
            callback_out(outdata, frames, time, status)

        io_stream = sd.Stream(
            samplerate=sample_rate,
            device=channel_in.device_index,
            callback=callback_io,
        )
        return SdPlayrecController(stream_io=io_stream)
    else:
        input_stream = sd.InputStream(
            samplerate=sample_rate,
            device=channel_in.device_index,
            callback=callback_in,
        )
        output_stream = sd.OutputStream(
            samplerate=sample_rate,
            device=channel_out.device_index,
            callback=callback_out,
        )
        return SdPlayrecController(stream_in=input_stream, stream_out=output_stream)


def get_input_devices() -> list[SdDevice]:
    return _get_devices(include_input=True, include_output=False)


def get_output_devices() -> list[SdDevice]:
    return _get_devices(include_input=False, include_output=True)
