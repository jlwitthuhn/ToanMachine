# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

import threading
from dataclasses import dataclass

import numpy as np
import sounddevice as sd

from toan.mix import concat_signals
from toan.soundio import SdChannel, SdIoController


@dataclass
class RecordWetProgress:
    samples_to_play: int = 0
    samples_played: int = 0
    samples_recorded: int = 0


class RecordWetController:
    sample_rate: int
    dry_signal: np.ndarray
    channel_in: SdChannel
    channel_out: SdChannel

    progress: RecordWetProgress

    controller: SdIoController

    _recorded_segment_lock: threading.Lock
    _recorded_segments: list[np.ndarray]

    def __init__(
        self,
        sample_rate: int,
        dry_signal: np.ndarray,
        channel_in: SdChannel,
        channel_out: SdChannel,
    ):
        self.sample_rate = sample_rate
        self.dry_signal = dry_signal
        self.channel_in = channel_in
        self.channel_out = channel_out
        self._recorded_segment_lock = threading.Lock()
        self._recorded_segments = []

        self.progress = RecordWetProgress(
            samples_to_play=len(dry_signal),
        )

        self.controller = SdIoController.from_callbacks(
            sample_rate,
            channel_in,
            channel_out,
            self._callback_input,
            self._callback_output,
        )

    def start(self) -> None:
        self.controller.start()

    def close(self) -> None:
        self.controller.close()

    def get_recorded_signal(self) -> np.ndarray:
        with self._recorded_segment_lock:
            full_result = concat_signals(self._recorded_segments)
            return full_result[: len(self.dry_signal)]

    def is_complete(self) -> bool:
        return self.progress is not None and self.progress.samples_recorded >= len(
            self.dry_signal
        )

    def _callback_input(
        self, data: np.ndarray, frames: int, time, status: sd.CallbackFlags
    ) -> None:
        if self.is_complete():
            return
        channel_data = data[:, self.channel_in.channel_index - 1]
        assert channel_data.ndim == 1
        with self._recorded_segment_lock:
            self._recorded_segments.append(channel_data.copy())
        self.progress.samples_recorded += len(channel_data)

    def _callback_output(
        self, data: np.ndarray, frames: int, time, status: sd.CallbackFlags
    ) -> None:
        data.fill(0)
        if self.progress.samples_played >= len(self.dry_signal):
            return
        segment = self.dry_signal[
            self.progress.samples_played : self.progress.samples_played + frames
        ]
        self.progress.samples_played += len(segment)
        channel = self.channel_out.channel_index - 1
        data[0 : len(segment), channel] = segment
