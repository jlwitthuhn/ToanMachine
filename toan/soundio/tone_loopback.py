# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

import threading
import time

import numpy as np
import sounddevice as sd

from toan.soundio import SdChannel, SdIoController

FADE_SECONDS = 0.01
POLL_SECONDS = 0.01
STALL_SECONDS = 2.0
# Let the buffered fade finish before closing the streams.
CLOSE_DELAY_SECONDS = 0.25


class LoopbackCancelled(Exception):
    pass


class ToneLoopback:
    """Worker-thread context manager; input index = output index + latency."""

    sample_rate: int
    tone_frequency: float
    channel_in: SdChannel
    channel_out: SdChannel

    _controller: SdIoController | None = None
    _lock: threading.Lock
    _cancelled: threading.Event

    # Callback state is guarded by _lock.
    _amplitude: float = 0.0
    _tone_position: int = 0
    _fade_queue: np.ndarray
    _output_length: int = 0

    _input_chunks: list[np.ndarray]
    _input_length: int = 0

    def __init__(
        self,
        sample_rate: int,
        tone_frequency: float,
        channel_in: SdChannel,
        channel_out: SdChannel,
    ):
        self.sample_rate = sample_rate
        self.tone_frequency = tone_frequency
        self.channel_in = channel_in
        self.channel_out = channel_out
        self._lock = threading.Lock()
        self._cancelled = threading.Event()
        self._fade_queue = np.zeros(0)
        self._input_chunks = []

    def __enter__(self) -> "ToneLoopback":
        if self._cancelled.is_set():
            raise LoopbackCancelled()
        self._controller = SdIoController.from_callbacks(
            self.sample_rate,
            self.channel_in,
            self.channel_out,
            self._input_callback,
            self._output_callback,
        )
        try:
            self._controller.start()
        except Exception:
            self._controller.close()
            raise
        return self

    def __exit__(self, *exc_info) -> None:
        try:
            self.stop_tone()
            time.sleep(CLOSE_DELAY_SECONDS)
        finally:
            self._controller.close()

    def cancel(self) -> None:
        self._cancelled.set()

    def play_tone(self, level_dbfs: float) -> int:
        """Fade out, then start a tone; return its first output sample index."""
        if self._cancelled.is_set():
            raise LoopbackCancelled()
        return self._change_amplitude(10.0 ** (level_dbfs / 20.0))

    def stop_tone(self) -> int:
        """Fade out; return the output sample index where silence begins."""
        return self._change_amplitude(0.0)

    def input_length(self) -> int:
        with self._lock:
            return self._input_length

    def wait_for_input(self, begin: int, end: int) -> np.ndarray:
        """Wait for and return input samples [begin, end)."""
        expected_seconds = (end - self.input_length()) / self.sample_rate
        deadline = time.monotonic() + expected_seconds + STALL_SECONDS
        while True:
            if self._cancelled.is_set():
                raise LoopbackCancelled()
            if self.input_length() >= end:
                break
            if time.monotonic() > deadline:
                raise TimeoutError("The audio interface stopped delivering input.")
            time.sleep(POLL_SECONDS)
        with self._lock:
            return np.concatenate(self._input_chunks)[begin:end]

    def _change_amplitude(self, amplitude: float) -> int:
        with self._lock:
            fade_length = round(self.sample_rate * FADE_SECONDS)
            fade = self._render_tone(fade_length) * np.linspace(1.0, 0.0, fade_length)
            self._fade_queue = np.concatenate((self._fade_queue, fade))
            self._amplitude = amplitude
            self._tone_position = 0
            return self._output_length + len(self._fade_queue)

    def _render_tone(self, length: int) -> np.ndarray:
        positions = np.arange(self._tone_position, self._tone_position + length)
        self._tone_position += length
        phase = 2.0 * np.pi * self.tone_frequency * positions / self.sample_rate
        return self._amplitude * np.sin(phase)

    def _input_callback(
        self, indata: np.ndarray, frames: int, time_info, status: sd.CallbackFlags
    ) -> None:
        channel = self.channel_in.channel_index - 1
        with self._lock:
            self._input_chunks.append(indata[:, channel].copy())
            self._input_length += frames

    def _output_callback(
        self, outdata: np.ndarray, frames: int, time_info, status: sd.CallbackFlags
    ) -> None:
        with self._lock:
            from_fade = self._fade_queue[:frames]
            self._fade_queue = self._fade_queue[frames:]
            from_tone = self._render_tone(frames - len(from_fade))
            self._output_length += frames
        outdata.fill(0)
        channel = self.channel_out.channel_index - 1
        outdata[:, channel] = np.concatenate((from_fade, from_tone))
