# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

import math

import numpy as np

from toan.soundio.tone_loopback import ToneLoopback

SIGNAL_THRESHOLD_DBFS = -50.0
CLIPPING_DBFS = -0.1
MAX_LATENCY_SECONDS = 1.5
SETTLE_SECONDS = 0.1
MEASURE_SECONDS = 0.25
# Start below full scale output, some interfaces are rated for higher output than input
START_OUTPUT_DBFS = -20.0
CLIPPING_STEP_DB = 5.0
MIN_OUTPUT_DBFS = -40.0
COMPRESSION_STEP_DB = 10.0
COMPRESSION_TOLERANCE_DB = 2.0


class CalibrationError(Exception):
    pass


def calibrate_input_level(loopback: ToneLoopback, output_max_dbu: float) -> float:
    """Return input full scale in dBu RMS of a sine, given output full scale."""
    if not math.isfinite(output_max_dbu):
        raise CalibrationError("The output voltage calibration must be finite.")
    check_input_is_silent(loopback)
    latency = measure_latency(loopback)
    output_dbfs, input_dbfs = find_unclipped_level(loopback, latency)
    check_no_compression(loopback, latency, output_dbfs, input_dbfs)
    output_dbu = output_max_dbu + output_dbfs
    return compute_input_max_dbu(output_dbu, input_dbfs)


def check_input_is_silent(loopback: ToneLoopback) -> None:
    # Existing audio would be mistaken for the tone's arrival.
    begin = loopback.input_length()
    end = begin + _samples(loopback, MEASURE_SECONDS)
    samples = loopback.wait_for_input(begin, end)
    if peak_dbfs(samples) > SIGNAL_THRESHOLD_DBFS:
        raise CalibrationError(
            "There is already a signal on the input before the test tone has started. Make sure nothing else is playing and check the cable for noise."
        )


def measure_latency(loopback: ToneLoopback) -> int:
    tone_begin = loopback.play_tone(START_OUTPUT_DBFS)
    tone_end = tone_begin + _samples(loopback, MAX_LATENCY_SECONDS)
    samples = loopback.wait_for_input(tone_begin, tone_end)
    loopback.stop_tone()

    threshold = 10.0 ** (SIGNAL_THRESHOLD_DBFS / 20.0)
    tone_indices = np.flatnonzero(np.abs(samples) > threshold)
    if len(tone_indices) == 0:
        raise CalibrationError(
            f"The test tone did not reach the input within {MAX_LATENCY_SECONDS} seconds. Make sure your output is connected directly to your input."
        )
    return int(tone_indices[0])


def find_unclipped_level(loopback: ToneLoopback, latency: int) -> tuple[float, float]:
    output_dbfs = START_OUTPUT_DBFS
    while output_dbfs >= MIN_OUTPUT_DBFS:
        input_dbfs = measure_input_level(loopback, latency, output_dbfs)
        if input_dbfs < CLIPPING_DBFS:
            return output_dbfs, input_dbfs
        output_dbfs -= CLIPPING_STEP_DB
    raise CalibrationError(
        f"The input is still clipping with the output at {MIN_OUTPUT_DBFS:g} dBFS. Make sure your output is connected directly to your input."
    )


def check_no_compression(
    loopback: ToneLoopback, latency: int, output_dbfs: float, input_dbfs: float
) -> None:
    quieter_output_dbfs = output_dbfs - COMPRESSION_STEP_DB
    quieter_input_dbfs = measure_input_level(loopback, latency, quieter_output_dbfs)
    input_drop_db = input_dbfs - quieter_input_dbfs
    if abs(input_drop_db - COMPRESSION_STEP_DB) > COMPRESSION_TOLERANCE_DB:
        raise CalibrationError(
            f"Lowering the output by {COMPRESSION_STEP_DB:g} dB lowered the input by {input_drop_db:.1f} dB. Something in your signal path may be compressing or distorting the signal."
        )


def measure_input_level(
    loopback: ToneLoopback, latency: int, output_dbfs: float
) -> float:
    tone_begin = loopback.play_tone(output_dbfs)
    measure_begin = tone_begin + latency + _samples(loopback, SETTLE_SECONDS)
    measure_end = measure_begin + _samples(loopback, MEASURE_SECONDS)
    samples = loopback.wait_for_input(measure_begin, measure_end)
    loopback.stop_tone()
    level_dbfs = peak_dbfs(samples)
    if not math.isfinite(level_dbfs):
        raise CalibrationError(
            "The test tone disappeared from the input. Check the cable and try again."
        )
    return level_dbfs


def compute_input_max_dbu(output_dbu: float, input_dbfs: float) -> float:
    return output_dbu - input_dbfs


def peak_dbfs(samples: np.ndarray) -> float:
    if not np.all(np.isfinite(samples)):
        raise CalibrationError("The audio interface returned invalid input samples.")
    peak = float(np.max(np.abs(samples)))
    if peak <= 0.0:
        return -math.inf
    return 20.0 * math.log10(peak)


def _samples(loopback: ToneLoopback, seconds: float) -> int:
    return round(seconds * loopback.sample_rate)
