# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

import io
import json
import threading
import zipfile
from dataclasses import dataclass

import numpy as np
from scipy.io import wavfile

from toan.model.metadata import ModelMetadata


class ZipLoaderContext:
    messages_lock = threading.Lock()
    messages_queue: list[str] = []

    signal_dry: np.ndarray | None = None
    signal_wet: np.ndarray | None = None
    signal_dry_test: np.ndarray | None = None
    signal_wet_test: np.ndarray | None = None
    metadata: ModelMetadata
    sample_rate: int = 0

    complete: bool = False
    errored: bool = False


@dataclass
class _PotentialClick:
    index_begin: int = 0
    index_peak: int = 0
    magnitude: float = 0.0
    width: int = 0


def _find_clicks(signal: np.ndarray, raw_noise_floor: float) -> list[_PotentialClick]:
    noise_threshold = raw_noise_floor * 10.0
    result = []
    silence_samples_required = 500
    silence_samples_remaining = silence_samples_required
    current_click = _PotentialClick()
    for i in range(len(signal)):
        current_click.width += 1
        this_sample = signal[i]
        this_sample_abs = np.abs(this_sample)
        if this_sample_abs > noise_threshold:
            if silence_samples_remaining <= 0:
                current_click = _PotentialClick(index_begin=i)
                result.append(current_click)
            silence_samples_remaining = silence_samples_required
            if this_sample_abs > current_click.magnitude:
                current_click.magnitude = this_sample_abs
                current_click.index_peak = i
        else:
            silence_samples_remaining -= 1
    return result


def _sort_clicks(click: _PotentialClick) -> float:
    return click.width * click.magnitude


def run_zip_loader(context: ZipLoaderContext, input_file: str | io.BytesIO):
    def print_status(message: str):
        with context.messages_lock:
            context.messages_queue.append(message)

    # So errored is true for any early exits, only gets set False at the end
    context.errored = True
    print_status("Loading as zip archive...")
    try:
        with zipfile.ZipFile(input_file, "r") as zip_file:
            print_status("Loading config...")

            try:
                config_json_bytes = zip_file.read("config.json")
                config_json = json.loads(config_json_bytes.decode("utf-8"))
            except KeyError:
                print_status("Error: config.json not found")
                return

            if "version" not in config_json:
                print_status("Error: config.json does not contain key 'version'")
                return
            if (
                not isinstance(config_json["version"], int)
                or config_json["version"] != 0
            ):
                print_status("Error: config.json has unknown version")
                return

            if "device_make" not in config_json or not isinstance(
                config_json["device_make"], str
            ):
                print_status("Error: config.json does not contain key 'device_make'")
                return

            print_status(f"Device make: {config_json["device_make"]}")

            if "device_model" not in config_json or not isinstance(
                config_json["device_model"], str
            ):
                print_status("Error: config.json does not contain key 'device_model'")
                return

            print_status(f"Device model: {config_json["device_model"]}")

            if "sample_rate" not in config_json or not isinstance(
                config_json["sample_rate"], int
            ):
                print_status("Error: config.json does not contain key 'sample_rate'")
                return

            print_status(f"Sample rate: {config_json["sample_rate"]}")

            if "test_offset" not in config_json or not isinstance(
                config_json["sample_rate"], int
            ):
                print_status("Error: config.json does not contain key 'test_offset'")
                return

            test_data_offset = config_json["test_offset"]
            if test_data_offset > 0:
                print_status("Recording contains test data")

            if "dry_signal" not in config_json or not isinstance(
                config_json["dry_signal"], str
            ):
                print_status("Error: config.json does not contain key 'dry_signal'")
                return
            if "wet_signal" not in config_json or not isinstance(
                config_json["wet_signal"], str
            ):
                print_status("Error: config.json does not contain key 'wet_signal'")
                return

            try:
                print_status(f"Loading dry signal: {config_json["dry_signal"]}")
                dry_signal_bytes = zip_file.read(config_json["dry_signal"])
                with io.BytesIO(dry_signal_bytes) as dry_bytes_io:
                    dry_sample_rate, dry_signal = wavfile.read(dry_bytes_io)
                    if dry_sample_rate != config_json["sample_rate"]:
                        print_status("Error: dry signal has unexpected sample rate")
                        return
            except:
                print_status(
                    f"Error: Failed to load dry signal from {config_json["dry_signal"]}"
                )
                return

            try:
                print_status(f"Loading wet signal: {config_json["wet_signal"]}")
                wet_signal_bytes = zip_file.read(config_json["wet_signal"])
                with io.BytesIO(wet_signal_bytes) as wet_bytes_io:
                    wet_sample_rate, wet_signal = wavfile.read(wet_bytes_io)
                    if wet_sample_rate != config_json["sample_rate"]:
                        print_status("Error: wet signal has unexpected sample rate")
                        return
            except:
                print_status(
                    f"Error: Failed to load wet signal from {config_json["wet_signal"]}"
                )
                return

            # Input file is half a second of silence, a click, quarter second silence, click, half second silence
            dry_silent = dry_signal[dry_sample_rate // 8 : 3 * dry_sample_rate // 8]
            dry_clicks = dry_signal[3 * dry_sample_rate // 8 : 5 * dry_sample_rate // 4]
            dry_noise_floor = np.max(np.abs(dry_silent))
            print_status(f"Dry noise floor: {dry_noise_floor}")

            wet_silent = wet_signal[wet_sample_rate // 8 : 3 * wet_sample_rate // 8]
            wet_clicks = wet_signal[3 * wet_sample_rate // 8 : 5 * wet_sample_rate // 4]
            wet_noise_floor = np.max(np.abs(wet_silent))
            print_status(f"Wet noise floor: {wet_noise_floor}")

            # Find all parts of the signal that exceed the noise threshold
            maybe_measured_dry_clicks = _find_clicks(dry_clicks, wet_noise_floor)
            if len(maybe_measured_dry_clicks) < 2:
                print_status(
                    f"Error: Found {len(maybe_measured_dry_clicks)} dry click(s), should be at least 2"
                )
                print_status(f"{maybe_measured_dry_clicks}")
                return

            # Grab the two 'biggest' clicks and use those
            maybe_measured_dry_clicks.sort(key=_sort_clicks)
            dry_click_indices = [
                click.index_begin for click in maybe_measured_dry_clicks[-2:]
            ]
            dry_click_indices.sort()

            maybe_measured_wet_clicks = _find_clicks(wet_clicks, wet_noise_floor)
            if len(maybe_measured_wet_clicks) < 2:
                print_status(
                    f"Error: Found {len(maybe_measured_wet_clicks)} wet click(s), should be at least 2"
                )
                print_status(f"{maybe_measured_wet_clicks}")
                return

            maybe_measured_wet_clicks.sort(key=_sort_clicks)
            wet_click_indices = [
                click.index_begin for click in maybe_measured_wet_clicks[-2:]
            ]
            wet_click_indices.sort()

            # Click delta is the time between two clicks on a single track
            # Click delta delta is the difference between the 'Click deltas' of the wet and dry tracks
            dry_click_delta = dry_click_indices[1] - dry_click_indices[0]
            wet_click_delta = wet_click_indices[1] - wet_click_indices[0]
            click_delta_delta = abs(dry_click_delta - wet_click_delta)
            print_status(f"Click delta delta: {click_delta_delta}")

            if click_delta_delta > 4:
                print_status("Error: Click delta delta is greater than 4")
                return

            latency_samples_a = wet_click_indices[0] - dry_click_indices[0]
            latency_samples_b = wet_click_indices[1] - dry_click_indices[1]
            latency_samples = min(latency_samples_a, latency_samples_b)
            print_status(f"Recording latency: {latency_samples} samples")

            # Cut the first second of timing calibration data off of the start
            # Also cut the test data off of the end if it exists
            if test_data_offset > 0:
                train_dry = dry_signal[dry_sample_rate:test_data_offset]
                train_wet = wet_signal[
                    wet_sample_rate
                    + latency_samples : test_data_offset
                    + latency_samples
                ]

                test_dry = dry_signal[test_data_offset:]
                test_wet = wet_signal[test_data_offset + latency_samples :]
            else:
                train_dry = dry_signal[dry_sample_rate:]
                train_wet = wet_signal[wet_sample_rate + latency_samples :]

                test_dry = None
                test_wet = None

            # Trim the end so we have a 1:1 mapping between the two
            train_trimmed_size = min(len(train_dry), len(train_wet))
            train_dry = train_dry[:train_trimmed_size]
            train_wet = train_wet[:train_trimmed_size]

            if test_dry is not None:
                assert test_wet is not None
                test_trimmed_size = min(len(test_dry), len(test_wet))
                test_dry = test_dry[:test_trimmed_size]
                test_wet = test_wet[:test_trimmed_size]

            assert len(train_dry) == len(train_wet)
            if test_dry is not None:
                assert len(test_dry) == len(test_wet)

            print_status(f"Training samples available: {len(train_dry)}")
            if test_dry is not None:
                print_status(f"Testing samples available: {len(test_dry)}")
            else:
                print_status("No testing samples available")
            context.signal_dry = train_dry
            context.signal_wet = train_wet
            context.signal_dry_test = test_dry
            context.signal_wet_test = test_wet

            gear_make = config_json["device_make"]
            gear_model = config_json["device_model"]
            context.metadata = ModelMetadata(
                name=f"{gear_make} -- {gear_model}",
                gear_make=gear_make,
                gear_model=gear_model,
                comment="",
            )

            context.sample_rate = config_json["sample_rate"]
            context.errored = False
            context.complete = True

    except zipfile.BadZipFile:
        print_status("Error: File is not a valid zip archive")
        return
    except:
        print_status("Error: Unknown error occurred")
        return
