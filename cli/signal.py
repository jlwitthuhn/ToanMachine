# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

# This script is used to evaluate the effectiveness of changes to the capture signal
# It can record and then train on that recording in a loop to measure test loss

import math
import time
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import numpy as np

from toan.mix import concat_signals
from toan.model.nam_wavenet_presets import get_wavenet_config
from toan.model.presets import ModelConfigPreset
from toan.persistence import get_user_wav_list
from toan.signal import generate_capture_signal
from toan.signal.capture_signal import CaptureSignalConfig
from toan.soundio import SdChannel, get_input_devices, get_output_devices
from toan.soundio.record_wet import RecordWetController
from toan.training.config import TrainingConfig
from toan.training.context import TrainingProgressContext
from toan.training.loop import run_training_loop
from toan.training.zip_loader import ZipLoaderContext, run_zip_loader
from toan.wav import load_and_resample_wav
from toan.zip import create_training_zip


def _parse_colon_syntax(device_str: str) -> SdChannel | None:
    colon_index = device_str.find(":")
    if colon_index == -1:
        return None
    before = device_str[:colon_index]
    after = device_str[colon_index + 1 :]
    try:
        return SdChannel(device_index=int(before), channel_index=int(after))
    except ValueError:
        return None


def _validate_sdchannel(channel: SdChannel, is_input: bool) -> str | None:
    devices = get_input_devices() if is_input else get_output_devices()
    if channel.channel_index <= 0:
        return "1 is the lowest valid channel"
    for device in devices:
        if channel.device_index != device.index:
            continue
        if is_input and channel.channel_index > device.channels_in:
            return "Selected device does not have enough input channels"
        if not is_input and channel.channel_index > device.channels_out:
            return "Selected device does not have enough output channels"
        return None

    return f"Index {channel.device_index} is not a valid device"


def do_iteration(
    sample_rate: int,
    channel_in: SdChannel,
    channel_out: SdChannel,
    signal_config: CaptureSignalConfig,
    extra_signal_test: np.ndarray | None,
) -> float:
    print("Generating signal...")
    signal_dry = generate_capture_signal(sample_rate, signal_config)

    test_offset = 0
    if extra_signal_test is not None:
        print(f"Adding extra test signal of {len(extra_signal_test)} samples...")
        assert extra_signal_test.ndim == 1
        test_offset = len(signal_dry)
        signal_dry = concat_signals([signal_dry, extra_signal_test], sample_rate // 2)

    print("Beginning recording...")
    record_controller = RecordWetController(
        sample_rate, signal_dry, channel_in, channel_out
    )
    record_controller.start()

    while not record_controller.is_complete():
        time.sleep(0.2)
    signal_wet = record_controller.get_recorded_signal()
    record_controller.close()
    print(f"Recording complete, got {len(signal_wet)} samples")

    print("Packaging recording...")
    zip_buffer = create_training_zip(
        sample_rate,
        signal_dry,
        signal_wet,
        "Test Make",
        "Test Model",
        test_offset,
    )

    print("Loading zip file...")
    zip_context = ZipLoaderContext()
    run_zip_loader(zip_context, zip_buffer)

    if zip_context.errored:
        print("Failed to load zip file, replaying log...")
        for line in zip_context.messages_queue:
            print(f">> {line}")
        return math.inf
    assert zip_context.complete

    print("Beginning training...")
    training_config = TrainingConfig()
    progress_context = TrainingProgressContext()
    progress_context.model_config = get_wavenet_config(ModelConfigPreset.NAM_STANDARD)
    progress_context.metadata = zip_context.metadata
    progress_context.sample_rate = sample_rate
    progress_context.signal_dry_train = zip_context.signal_dry
    progress_context.signal_wet_train = zip_context.signal_wet
    progress_context.signal_dry_test = zip_context.signal_dry_test
    progress_context.signal_wet_test = zip_context.signal_wet_test
    run_training_loop(progress_context, training_config)
    print("Training complete")
    print(f"train loss: {progress_context.loss_train}")
    print(f"test loss: {progress_context.loss_test}")
    return progress_context.loss_train


def main() -> None:
    arg_parser = ArgumentParser(
        description="Script to record a device and then train from that recording",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    arg_parser.add_argument(
        "--input",
        type=str,
        help="Input device in the format '[index]:[channel]' ex: '0:1'",
        required=True,
    )
    arg_parser.add_argument(
        "--output",
        type=str,
        help="Output device in the format '[index]:[channel]' ex: '0:1'",
        required=True,
    )
    arg_parser.add_argument(
        "--samplerate",
        type=int,
        default=48000,
        help="Desired sample rate of the recording and model",
    )
    arg_parser.add_argument(
        "--testwavs",
        type=str,
        help="Comma separated list of test wavs",
    )
    args = arg_parser.parse_args()

    input_channel = _parse_colon_syntax(args.input)
    if input_channel is None:
        print(f"Failed to parse input device: {args.input}")
        return
    input_err = _validate_sdchannel(input_channel, True)
    if input_err is not None:
        print(f"Failed to find input device: {input_err}")
        return

    output_channel = _parse_colon_syntax(args.output)
    if output_channel is None:
        print(f"Failed to parse output device: {args.output}")
        return
    output_err = _validate_sdchannel(output_channel, False)
    if output_err is not None:
        print(f"Failed to find output device: {output_err}")
        return

    test_signal = None
    if args.testwavs is not None:
        print("Loading test wavs...")
        wav_set = set(args.testwavs.split(","))
        available_wavs = get_user_wav_list()
        paths: list[str] = []
        for user_wav in available_wavs:
            if user_wav.filename in wav_set:
                paths.append(user_wav.path)
                wav_set.remove(user_wav.filename)
        if len(wav_set) > 0:
            print(f"Warning: Could not find {wav_set}")
        signal_list: list[np.ndarray] = []
        for wav_path in paths:
            signal_list.append(load_and_resample_wav(args.samplerate, wav_path))
        test_signal = concat_signals(signal_list, args.samplerate // 4)
        test_signal = test_signal.astype(np.float32) / np.abs(test_signal).max()

    print("Beginning iteration...")
    do_iteration(
        args.samplerate,
        input_channel,
        output_channel,
        CaptureSignalConfig(),
        test_signal,
    )


if __name__ == "__main__":
    main()
