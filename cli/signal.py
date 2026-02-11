# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

# This script is used to evaluate the effectiveness of changes to the capture signal
# It can record and then train on that recording in a loop to measure test loss

import time
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

from toan.signal import generate_capture_signal
from toan.signal.capture_signal import CaptureSignalConfig
from toan.soundio import SdChannel, get_input_devices, get_output_devices
from toan.soundio.record_wet import RecordWetController


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
    for device in devices:
        if channel.device_index != device.index:
            continue
        if is_input and channel.channel_index >= device.channels_in:
            return "Selected device does not have enough input channels"
        if not is_input and channel.channel_index >= device.channels_out:
            return "Selected device does not have enough output channels"
        return None

    return f"Index {channel.device_index} is not a valid device"


def do_iteration(
    sample_rate: int,
    channel_in: SdChannel,
    channel_out: SdChannel,
    signal_config: CaptureSignalConfig,
) -> None:
    print("Generating signal...")
    signal_dry = generate_capture_signal(sample_rate, signal_config)

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


def main() -> None:
    arg_parser = ArgumentParser(
        description="Script to record a device and then train from that recording",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    arg_parser.add_argument(
        "--input",
        type=str,
        help="Input device in the format '[index]:[channel]' ex: '0:1'",
    )
    arg_parser.add_argument(
        "--output",
        type=str,
        help="Output device in the format '[index]:[channel]' ex: '0:1'",
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

    print("Beginning iteration...")
    do_iteration(48000, input_channel, output_channel, CaptureSignalConfig())


if __name__ == "__main__":
    main()
