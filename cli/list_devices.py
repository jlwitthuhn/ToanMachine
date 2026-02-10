# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

from toan.soundio import get_input_devices, get_output_devices


def main():
    print("Input devices:")
    for device in get_input_devices():
        print(f"{device.index}: {device.name} - {device.channels_in} channel(s)")
    print("Output devices:")
    for device in get_output_devices():
        print(f"{device.index}: {device.name} - {device.channels_out} channel(s)")


if __name__ == "__main__":
    main()
