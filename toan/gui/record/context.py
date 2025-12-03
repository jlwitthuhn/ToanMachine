# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

from toan.soundio import SdChannel


class RecordingContext:
    input_channel: SdChannel
    output_channel: SdChannel
    sample_rate: int = 44100
