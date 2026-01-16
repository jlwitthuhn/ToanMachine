# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

import numpy as np

from toan.soundio import SdChannel


class RecordingContext:
    device_make: str = "The Toan Zoan"
    device_model: str = "Toan Device"
    sample_rate: int = 44100
    input_channel: SdChannel
    output_channel: SdChannel
    extra_signal_dry: np.ndarray | None = None
    signal_dry: np.ndarray | None = None
    signal_recorded: np.ndarray | None = None
