# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

import numpy as np

from toan.generate.chirp import generate_chirp
from toan.generate.scale import generate_semitone_scale
from toan.mix import concat_signals
from toan.music import get_note_frequency_by_name


def generate_capture_signal(sample_rate: int) -> np.ndarray:
    AMPLITUDE = 0.6
    c2_pitch = get_note_frequency_by_name("C", 2, 440.0)

    sweep_up = generate_chirp(sample_rate, 20.0, 20000.0, AMPLITUDE, 4.0)
    sweep_down = generate_chirp(sample_rate, 20000.0, 20.0, AMPLITUDE, 4.0)

    scale_list = generate_semitone_scale(sample_rate, c2_pitch, 72, AMPLITUDE, 0.5)
    scale = concat_signals(scale_list, -sample_rate // 6)

    return concat_signals([sweep_up, sweep_down, scale], sample_rate // 2)
