# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

import math


def increase_frequency_by_semitones(frequency: float, semitones: int) -> float:
    return frequency * math.pow(2, semitones / 12)
