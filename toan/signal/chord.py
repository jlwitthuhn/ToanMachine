# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

from enum import Enum


class ChordType(Enum):
    # 2 notes
    Octave = 1
    Tritone = 2
    # 3 notes
    Major = 3
    Minor = 4
    Diminished = 5
    # 4 notes
    MajorSeventh = 6
    # 5 notes
    MinorNinth = 7
    # 6 notes
    GuitarStrings = 8
