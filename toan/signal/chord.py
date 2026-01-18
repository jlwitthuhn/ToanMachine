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

    def get_shape(self) -> list[int]:
        match self:
            case ChordType.Octave:
                return [12]
            case ChordType.Tritone:
                return [6]
            case ChordType.Major:
                return [4, 7]
            case ChordType.Minor:
                return [3, 7]
            case ChordType.Diminished:
                return [3, 6]
            case ChordType.MajorSeventh:
                return [4, 7, 11]
            case ChordType.MinorNinth:
                return [3, 7, 10, 14]
            case ChordType.GuitarStrings:
                return [5, 10, 15, 19, 24]
            case _:
                assert False
