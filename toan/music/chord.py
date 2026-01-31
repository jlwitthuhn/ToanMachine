# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

import enum


class ChordType(enum.Enum):
    # Not actually a chord
    RootOnly = enum.auto()
    # 2 notes
    Octave = enum.auto()
    PerfectFifth = enum.auto()
    Tritone = enum.auto()
    # 3 notes
    Major = enum.auto()
    Minor = enum.auto()
    Diminished = enum.auto()
    # 4 notes
    MajorSeventh = enum.auto()
    MinorSeventh = enum.auto()
    # 5 notes
    MinorNinth = enum.auto()
    # 6 notes
    GuitarStrings = enum.auto()

    def get_shape(self) -> list[int]:
        match self:
            case ChordType.RootOnly:
                return []
            case ChordType.Octave:
                return [12]
            case ChordType.PerfectFifth:
                return [7]
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
            case ChordType.MinorSeventh:
                return [3, 7, 10]
            case ChordType.MinorNinth:
                return [3, 7, 10, 14]
            case ChordType.GuitarStrings:
                return [5, 10, 15, 19, 24]
            case _:
                assert False
