# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

import enum


class ChordType(enum.Enum):
    # Not actually a chord
    RootOnly = enum.auto()
    # 2 notes
    MajorSecond = enum.auto()
    MajorSeventh = enum.auto()
    MajorSixth = enum.auto()
    MajorThird = enum.auto()
    MinorSecond = enum.auto()
    MinorSeventh = enum.auto()
    MinorSixth = enum.auto()
    MinorThird = enum.auto()
    Octave = enum.auto()
    PerfectFifth = enum.auto()
    PerfectFourth = enum.auto()
    Tritone = enum.auto()
    # 3 notes
    AugmentedTriad = enum.auto()
    DiminishedTriad = enum.auto()
    MajorTriad = enum.auto()
    MinorTriad = enum.auto()
    # 4 notes
    AugmentedMajorSeventhTetrad = enum.auto()
    DiminishedSeventhTetrad = enum.auto()
    DominantSeventhTetrad = enum.auto()
    HalfDiminishedSeventhTetrad = enum.auto()
    MajorSeventhTetrad = enum.auto()
    MinorSeventhTetrad = enum.auto()
    MinorMajorSeventhTetrad = enum.auto()
    # 5 notes
    MajorNinthPentad = enum.auto()
    MinorNinthPentad = enum.auto()
    # 6 notes
    GuitarStrings = enum.auto()

    def get_shape(self) -> list[int]:
        match self:
            case ChordType.RootOnly:
                return []
            case ChordType.MajorThird:
                return [4]
            case ChordType.MajorSecond:
                return [2]
            case ChordType.MajorSeventh:
                return [11]
            case ChordType.MajorSixth:
                return [9]
            case ChordType.MinorThird:
                return [3]
            case ChordType.MinorSecond:
                return [1]
            case ChordType.MinorSeventh:
                return [10]
            case ChordType.MinorSixth:
                return [8]
            case ChordType.Octave:
                return [12]
            case ChordType.PerfectFifth:
                return [7]
            case ChordType.PerfectFourth:
                return [5]
            case ChordType.Tritone:
                return [6]
            case ChordType.AugmentedTriad:
                return [4, 8]
            case ChordType.DiminishedTriad:
                return [3, 6]
            case ChordType.MajorTriad:
                return [4, 7]
            case ChordType.MinorTriad:
                return [3, 7]
            case ChordType.AugmentedMajorSeventhTetrad:
                return [4, 8, 11]
            case ChordType.DiminishedSeventhTetrad:
                return [3, 6, 9]
            case ChordType.DominantSeventhTetrad:
                return [4, 7, 10]
            case ChordType.HalfDiminishedSeventhTetrad:
                return [3, 6, 10]
            case ChordType.MajorSeventhTetrad:
                return [4, 7, 11]
            case ChordType.MinorSeventhTetrad:
                return [3, 7, 10]
            case ChordType.MinorMajorSeventhTetrad:
                return [3, 7, 11]
            case ChordType.MajorNinthPentad:
                return [4, 7, 11, 14]
            case ChordType.MinorNinthPentad:
                return [3, 7, 10, 13]
            case ChordType.GuitarStrings:
                return [5, 10, 15, 19, 24]
            case _:
                assert False
