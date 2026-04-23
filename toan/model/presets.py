# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

import enum
from enum import Enum


class ModelConfigPreset(Enum):
    NAM_A1_STANDARD = enum.auto()
    NAM_A1_LITE = enum.auto()
    NAM_A1_FEATHER = enum.auto()
    NAM_A1_NANO = enum.auto()
    CUSTOM_A1_XSTD = enum.auto()
    TOAN_A2_TEST = enum.auto()

    def get_label(self) -> str:
        match self:
            case ModelConfigPreset.NAM_A1_STANDARD:
                return "NAM A1 Standard - 13801p"
            case ModelConfigPreset.NAM_A1_LITE:
                return "NAM A1 Lite - 6553p"
            case ModelConfigPreset.NAM_A1_FEATHER:
                return "NAM A1 Feather - 3025p"
            case ModelConfigPreset.NAM_A1_NANO:
                return "NAM A1 Nano"
            case ModelConfigPreset.CUSTOM_A1_XSTD:
                return "Custom A1 xSTD"
            case ModelConfigPreset.TOAN_A2_TEST:
                return "Toan A2 Test"
            case _:
                raise NotImplementedError
