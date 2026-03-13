# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

from enum import Enum


class ModelConfigPreset(Enum):
    NAM_A1_STANDARD = 0
    NAM_A1_LITE = 1
    NAM_A1_FEATHER = 2
    NAM_A1_NANO = 3
    TOAN_A1_STANDARD_PLUS = 4
    TOAN_A2_TEST = 5

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
            case ModelConfigPreset.TOAN_A1_STANDARD_PLUS:
                return "Toan A1 Standard Plus - 30493p"
            case ModelConfigPreset.TOAN_A2_TEST:
                return "Toan A2 Test"
            case _:
                assert False
