# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

from enum import Enum


class ModelConfigPreset(Enum):
    NAM_STANDARD = 0
    NAM_LITE = 1
    NAM_FEATHER = 2
    NAM_NANO = 3
    TOAN_STANDARD_PLUS = 4

    def get_label(self) -> str:
        match self:
            case ModelConfigPreset.NAM_STANDARD:
                return "NAM A1 Standard - 13801p"
            case ModelConfigPreset.NAM_LITE:
                return "NAM A1 Lite - 6553p"
            case ModelConfigPreset.NAM_FEATHER:
                return "NAM A1 Feather"
            case ModelConfigPreset.NAM_NANO:
                return "NAM A1 Nano"
            case ModelConfigPreset.TOAN_STANDARD_PLUS:
                return "Toan A1 Standard Plus - 30493p"
            case _:
                assert False
