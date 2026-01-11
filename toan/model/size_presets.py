# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

from enum import Enum


class ModelSizePreset(Enum):
    NAM_STANDARD = 0
    NAM_LITE = 1
    NAM_FEATHER = 2
    NAM_NANO = 3

    def get_label(self) -> str:
        match self:
            case ModelSizePreset.NAM_STANDARD:
                return "NAM Standard"
            case ModelSizePreset.NAM_LITE:
                return "NAM Lite"
            case ModelSizePreset.NAM_FEATHER:
                return "NAM Feather"
            case ModelSizePreset.NAM_NANO:
                return "NAM Nano"
            case _:
                assert False
