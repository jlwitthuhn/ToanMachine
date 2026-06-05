# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

import enum
from enum import Enum


class ModelConfigPreset(Enum):
    A1_NAM_STANDARD = enum.auto()
    A1_NAM_LITE = enum.auto()
    A1_NAM_FEATHER = enum.auto()
    A1_CUSTOM_XSTD = enum.auto()
    A1_CUSTOM_REVYSTD = enum.auto()
    A2_NAM = enum.auto()

    def get_label(self) -> str:
        match self:
            case ModelConfigPreset.A1_NAM_STANDARD:
                return "A1 NAM Standard - 13801p"
            case ModelConfigPreset.A1_NAM_LITE:
                return "A1 NAM Lite - 6553p"
            case ModelConfigPreset.A1_NAM_FEATHER:
                return "A1 NAM Feather - 3025p"
            case ModelConfigPreset.A1_CUSTOM_XSTD:
                return "A1 Custom xSTD - 12409p"
            case ModelConfigPreset.A1_CUSTOM_REVYSTD:
                return "A1 Custom REVySTD - 12768p"
            case ModelConfigPreset.A2_NAM:
                return "A2 NAM Slimmable"
            case _:
                raise NotImplementedError
