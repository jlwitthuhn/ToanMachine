# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

import enum
from enum import Enum


class ModelConfigPreset(Enum):
    A2_NAM = enum.auto()

    def get_label(self) -> str:
        match self:
            case ModelConfigPreset.A2_NAM:
                return "A2 NAM Slimmable"
            case _:
                raise NotImplementedError
