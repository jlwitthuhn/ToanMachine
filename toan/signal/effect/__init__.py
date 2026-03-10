# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

import enum
from enum import Enum


class EffectType(Enum):
    Nothing = enum.auto()
    Delay01 = enum.auto()
    Delay02 = enum.auto()
