# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only
import enum


class LossFunction(enum.Enum):
    ESR = enum.auto()
    MSE = enum.auto()
    RMSE = enum.auto()
    FFT_MSE = enum.auto()
    # Loss used by original neural-amp-modeler repo
    NamOriginal = enum.auto()
