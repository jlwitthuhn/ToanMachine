# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

import warnings

import mlx.core as mx
import mlx.nn as nn


# Based on `fast_tanh` from NeuralAmpModelerCore
# https://github.com/sdatkinson/NeuralAmpModelerCore/blob/v0.4.0.rc2/NAM/activations.h#L89
class FastTanh(nn.Module):
    def __init__(self):
        warnings.warn("Using unverified fast tanh implementation")
        super().__init__()

    def __call__(self, x: mx.array) -> mx.array:
        ax = mx.abs(x)
        x2 = x * x
        return (
            x
            * (
                2.45550750702956
                + 2.45550750702956 * ax
                + (0.893229853513558 + 0.821226666969744 * ax) * x2
            )
            / (
                2.44506634652299
                + (2.44506634652299 + x2) * mx.abs(x + 0.814642734961073 * x * ax)
            )
        )


# Based on `leaky_hardtanh` from NeuralAmpModelerCore
# https://github.com/sdatkinson/NeuralAmpModelerCore/blob/v0.4.0.rc2/NAM/activations.h#L73
class LeakyHardTanh(nn.Module):
    min_val: float
    max_val: float
    min_slope: float
    max_slope: float

    def __init__(
        self,
        min_val: float = -1.0,
        max_val: float = 1.0,
        min_slope: float = 0.01,
        max_slope: float = 0.01,
    ):
        warnings.warn("Using unverified leaky hard tanh implementation")
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.min_slope = min_slope
        self.max_slope = max_slope

    def __call__(self, x: mx.array) -> mx.array:
        if x < self.min_val:
            return (x - self.min_val) * self.min_slope + self.min_val
        elif x > self.max_val:
            return (x - self.max_val) * self.max_slope + self.max_val
        else:
            return x
