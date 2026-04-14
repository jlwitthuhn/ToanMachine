# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

import enum

import mlx.core as mx
from mlx import nn


class LossFunction(enum.Enum):
    ESR = enum.auto()
    MSE = enum.auto()
    RMSE = enum.auto()
    FFT_MSE = enum.auto()


def _loss_esr(output: mx.array, target: mx.array) -> mx.array:
    eps = 1e-6
    delta2 = (target - output) ** 2
    target2 = target**2
    delta2_mean = delta2.mean(axis=-1)
    target2_mean = target2.mean(axis=-1)
    loss_per_batch_item = delta2_mean / (target2_mean + eps)
    return loss_per_batch_item.mean()


def _loss_mse(output: mx.array, target: mx.array) -> mx.array:
    delta = target - output
    delta2 = delta**2
    return delta2.mean()


def _loss_rmse(output: mx.array, target: mx.array) -> mx.array:
    return mx.sqrt(_loss_mse(output, target))


def _loss_fft_mse(output: mx.array, target: mx.array) -> mx.array:
    output_fft = mx.fft.fft(output)
    target_fft = mx.fft.fft(target)
    delta = target_fft - output_fft
    delta2 = (delta.abs()) ** 2
    return delta2.mean()


def calculate_loss(
    model: nn.Module, loss_fn: LossFunction, input: mx.array, target: mx.array
) -> mx.array:
    output = model(input)
    match loss_fn:
        case LossFunction.ESR:
            return _loss_esr(output, target)
        case LossFunction.MSE:
            return _loss_mse(output, target)
        case LossFunction.RMSE:
            return _loss_rmse(output, target)
        case LossFunction.FFT_MSE:
            return _loss_fft_mse(output, target)
        case _:
            assert False
