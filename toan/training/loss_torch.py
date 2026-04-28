# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

import warnings

import torch

from toan.training.loss import LossFunction


def _loss_esr_torch(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    eps = 1e-6
    delta2 = (target - output) ** 2
    target2 = target**2
    delta2_mean = delta2.mean(dim=-1)
    target2_mean = target2.mean(dim=-1)
    loss_per_batch_item = delta2_mean / (target2_mean + eps)
    return loss_per_batch_item.mean()


def _loss_mse_torch(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    delta = target - output
    delta2 = delta**2
    return delta2.mean()


def _loss_rmse_torch(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(_loss_mse_torch(output, target))


def _loss_fft_mse_torch(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*was resized since it had shape.*")
        output_fft = torch.fft.rfft(output.contiguous())
        target_fft = torch.fft.rfft(target.contiguous())
    delta = target_fft - output_fft
    delta2 = delta.abs() ** 2
    return delta2.mean()


def calculate_loss_torch(
    loss_fn: LossFunction, model_output: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    match loss_fn:
        case LossFunction.ESR:
            return _loss_esr_torch(model_output, target)
        case LossFunction.MSE:
            return _loss_mse_torch(model_output, target)
        case LossFunction.RMSE:
            return _loss_rmse_torch(model_output, target)
        case LossFunction.FFT_MSE:
            return _loss_fft_mse_torch(model_output, target)
        case _:
            assert False
