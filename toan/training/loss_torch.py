# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

# This file contains code adapted from AuraLoss under the terms of
# the Apache 2 license.

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


# Multi-resolution STFT parameters matching auraloss
_MRSTFT_FFT_SIZES = (1024, 2048, 512)
_MRSTFT_HOP_SIZES = (120, 240, 50)
_MRSTFT_WIN_LENGTHS = (600, 1200, 240)
_MRSTFT_EPS = 1e-8

_NAM_MSE_WEIGHT = 1.0
_NAM_MRSTFT_WEIGHT = 5e-4

# Cache Hann windows per (length, device, dtype) so we don't rebuild them every training step.
_hann_window_cache: dict[tuple[int, torch.device, torch.dtype], torch.Tensor] = {}


def _get_hann_window(
    win_length: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    key = (win_length, device, dtype)
    window = _hann_window_cache.get(key)
    if window is None:
        window = torch.hann_window(win_length, device=device, dtype=dtype)
        _hann_window_cache[key] = window
    return window


def _stft_magnitude(
    x: torch.Tensor, fft_size: int, hop_size: int, win_length: int
) -> torch.Tensor:
    # x: (B, L) -> magnitude spectrogram (B, freq, frames)
    window = _get_hann_window(win_length, x.device, x.dtype)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*was resized since it had shape.*")
        x_stft = torch.stft(
            x.contiguous(),
            fft_size,
            hop_size,
            win_length,
            window,
            return_complex=True,
        )
    return torch.sqrt(torch.clamp(x_stft.real**2 + x_stft.imag**2, min=_MRSTFT_EPS))


# MRSTFT implementation matching auraloss
def _loss_mrstft_torch(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    total = output.new_zeros(())
    for fft_size, hop_size, win_length in zip(
        _MRSTFT_FFT_SIZES, _MRSTFT_HOP_SIZES, _MRSTFT_WIN_LENGTHS
    ):
        x_mag = _stft_magnitude(output, fft_size, hop_size, win_length)
        y_mag = _stft_magnitude(target, fft_size, hop_size, win_length)
        # Spectral convergence (Frobenius norm over the whole tensor)
        sc_loss = torch.norm(y_mag - x_mag, p="fro") / torch.norm(y_mag, p="fro")
        # Log-magnitude L1 distance
        log_mag_loss = torch.nn.functional.l1_loss(torch.log(x_mag), torch.log(y_mag))
        total = total + sc_loss + log_mag_loss
    return total / len(_MRSTFT_FFT_SIZES)


def _loss_nam_original_torch(
    output: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    return _NAM_MSE_WEIGHT * _loss_mse_torch(
        output, target
    ) + _NAM_MRSTFT_WEIGHT * _loss_mrstft_torch(output, target)


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
        case LossFunction.NamOriginal:
            return _loss_nam_original_torch(model_output, target)
        case _:
            assert False
