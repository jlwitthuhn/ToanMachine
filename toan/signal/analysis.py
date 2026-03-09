# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

from dataclasses import dataclass

import numpy as np


@dataclass
class SignalClickDetails:
    first_click: int
    delta: int
    noise_floor: float


@dataclass
class _PotentialClick:
    index_begin: int = 0
    index_peak: int = 0
    magnitude: float = 0.0
    width: int = 0


def _sort_clicks(click: _PotentialClick) -> float:
    return click.width * click.magnitude


def _find_clicks(signal: np.ndarray, noise_floor: float) -> list[_PotentialClick]:
    noise_threshold = noise_floor * 12.0
    result = []
    silence_samples_required = 500
    silence_samples_remaining = silence_samples_required
    current_click = _PotentialClick()
    for i in range(len(signal)):
        current_click.width += 1
        this_sample = signal[i]
        this_sample_abs = np.abs(this_sample)
        if this_sample_abs > noise_threshold:
            if silence_samples_remaining <= 0:
                current_click = _PotentialClick(index_begin=i)
                result.append(current_click)
            silence_samples_remaining = silence_samples_required
            if this_sample_abs > current_click.magnitude:
                current_click.magnitude = this_sample_abs
                current_click.index_peak = i
        else:
            silence_samples_remaining -= 1
    return result


def find_dry_clicks(signal: np.ndarray) -> SignalClickDetails | None:
    assert signal.ndim == 1
    eps = 0.001

    click_indices: list[int] = []
    for i in range(len(signal)):
        if signal[i] > eps:
            click_indices.append(i)

    if len(click_indices) != 2:
        return None

    return SignalClickDetails(
        first_click=click_indices[0],
        delta=click_indices[1] - click_indices[0],
        noise_floor=eps,
    )


def find_wet_clicks(
    signal: np.ndarray, quiet_samples: int
) -> SignalClickDetails | None:
    noise_floor = np.max(np.abs(signal[:quiet_samples]))
    maybe_clicks = _find_clicks(signal, noise_floor)
    maybe_clicks.sort(key=_sort_clicks)
    if len(maybe_clicks) < 2:
        return None
    click_indices = [click.index_begin for click in maybe_clicks[-2:]]
    click_indices.sort()
    return SignalClickDetails(
        first_click=click_indices[0],
        delta=click_indices[1] - click_indices[0],
        noise_floor=noise_floor,
    )
