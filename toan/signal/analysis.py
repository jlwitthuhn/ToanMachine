# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only
import math
from dataclasses import dataclass

import numpy as np
import scipy
from matplotlib import pyplot as plt


@dataclass
class SignalClickDetails:
    first_click: int
    delta: int
    magnitude: float


@dataclass
class _PotentialClick:
    index_begin: int = 0
    index_peak: int = 0
    magnitude: float = 0.0
    width: int = 0


def _sort_clicks(click: _PotentialClick) -> float:
    return click.width * click.magnitude


def _find_clicks(signal: np.ndarray, noise_threshold: float) -> list[_PotentialClick]:
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


def _find_wet_clicks(
    signal: np.ndarray, noise_threshold: float
) -> SignalClickDetails | None:
    maybe_clicks = _find_clicks(signal, noise_threshold)
    maybe_clicks.sort(key=_sort_clicks)
    if len(maybe_clicks) < 2:
        return None
    click_indices = [click.index_begin for click in maybe_clicks[-2:]]
    click_indices.sort()
    return SignalClickDetails(
        first_click=click_indices[0],
        delta=click_indices[1] - click_indices[0],
        magnitude=noise_threshold,
    )


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
        magnitude=eps,
    )


def find_wet_clicks(
    signal: np.ndarray, quiet_samples: int, target_delta: int
) -> SignalClickDetails | None:
    noise_floor = np.max(np.abs(signal[:quiet_samples]))
    # Attempt finding clicks at all of these thresholds and choose best
    magnitudes = [1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0]
    best_match: SignalClickDetails | None = None
    # This default must exceed our desired click delta delta
    best_score: int = 1000
    for magnitude in magnitudes:
        maybe_result = _find_wet_clicks(signal, magnitude * noise_floor)
        if maybe_result is None:
            continue
        delta_delta = abs(target_delta - maybe_result.delta)
        if delta_delta < best_score:
            best_match = maybe_result
            best_score = delta_delta
        if best_score == 0:
            break
    return best_match


def generate_spectrogram(sample_rate: int, signal: np.ndarray) -> plt.Figure:
    freq, t, Sxx = scipy.signal.spectrogram(
        signal,
        sample_rate,
        window="hann",
        nperseg=2048,
        noverlap=1024,
    )

    fig, ax = plt.subplots()
    ax.set_title("Spectrogram")

    mesh = ax.pcolormesh(t, freq, Sxx, shading="gouraud")
    ax.set_xlabel("Seconds")
    ax.set_ylabel("Frequency")
    fig.colorbar(mesh, ax=ax)
    return fig
