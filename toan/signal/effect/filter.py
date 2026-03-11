# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

import numpy as np
from scipy.signal import iirfilter, sosfilt


def _do_iirfilter(
    signal: np.ndarray,
    iir_wn: float | list[float],
    iir_type: str,
    iir_order: int,
) -> np.ndarray:
    sos_filter = iirfilter(
        N=iir_order,
        Wn=iir_wn,
        btype=iir_type,
        ftype="butter",
        output="sos",
    )
    return sosfilt(sos_filter, signal)


def effect_filter_band_pass(
    signal: np.ndarray,
    sample_rate: int,
    low_cut_freq: float,
    high_cut_freq: float,
    order: int = 4,
    dry_mix: float = 0.0,
) -> None:
    low = low_cut_freq / (sample_rate / 2.0)
    high = high_cut_freq / (sample_rate / 2.0)
    wet_signal = _do_iirfilter(
        signal,
        [low, high],
        "band",
        order,
    )
    signal[:] = dry_mix * signal + (1.0 - dry_mix) * wet_signal


def effect_filer_high_pass(
    signal: np.ndarray,
    sample_rate: int,
    freq: float,
    order: int = 4,
    dry_mix: float = 0.0,
) -> None:
    cutoff: float = freq / (sample_rate / 2.0)
    wet_signal = _do_iirfilter(
        signal,
        cutoff,
        "highpass",
        order,
    )
    signal[:] = dry_mix * signal + (1.0 - dry_mix) * wet_signal


def effect_filter_low_pass(
    signal: np.ndarray,
    sample_rate: int,
    freq: float,
    order: int = 4,
    dry_mix: float = 0.0,
) -> None:
    cutoff: float = freq / (sample_rate / 2.0)
    wet_signal = _do_iirfilter(
        signal,
        cutoff,
        "lowpass",
        order,
    )
    signal[:] = dry_mix * signal + (1.0 - dry_mix) * wet_signal


def effect_filter_notch(
    signal: np.ndarray,
    sample_rate: int,
    low_cut_freq: float,
    high_cut_freq: float,
    order: int = 4,
    dry_mix: float = 0.0,
) -> None:
    low = low_cut_freq / (sample_rate / 2.0)
    high = high_cut_freq / (sample_rate / 2.0)
    wet_signal = _do_iirfilter(
        signal,
        [low, high],
        "bandstop",
        order,
    )
    signal[:] = dry_mix * signal + (1.0 - dry_mix) * wet_signal
