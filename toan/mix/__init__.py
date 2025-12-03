# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

import numpy as np


def concat_signals(signals: list[np.ndarray], padding_samples: int = 0) -> np.ndarray:
    if padding_samples == 0:
        return np.concat(signals)
    elif padding_samples > 0:
        padding_array = np.zeros(padding_samples)
        signals_with_padding = []
        for signal in signals:
            signals_with_padding.append(signal)
            signals_with_padding.append(padding_array)
        signals_with_padding.pop()
        return np.concat(signals_with_padding)
    else:
        overlap_samples = -padding_samples
        output = signals.pop(0)
        for signal in signals:
            before = output[:-overlap_samples]
            overlap = output[-overlap_samples:] + signal[:overlap_samples]
            after = signal[overlap_samples:]
            output = np.concat([before, overlap, after])
        return output
