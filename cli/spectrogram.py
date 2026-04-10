# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

import sys

import soundfile as sf

from toan.signal.analysis import generate_spectrogram

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python -m cli.spectrogram <input.wav> <output.png>")
        sys.exit(1)

    path_in = sys.argv[1]
    signal, sample_rate = sf.read(path_in)
    fig = generate_spectrogram(sample_rate, signal)
    fig.savefig(sys.argv[2])
