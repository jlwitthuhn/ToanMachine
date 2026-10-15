# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

import sys
from argparse import ArgumentParser

import numpy as np
import soundfile as sf

CHUNK_SECONDS = 0.1
PERCENTILES = (1, 10, 25, 50, 75, 90, 99)


def analyze_wav(path: str) -> np.ndarray:
    """Return percentiles of non-silent chunk peaks across all channels in dBFS."""
    samples, sample_rate = sf.read(path, always_2d=True)
    chunk_frames = round(sample_rate * CHUNK_SECONDS)
    if chunk_frames < 1:
        raise ValueError(f"Sample rate is too low for {CHUNK_SECONDS}-second chunks.")
    chunk_count = len(samples) // chunk_frames
    if chunk_count == 0:
        raise ValueError(
            f"Input must contain at least one complete {CHUNK_SECONDS}-second chunk."
        )

    chunks = samples[: chunk_count * chunk_frames].reshape(
        chunk_count, chunk_frames, samples.shape[1]
    )
    if not np.all(np.isfinite(chunks)):
        raise ValueError("Input contains non-finite audio samples.")
    peaks = np.max(np.abs(chunks), axis=(1, 2))
    peaks = peaks[peaks > 0]
    if peaks.size == 0:
        raise ValueError(
            f"Input contains no non-silent complete {CHUNK_SECONDS}-second chunks."
        )
    peaks_dbfs = 20.0 * np.log10(peaks)

    return np.percentile(peaks_dbfs, PERCENTILES)


def main() -> int:
    parser = ArgumentParser(
        description=("Print dBFS at various percentiles for the content of a wav file.")
    )
    parser.add_argument("wav_path", help="Path to the input WAV file")
    args = parser.parse_args()
    try:
        values = analyze_wav(args.wav_path)
    except (OSError, ValueError, sf.LibsndfileError) as error:
        parser.error(str(error))
    for percentile, value in zip(PERCENTILES, values):
        print(f"Percentile {percentile:2d}: {value:.6f} dBFS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
