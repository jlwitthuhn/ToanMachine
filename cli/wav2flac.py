# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

import os
import sys

import soundfile as sf


def load_and_convert(input_path, output_path):

    data, sample_rate = sf.read(input_path, always_2d=True)
    subtype = sf.info(input_path).subtype  # "PCM_16", "PCM_24", "PCM_32"

    flac_formats = sf.available_subtypes("FLAC")
    assert subtype in flac_formats

    sf.write(output_path, data, sample_rate, format="FLAC", subtype=subtype)

    print(f"Wrote {output_path} with format {subtype}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python -m cli.wav2flac <input.wav> <output.flac>")
        sys.exit(1)

    input_path = sys.argv[1]
    if os.path.isfile(input_path) == False:
        print("Input file does not exist")
        sys.exit(1)
    output_path = sys.argv[2]

    load_and_convert(input_path, output_path)
