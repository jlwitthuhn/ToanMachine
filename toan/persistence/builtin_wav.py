# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

import enum
from pathlib import Path

import numpy as np

from toan.wav import load_and_resample_wav


class BuiltinWav(enum.Enum):
    DUMMY = 0


def _get_builtin_wav_dir() -> Path:
    script_dir = Path(__file__).resolve().parent
    root_dir = script_dir.parent.parent
    return root_dir.joinpath("data").joinpath("training_wav").resolve()


def _get_builtin_wav_filename(type: BuiltinWav) -> str:
    match type:
        case BuiltinWav.DUMMY:
            return "dummy.wav"
        case _:
            raise NotImplementedError


def get_builtin_wav_signal(sample_rate: int, type: BuiltinWav) -> np.ndarray:
    file_name = _get_builtin_wav_filename(type)
    file_path = _get_builtin_wav_dir().joinpath(file_name).resolve()
    return load_and_resample_wav(sample_rate, str(file_path))
