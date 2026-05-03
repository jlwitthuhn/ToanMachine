# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

import os
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import platformdirs
from scipy.io import wavfile

from toan.signal.mix import concat_signals
from toan.wav import load_and_resample_wav


@dataclass
class UserWavDesc:
    path: str
    filename: str
    sample_rate: int
    duration: float


def create_user_wav_dir() -> None:
    os.makedirs(get_user_wav_dir(), exist_ok=True)


def get_user_wav_dir() -> str:
    root_dir = platformdirs.user_data_dir("toan", "toan")
    return os.path.join(root_dir, "extra")


def do_user_wavs_exist() -> bool:
    wav_dir = Path(get_user_wav_dir())
    for _ in wav_dir.glob("*.wav"):
        return True
    return False


def get_user_wav_list() -> list[UserWavDesc]:
    result = []
    wav_dir = Path(get_user_wav_dir())
    for file in wav_dir.glob("*.wav"):
        path = wav_dir / file.name
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=wavfile.WavFileWarning)
            sample_rate, wav_data = wavfile.read(path)
        duration = len(wav_data) / sample_rate
        result.append(UserWavDesc(str(path), file.name, sample_rate, duration))
    return result


def load_user_wav_list(sample_rate: int, file_names: list[str]) -> np.ndarray:
    wav_dir = get_user_wav_dir()
    signal_list: list[np.ndarray] = []
    for file_name in file_names:
        file_path = os.path.join(wav_dir, file_name)
        file_signal = load_and_resample_wav(sample_rate, file_path)
        file_signal = file_signal.astype(np.float32) / np.abs(file_signal).max()
        signal_list.append(file_signal)
    return concat_signals(signal_list, sample_rate // 4)
