# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

import os
from dataclasses import dataclass
from pathlib import Path

import platformdirs
from scipy.io import wavfile


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
        sample_rate, wav_data = wavfile.read(path)
        duration = len(wav_data) / sample_rate
        result.append(UserWavDesc(str(path), file.name, sample_rate, duration))
    return result
