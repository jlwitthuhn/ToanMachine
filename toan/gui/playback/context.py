# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

from toan.model.nam_a1_wavenet import NamA1WaveNet


class PlaybackContext:
    nam_model_path: str | None = None
    nam_model: NamA1WaveNet | None = None
    sample_rate: int = 0
