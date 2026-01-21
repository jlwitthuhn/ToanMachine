# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only


def format_seconds_as_mmss(seconds: float) -> str:
    assert seconds >= 0
    sec_int = int(seconds)
    minutes, seconds = divmod(sec_int, 60)
    return f"{minutes:02d}:{seconds:02d}"
