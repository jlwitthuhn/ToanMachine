# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identified: GPL-3.0-only

import math


def _get_note_offset(note: str) -> int:
    match note:
        case "C":
            return 0
        case "C#" | "Db":
            return 1
        case "D":
            return 2
        case "D#" | "Eb":
            return 3
        case "E":
            return 4
        case "F":
            return 5
        case "F#" | "Gb":
            return 6
        case "G":
            return 7
        case "G#" | "Ab":
            return 8
        case "A":
            return 9
        case "A#" | "Bb":
            return 10
        case "B":
            return 11
    raise Exception("Invalid note")


def get_note_index_by_name(note: str, octave: int) -> int:
    return _get_note_offset(note) + 12 * octave


def get_note_frequency_by_name(note: str, octave: int, a_freq: float) -> float:
    a4_index = get_note_index_by_name("A", 4)
    note_index = get_note_index_by_name(note, octave)
    a4_semitone_offset = note_index - a4_index
    pitch_adjustment = math.pow(2, a4_semitone_offset / 12)
    return pitch_adjustment * a_freq
