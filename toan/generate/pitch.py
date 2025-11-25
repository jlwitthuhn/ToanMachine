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


def generate_pitch(note: str, octave: int, a_freq: float) -> float:
    a4_octave_offset = octave - 4
    a4_note_offset = _get_note_offset(note) - _get_note_offset("A")
    a4_semitone_offset = a4_octave_offset * 12 + a4_note_offset
    pitch_adjustment = math.pow(2, a4_semitone_offset / 12)
    return pitch_adjustment * a_freq
