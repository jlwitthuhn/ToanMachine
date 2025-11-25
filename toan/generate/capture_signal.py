import numpy as np

from toan.generate.chirp import generate_chirp
from toan.generate.pitch import generate_pitch
from toan.generate.scale import generate_semitone_scale
from toan.mix import concat_signals


def generate_capture_signal(sample_rate: int) -> np.ndarray:
    AMPLITUDE = 0.6
    c2_pitch = generate_pitch("C", 2, 440.0)

    sweep_up = generate_chirp(sample_rate, 20.0, 20000.0, AMPLITUDE, 10.0)
    sweep_down = generate_chirp(sample_rate, 20000.0, 20.0, AMPLITUDE, 10.0)

    scale_list = generate_semitone_scale(sample_rate, c2_pitch, 72, AMPLITUDE, 1.0)
    scale = concat_signals(scale_list, -sample_rate // 4)

    return concat_signals([sweep_up, sweep_down, scale], sample_rate // 4)
