from toan.soundio import SdChannel


class RecordingContext:
    input_channel: SdChannel
    output_channel: SdChannel
    sample_rate: int = 44100
