from PySide6 import QtWidgets

from toan.gui.record import RecordingContext
from toan.soundio import SdChannel, SdDevice, get_input_devices, get_output_devices

DEVICE_TEXT = [
    "Choose both the output device that will send a signal to your pedal as well as the input device that will record the signal coming back from your pedal."
]


def _extract_lists(
    device_list: list[SdDevice], include_in: bool = True, include_out: bool = True
) -> tuple[list[str], dict[str, SdChannel]]:
    out_labels = []
    out_data = {}

    for device in device_list:
        if include_in:
            for chan in range(device.channels_in):
                chan += 1  # 1-indexed
                label = f"Input: {device.name}: Device {device.index} Channel {chan}"
                out_labels.append(label)
                out_data[label] = SdChannel(device.index, chan)
        if include_out:
            for chan in range(device.channels_out):
                chan += 1  # 1-indexed
                label = f"Output: {device.name}: Device {device.index} Channel {chan}"
                out_labels.append(label)
                out_data[label] = SdChannel(device.index, chan)

    return out_labels, out_data


class RecordDevicePage(QtWidgets.QWizardPage):
    context: RecordingContext
    input_channels: dict[str, SdChannel]
    output_channels: dict[str, SdChannel]

    combo_output: QtWidgets.QComboBox
    combo_input: QtWidgets.QComboBox

    def __init__(self, parent, context: RecordingContext):
        super().__init__(parent)
        self.context = context

        self.setTitle("Choose Device")
        layout = QtWidgets.QVBoxLayout(self)

        label = QtWidgets.QLabel("\n\n".join(DEVICE_TEXT), self)
        label.setWordWrap(True)
        layout.addWidget(label)

        hline = QtWidgets.QFrame(self)
        hline.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        hline.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        layout.addWidget(hline)

        label_output = QtWidgets.QLabel("Output Device:", self)
        layout.addWidget(label_output)

        def create_output_combo():
            result = QtWidgets.QComboBox(self)
            output_devices = get_output_devices()
            label_list, self.output_channels = _extract_lists(
                output_devices, include_in=False, include_out=True
            )
            result.addItems(label_list)
            return result

        self.combo_output = create_output_combo()
        layout.addWidget(self.combo_output)

        label_input = QtWidgets.QLabel("Input Device:", self)
        layout.addWidget(label_input)

        def create_input_combo():
            result = QtWidgets.QComboBox(self)
            input_devices = get_input_devices()
            label_list, self.input_channels = _extract_lists(
                input_devices, include_in=True, include_out=False
            )
            result.addItems(label_list)
            return result

        self.combo_input = create_input_combo()
        layout.addWidget(self.combo_input)

    def validatePage(self, /) -> bool:
        assert self.combo_input.currentText() in self.input_channels
        assert self.combo_output.currentText() in self.output_channels
        self.context.input_channel = self.input_channels[self.combo_input.currentText()]
        self.context.output_channel = self.output_channels[
            self.combo_output.currentText()
        ]
        return True
