import tkinter as tk
from tkinter import ttk

from toan.gui.calibrate import VolumeCalibrationDialog
from toan.soundio import SdChannel, SdDevice, get_input_devices, get_output_devices


def _extract_lists(
    device_list: list[SdDevice], include_in: bool = True, include_out: bool = True
) -> tuple[list[str], dict[str, SdChannel]]:
    out_labels = []
    out_data = {}

    for device in device_list:
        if include_in:
            for chan in range(device.channels_in):
                chan += 1  # 1-indexed
                label = f"In - {device.index}: {device.name} [ch {chan}]"
                out_labels.append(label)
                out_data[label] = SdChannel(device.index, chan)
        if include_out:
            for chan in range(device.channels_out):
                chan += 1  # 1-indexed
                label = f"Out - {device.index}: {device.name} [ch {chan}]"
                out_labels.append(label)
                out_data[label] = SdChannel(device.index, chan)

    return out_labels, out_data


class CaptureFrame(ttk.Frame):
    parent: ttk.Notebook
    active_dialog: tk.Toplevel

    sample_rate: int

    input_combo: ttk.Combobox
    input_device_data: dict[str, SdChannel]

    output_combo: ttk.Combobox
    output_device_data: dict[str, SdChannel]

    def __init__(
        self,
        parent: ttk.Notebook,
        sample_rate: int,
    ):
        super().__init__(parent)

        self.parent = parent
        self.sample_rate = sample_rate

        input_devices = get_input_devices()
        input_device_labels, self.input_device_data = _extract_lists(input_devices)

        output_devices = get_output_devices()
        output_device_labels, self.output_device_data = _extract_lists(output_devices)

        root = self

        output_label = ttk.Label(root, text="Send device:")
        output_label.pack()

        self.output_combo = ttk.Combobox(root, state="readonly")
        self.output_combo["values"] = output_device_labels
        self.output_combo.bind("<<ComboboxSelected>>", self._combobox_changed_output)
        self.output_combo.pack(fill="x")

        input_label = ttk.Label(root, text="Return device:")
        input_label.pack()

        self.input_combo = ttk.Combobox(root, state="readonly")
        self.input_combo["values"] = input_device_labels
        self.input_combo.bind("<<ComboboxSelected>>", self._combobox_changed_input)
        self.input_combo.pack(fill="x")

        spacer_label_1 = ttk.Label(root, text="")
        spacer_label_1.pack()

        begin_button = ttk.Button(
            root, text="Begin Capture", command=self._begin_capture
        )
        begin_button.pack()

    def _begin_capture(self):
        if self.input_combo.get() == "":
            print("Error: No input device selected")
            return
        if self.output_combo.get() == "":
            print("Error: No output device selected")
            return
        assert self.input_combo.get() in self.input_device_data
        assert self.output_combo.get() in self.output_device_data
        self.active_dialog = VolumeCalibrationDialog(
            self.parent,
            self.output_device_data[self.output_combo.get()],
            self.input_device_data[self.input_combo.get()],
            self.sample_rate,
        )

    def _combobox_changed_input(self, _):
        self.input_combo.selection_clear()

    def _combobox_changed_output(self, _):
        self.output_combo.selection_clear()
