from tkinter import ttk

from toan.gui.calibrate import VolumeCalibrationDialog
from toan.soundio import SdChannel, SdDevice, get_input_devices, get_output_devices

_parent: ttk.Notebook
_sample_rate: int

_input_combo: ttk.Combobox
_input_device_data: dict[str, SdChannel]

_output_combo: ttk.Combobox
_output_device_data: dict[str, SdChannel]


def _input_combo_changed(_):
    _input_combo.selection_clear()


def _output_combo_changed(_):
    _output_combo.selection_clear()


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


the_dialog = None


def _begin_capture():
    global _input_combo
    if _input_combo.get() == "":
        print("Error: No input device selected")
        return
    global _output_combo
    if _output_combo.get() == "":
        print("Error: No output device selected")
        return
    global _input_device_data
    assert _input_combo.get() in _input_device_data
    global _output_device_data
    assert _output_combo.get() in _output_device_data
    global the_dialog
    global _sample_rate
    the_dialog = VolumeCalibrationDialog(
        _parent,
        _output_device_data[_output_combo.get()],
        _input_device_data[_input_combo.get()],
        _sample_rate,
    )


def create_capture_tab(notebook: ttk.Notebook, sample_rate: int) -> ttk.Frame:
    global _parent
    _parent = notebook
    global _sample_rate
    _sample_rate = sample_rate

    input_devices = get_input_devices()
    global _input_device_data
    input_device_labels, _input_device_data = _extract_lists(input_devices)

    output_devices = get_output_devices()
    global _output_device_data
    output_device_labels, _output_device_data = _extract_lists(output_devices)

    root = ttk.Frame(notebook)

    output_label = ttk.Label(root, text="Send device:")
    output_label.pack()

    global _output_combo
    _output_combo = ttk.Combobox(root, state="readonly")
    _output_combo["values"] = output_device_labels
    _output_combo.bind("<<ComboboxSelected>>", _output_combo_changed)
    _output_combo.pack(fill="x")

    input_label = ttk.Label(root, text="Return device:")
    input_label.pack()

    global _input_combo
    _input_combo = ttk.Combobox(root, state="readonly")
    _input_combo["values"] = input_device_labels
    _input_combo.bind("<<ComboboxSelected>>", _input_combo_changed)
    _input_combo.pack(fill="x")

    _spacer_label_1 = ttk.Label(root, text="")
    _spacer_label_1.pack()

    begin_button = ttk.Button(root, text="Begin Capture", command=_begin_capture)
    begin_button.pack()

    return root
