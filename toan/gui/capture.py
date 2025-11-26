from tkinter import ttk

from toan.soundio import SdDevice, get_input_devices, get_output_devices

_input_combo: ttk.Combobox
_input_device_data: list[dict]

_output_combo: ttk.Combobox
_output_device_data: list[dict]


def _input_combo_changed(_):
    _input_combo.selection_clear()


def _output_combo_changed(_):
    _output_combo.selection_clear()


def _extract_lists(
    device_list: list[SdDevice], include_in: bool = True, include_out: bool = True
) -> tuple[list[str], list[dict]]:
    out_labels = []
    out_data = []

    for device in device_list:
        if include_in:
            for chan in range(device.channels_in):
                chan += 1  # 1-indexed
                label = f"In - {device.index}: {device.name} [ch {chan}]"
                out_labels.append(label)
                data = {
                    "label": label,
                    "index": device.index,
                    "name": device.name,
                    "channel": chan,
                    "dir": "input",
                }
                out_data.append(data)
        if include_out:
            for chan in range(device.channels_out):
                chan += 1  # 1-indexed
                label = f"Out - {device.index}: {device.name} [ch {chan}]"
                out_labels.append(label)
                data = {
                    "label": label,
                    "index": device.index,
                    "name": device.name,
                    "channel": chan,
                    "dir": "output",
                }
                out_data.append(data)

    return out_labels, out_data


def create_capture_tab(notebook: ttk.Notebook) -> ttk.Frame:
    input_devices = get_input_devices()
    print(len(input_devices))
    global _input_device_data
    input_device_labels, _input_device_data = _extract_lists(input_devices)

    output_devices = get_output_devices()
    global _output_device_data
    output_device_labels, _output_device_data = _extract_lists(output_devices)

    root = ttk.Frame(notebook)

    output_label = ttk.Label(root, text="Capture send:")
    output_label.pack()

    global _output_combo
    _output_combo = ttk.Combobox(root, state="readonly")
    _output_combo["values"] = output_device_labels
    _output_combo.bind("<<ComboboxSelected>>", _output_combo_changed)
    _output_combo.pack(fill="x")

    input_label = ttk.Label(root, text="Capture return:")
    input_label.pack()

    global _input_combo
    _input_combo = ttk.Combobox(root, state="readonly")
    _input_combo["values"] = input_device_labels
    _input_combo.bind("<<ComboboxSelected>>", _input_combo_changed)
    _input_combo.pack(fill="x")

    _spacer_label_1 = ttk.Label(root, text="")
    _spacer_label_1.pack()

    begin_button = ttk.Button(root, text="Begin Capture")
    begin_button.pack()

    return root
