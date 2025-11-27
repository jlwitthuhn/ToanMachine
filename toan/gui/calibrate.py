import tkinter as tk
from tkinter import ttk

INSTRUCTIONS = [
    "First, you must calibrate your input gain to ensure your interface is recording as loud as possible without clipping.",
    "",
    "When you click play below, a tone will be played for several seconds. To ensure a lack of clipping you will want to adjust your interface's input gain so the receive volume is as close as possible to the send volume without exceeding it.",
]


class VolumeCalibrationDialog(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.geometry("400x250")
        self.title("Calibration")

        mainframe = ttk.Frame(self)
        mainframe.pack(fill="both", expand=True, padx=10, pady=10)

        instructions_area = tk.Text(mainframe, height=4, width=40, wrap="word")
        instructions_area.insert(tk.END, "\n".join(INSTRUCTIONS))
        instructions_area.configure(state="disabled")
        instructions_area.pack(fill="both", expand=True)

        test_button = ttk.Button(mainframe, text="Play test signal")
        test_button.pack()

        output_label = ttk.Label(mainframe, text="Send volume: 0.000")
        output_label.pack()

        input_label = ttk.Label(mainframe, text="Receive volume: 0.000")
        input_label.pack()

        self.transient(parent)
        self.grab_set()
        self.wait_window(self)
