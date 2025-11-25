import tkinter as tk
from tkinter import ttk

import sounddevice as sd

from toan.generate import generate_capture_signal

THE_SAMPLE_RATE = 44100


def _play_test_signal() -> None:
    signal = generate_capture_signal(THE_SAMPLE_RATE)
    sd.play(signal, THE_SAMPLE_RATE)


def _create_debug_tab(notebook: ttk.Notebook) -> ttk.Frame:
    root = ttk.Frame(notebook)
    root.grid(column=0, row=0, sticky="NSEW")

    button_test_signal = ttk.Button(
        root, text="Play Test Signal", command=_play_test_signal
    )
    button_test_signal.grid(column=0, row=0, sticky="EW")
    return root


def _create_capture_tab(notebook: ttk.Notebook) -> ttk.Frame:
    root = ttk.Frame(notebook)
    return root


def run_gui() -> None:
    root = tk.Tk()
    root.geometry("300x300")
    root.title("Toan Machine")

    mainframe = ttk.Frame(root)
    mainframe.pack()

    notebook = ttk.Notebook(mainframe)

    tab_debug = _create_debug_tab(notebook)
    notebook.add(tab_debug, text="Debug")

    tab_capture = _create_capture_tab(notebook)
    notebook.add(tab_capture, text="Capture")

    notebook.pack(fill="both", expand=True)

    root.lift()
    root.mainloop()
