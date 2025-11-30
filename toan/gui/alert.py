import tkinter as tk
from tkinter import ttk


class AlertDialog(tk.Toplevel):
    def __init__(self, parent, title: str, message: str):
        super().__init__(parent)

        self.title(title)

        mainframe = ttk.Frame(self)
        mainframe.pack(fill="both", expand=True, padx=10, pady=10)

        label_message = ttk.Label(
            mainframe, text=message, wraplength=300, justify="left"
        )
        label_message.pack()

        button_close = ttk.Button(
            mainframe, text="Close", command=lambda: self.destroy()
        )
        button_close.pack()

        # Center the alert
        self.update_idletasks()
        w = self.winfo_width()
        h = self.winfo_height()
        x = (self.winfo_screenwidth() // 2) - (w // 2)
        y = (self.winfo_screenheight() // 2) - (h // 2)
        self.geometry(f"{w}x{h}+{x}+{y}")

        self.transient(parent)
        self.grab_set()
        self.wait_window(self)
