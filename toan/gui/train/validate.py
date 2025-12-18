# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only
import threading
import zipfile

from PySide6 import QtWidgets

from toan.gui.train import TrainingContext


class TrainValidatePage(QtWidgets.QWizardPage):
    context: TrainingContext
    text_edit: QtWidgets.QTextEdit

    def __init__(self, parent, context: TrainingContext):
        super().__init__(parent)
        self.context = context

        self.setTitle("Checking Input")
        layout = QtWidgets.QVBoxLayout(self)

        self.text_edit = QtWidgets.QTextEdit()
        self.text_edit.setReadOnly(True)
        layout.addWidget(self.text_edit)

    def initializePage(self):
        self.text_edit.clear()
        self.text_edit.append(f"Analyzing input file: {self.context.input_path}")
        threading.Thread(target=self._run_thread).start()

    def _run_thread(self):
        self.text_edit.append("Loading as zip archive...")
        try:
            with zipfile.ZipFile(self.context.input_path, "r") as zip_file:
                self.text_edit.append("Loading file: config.json")
                try:
                    config_json_bytes = zip_file.read("config.json")
                except KeyError:
                    self.text_edit.append("Error: config.json not found")
        except zipfile.BadZipFile:
            self.text_edit.append("Error: File is not a valid zip archive")
        except:
            self.text_edit.append("Error: Unknown error occurred")
