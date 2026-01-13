# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

from PySide6 import QtWidgets

from toan.gui.record import RecordingContext


class RecordOutputLevelPage(QtWidgets.QWizardPage):
    context: RecordingContext

    def __init__(self, parent, context: RecordingContext):
        super().__init__(parent)

        self.setTitle("Output Level")
