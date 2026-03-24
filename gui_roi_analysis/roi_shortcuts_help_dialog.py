# -*- coding: utf-8 -*-
"""Floating help dialog listing ROI definition keyboard shortcuts."""

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QTextEdit


SHORTCUTS_TEXT = """ROI window shortcuts (when ROI tool is open)

F1 — Analysis Complete
F2 — Previous frame
F3 — Next frame
F4 — Review mode on/off
F5 — (reserved)
F6 — Cancel, then Launch All on previous set
F7 — (reserved)
F8 — Cancel, then Launch All on next set
F9 — Reject set, then Launch All on next set
Esc — Cancel (same as closing without save). Stops Chain sets for further auto-opens.

Arrow keys — Nudge ROI by 1 pixel (Define mode only)
"""


class RoiKeyboardShortcutsHelpDialog(QDialog):
    """Non-modal help window; parent is the file-selection GUI for lifetime."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ROI keyboard shortcuts")
        self.setModal(False)
        self.setWindowFlags(
            (self.windowFlags() | Qt.Tool | Qt.WindowStaysOnTopHint)
            & ~Qt.WindowContextHelpButtonHint
        )
        layout = QVBoxLayout(self)
        text = QTextEdit()
        text.setReadOnly(True)
        text.setPlainText(SHORTCUTS_TEXT.strip())
        text.setMinimumWidth(420)
        text.setMinimumHeight(320)
        layout.addWidget(text)
