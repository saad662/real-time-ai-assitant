"""Small reusable UI pieces.

Kept separate from main_window.py purely so the window file stays readable.
None of these do any work of their own - they are told what to show.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import QLabel, QSizePolicy, QWidget

# One place for every colour in the app.
COLORS = {
    "bg": "#14161a",
    "panel": "#1c1f26",
    "border": "#2a2f3a",
    "text": "#e6e9ef",
    "muted": "#8b93a7",
    "accent": "#4da3ff",
    "green": "#3ecf8e",
    "amber": "#f0b429",
    "red": "#ff6b6b",
}

STATUS_STYLES = {
    "idle":      ("OFF",        COLORS["muted"]),
    "listening": ("LISTENING",  COLORS["green"]),
    "speech":    ("SPEECH",     COLORS["accent"]),
    "thinking":  ("THINKING",   COLORS["amber"]),
    "answering": ("ANSWERING",  COLORS["accent"]),
    "ready":     ("READY",      COLORS["green"]),
    "error":     ("ERROR",      COLORS["red"]),
}


class StatusPill(QLabel):
    """The coloured dot + word in the header."""

    def __init__(self, parent: QWidget = None) -> None:
        super().__init__(parent)
        self.setAlignment(Qt.AlignVCenter | Qt.AlignRight)
        self.set_status("idle")

    def set_status(self, status: str) -> None:
        label, color = STATUS_STYLES.get(status, STATUS_STYLES["idle"])
        self.setText("● %s" % label)
        self.setStyleSheet(
            "color: %s; font-weight: 600; letter-spacing: 0.5px;" % color
        )


class LevelMeter(QWidget):
    """A thin bar showing input level against the VAD threshold.

    This is the fastest way to diagnose "it isn't hearing anything": if the bar
    never moves while audio plays, the wrong device is selected.
    """

    def __init__(self, parent: QWidget = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(6)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._level = 0.0       # 0..1
        self._threshold = 0.3   # 0..1
        self._speech = False

    def update_level(self, rms_db: float, noise_floor_db: float, is_speech: bool) -> None:
        # Map -70..-10 dBFS onto 0..1; anything outside clamps.
        self._level = max(0.0, min(1.0, (rms_db + 70.0) / 60.0))
        self._threshold = max(0.0, min(1.0, (noise_floor_db + 70.0) / 60.0))
        self._speech = is_speech
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(COLORS["border"]))
        width = int(self.width() * self._level)
        color = QColor(COLORS["accent"] if self._speech else COLORS["muted"])
        painter.fillRect(0, 0, width, self.height(), color)
        marker = int(self.width() * self._threshold)
        painter.fillRect(marker, 0, 1, self.height(), QColor(COLORS["amber"]))
        painter.end()


class MutedLabel(QLabel):
    """Secondary text: partial transcripts, latency, hints."""

    def __init__(self, text: str = "", parent: QWidget = None, size_delta: int = -1) -> None:
        super().__init__(text, parent)
        self._size_delta = size_delta
        self.setWordWrap(True)
        self.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.apply_font_size(13)

    def apply_font_size(self, base: int) -> None:
        self.setStyleSheet(
            "color: %s; font-size: %dpx;" % (COLORS["muted"], max(9, base + self._size_delta))
        )


def section_label(text: str) -> QLabel:
    label = QLabel(text)
    label.setStyleSheet(
        "color: %s; font-size: 10px; font-weight: 700; letter-spacing: 1.2px;"
        % COLORS["muted"]
    )
    return label


APP_STYLESHEET = """
QWidget#root {{
    background: {bg};
}}
QWidget {{
    color: {text};
    font-family: "Segoe UI", "Inter", sans-serif;
}}
QFrame#panel {{
    background: {panel};
    border: 1px solid {border};
    border-radius: 8px;
}}
QPushButton {{
    background: {panel};
    border: 1px solid {border};
    border-radius: 6px;
    padding: 5px 12px;
    color: {text};
}}
QPushButton:hover {{
    border-color: {accent};
}}
QPushButton:pressed {{
    background: {border};
}}
QPushButton#primary {{
    background: {accent};
    border-color: {accent};
    color: #0b1220;
    font-weight: 600;
}}
QPushButton#primary[listening="true"] {{
    background: {red};
    border-color: {red};
    color: #1a0b0b;
}}
QComboBox, QSpinBox {{
    background: {panel};
    border: 1px solid {border};
    border-radius: 6px;
    padding: 4px 8px;
    color: {text};
}}
QComboBox QAbstractItemView {{
    background: {panel};
    border: 1px solid {border};
    selection-background-color: {accent};
    selection-color: #0b1220;
}}
QTextBrowser, QLineEdit {{
    background: {panel};
    border: 1px solid {border};
    border-radius: 8px;
    padding: 8px;
    color: {text};
}}
QScrollBar:vertical {{
    background: transparent;
    width: 8px;
    margin: 2px;
}}
QScrollBar::handle:vertical {{
    background: {border};
    border-radius: 4px;
    min-height: 24px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}
""".format(**COLORS)
