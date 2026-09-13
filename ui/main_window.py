"""The always-on-top assistant window (PySide6).

Why PySide6 over Tkinter
------------------------
Everything in this app happens on a background thread, and Qt's queued
signal/slot connections are a correct, boring way to hand data to the GUI
thread. Tkinter has no equivalent - you end up polling with `after()` and
hoping. Qt also gives us rich text with code blocks, a real always-on-top flag,
and per-widget stylesheets, all of which would be work elsewhere.

Two rules keep the window responsive:

1. **Nothing blocking runs here.** The UI only ever reads events.
2. **Streaming text is buffered and flushed on a timer** (60 ms). Re-rendering
   a QTextBrowser on every token would spend more time laying out text than the
   model spends generating it.
"""

from __future__ import annotations

import logging

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from audio.devices import AudioDeviceError, list_devices
from config.settings import LOG_DIR, Settings
from core import events as ev
from core.pipeline import Pipeline

from .widgets import APP_STYLESHEET, COLORS, LevelMeter, MutedLabel, StatusPill, section_label

log = logging.getLogger(__name__)

# Models offered in the dropdown. The list is editable, so anything the account
# has access to can be typed in directly.
SUGGESTED_MODELS = {
    "openai": ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1"],
    "anthropic": ["claude-haiku-4-5-20251001", "claude-sonnet-5", "claude-opus-5"],
}

RENDER_INTERVAL_MS = 60


class MainWindow(QWidget):
    """Signals exist so background threads never touch a widget directly."""

    sig_status = Signal(object)
    sig_partial = Signal(object)
    sig_final = Signal(object)
    sig_question = Signal(object)
    sig_answer_started = Signal(object)
    sig_answer_chunk = Signal(object)
    sig_answer_done = Signal(object)
    sig_latency = Signal(object)
    sig_error = Signal(object)
    sig_level = Signal(object)

    def __init__(self, settings: Settings, pipeline: Pipeline) -> None:
        super().__init__()
        self.settings = settings
        self.pipeline = pipeline
        self._answer_buffer = ""      # text waiting to be painted
        self._answer_text = ""        # everything received for this answer
        self._streaming = False
        self._provisional = False     # answering a question that is not finished
        self._devices = []

        self._build_ui()
        self._wire_events()
        self._install_shortcuts()
        self._install_global_hotkeys()
        self.apply_font_size(settings.font_size)

        self._render_timer = QTimer(self)
        self._render_timer.setInterval(RENDER_INTERVAL_MS)
        self._render_timer.timeout.connect(self._flush_answer)
        self._render_timer.start()

        for warning in settings.warnings:
            self._append_notice(warning)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        self.setObjectName("root")
        self.setWindowTitle("Real-Time AI Assistant")
        self.setStyleSheet(APP_STYLESHEET)
        self.resize(460, 620)
        self.setMinimumSize(360, 320)
        if self.settings.always_on_top:
            self.setWindowFlag(Qt.WindowStaysOnTopHint, True)

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 10, 12, 10)
        root.setSpacing(8)

        # -- header --------------------------------------------------------
        header = QHBoxLayout()
        title = QLabel("REAL-TIME AI ASSISTANT")
        title.setStyleSheet(
            "font-size: 11px; font-weight: 800; letter-spacing: 1.4px; color: %s;"
            % COLORS["text"]
        )
        self.status_pill = StatusPill()
        header.addWidget(title)
        header.addStretch(1)
        header.addWidget(self.status_pill)
        root.addLayout(header)

        self.level_meter = LevelMeter()
        root.addWidget(self.level_meter)

        # -- controls row 1 ------------------------------------------------
        controls = QHBoxLayout()
        controls.setSpacing(6)
        self.start_button = QPushButton("Start")
        self.start_button.setObjectName("primary")
        self.start_button.setProperty("listening", "false")
        self.start_button.clicked.connect(self.toggle_listening)
        controls.addWidget(self.start_button)

        self.device_combo = QComboBox()
        self.device_combo.setToolTip(
            "Pick a 'System Audio' entry to hear the other person in the call."
        )
        self.device_combo.setMinimumWidth(150)
        controls.addWidget(self.device_combo, 1)

        self.refresh_button = QPushButton("⟳")
        self.refresh_button.setFixedWidth(30)
        self.refresh_button.setToolTip("Re-scan audio devices")
        self.refresh_button.clicked.connect(self.reload_devices)
        controls.addWidget(self.refresh_button)
        root.addLayout(controls)

        # -- controls row 2 ------------------------------------------------
        controls2 = QHBoxLayout()
        controls2.setSpacing(6)

        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        self.model_combo.addItems(SUGGESTED_MODELS.get(self.settings.llm_provider, []))
        self.model_combo.setCurrentText(self.settings.llm_model)
        self.model_combo.currentTextChanged.connect(self.pipeline.set_model)
        self.model_combo.setToolTip("LLM model (LLM_MODEL in .env)")
        controls2.addWidget(self.model_combo, 2)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["SHORT", "NORMAL", "DETAILED"])
        self.mode_combo.setCurrentText(self.settings.answer_mode)
        self.mode_combo.currentTextChanged.connect(self.pipeline.set_answer_mode)
        self.mode_combo.setToolTip("Answer length (Ctrl+Shift+S cycles)")
        controls2.addWidget(self.mode_combo, 1)

        self.font_spin = QSpinBox()
        self.font_spin.setRange(9, 28)
        self.font_spin.setValue(self.settings.font_size)
        self.font_spin.setSuffix(" px")
        self.font_spin.valueChanged.connect(self.apply_font_size)
        self.font_spin.setToolTip("Font size")
        controls2.addWidget(self.font_spin)
        root.addLayout(controls2)

        # -- question ------------------------------------------------------
        root.addWidget(section_label("QUESTION"))
        self.question_label = QLabel("Waiting for a question...")
        self.question_label.setWordWrap(True)
        self.question_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        root.addWidget(self.question_label)

        self.partial_label = MutedLabel("")
        root.addWidget(self.partial_label)

        # -- answer --------------------------------------------------------
        root.addWidget(section_label("ANSWER"))
        self.answer_view = QTextBrowser()
        self.answer_view.setOpenExternalLinks(True)
        self.answer_view.setPlaceholderText(
            "Answers stream in here.\n\n"
            "Press Start, then play some audio. Or type a question below to test "
            "without audio."
        )
        root.addWidget(self.answer_view, 1)

        # -- manual entry --------------------------------------------------
        self.manual_input = QLineEdit()
        self.manual_input.setPlaceholderText("Type a question and press Enter...")
        self.manual_input.returnPressed.connect(self._submit_manual)
        root.addWidget(self.manual_input)

        # -- footer --------------------------------------------------------
        footer = QHBoxLayout()
        self.latency_label = MutedLabel("Latency: -")
        footer.addWidget(self.latency_label, 1)

        self.clear_button = QPushButton("Clear")
        self.clear_button.setToolTip("Clear the answer and conversation memory (Ctrl+Shift+C)")
        self.clear_button.clicked.connect(self.clear_conversation)
        footer.addWidget(self.clear_button)

        self.settings_button = QPushButton("Settings")
        self.settings_button.clicked.connect(self.show_settings)
        footer.addWidget(self.settings_button)
        root.addLayout(footer)

        self.reload_devices()

    # ------------------------------------------------------------------
    # Event plumbing
    # ------------------------------------------------------------------
    def _wire_events(self) -> None:
        self.sig_status.connect(self._on_status)
        self.sig_partial.connect(self._on_partial)
        self.sig_final.connect(self._on_final)
        self.sig_question.connect(self._on_question)
        self.sig_answer_started.connect(self._on_answer_started)
        self.sig_answer_chunk.connect(self._on_answer_chunk)
        self.sig_answer_done.connect(self._on_answer_done)
        self.sig_latency.connect(self._on_latency)
        self.sig_error.connect(self._on_error)
        self.sig_level.connect(self._on_level)

        routes = {
            ev.StatusEvent: self.sig_status,
            ev.PartialTranscript: self.sig_partial,
            ev.FinalTranscript: self.sig_final,
            ev.QuestionDetected: self.sig_question,
            ev.AnswerStarted: self.sig_answer_started,
            ev.AnswerChunk: self.sig_answer_chunk,
            ev.AnswerCompleted: self.sig_answer_done,
            ev.LatencyReport: self.sig_latency,
            ev.ErrorEvent: self.sig_error,
            ev.AudioLevel: self.sig_level,
        }

        def dispatch(event) -> None:
            # Runs on a worker thread. Emitting a Qt signal from here is safe:
            # Qt queues it onto the GUI thread for us.
            signal = routes.get(type(event))
            if signal is not None:
                signal.emit(event)

        self._unsubscribe = self.pipeline.bus.subscribe(dispatch)

    def _install_shortcuts(self) -> None:
        """Window-local shortcuts (work when the window has focus)."""
        for keys, handler in (
            ("Ctrl+Shift+Space", self.toggle_listening),
            ("Ctrl+Shift+C", self.clear_conversation),
            ("Ctrl+Shift+H", self.toggle_visibility),
            ("Ctrl+Shift+S", self.cycle_answer_mode),
            ("Ctrl+Q", self.close),
        ):
            QShortcut(QKeySequence(keys), self, activated=handler)

    def _install_global_hotkeys(self) -> None:
        """System-wide hotkeys, so they work while Meet has focus.

        pynput is used rather than the `keyboard` package because it does not
        need administrator rights on Windows. If it is unavailable we degrade
        to the window-local shortcuts above rather than failing to start.
        """
        self._hotkey_listener = None
        if not self.settings.hotkeys_enabled:
            return
        try:
            from pynput import keyboard
        except Exception as exc:
            log.info("Global hotkeys unavailable (%s); window shortcuts still work", exc)
            return

        def on_main_thread(fn):
            # pynput callbacks arrive on its own thread.
            return lambda: QTimer.singleShot(0, fn)

        try:
            self._hotkey_listener = keyboard.GlobalHotKeys({
                "<ctrl>+<shift>+<space>": on_main_thread(self.toggle_listening),
                "<ctrl>+<shift>+c": on_main_thread(self.clear_conversation),
                "<ctrl>+<shift>+h": on_main_thread(self.toggle_visibility),
                "<ctrl>+<shift>+s": on_main_thread(self.cycle_answer_mode),
            })
            self._hotkey_listener.daemon = True
            self._hotkey_listener.start()
            log.info("Global hotkeys registered")
        except Exception as exc:
            log.warning("Could not register global hotkeys: %s", exc)
            self._hotkey_listener = None

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def reload_devices(self) -> None:
        self.device_combo.blockSignals(True)
        self.device_combo.clear()
        try:
            self._devices = list_devices()
            for device in self._devices:
                self.device_combo.addItem(device.label, device)
            # Pre-select whatever .env asked for.
            spec = (self.settings.audio_device or "").strip().lower()
            if spec:
                for i, device in enumerate(self._devices):
                    if spec == str(device.index) or spec in device.name.lower():
                        self.device_combo.setCurrentIndex(i)
                        break
        except AudioDeviceError as exc:
            self.device_combo.addItem("No audio devices found")
            self._append_notice(str(exc))
        finally:
            self.device_combo.blockSignals(False)

    def selected_device(self):
        data = self.device_combo.currentData()
        return data if data is not None else None

    def toggle_listening(self) -> None:
        if self.pipeline.running:
            self.pipeline.stop()
            self._set_listening_button(False)
        else:
            device = self.selected_device()
            if device is None:
                self._append_notice(
                    "No audio device selected. Press ⟳ to re-scan, or check "
                    "Windows sound settings."
                )
                return
            self.device_combo.setEnabled(False)
            self._set_listening_button(True)
            self.pipeline.start(device)

    def _set_listening_button(self, listening: bool) -> None:
        self.start_button.setText("Stop" if listening else "Start")
        self.start_button.setProperty("listening", "true" if listening else "false")
        self.device_combo.setEnabled(not listening)
        # Re-apply the stylesheet so the property selector takes effect.
        self.start_button.style().unpolish(self.start_button)
        self.start_button.style().polish(self.start_button)

    def clear_conversation(self) -> None:
        self.pipeline.clear_conversation()
        self._answer_buffer = ""
        self._answer_text = ""
        self._streaming = False
        self.answer_view.clear()
        self.question_label.setText("Waiting for a question...")
        self.partial_label.setText("")
        self.latency_label.setText("Latency: -")

    def toggle_visibility(self) -> None:
        if self.isVisible():
            self.hide()
        else:
            self.show()
            self.raise_()
            self.activateWindow()

    def cycle_answer_mode(self) -> None:
        index = (self.mode_combo.currentIndex() + 1) % self.mode_combo.count()
        self.mode_combo.setCurrentIndex(index)

    def apply_font_size(self, size: int) -> None:
        self.settings.font_size = size
        self.question_label.setStyleSheet(
            "font-size: %dpx; font-weight: 600; color: %s;" % (size + 1, COLORS["text"])
        )
        self.answer_view.setStyleSheet(
            "QTextBrowser { background: %s; border: 1px solid %s; border-radius: 8px;"
            " padding: 8px; color: %s; font-size: %dpx; }"
            % (COLORS["panel"], COLORS["border"], COLORS["text"], size)
        )
        self.partial_label.apply_font_size(size)
        self.latency_label.apply_font_size(size)

    def show_settings(self) -> None:
        s = self.settings
        QMessageBox.information(
            self,
            "Settings",
            "Settings live in the .env file next to app.py.\n\n"
            "Current configuration\n"
            "  LLM:          %s / %s\n"
            "  Speech:       %s / %s\n"
            "  Audio mode:   %s\n"
            "  Answer mode:  %s\n"
            "  Context:      last %d exchanges\n"
            "  Speculative:  %s\n"
            "  End-of-speech silence: %.2f s\n\n"
            "Hotkeys\n"
            "  Ctrl+Shift+Space  start / stop\n"
            "  Ctrl+Shift+C      clear\n"
            "  Ctrl+Shift+H      hide / show\n"
            "  Ctrl+Shift+S      cycle answer mode\n\n"
            "Logs: %s"
            % (
                s.llm_provider, s.llm_model,
                s.stt_provider, s.stt_model,
                s.audio_mode, s.answer_mode, s.context_turns,
                "on" if s.speculative_start else "off",
                s.end_of_utterance_silence,
                LOG_DIR / "app.log",
            ),
        )

    def _submit_manual(self) -> None:
        text = self.manual_input.text().strip()
        if not text:
            return
        self.manual_input.clear()
        self.pipeline.submit_text(text, force=True)

    # ------------------------------------------------------------------
    # Slots (GUI thread)
    # ------------------------------------------------------------------
    def _on_status(self, event: ev.StatusEvent) -> None:
        self.status_pill.set_status(event.status)
        if event.detail:
            self.setWindowTitle("Real-Time AI Assistant - %s" % event.detail[:60])
        if event.status == "idle":
            self._set_listening_button(False)

    def _on_level(self, event: ev.AudioLevel) -> None:
        self.level_meter.update_level(event.rms_db, event.noise_floor_db, event.is_speech)

    def _on_partial(self, event: ev.PartialTranscript) -> None:
        self.partial_label.setText("… %s" % event.text)

    def _on_final(self, event: ev.FinalTranscript) -> None:
        self.partial_label.setText("heard: %s" % event.text)

    def _on_question(self, event: ev.QuestionDetected) -> None:
        self._provisional = event.speculative
        self.question_label.setText(event.text)
        # An early answer is a guess about a question that is not finished yet.
        # Say so, rather than letting it read as settled and then swapping it.
        if event.speculative:
            self.question_label.setStyleSheet(
                "font-size: %dpx; font-weight: 600; color: %s;"
                % (self.settings.font_size + 1, COLORS["amber"])
            )
            self.partial_label.setText("answering early - may update when they finish")
        else:
            self.apply_font_size(self.settings.font_size)
            self.partial_label.setText("")

    def _on_answer_started(self, event: ev.AnswerStarted) -> None:
        self._answer_buffer = ""
        self._answer_text = ""
        self._streaming = True
        self.answer_view.clear()

    def _on_answer_chunk(self, event: ev.AnswerChunk) -> None:
        # Buffered: painted by _flush_answer on the render timer.
        self._answer_buffer += event.text

    def _on_answer_done(self, event: ev.AnswerCompleted) -> None:
        self._flush_answer()
        self._streaming = False
        if event.cancelled:
            # The speaker changed the question mid-answer. Clear the stale text
            # immediately so it cannot be misread as the answer to what they
            # actually asked - the replacement is already on its way.
            self.answer_view.clear()
            self._answer_text = ""
            self._answer_buffer = ""
            self.partial_label.setText("question changed - re-answering")
            return
        if self._provisional:
            self._provisional = False
            self.apply_font_size(self.settings.font_size)
            self.partial_label.setText("")
        if event.full_text.strip():
            # Only now do we pay for markdown layout - code blocks, bullets and
            # emphasis all appear at once when the answer settles.
            self.answer_view.setMarkdown(event.full_text)
            self._scroll_to_bottom()

    def _on_latency(self, event: ev.LatencyReport) -> None:
        self.latency_label.setText("Latency:  %s" % event.as_line())

    def _on_error(self, event: ev.ErrorEvent) -> None:
        self._append_notice(event.message)
        if event.fatal:
            self._set_listening_button(False)

    # ------------------------------------------------------------------
    def _flush_answer(self) -> None:
        if not self._answer_buffer:
            return
        chunk, self._answer_buffer = self._answer_buffer, ""
        self._answer_text += chunk
        # Plain text while streaming: appending is O(chunk) whereas re-rendering
        # markdown is O(whole answer) and would stutter on every token.
        cursor = self.answer_view.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(chunk)
        self._scroll_to_bottom()

    def _scroll_to_bottom(self) -> None:
        bar = self.answer_view.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _append_notice(self, message: str) -> None:
        log.warning("UI notice: %s", message)
        self.partial_label.setText("⚠ %s" % message)

    # ------------------------------------------------------------------
    def closeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        try:
            if getattr(self, "_hotkey_listener", None) is not None:
                self._hotkey_listener.stop()
        except Exception:
            pass
        try:
            self._unsubscribe()
        except Exception:
            pass
        self.pipeline.shutdown()
        super().closeEvent(event)


def run_gui(settings: Settings, pipeline: Pipeline, autostart: bool = False) -> int:
    app = QApplication.instance() or QApplication([])
    app.setApplicationName("Real-Time AI Assistant")
    window = MainWindow(settings, pipeline)
    window.show()
    if autostart:
        QTimer.singleShot(300, window.toggle_listening)
    return app.exec()
