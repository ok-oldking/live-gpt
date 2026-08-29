from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from PySide6.QtCore import (
    QEvent,
    QPoint,
    QRect,
    QSettings,
    QSize,
    QTimer,
    Signal,
    Qt,
)
from PySide6.QtGui import (
    QAction,
    QCloseEvent,
    QColor,
    QCursor,
    QIcon,
    QKeySequence,
    QMouseEvent,
    QPalette,
    QTextCursor,
)
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QFrame,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QKeySequenceEdit,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSystemTrayIcon,
    QVBoxLayout,
    QWidget,
)

from .browser import (
    BrowserMonitor,
    discover_cdp_endpoint,
    open_remote_debugging_settings,
)
from .logger import Logger, config_logger, shutdown_logger
from .hotkeys import GlobalHotkeyMonitor, HotkeyBinding
from .screen_capture import CaptureSource, capture_webp, list_capture_sources
from .window_focus import ForegroundWindowRestorer


ASSET_DIRECTORY = Path(__file__).resolve().parent / "assets"
ICON_PATH = ASSET_DIRECTORY / "app-icon.ico"
MICROPHONE_ICON_PATH = ASSET_DIRECTORY / "microphone.svg"
HIDE_ICON_PATH = ASSET_DIRECTORY / "hide.svg"
EXIT_ICON_PATH = ASSET_DIRECTORY / "exit.svg"
SEND_ICON_PATH = ASSET_DIRECTORY / "send.svg"
SETTINGS_ICON_PATH = ASSET_DIRECTORY / "settings.svg"
LOCK_ICON_PATH = ASSET_DIRECTORY / "lock.svg"
UNLOCK_ICON_PATH = ASSET_DIRECTORY / "unlock.svg"
logger = Logger.get_logger(__name__)

DEFAULT_HOLD_MIC_HOTKEY = "CapsLock"
DEFAULT_SEND_HOTKEY = "Ctrl+S"
DEFAULT_SEND_WITHOUT_SCREENSHOT_HOTKEY = "Ctrl+D"
HOTKEY_SETTING_KEYS = {
    "hold": "hotkeys/hold_microphone",
    "send": "hotkeys/send",
    "send_without_screenshot": "hotkeys/send_without_screenshot",
}


class HotkeyConfigDialog(QDialog):
    """Edit the pass-through global shortcuts used by the overlay."""

    def __init__(
        self,
        hold_microphone: QKeySequence,
        send: QKeySequence,
        send_without_screenshot: QKeySequence,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Live GPT configuration")
        self.setMinimumWidth(430)

        self.hold_microphone_edit = self._sequence_edit(hold_microphone)
        self.send_edit = self._sequence_edit(send)
        self.send_without_screenshot_edit = self._sequence_edit(
            send_without_screenshot
        )

        form = QFormLayout()
        form.addRow("Hold microphone:", self.hold_microphone_edit)
        form.addRow("Send:", self.send_edit)
        form.addRow(
            "Send without screenshot:",
            self.send_without_screenshot_edit,
        )

        note = QLabel(
            "These shortcuts work globally and are still passed to the "
            "foreground program."
        )
        note.setWordWrap(True)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(note)
        layout.addWidget(buttons)

    @staticmethod
    def _sequence_edit(sequence: QKeySequence) -> QKeySequenceEdit:
        edit = QKeySequenceEdit(sequence)
        edit.setMaximumSequenceLength(1)
        return edit

    def sequences(self) -> dict[str, QKeySequence]:
        return {
            "hold": self.hold_microphone_edit.keySequence(),
            "send": self.send_edit.keySequence(),
            "send_without_screenshot": (
                self.send_without_screenshot_edit.keySequence()
            ),
        }

    def bindings(self) -> dict[str, HotkeyBinding]:
        bindings = {
            name: HotkeyBinding.from_sequence(sequence)
            for name, sequence in self.sequences().items()
        }
        texts = [binding.text.casefold() for binding in bindings.values()]
        if len(set(texts)) != len(texts):
            raise ValueError("Each action must use a different hotkey")
        return bindings

    def accept(self) -> None:
        try:
            self.bindings()
        except ValueError as error:
            QMessageBox.warning(self, "Invalid hotkey", str(error))
            return
        super().accept()


class TranscriptEditor(QPlainTextEdit):
    """Editable transcript with contextual actions inside the input."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._response_mode = False
        self._response_complete = False
        self._full_response_text = ""
        self._screenshot_selected = False

        self.clear_button = QPushButton(self)
        self.clear_button.setObjectName("clearButton")
        self.clear_button.setIcon(QIcon(str(EXIT_ICON_PATH)))
        self.clear_button.setIconSize(QSize(16, 16))
        self.clear_button.setFixedSize(32, 32)
        self.clear_button.setAccessibleName("Delete text")
        self.clear_button.setToolTip("Delete text")

        self.send_button = QPushButton(self)
        self.send_button.setObjectName("sendButton")
        self.send_button.setIcon(QIcon(str(SEND_ICON_PATH)))
        self.send_button.setIconSize(QSize(16, 16))
        self.send_button.setFixedSize(32, 32)
        self.send_button.setAccessibleName("Send")
        self.send_button.setToolTip("Send")

        self.send_without_screenshot_button = QPushButton(
            "No screenshot",
            self,
        )
        self.send_without_screenshot_button.setObjectName(
            "sendWithoutScreenshotButton"
        )
        self.send_without_screenshot_button.setFixedSize(108, 32)
        self.send_without_screenshot_button.setAccessibleName(
            "Send without screenshot"
        )
        self.send_without_screenshot_button.setToolTip(
            "Send the text without the selected screenshot"
        )

        self.textChanged.connect(self._sync_action_visibility)
        self._sync_action_visibility()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._position_action_buttons()

    def _position_action_buttons(self) -> None:
        margin = 8
        spacing = 6
        y = self.height() - self.send_button.height() - margin
        send_x = self.width() - self.send_button.width() - margin
        if self._screenshot_selected:
            no_screenshot_x = (
                send_x
                - self.send_without_screenshot_button.width()
                - spacing
            )
            clear_x = no_screenshot_x - self.clear_button.width() - spacing
            self.send_without_screenshot_button.move(no_screenshot_x, y)
        else:
            clear_x = send_x - self.clear_button.width() - spacing
        self.clear_button.move(clear_x, y)
        self.send_button.move(send_x, y)
        self.clear_button.raise_()
        self.send_without_screenshot_button.raise_()
        self.send_button.raise_()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if self._response_mode:
            if not self._response_complete:
                event.accept()
                return
            self.begin_composing()
        super().mousePressEvent(event)

    def keyPressEvent(self, event) -> None:  # noqa: N802
        is_enter = event.key() in (
            Qt.Key.Key_Return,
            Qt.Key.Key_Enter,
        )
        wants_newline = bool(
            event.modifiers() & Qt.KeyboardModifier.ShiftModifier
        )
        if is_enter and not wants_newline and not self._response_mode:
            if self.toPlainText().strip():
                self.send_button.click()
            event.accept()
            return
        super().keyPressEvent(event)

    @property
    def is_showing_response(self) -> bool:
        return self._response_mode

    def begin_response(self) -> None:
        self._response_mode = True
        self._response_complete = False
        self._full_response_text = ""
        self.setReadOnly(True)
        self.clear()
        self._sync_action_visibility()

    def update_response(self, text: str) -> None:
        if not self._response_mode:
            self.begin_response()
        if self.toPlainText() == text:
            return
        self._full_response_text = text
        self.setPlainText(text)
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.setTextCursor(cursor)

    def finish_response(self) -> None:
        self._response_complete = True

    def begin_reading(self) -> None:
        self._response_mode = True
        self._response_complete = False
        self.setReadOnly(True)
        self.clear()

    def finish_reading(self) -> None:
        self.setPlainText(self._full_response_text)
        self._response_complete = True

    def begin_composing(self) -> None:
        self._response_mode = False
        self._response_complete = False
        self._full_response_text = ""
        self.setReadOnly(False)
        self.clear()
        self._sync_action_visibility()

    def set_screenshot_selected(self, selected: bool) -> None:
        self._screenshot_selected = selected
        self._sync_action_visibility()
        self._position_action_buttons()

    def set_hint(self, message: str, *, error: bool = False) -> None:
        self.setPlaceholderText(message)
        color = QColor("#ff667a" if error else "#aeb9d5")
        palette = self.palette()
        for group in (
            QPalette.ColorGroup.Active,
            QPalette.ColorGroup.Inactive,
            QPalette.ColorGroup.Disabled,
        ):
            palette.setColor(group, QPalette.ColorRole.PlaceholderText, color)
        self.setPalette(palette)

    def _sync_action_visibility(self) -> None:
        has_text = bool(self.toPlainText().strip()) and not self._response_mode
        self.clear_button.setVisible(has_text)
        self.send_button.setVisible(has_text)
        self.send_without_screenshot_button.setVisible(
            has_text and self._screenshot_selected
        )
        right_margin = 190 if has_text and self._screenshot_selected else 76
        self.setViewportMargins(0, 0, right_margin if has_text else 0, 0)


class OverlayWindow(QMainWindow):
    exit_requested = Signal()
    hide_requested = Signal()
    dictation_requested = Signal()
    dictation_finish_requested = Signal()
    clear_requested = Signal()
    send_requested = Signal(str, object)
    configure_requested = Signal()
    open_remote_debugging_requested = Signal()
    chatgpt_tab_selected = Signal(str)
    chatgpt_connection_changed = Signal(bool)

    def __init__(self) -> None:
        super().__init__()
        logger.debug("Creating overlay window")
        self._drag_offset: QPoint | None = None
        self._position_locked = False
        self._resize_edges = Qt.Edges()
        self._resize_start_global: QPoint | None = None
        self._resize_start_geometry: QRect | None = None
        self._chrome_visible = False
        self._focus_restorer = ForegroundWindowRestorer()
        self.setWindowTitle("Live GPT")
        self.setMinimumSize(760, 180)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        self._focus_history_timer = QTimer(self)
        self._focus_history_timer.timeout.connect(
            self._focus_restorer.remember_foreground
        )
        self._focus_history_timer.start(75)

        container = QWidget(self)
        container.setObjectName("overlayContainer")
        container.setProperty("chromeVisible", False)
        container.setMouseTracking(True)
        container.installEventFilter(self)
        self._resize_surface = container
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(12, 12, 12, 12)

        self.panel = QFrame(container)
        self.panel.setObjectName("overlayPanel")
        self.panel.setProperty("chromeVisible", False)
        self.panel.setMouseTracking(True)
        self.panel.installEventFilter(self)
        panel_layout = QVBoxLayout(self.panel)
        panel_layout.setContentsMargins(20, 16, 16, 18)
        panel_layout.setSpacing(14)

        self.title_bar = QWidget(self.panel)
        self.title_bar.setObjectName("overlayTitleBar")
        title_layout = QHBoxLayout(self.title_bar)
        title_layout.setContentsMargins(0, 0, 0, 0)
        title = QLabel("Live GPT")
        title.setObjectName("overlayTitle")
        title_layout.addWidget(title)

        self.chatgpt_tab_combo = QComboBox()
        self.chatgpt_tab_combo.setObjectName("chatgptTabCombo")
        self.chatgpt_tab_combo.setAccessibleName("ChatGPT window")
        self.chatgpt_tab_combo.setMinimumWidth(220)
        self.chatgpt_tab_combo.setMaximumWidth(300)
        self.chatgpt_tab_combo.addItem("Looking for ChatGPT windows…")
        self.chatgpt_tab_combo.setEnabled(False)
        self.chatgpt_tab_combo.currentIndexChanged.connect(
            self._chatgpt_tab_changed
        )
        title_layout.addWidget(self.chatgpt_tab_combo, 1)

        self.capture_source_combo = QComboBox()
        self.capture_source_combo.setObjectName("captureSourceCombo")
        self.capture_source_combo.setAccessibleName("Screenshot source")
        self.capture_source_combo.setMinimumWidth(150)
        self.capture_source_combo.setMaximumWidth(220)
        self.capture_source_combo.addItem("No screenshot", None)
        self.capture_source_combo.setToolTip(
            "Choose a desktop or visible window to attach when sending"
        )
        self.capture_source_combo.currentIndexChanged.connect(
            self._capture_source_changed
        )
        title_layout.addWidget(self.capture_source_combo, 1)

        self.remote_debugging_button = QPushButton("Enable Debugging")
        self.remote_debugging_button.setObjectName("remoteDebuggingButton")
        self.remote_debugging_button.setAccessibleName(
            "Open remote debugging settings"
        )
        self.remote_debugging_button.setToolTip(
            "Open the browser's remote debugging settings"
        )
        self.remote_debugging_button.clicked.connect(
            self.open_remote_debugging_requested.emit
        )
        title_layout.addWidget(self.remote_debugging_button)

        title_layout.addStretch()

        self.configure_button = QPushButton()
        self.configure_button.setObjectName("configureButton")
        self._configure_icon_button(
            self.configure_button,
            SETTINGS_ICON_PATH,
            "Configure hotkeys",
        )
        self.configure_button.clicked.connect(self.configure_requested.emit)
        title_layout.addWidget(self.configure_button)

        self.lock_button = QPushButton()
        self.lock_button.setObjectName("lockButton")
        self.lock_button.setCheckable(True)
        self._configure_icon_button(
            self.lock_button,
            UNLOCK_ICON_PATH,
            "Lock overlay position",
        )
        self.lock_button.toggled.connect(self._set_position_locked)
        title_layout.addWidget(self.lock_button)

        self.hide_button = QPushButton()
        self.hide_button.setObjectName("hideButton")
        self._configure_icon_button(
            self.hide_button,
            HIDE_ICON_PATH,
            "Hide",
        )
        self.hide_button.clicked.connect(self.hide_requested.emit)
        title_layout.addWidget(self.hide_button)

        self.exit_button = QPushButton()
        self.exit_button.setObjectName("exitButton")
        self._configure_icon_button(
            self.exit_button,
            EXIT_ICON_PATH,
            "Exit",
        )
        self.exit_button.clicked.connect(self.exit_requested.emit)
        title_layout.addWidget(self.exit_button)

        self.transcript_area = TranscriptEditor()
        self.transcript_area.setObjectName("transcriptArea")
        self.transcript_area.setEnabled(False)
        self.transcript_area.set_hint("Looking for ChatGPT windows…")

        self.subtitle_panel = QFrame()
        self.subtitle_panel.setObjectName("subtitlePanel")
        subtitle_layout = QVBoxLayout(self.subtitle_panel)
        subtitle_layout.setContentsMargins(16, 8, 16, 8)
        subtitle_layout.setSpacing(0)
        self.subtitle_line_one = QLabel()
        self.subtitle_line_one.setObjectName("subtitleLine")
        self.subtitle_line_one.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        self.subtitle_line_one.setWordWrap(False)
        self.subtitle_line_two = QLabel()
        self.subtitle_line_two.setObjectName("subtitleLine")
        self.subtitle_line_two.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        self.subtitle_line_two.setWordWrap(False)
        self.subtitle_full_text = QPlainTextEdit()
        self.subtitle_full_text.setObjectName("subtitleFullText")
        self.subtitle_full_text.setReadOnly(True)
        self.subtitle_full_text.setMouseTracking(True)
        self.subtitle_full_text.hide()
        self._reading_full_text = ""
        self._reading_fraction = 0.0
        self._subtitle_line_index = -1
        self._subtitle_mode_active = False
        self._subtitle_dismissed = False
        self._subtitle_expanded = False
        self._subtitle_reading_active = False
        self._subtitle_status_text = ""
        self._subtitle_collapsed_geometry: QRect | None = None
        subtitle_layout.addWidget(self.subtitle_line_one, 1)
        subtitle_layout.addWidget(self.subtitle_line_two, 1)
        subtitle_layout.addWidget(self.subtitle_full_text, 1)
        self.subtitle_panel.hide()
        self._subtitle_hover_widgets = (
            self.subtitle_panel,
            self.subtitle_line_one,
            self.subtitle_line_two,
            self.subtitle_full_text,
            self.subtitle_full_text.viewport(),
        )
        for widget in self._subtitle_hover_widgets:
            widget.setMouseTracking(True)
            widget.installEventFilter(self)

        self.dictation_panel = QFrame()
        self.dictation_panel.setObjectName("dictationPanel")
        dictation_layout = QVBoxLayout(self.dictation_panel)
        dictation_layout.setContentsMargins(16, 8, 16, 8)
        self.dictation_state_label = QLabel()
        self.dictation_state_label.setObjectName("dictationState")
        self.dictation_state_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.dictation_state_label.setWordWrap(True)
        dictation_layout.addWidget(self.dictation_state_label, 1)
        self.dictation_panel.hide()

        self.microphone_button = QPushButton()
        self.microphone_button.setObjectName("microphoneButton")
        self.microphone_button.setProperty("recordingState", "idle")
        self.microphone_button.setIcon(QIcon(str(MICROPHONE_ICON_PATH)))
        self.microphone_button.setIconSize(QSize(26, 26))
        self.microphone_button.setFixedSize(56, 56)
        self.microphone_button.setAccessibleName("Hold to dictate")
        self.microphone_button.setEnabled(False)
        self.microphone_button.setToolTip(
            "Press and hold to use ChatGPT dictation"
        )
        self.microphone_button.pressed.connect(self.dictation_requested.emit)
        self.microphone_button.released.connect(
            self.dictation_finish_requested.emit
        )

        self.send_button = self.transcript_area.send_button
        self.send_button.clicked.connect(lambda: self._request_send(True))

        self.send_without_screenshot_button = (
            self.transcript_area.send_without_screenshot_button
        )
        self.send_without_screenshot_button.clicked.connect(
            self._request_send_without_screenshot
        )

        self.clear_button = self.transcript_area.clear_button
        self.clear_button.clicked.connect(self._request_clear)

        panel_layout.addWidget(self.title_bar)
        recording_layout = QHBoxLayout()
        recording_layout.setSpacing(16)
        recording_layout.addWidget(self.transcript_area, 1)
        recording_layout.addWidget(self.subtitle_panel, 1)
        recording_layout.addWidget(self.dictation_panel, 1)

        recording_layout.addWidget(
            self.microphone_button,
            0,
            Qt.AlignmentFlag.AlignVCenter,
        )

        panel_layout.addLayout(recording_layout, 1)
        container_layout.addWidget(self.panel)
        self.setCentralWidget(container)
        self.resize(1140, 240)

        self._title_opacity = QGraphicsOpacityEffect(self.title_bar)
        self.title_bar.setGraphicsEffect(self._title_opacity)
        self._microphone_opacity = QGraphicsOpacityEffect(
            self.microphone_button
        )
        self.microphone_button.setGraphicsEffect(self._microphone_opacity)

        self.setStyleSheet(
            """
            QWidget#overlayContainer {
                background: transparent;
            }
            QFrame#overlayPanel {
                background-color: rgba(12, 20, 48, 224);
                border: 1px solid rgba(66, 220, 255, 150);
                border-radius: 18px;
            }
            QFrame#overlayPanel[chromeVisible="false"] {
                background-color: transparent;
                border-color: transparent;
            }
            QLabel#overlayTitle {
                color: #f5f7ff;
                font-size: 20px;
                font-weight: 700;
            }
            QComboBox#chatgptTabCombo,
            QComboBox#captureSourceCombo {
                min-height: 34px;
                padding: 0 10px;
                color: #f5f7ff;
                background-color: rgba(5, 10, 28, 145);
                border: 1px solid rgba(130, 165, 230, 75);
                border-radius: 8px;
            }
            QComboBox#chatgptTabCombo:disabled,
            QComboBox#captureSourceCombo:disabled {
                color: rgba(228, 235, 255, 155);
            }
            QComboBox#chatgptTabCombo QAbstractItemView,
            QComboBox#captureSourceCombo QAbstractItemView {
                color: #f5f7ff;
                background-color: rgb(18, 28, 58);
                selection-background-color: rgb(38, 112, 145);
            }
            QPlainTextEdit#transcriptArea {
                color: #f5f7ff;
                background-color: rgba(5, 10, 28, 145);
                border: 1px solid rgba(130, 165, 230, 75);
                border-radius: 10px;
                padding: 8px;
                font-size: 15px;
                selection-background-color: rgba(76, 201, 240, 130);
            }
            QFrame#subtitlePanel,
            QFrame#dictationPanel {
                background-color: rgba(5, 10, 28, 145);
                border: 1px solid rgba(130, 165, 230, 75);
                border-radius: 10px;
            }
            QLabel#subtitleLine {
                color: #f5f7ff;
                background: transparent;
                border: none;
                font-size: 22px;
                font-weight: 600;
            }
            QPlainTextEdit#subtitleFullText {
                color: #f5f7ff;
                background: transparent;
                border: none;
                padding: 4px;
                font-size: 16px;
                selection-background-color: rgba(76, 201, 240, 130);
            }
            QLabel#dictationState {
                color: #f5f7ff;
                background: transparent;
                border: none;
                font-size: 20px;
                font-weight: 600;
            }
            QPushButton {
                min-height: 30px;
                padding: 0 12px;
                color: #f5f7ff;
                background-color: rgba(70, 88, 140, 125);
                border: 1px solid rgba(170, 195, 255, 90);
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: rgba(76, 201, 240, 150);
            }
            QPushButton#hideButton,
            QPushButton#configureButton,
            QPushButton#lockButton,
            QPushButton#exitButton {
                min-width: 36px;
                max-width: 36px;
                min-height: 36px;
                max-height: 36px;
                padding: 0;
                border-radius: 9px;
            }
            QPushButton#remoteDebuggingButton {
                min-height: 34px;
                max-height: 34px;
            }
            QPushButton#sendButton,
            QPushButton#clearButton,
            QPushButton#sendWithoutScreenshotButton {
                min-width: 32px;
                min-height: 32px;
                max-height: 32px;
                padding: 0;
                border-radius: 8px;
            }
            QPushButton#sendButton,
            QPushButton#clearButton {
                max-width: 32px;
            }
            QPushButton#exitButton:hover {
                background-color: rgba(239, 68, 88, 190);
            }
            QPushButton#lockButton:checked {
                background-color: rgba(35, 155, 116, 190);
                border-color: rgba(130, 255, 195, 190);
            }
            QPushButton#sendButton {
                background-color: rgba(35, 155, 116, 190);
            }
            QPushButton#sendButton:hover {
                background-color: rgba(40, 190, 140, 220);
            }
            QPushButton#sendWithoutScreenshotButton {
                min-width: 108px;
                max-width: 108px;
                background-color: rgba(38, 112, 145, 190);
            }
            QPushButton#sendWithoutScreenshotButton:hover {
                background-color: rgba(48, 145, 185, 220);
            }
            QPushButton#clearButton:hover {
                background-color: rgba(210, 116, 34, 190);
            }
            QPushButton#microphoneButton {
                min-width: 56px;
                max-width: 56px;
                min-height: 56px;
                max-height: 56px;
                padding: 0;
                background-color: rgba(48, 72, 128, 175);
                border-radius: 28px;
            }
            QPushButton#microphoneButton[recordingState="recording"] {
                color: white;
                background-color: rgba(220, 48, 72, 220);
                border-color: rgba(255, 150, 165, 220);
            }
            QPushButton#microphoneButton[recordingState="saved"] {
                background-color: rgba(34, 160, 105, 210);
                border-color: rgba(130, 255, 195, 190);
            }
            QPushButton#microphoneButton[recordingState="error"] {
                background-color: rgba(210, 116, 34, 215);
                border-color: rgba(255, 205, 130, 200);
            }
            """
        )
        self._set_chrome_visible(False)

    @staticmethod
    def _configure_icon_button(
        button: QPushButton,
        icon_path: Path,
        accessible_name: str,
    ) -> None:
        button.setIcon(QIcon(str(icon_path)))
        button.setIconSize(QSize(18, 18))
        button.setAccessibleName(accessible_name)
        button.setToolTip(accessible_name)

    def set_microphone_state(
        self,
        state: str,
        message: str | None = None,
    ) -> None:
        labels = {
            "idle": "Press and hold the microphone to dictate",
            "recording": "ChatGPT is listening… release to finish",
            "saved": "Dictation copied from ChatGPT",
            "error": "Browser dictation unavailable",
        }
        label = message or labels[state]
        self.set_status(label, error=state == "error")
        self.microphone_button.setProperty("recordingState", state)
        self.microphone_button.setAccessibleName(label)
        self.microphone_button.setToolTip(label)
        style = self.microphone_button.style()
        style.unpolish(self.microphone_button)
        style.polish(self.microphone_button)
        self.microphone_button.update()

    def set_status(self, message: str, *, error: bool = False) -> None:
        self.transcript_area.set_hint(message, error=error)

    def set_transcript(self, text: str) -> None:
        self.transcript_area.setPlainText(text)
        cursor = self.transcript_area.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.transcript_area.setTextCursor(cursor)

    def set_chatgpt_tabs(self, tabs: list[dict[str, str]]) -> None:
        selected_id = self.chatgpt_tab_combo.currentData()
        self.chatgpt_tab_combo.blockSignals(True)
        self.chatgpt_tab_combo.clear()
        if not tabs:
            self.chatgpt_tab_combo.addItem("No ChatGPT windows")
            self.chatgpt_tab_combo.setEnabled(False)
            self.remote_debugging_button.setVisible(True)
            self.microphone_button.setEnabled(False)
            self.transcript_area.setEnabled(False)
            self.set_status("Connect to a ChatGPT window to begin")
        else:
            for tab in tabs:
                self.chatgpt_tab_combo.addItem(tab["title"], tab["id"])
                index = self.chatgpt_tab_combo.count() - 1
                self.chatgpt_tab_combo.setItemData(
                    index,
                    tab["url"],
                    Qt.ItemDataRole.ToolTipRole,
                )
            self.chatgpt_tab_combo.setEnabled(True)
            self.remote_debugging_button.setVisible(False)
            self.microphone_button.setEnabled(True)
            self.transcript_area.setEnabled(True)
            self.set_status("Hold the microphone or enter a message")
            selected_index = self.chatgpt_tab_combo.findData(selected_id)
            self.chatgpt_tab_combo.setCurrentIndex(
                selected_index if selected_index >= 0 else 0
            )
        self.chatgpt_tab_combo.blockSignals(False)
        self.chatgpt_connection_changed.emit(bool(tabs))
        if tabs:
            self._chatgpt_tab_changed(self.chatgpt_tab_combo.currentIndex())

    def set_browser_status(self, status: str) -> None:
        self.chatgpt_tab_combo.setToolTip(status)
        status_lower = status.casefold()
        self.set_status(
            status,
            error=any(
                word in status_lower
                for word in ("not installed", "disconnected", "unable", "could not")
            ),
        )
        if not self.chatgpt_tab_combo.isEnabled():
            self.chatgpt_tab_combo.setItemText(0, status)

    def set_capture_sources(self, sources: list[CaptureSource]) -> None:
        selected = self.capture_source_combo.currentData()
        selected_key = selected.key if isinstance(selected, CaptureSource) else None
        self.capture_source_combo.blockSignals(True)
        self.capture_source_combo.clear()
        self.capture_source_combo.addItem("No screenshot", None)
        for source in sources:
            self.capture_source_combo.addItem(source.label, source)
        if selected_key is not None:
            for index in range(1, self.capture_source_combo.count()):
                source = self.capture_source_combo.itemData(index)
                if isinstance(source, CaptureSource) and source.key == selected_key:
                    self.capture_source_combo.setCurrentIndex(index)
                    break
        self.capture_source_combo.blockSignals(False)
        self._capture_source_changed(self.capture_source_combo.currentIndex())

    def set_send_result(
        self,
        success: bool,
        sent_text: str,
        message: str,
    ) -> None:
        self.send_button.setEnabled(True)
        self.send_without_screenshot_button.setEnabled(True)
        if not success:
            self.dismiss_subtitle_mode()
            self.set_transcript(sent_text)
            self.microphone_button.setVisible(True)
            self.set_status(message, error=True)
            return

        del sent_text, message
        if not self._subtitle_mode_active:
            self.begin_response_display()
        self._set_subtitle_status("Waiting for ChatGPT…")
        self.set_status("Waiting for ChatGPT…")

    def begin_response_display(
        self,
        sent_text: str = "",
        message: str = "Sending to ChatGPT…",
    ) -> None:
        del sent_text
        self._collapse_subtitle()
        self._subtitle_mode_active = True
        self._subtitle_dismissed = False
        self._subtitle_reading_active = False
        self._reading_full_text = ""
        self._reading_fraction = 0.0
        self._subtitle_line_index = -1
        self.transcript_area.begin_response()
        self.transcript_area.hide()
        self.dictation_panel.hide()
        self.subtitle_panel.show()
        self.microphone_button.setVisible(True)
        self._set_subtitle_status(message)
        self.set_status(message)

    def set_response_update(self, status: str, text: str) -> None:
        self.set_status(status)
        if self._subtitle_dismissed:
            return
        self.transcript_area.update_response(text)
        if not self._subtitle_mode_active:
            self.begin_response_display(message=status)
        self._reading_full_text = text
        self._reading_fraction = 1.0
        if text:
            self._render_reading_subtitle(resized=True, latest=True)
            self._update_expanded_subtitle()
        else:
            self._set_subtitle_status(status)

    def set_response_finished(self, success: bool, message: str) -> None:
        if self._subtitle_dismissed:
            return
        self.transcript_area.finish_response()
        self.microphone_button.setVisible(True)
        self.set_status(message, error=not success)
        if self._subtitle_mode_active and not self._reading_full_text:
            self._set_subtitle_status(message)

    def begin_reading(self, message: str) -> None:
        if self._subtitle_dismissed:
            return
        self.transcript_area.begin_reading()
        self._subtitle_reading_active = True
        self._reading_fraction = 0.0
        self._subtitle_line_index = -1
        self._subtitle_mode_active = True
        self.transcript_area.hide()
        self.subtitle_panel.show()
        self.microphone_button.setVisible(True)
        if not self._reading_full_text:
            self._set_subtitle_status(message)
        else:
            self._render_reading_subtitle(resized=True)
        self.set_status(message)

    def set_reading_subtitle(self, update: object) -> None:
        if isinstance(update, dict):
            self._reading_full_text = str(update.get("text") or "")
            try:
                self._reading_fraction = min(
                    max(float(update.get("fraction") or 0.0), 0.0),
                    1.0,
                )
            except (TypeError, ValueError):
                self._reading_fraction = 0.0
        else:
            self._reading_full_text = str(update or "")
            self._reading_fraction = 0.0
        if self._subtitle_dismissed:
            return
        self._render_reading_subtitle()
        self._update_expanded_subtitle()
        self.set_status("Reading aloud…")

    def _subtitle_lines(self) -> list[str]:
        words = self._reading_full_text.split()
        if not words:
            return []

        available_width = max(self.subtitle_panel.width() - 32, 1)
        metrics = self.subtitle_line_one.fontMetrics()
        lines: list[str] = []
        current = ""
        for word in words:
            candidate = f"{current} {word}".strip()
            if current and metrics.horizontalAdvance(candidate) > available_width:
                lines.append(current)
                current = word
            else:
                current = candidate
        if current:
            lines.append(current)
        return lines

    def _subtitle_index_at_progress(self, lines: list[str]) -> int:
        if not lines:
            return 0
        weights = [max(len(line), 12) for line in lines]
        target = self._reading_fraction * sum(weights)
        cumulative = 0
        for index, weight in enumerate(weights):
            cumulative += weight
            if target < cumulative:
                return index
        return len(lines) - 1

    def _render_reading_subtitle(
        self,
        *,
        resized: bool = False,
        latest: bool = False,
    ) -> None:
        lines = self._subtitle_lines()
        if not lines:
            self.subtitle_line_one.clear()
            self.subtitle_line_two.clear()
            self._subtitle_line_index = -1
            return

        target_index = (
            max(len(lines) - 2, 0)
            if latest
            else self._subtitle_index_at_progress(lines)
        )
        if (
            not resized
            and self._subtitle_line_index >= 0
            and target_index > self._subtitle_line_index + 1
        ):
            target_index = self._subtitle_line_index + 1
        if not resized and target_index == self._subtitle_line_index:
            return

        self._subtitle_line_index = target_index
        visible_lines = lines[target_index:target_index + 2]
        self.subtitle_line_one.setText(visible_lines[0])
        self.subtitle_line_two.setText(
            visible_lines[1] if len(visible_lines) > 1 else ""
        )

    def finish_reading(self, success: bool, message: str) -> None:
        if self._subtitle_dismissed:
            return
        self.transcript_area.finish_reading()
        self.microphone_button.setVisible(True)
        self.set_status(message, error=not success)

    def _set_subtitle_status(self, message: str) -> None:
        self._subtitle_status_text = message
        self.subtitle_line_one.setText(message)
        self.subtitle_line_two.clear()
        self.subtitle_full_text.setPlainText(
            self._reading_full_text or message
        )

    def _update_expanded_subtitle(self) -> None:
        if not self._subtitle_expanded:
            return
        self.subtitle_full_text.setPlainText(
            self._reading_full_text or self._subtitle_status_text
        )
        cursor = self.subtitle_full_text.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.subtitle_full_text.setTextCursor(cursor)

    def _expand_subtitle(self) -> None:
        if not self._subtitle_mode_active or self._subtitle_expanded:
            return
        self._subtitle_expanded = True
        self._subtitle_collapsed_geometry = QRect(self.geometry())
        self.subtitle_full_text.setPlainText(
            self._reading_full_text or self._subtitle_status_text
        )
        self.subtitle_line_one.hide()
        self.subtitle_line_two.hide()
        self.subtitle_full_text.show()

        available = self.screen().availableGeometry()
        target_height = min(
            max(self.height() * 2, 420),
            max(available.height() - 40, self.height()),
        )
        if target_height > self.height() and not self._position_locked:
            geometry = QRect(self.geometry())
            geometry.setTop(
                max(available.top(), geometry.bottom() - target_height + 1)
            )
            geometry.setHeight(target_height)
            self.setGeometry(geometry)

    def _collapse_subtitle(self) -> None:
        if not self._subtitle_expanded:
            return
        self._subtitle_expanded = False
        self.subtitle_full_text.hide()
        self.subtitle_line_one.show()
        self.subtitle_line_two.show()
        if (
            self._subtitle_collapsed_geometry is not None
            and not self._position_locked
        ):
            self.setGeometry(self._subtitle_collapsed_geometry)
        self._subtitle_collapsed_geometry = None

    def _collapse_subtitle_if_outside(self) -> None:
        if not self._subtitle_expanded:
            return
        position = self.subtitle_panel.mapFromGlobal(QCursor.pos())
        if not self.subtitle_panel.rect().contains(position):
            self._collapse_subtitle()

    def dismiss_subtitle_mode(self) -> bool:
        if not self._subtitle_mode_active:
            return False
        self._collapse_subtitle()
        self._subtitle_mode_active = False
        self._subtitle_dismissed = True
        self._subtitle_reading_active = False
        self.subtitle_panel.hide()
        self.dictation_panel.hide()
        self.transcript_area.begin_composing()
        self.transcript_area.show()
        self.microphone_button.setVisible(True)
        self.set_status("Hold the microphone or enter a message")
        return True

    def begin_dictation_waiting(self) -> None:
        self.dismiss_subtitle_mode()
        self.subtitle_panel.hide()
        self.transcript_area.hide()
        self.dictation_state_label.setText(
            "Waiting for the browser to start listening…"
        )
        self.dictation_panel.show()

    def set_dictation_listening(self) -> None:
        self.dictation_state_label.setText("Listening…")

    def set_dictation_finishing(self) -> None:
        self.dictation_state_label.setText("Finishing dictation…")

    def set_dictation_cancelling(self) -> None:
        self.dictation_state_label.setText("Cancelling short dictation…")

    def end_dictation_display(self) -> None:
        self.dictation_panel.hide()
        self.transcript_area.show()

    def _chatgpt_tab_changed(self, index: int) -> None:
        tab_id = self.chatgpt_tab_combo.itemData(index)
        if tab_id:
            self.chatgpt_tab_selected.emit(str(tab_id))

    def _capture_source_changed(self, index: int) -> None:
        self.transcript_area.set_screenshot_selected(
            self.capture_source_combo.itemData(index) is not None
        )

    def clear_transcript(self) -> None:
        self.transcript_area.clear()
        self.set_status("Text cleared")

    def _request_clear(self) -> None:
        self.clear_transcript()
        self.clear_requested.emit()

    def _request_send(
        self,
        include_screenshot: bool = True,
        *,
        restore_focus: bool = True,
    ) -> None:
        if self.transcript_area.is_showing_response:
            self.set_status("Wait for the current response to finish")
            return
        text = self.transcript_area.toPlainText().strip()
        if not text:
            self.set_status("Enter text before sending", error=True)
            return
        capture_source = (
            self.capture_source_combo.currentData()
            if include_screenshot
            else None
        )
        self.send_requested.emit(text, capture_source)
        if restore_focus:
            QTimer.singleShot(0, self._restore_previous_focus)

    def _request_send_without_screenshot(self) -> None:
        self._request_send(include_screenshot=False)

    def request_send_from_hotkey(self, include_screenshot: bool) -> None:
        self._request_send(
            include_screenshot=include_screenshot,
            restore_focus=False,
        )

    def remember_foreground_app(self) -> None:
        self._focus_restorer.remember_foreground()

    def _restore_previous_focus(self) -> None:
        if self._focus_restorer.restore_previous():
            logger.debug("Restored focus to the previous application")
        else:
            logger.debug("No external application was available to restore")

    def _set_position_locked(self, locked: bool) -> None:
        self._position_locked = locked
        self._drag_offset = None
        self._end_border_resize()
        if locked:
            self.setFixedSize(self.size())
            icon_path = LOCK_ICON_PATH
            label = "Unlock overlay position"
        else:
            self.setMinimumSize(760, 180)
            self.setMaximumSize(16_777_215, 16_777_215)
            icon_path = UNLOCK_ICON_PATH
            label = "Lock overlay position"
        self.lock_button.setIcon(QIcon(str(icon_path)))
        self.lock_button.setAccessibleName(label)
        self.lock_button.setToolTip(label)

    def _set_chrome_visible(self, visible: bool) -> None:
        visible = visible or bool(self._resize_edges)
        self._chrome_visible = visible
        opacity = 1.0 if visible else 0.0
        self._title_opacity.setOpacity(opacity)
        self._microphone_opacity.setOpacity(opacity)
        self.panel.setProperty("chromeVisible", visible)
        self._resize_surface.setProperty("chromeVisible", visible)
        for widget in (self.panel, self._resize_surface):
            style = widget.style()
            style.unpolish(widget)
            style.polish(widget)
            widget.update()

    def enterEvent(self, event) -> None:  # noqa: N802
        self._set_chrome_visible(True)
        super().enterEvent(event)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if hasattr(self, "_reading_full_text"):
            self._render_reading_subtitle(
                resized=True,
                latest=not self._subtitle_reading_active,
            )

    def leaveEvent(self, event) -> None:  # noqa: N802
        QTimer.singleShot(0, self._hide_chrome_if_outside)
        super().leaveEvent(event)

    def _hide_chrome_if_outside(self) -> None:
        if not self._resize_edges and not self.underMouse():
            self._set_chrome_visible(False)

    def eventFilter(self, watched, event) -> bool:  # noqa: N802
        subtitle_widgets = getattr(self, "_subtitle_hover_widgets", ())
        if watched in subtitle_widgets:
            event_type = event.type()
            if event_type == QEvent.Type.Enter:
                self._expand_subtitle()
            elif event_type == QEvent.Type.Leave:
                QTimer.singleShot(0, self._collapse_subtitle_if_outside)
            elif (
                event_type == QEvent.Type.MouseButtonPress
                and event.button() == Qt.MouseButton.LeftButton
            ):
                if self.dismiss_subtitle_mode():
                    event.accept()
                    return True

        resize_surface = getattr(self, "_resize_surface", None)
        panel = getattr(self, "panel", None)
        if watched is not resize_surface and watched is not panel:
            return super().eventFilter(watched, event)

        position = watched.mapTo(
            resize_surface,
            event.position().toPoint(),
        ) if hasattr(event, "position") else QPoint()

        event_type = event.type()
        if event_type == QEvent.Type.MouseButtonPress:
            if (
                event.button() == Qt.MouseButton.LeftButton
                and not self._position_locked
            ):
                edges = self._resize_edges_at(position)
                if edges:
                    self._begin_border_resize(
                        edges,
                        event.globalPosition().toPoint(),
                    )
                    event.accept()
                    return True
        elif event_type == QEvent.Type.MouseMove:
            if (
                self._resize_edges
                and event.buttons() & Qt.MouseButton.LeftButton
            ):
                self._update_border_resize(event.globalPosition().toPoint())
                event.accept()
                return True
            if self._resize_edges_at(position):
                self._set_chrome_visible(True)
            self._update_border_cursor(position)
        elif event_type == QEvent.Type.MouseButtonRelease:
            if self._resize_edges:
                self._end_border_resize()
                self._update_border_cursor(position)
                event.accept()
                return True
        return super().eventFilter(watched, event)

    def _resize_edges_at(self, position: QPoint) -> Qt.Edges:
        if self._position_locked:
            return Qt.Edges()
        margin = 20
        edges = Qt.Edges()
        if position.x() <= margin:
            edges |= Qt.Edge.LeftEdge
        elif position.x() >= self._resize_surface.width() - margin - 1:
            edges |= Qt.Edge.RightEdge
        if position.y() <= margin:
            edges |= Qt.Edge.TopEdge
        elif position.y() >= self._resize_surface.height() - margin - 1:
            edges |= Qt.Edge.BottomEdge
        return edges

    def _begin_border_resize(
        self,
        edges: Qt.Edges,
        global_position: QPoint,
    ) -> None:
        self._resize_edges = edges
        self._resize_start_global = global_position
        self._resize_start_geometry = self.geometry()
        self._drag_offset = None
        self._set_chrome_visible(True)

    def _update_border_resize(self, global_position: QPoint) -> None:
        if (
            not self._resize_edges
            or self._resize_start_global is None
            or self._resize_start_geometry is None
        ):
            return
        delta = global_position - self._resize_start_global
        start = self._resize_start_geometry
        resized = QRect(start)
        if self._resize_edges & Qt.Edge.LeftEdge:
            resized.setLeft(
                min(start.left() + delta.x(), start.right() - self.minimumWidth() + 1)
            )
        if self._resize_edges & Qt.Edge.RightEdge:
            resized.setRight(
                max(start.right() + delta.x(), start.left() + self.minimumWidth() - 1)
            )
        if self._resize_edges & Qt.Edge.TopEdge:
            resized.setTop(
                min(start.top() + delta.y(), start.bottom() - self.minimumHeight() + 1)
            )
        if self._resize_edges & Qt.Edge.BottomEdge:
            resized.setBottom(
                max(start.bottom() + delta.y(), start.top() + self.minimumHeight() - 1)
            )
        self.setGeometry(resized)
        self._set_chrome_visible(True)

    def _end_border_resize(self) -> None:
        was_resizing = bool(self._resize_edges)
        self._resize_edges = Qt.Edges()
        self._resize_start_global = None
        self._resize_start_geometry = None
        if was_resizing and not self.underMouse():
            self._set_chrome_visible(False)

    def _update_border_cursor(self, position: QPoint) -> None:
        edges = self._resize_edges or self._resize_edges_at(position)
        if edges in (
            Qt.Edge.TopEdge | Qt.Edge.LeftEdge,
            Qt.Edge.BottomEdge | Qt.Edge.RightEdge,
        ):
            cursor = Qt.CursorShape.SizeFDiagCursor
        elif edges in (
            Qt.Edge.TopEdge | Qt.Edge.RightEdge,
            Qt.Edge.BottomEdge | Qt.Edge.LeftEdge,
        ):
            cursor = Qt.CursorShape.SizeBDiagCursor
        elif edges & (Qt.Edge.LeftEdge | Qt.Edge.RightEdge):
            cursor = Qt.CursorShape.SizeHorCursor
        elif edges & (Qt.Edge.TopEdge | Qt.Edge.BottomEdge):
            cursor = Qt.CursorShape.SizeVerCursor
        else:
            cursor = Qt.CursorShape.ArrowCursor
        self._resize_surface.setCursor(cursor)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        """Keep the application running in the system tray."""
        logger.info("Window closed; hiding it in the system tray")
        self.hide()
        event.ignore()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if (
            not self._position_locked
            and event.button() == Qt.MouseButton.LeftButton
        ):
            clicked_widget = self.childAt(event.position().toPoint())
            if not isinstance(clicked_widget, (QPushButton, QComboBox)):
                self._drag_offset = (
                    event.globalPosition().toPoint()
                    - self.frameGeometry().topLeft()
                )
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if (
            not self._position_locked
            and self._drag_offset is not None
            and event.buttons() & Qt.MouseButton.LeftButton
        ):
            self.move(event.globalPosition().toPoint() - self._drag_offset)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self._drag_offset is not None
        ):
            self._drag_offset = None
            logger.debug(f"Overlay moved to {self.pos().x()},{self.pos().y()}")
            event.accept()
            return
        super().mouseReleaseEvent(event)


class TrayController:
    def __init__(self, application: QApplication) -> None:
        logger.debug("Creating tray controller")
        self.application = application
        self.icon = QIcon(str(ICON_PATH))
        self.window = OverlayWindow()
        self.browser_monitor = BrowserMonitor()
        self.selected_chatgpt_tab_id: str | None = None
        self.dictation_tab_id: str | None = None
        self._dictation_state = "idle"
        self._dictation_input_held = False
        self._dictation_listening_since: float | None = None
        self.settings = QSettings("Live GPT", "Live GPT")
        self._hotkey_sequences = self._load_hotkey_sequences()
        bindings = self._bindings_for_sequences(self._hotkey_sequences)
        self.hotkey_monitor = GlobalHotkeyMonitor(
            bindings["hold"],
            bindings["send"],
            bindings["send_without_screenshot"],
            self.window,
        )

        self.application.setWindowIcon(self.icon)
        self.window.setWindowIcon(self.icon)
        self.window.exit_requested.connect(self._exit_application)
        self.window.hide_requested.connect(self.hide_window)
        self.window.dictation_requested.connect(self.start_dictation)
        self.window.dictation_finish_requested.connect(self.finish_dictation)
        self.window.clear_requested.connect(self._handle_clear_requested)
        self.window.send_requested.connect(self._handle_send_requested)
        self.window.configure_requested.connect(self._open_configuration)
        self.window.open_remote_debugging_requested.connect(
            self._open_remote_debugging_settings
        )
        self.window.chatgpt_tab_selected.connect(
            self._select_chatgpt_tab
        )
        self.window.chatgpt_connection_changed.connect(
            self._set_chatgpt_connection
        )
        self.browser_monitor.tabs_changed.connect(self.window.set_chatgpt_tabs)
        self.browser_monitor.status_changed.connect(
            self.window.set_browser_status
        )
        self.browser_monitor.send_finished.connect(
            self.window.set_send_result
        )
        self.browser_monitor.response_changed.connect(
            self.window.set_response_update
        )
        self.browser_monitor.response_finished.connect(
            self.window.set_response_finished
        )
        self.browser_monitor.reading_started.connect(
            self.window.begin_reading
        )
        self.browser_monitor.reading_changed.connect(
            self.window.set_reading_subtitle
        )
        self.browser_monitor.reading_finished.connect(
            self.window.finish_reading
        )
        self.browser_monitor.dictation_started.connect(
            self._on_dictation_started
        )
        self.browser_monitor.dictation_finished.connect(
            self._on_dictation_finished
        )
        self.browser_monitor.clear_finished.connect(
            self._on_clear_finished
        )
        self.hotkey_monitor.hold_pressed.connect(self.start_dictation)
        self.hotkey_monitor.hold_released.connect(self.finish_dictation)
        self.hotkey_monitor.send_pressed.connect(
            lambda: self.window.request_send_from_hotkey(True)
        )
        self.hotkey_monitor.send_without_screenshot_pressed.connect(
            lambda: self.window.request_send_from_hotkey(False)
        )

        self.browser_monitor.start()
        self.hotkey_monitor.start()

        self.capture_refresh_timer = QTimer(self.window)
        self.capture_refresh_timer.timeout.connect(
            self._refresh_capture_sources
        )
        self.capture_refresh_timer.start(2_000)
        self._refresh_capture_sources()

        self.menu = QMenu()
        self.exit_action = QAction("Exit", self.menu)
        self.exit_action.triggered.connect(self._exit_application)
        self.menu.addAction(self.exit_action)

        self.tray_icon = QSystemTrayIcon(self.icon, self.application)
        self.tray_icon.setToolTip("Live GPT")
        self.tray_icon.setContextMenu(self.menu)
        self.tray_icon.activated.connect(self._handle_activation)
        self.tray_icon.show()
        logger.info("System tray icon is ready")
        self._position_overlay()
        self.show_window()

    @staticmethod
    def _bindings_for_sequences(
        sequences: dict[str, QKeySequence],
    ) -> dict[str, HotkeyBinding]:
        bindings = {
            name: HotkeyBinding.from_sequence(sequence)
            for name, sequence in sequences.items()
        }
        if len({binding.text.casefold() for binding in bindings.values()}) != 3:
            raise ValueError("Each action must use a different hotkey")
        return bindings

    def _load_hotkey_sequences(self) -> dict[str, QKeySequence]:
        defaults = {
            "hold": DEFAULT_HOLD_MIC_HOTKEY,
            "send": DEFAULT_SEND_HOTKEY,
            "send_without_screenshot": (
                DEFAULT_SEND_WITHOUT_SCREENSHOT_HOTKEY
            ),
        }
        sequences = {
            name: QKeySequence(
                str(self.settings.value(HOTKEY_SETTING_KEYS[name], default))
            )
            for name, default in defaults.items()
        }
        try:
            self._bindings_for_sequences(sequences)
        except ValueError as error:
            logger.warning(f"Invalid saved hotkey configuration: {error}")
            return {
                name: QKeySequence(default)
                for name, default in defaults.items()
            }
        return sequences

    def _open_configuration(self) -> None:
        self.hotkey_monitor.stop()
        try:
            dialog = HotkeyConfigDialog(
                self._hotkey_sequences["hold"],
                self._hotkey_sequences["send"],
                self._hotkey_sequences["send_without_screenshot"],
                self.window,
            )
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return

            sequences = dialog.sequences()
            bindings = dialog.bindings()
            self._hotkey_sequences = sequences
            for name, sequence in sequences.items():
                self.settings.setValue(
                    HOTKEY_SETTING_KEYS[name],
                    sequence.toString(
                        QKeySequence.SequenceFormat.PortableText
                    ),
                )
            self.settings.sync()
            self.hotkey_monitor.update_bindings(
                bindings["hold"],
                bindings["send"],
                bindings["send_without_screenshot"],
            )
            self.window.set_status("Global hotkeys updated")
            logger.info(
                "Updated global hotkeys "
                + ", ".join(
                    f"{name}={binding.text!r}"
                    for name, binding in bindings.items()
                )
            )
        finally:
            self.hotkey_monitor.start()

    def _handle_activation(
        self, reason: QSystemTrayIcon.ActivationReason
    ) -> None:
        logger.debug(f"Tray icon activated: {reason.name}")
        if reason == QSystemTrayIcon.ActivationReason.DoubleClick:
            self.show_window()

    def show_window(self) -> None:
        logger.info("Showing the overlay window")
        self.window.remember_foreground_app()
        self.window.showNormal()
        self.window.raise_()
        self.window.activateWindow()

    def hide_window(self) -> None:
        logger.info("Hiding the overlay window")
        self.window.hide()

    def start_dictation(self) -> None:
        self._dictation_input_held = True
        state = getattr(self, "_dictation_state", "idle")
        if state != "idle":
            return

        dismissed = self.window.dismiss_subtitle_mode()
        if (
            not dismissed
            and self.window.transcript_area.is_showing_response
        ):
            self.window.transcript_area.begin_composing()

        tab_id = self.selected_chatgpt_tab_id
        if tab_id is None:
            self._dictation_input_held = False
            self.window.set_microphone_state(
                "error",
                "Select a ChatGPT window first",
            )
            return

        logger.info(f"Starting ChatGPT dictation tab_id={tab_id!r}")
        self.dictation_tab_id = tab_id
        self._dictation_state = "starting"
        self._dictation_listening_since = None
        self.window.begin_dictation_waiting()
        self.window.set_microphone_state(
            "recording",
            "Waiting for the browser to start listening…",
        )
        self.browser_monitor.request_start_dictation(tab_id)

    def finish_dictation(self) -> None:
        self._dictation_input_held = False
        state = getattr(self, "_dictation_state", "idle")
        tab_id = self.dictation_tab_id
        if tab_id is None or state == "idle":
            return

        if state == "starting":
            self.window.set_dictation_cancelling()
            return
        if state != "listening":
            return

        listening_since = self._dictation_listening_since or time.monotonic()
        if time.monotonic() - listening_since < 0.5:
            self._cancel_dictation(tab_id)
            return

        self._dictation_state = "finishing"
        logger.info(f"Finishing ChatGPT dictation tab_id={tab_id!r}")
        self.window.set_dictation_finishing()
        self.window.set_microphone_state(
            "recording",
            "Finishing ChatGPT dictation…",
        )
        self.browser_monitor.request_finish_dictation(tab_id)

    def _cancel_dictation(self, tab_id: str) -> None:
        self._dictation_state = "cancelling"
        logger.info(f"Cancelling short ChatGPT dictation tab_id={tab_id!r}")
        self.window.set_dictation_cancelling()
        self.window.set_microphone_state(
            "recording",
            "Dictation was too short; cancelling…",
        )
        self.browser_monitor.request_cancel_dictation(tab_id)

    def _on_dictation_started(self, success: bool, message: str) -> None:
        if success:
            if (
                self.dictation_tab_id is None
                or getattr(self, "_dictation_state", "idle") != "starting"
            ):
                return
            self._dictation_state = "listening"
            self._dictation_listening_since = time.monotonic()
            if not getattr(self, "_dictation_input_held", False):
                self._cancel_dictation(self.dictation_tab_id)
                return
            self.window.set_dictation_listening()
            self.window.set_microphone_state("recording", message)
            return

        self._dictation_state = "idle"
        self._dictation_input_held = False
        self._dictation_listening_since = None
        self.dictation_tab_id = None
        self.window.end_dictation_display()
        self.window.set_microphone_state("error", message)

    def _on_dictation_finished(
        self,
        success: bool,
        text: str,
        message: str,
    ) -> None:
        was_cancelled = getattr(self, "_dictation_state", "idle") == "cancelling"
        restart = getattr(self, "_dictation_input_held", False)
        self._dictation_state = "idle"
        self._dictation_listening_since = None
        self.dictation_tab_id = None
        self.window.end_dictation_display()
        if not success:
            self._dictation_input_held = False
            self.window.set_microphone_state("error", message)
            return

        if was_cancelled:
            self.window.set_microphone_state("idle", message)
        else:
            self.window.set_transcript(text)
            self.window.set_microphone_state("saved", message)
        if restart:
            self.start_dictation()

    def _handle_send_requested(
        self,
        text: str,
        capture_source: CaptureSource | None,
    ) -> None:
        tab_id = self.selected_chatgpt_tab_id
        if tab_id is None:
            self.window.set_status(
                "Select a ChatGPT window first",
                error=True,
            )
            return

        self.window.send_button.setEnabled(False)
        self.window.send_without_screenshot_button.setEnabled(False)
        screenshot = None
        if capture_source is not None:
            self.window.set_status("Capturing screenshot…")
            try:
                screenshot = capture_webp(capture_source)
            except Exception as error:
                logger.error("Unable to capture screenshot", error)
                self.window.send_button.setEnabled(True)
                self.window.send_without_screenshot_button.setEnabled(True)
                self.window.set_status(
                    f"Could not capture screenshot: {error}",
                    error=True,
                )
                return

        logger.info(
            "Send requested "
            f"tab_id={tab_id!r} characters={len(text)} "
            f"screenshot={capture_source.key if capture_source else None!r}"
        )
        self.window.begin_response_display(text)
        self.window.set_status("Sending to ChatGPT…")
        self.browser_monitor.request_send(tab_id, text, screenshot)

    def _refresh_capture_sources(self) -> None:
        if self.window.capture_source_combo.view().isVisible():
            return
        try:
            sources = list_capture_sources({int(self.window.winId())})
        except Exception as error:
            logger.error("Unable to list screenshot sources", error)
            return
        self.window.set_capture_sources(sources)

    def _handle_clear_requested(self) -> None:
        tab_id = self.selected_chatgpt_tab_id
        if tab_id is None:
            self.window.set_status(
                "Text cleared locally; no ChatGPT window selected",
                error=True,
            )
            return

        logger.info(f"Clearing ChatGPT input tab_id={tab_id!r}")
        self.window.set_status("Clearing ChatGPT input…")
        self.browser_monitor.request_clear(tab_id)

    def _on_clear_finished(self, success: bool, message: str) -> None:
        self.window.set_status(message, error=not success)

    def _select_chatgpt_tab(self, tab_id: str) -> None:
        self.selected_chatgpt_tab_id = tab_id
        logger.info(f"Selected ChatGPT tab id={tab_id}")

    def _set_chatgpt_connection(self, connected: bool) -> None:
        if connected:
            return
        self.selected_chatgpt_tab_id = None

    def _open_remote_debugging_settings(self) -> None:
        endpoint = discover_cdp_endpoint()
        if endpoint is not None:
            self.window.set_browser_status(
                "Retrying connection… approve it in the browser"
            )
            self.browser_monitor.request_retry_connection()
            return

        try:
            settings_url = open_remote_debugging_settings()
        except Exception as error:
            logger.error("Unable to open remote debugging settings", error)
            self.window.set_browser_status(str(error))
            return
        self.window.set_browser_status(
            f"Remote debugging enabled at {settings_url}"
        )
        self.browser_monitor.request_retry_connection()

    def _position_overlay(self) -> None:
        screen = self.application.primaryScreen()
        if screen is None:
            return

        available = screen.availableGeometry()
        bottom_margin = 32
        self.window.move(
            available.left() + (available.width() - self.window.width()) // 2,
            available.bottom() - self.window.height() - bottom_margin + 1,
        )

    def _exit_application(self, checked: bool = False) -> None:
        del checked
        logger.info("Exit requested")
        self.capture_refresh_timer.stop()
        self.hotkey_monitor.stop()
        self.browser_monitor.request_stop()
        if not self.browser_monitor.wait(17_000):
            logger.warning("Browser monitor did not stop before application exit")
        self.application.quit()


def main() -> int:
    config_logger({"debug": True}, name="live-gpt")
    logger.info(f"Starting Live GPT with pid={os.getpid()} args={sys.argv}")

    try:
        application = QApplication(sys.argv)
        application.setApplicationName("Live GPT")
        application.setQuitOnLastWindowClosed(False)

        if not QSystemTrayIcon.isSystemTrayAvailable():
            logger.error("No system tray is available on this desktop")
            QMessageBox.critical(
                None,
                "Live GPT",
                "No system tray is available on this desktop.",
            )
            return 1

        controller = TrayController(application)
        exit_code = application.exec()
        logger.info(f"Application event loop stopped with code={exit_code}")
        return exit_code
    except Exception as error:
        logger.error("Application startup failed", error)
        raise
    finally:
        shutdown_logger()
