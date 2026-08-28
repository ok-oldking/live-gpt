from __future__ import annotations

import os
import sys
from pathlib import Path

from PySide6.QtCore import QPoint, QSize, Signal, Qt
from PySide6.QtGui import (
    QAction,
    QCloseEvent,
    QIcon,
    QMouseEvent,
    QTextCursor,
)
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QHBoxLayout,
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


ASSET_DIRECTORY = Path(__file__).resolve().parent / "assets"
ICON_PATH = ASSET_DIRECTORY / "app-icon.ico"
MICROPHONE_ICON_PATH = ASSET_DIRECTORY / "microphone.svg"
HIDE_ICON_PATH = ASSET_DIRECTORY / "hide.svg"
EXIT_ICON_PATH = ASSET_DIRECTORY / "exit.svg"
SEND_ICON_PATH = ASSET_DIRECTORY / "send.svg"
logger = Logger.get_logger(__name__)


class TranscriptEditor(QPlainTextEdit):
    """Editable transcript with contextual actions inside the input."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._response_mode = False
        self._response_complete = False
        self._full_response_text = ""

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

        self.textChanged.connect(self._sync_action_visibility)
        self._sync_action_visibility()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        margin = 8
        spacing = 6
        y = self.height() - self.send_button.height() - margin
        send_x = self.width() - self.send_button.width() - margin
        clear_x = send_x - self.clear_button.width() - spacing
        self.clear_button.move(clear_x, y)
        self.send_button.move(send_x, y)
        self.clear_button.raise_()
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

    def _sync_action_visibility(self) -> None:
        has_text = bool(self.toPlainText().strip()) and not self._response_mode
        self.clear_button.setVisible(has_text)
        self.send_button.setVisible(has_text)
        self.setViewportMargins(0, 0, 76 if has_text else 0, 0)


class OverlayWindow(QMainWindow):
    exit_requested = Signal()
    hide_requested = Signal()
    dictation_requested = Signal()
    dictation_finish_requested = Signal()
    clear_requested = Signal()
    send_requested = Signal(str)
    open_remote_debugging_requested = Signal()
    chatgpt_tab_selected = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        logger.debug("Creating overlay window")
        self._drag_offset: QPoint | None = None
        self.setWindowTitle("Live GPT")
        self.setFixedSize(760, 240)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        container = QWidget(self)
        container.setObjectName("overlayContainer")
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(12, 12, 12, 12)

        panel = QFrame(container)
        panel.setObjectName("overlayPanel")
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(20, 16, 16, 18)
        panel_layout.setSpacing(14)

        title_layout = QHBoxLayout()
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

        self.status_label = QLabel("Looking for ChatGPT windows…")
        self.status_label.setObjectName("overlayStatus")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.transcript_area = TranscriptEditor()
        self.transcript_area.setObjectName("transcriptArea")
        self.transcript_area.setPlaceholderText(
            "Hold the microphone to dictate through ChatGPT"
        )

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
        subtitle_layout.addWidget(self.subtitle_line_one, 1)
        subtitle_layout.addWidget(self.subtitle_line_two, 1)
        self.subtitle_panel.hide()

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
        self.send_button.clicked.connect(self._request_send)

        self.clear_button = self.transcript_area.clear_button
        self.clear_button.clicked.connect(self._request_clear)

        panel_layout.addLayout(title_layout)
        recording_layout = QHBoxLayout()
        recording_layout.setSpacing(16)
        recording_layout.addWidget(self.transcript_area, 1)
        recording_layout.addWidget(self.subtitle_panel, 1)

        recording_layout.addWidget(
            self.microphone_button,
            0,
            Qt.AlignmentFlag.AlignVCenter,
        )

        panel_layout.addWidget(self.status_label)
        panel_layout.addLayout(recording_layout, 1)
        container_layout.addWidget(panel)
        self.setCentralWidget(container)

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
            QLabel#overlayTitle {
                color: #f5f7ff;
                font-size: 20px;
                font-weight: 700;
            }
            QLabel#overlayStatus {
                color: rgba(228, 235, 255, 210);
                font-size: 13px;
            }
            QComboBox#chatgptTabCombo {
                min-height: 34px;
                padding: 0 10px;
                color: #f5f7ff;
                background-color: rgba(5, 10, 28, 145);
                border: 1px solid rgba(130, 165, 230, 75);
                border-radius: 8px;
            }
            QComboBox#chatgptTabCombo:disabled {
                color: rgba(228, 235, 255, 155);
            }
            QComboBox#chatgptTabCombo QAbstractItemView {
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
            QFrame#subtitlePanel {
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
            QPushButton#clearButton {
                min-width: 32px;
                max-width: 32px;
                min-height: 32px;
                max-height: 32px;
                padding: 0;
                border-radius: 8px;
            }
            QPushButton#exitButton:hover {
                background-color: rgba(239, 68, 88, 190);
            }
            QPushButton#sendButton {
                background-color: rgba(35, 155, 116, 190);
            }
            QPushButton#sendButton:hover {
                background-color: rgba(40, 190, 140, 220);
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
        self.status_label.setText(label)
        self.microphone_button.setProperty("recordingState", state)
        self.microphone_button.setAccessibleName(label)
        self.microphone_button.setToolTip(label)
        style = self.microphone_button.style()
        style.unpolish(self.microphone_button)
        style.polish(self.microphone_button)
        self.microphone_button.update()

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
            selected_index = self.chatgpt_tab_combo.findData(selected_id)
            self.chatgpt_tab_combo.setCurrentIndex(
                selected_index if selected_index >= 0 else 0
            )
        self.chatgpt_tab_combo.blockSignals(False)
        if tabs:
            self._chatgpt_tab_changed(self.chatgpt_tab_combo.currentIndex())

    def set_browser_status(self, status: str) -> None:
        self.chatgpt_tab_combo.setToolTip(status)
        if not self.chatgpt_tab_combo.isEnabled():
            self.chatgpt_tab_combo.setItemText(0, status)

    def set_send_result(
        self,
        success: bool,
        sent_text: str,
        message: str,
    ) -> None:
        self.send_button.setEnabled(True)
        if not success:
            self.microphone_button.setVisible(True)
            self.status_label.setText(message)
            return

        del sent_text, message
        self.transcript_area.begin_response()
        self.microphone_button.setVisible(False)
        self.status_label.setText("Waiting for ChatGPT…")

    def set_response_update(self, status: str, text: str) -> None:
        self.status_label.setText(status)
        self.transcript_area.update_response(text)

    def set_response_finished(self, success: bool, message: str) -> None:
        del success
        self.transcript_area.finish_response()
        self.microphone_button.setVisible(True)
        self.status_label.setText(message)

    def begin_reading(self, message: str) -> None:
        self.transcript_area.begin_reading()
        self.subtitle_line_one.clear()
        self.subtitle_line_two.clear()
        self.transcript_area.hide()
        self.subtitle_panel.show()
        self.microphone_button.setVisible(False)
        self.status_label.setText(message)

    def set_reading_subtitle(self, subtitle: str) -> None:
        lines = subtitle.splitlines()
        self.subtitle_line_one.setText(lines[0] if lines else "")
        self.subtitle_line_two.setText(lines[1] if len(lines) > 1 else "")
        self.status_label.setText("Reading aloud…")

    def finish_reading(self, success: bool, message: str) -> None:
        del success
        self.transcript_area.finish_reading()
        self.subtitle_panel.hide()
        self.transcript_area.show()
        self.microphone_button.setVisible(True)
        self.status_label.setText(message)

    def _chatgpt_tab_changed(self, index: int) -> None:
        tab_id = self.chatgpt_tab_combo.itemData(index)
        if tab_id:
            self.chatgpt_tab_selected.emit(str(tab_id))

    def clear_transcript(self) -> None:
        self.transcript_area.clear()
        self.status_label.setText("Text cleared")

    def _request_clear(self) -> None:
        self.clear_transcript()
        self.clear_requested.emit()

    def _request_send(self) -> None:
        text = self.transcript_area.toPlainText().strip()
        if not text:
            self.status_label.setText("Enter text before sending")
            return
        self.send_requested.emit(text)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        """Keep the application running in the system tray."""
        logger.info("Window closed; hiding it in the system tray")
        self.hide()
        event.ignore()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
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
            self._drag_offset is not None
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

        self.application.setWindowIcon(self.icon)
        self.window.setWindowIcon(self.icon)
        self.window.exit_requested.connect(self._exit_application)
        self.window.hide_requested.connect(self.hide_window)
        self.window.dictation_requested.connect(self.start_dictation)
        self.window.dictation_finish_requested.connect(self.finish_dictation)
        self.window.clear_requested.connect(self._handle_clear_requested)
        self.window.send_requested.connect(self._handle_send_requested)
        self.window.open_remote_debugging_requested.connect(
            self._open_remote_debugging_settings
        )
        self.window.chatgpt_tab_selected.connect(
            self._select_chatgpt_tab
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

        self.browser_monitor.start()

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

    def _handle_activation(
        self, reason: QSystemTrayIcon.ActivationReason
    ) -> None:
        logger.debug(f"Tray icon activated: {reason.name}")
        if reason == QSystemTrayIcon.ActivationReason.DoubleClick:
            self.show_window()

    def show_window(self) -> None:
        logger.info("Showing the overlay window")
        self.window.showNormal()
        self.window.raise_()
        self.window.activateWindow()

    def hide_window(self) -> None:
        logger.info("Hiding the overlay window")
        self.window.hide()

    def start_dictation(self) -> None:
        tab_id = self.selected_chatgpt_tab_id
        if tab_id is None:
            self.window.set_microphone_state(
                "error",
                "Select a ChatGPT window first",
            )
            return

        logger.info(f"Starting ChatGPT dictation tab_id={tab_id!r}")
        if self.window.transcript_area.is_showing_response:
            self.window.transcript_area.begin_composing()
        self.dictation_tab_id = tab_id
        self.window.set_microphone_state(
            "recording",
            "Starting ChatGPT dictation…",
        )
        self.browser_monitor.request_start_dictation(tab_id)

    def finish_dictation(self) -> None:
        tab_id = self.dictation_tab_id
        if tab_id is None:
            return

        self.dictation_tab_id = None
        logger.info(f"Finishing ChatGPT dictation tab_id={tab_id!r}")
        self.window.set_microphone_state(
            "recording",
            "Finishing ChatGPT dictation…",
        )
        self.browser_monitor.request_finish_dictation(tab_id)

    def _on_dictation_started(self, success: bool, message: str) -> None:
        if success:
            self.window.set_microphone_state("recording", message)
            return

        self.dictation_tab_id = None
        self.window.set_microphone_state("error", message)

    def _on_dictation_finished(
        self,
        success: bool,
        text: str,
        message: str,
    ) -> None:
        if not success:
            self.window.set_microphone_state("error", message)
            return

        self.window.set_transcript(text)
        self.window.set_microphone_state("saved", message)

    def _handle_send_requested(self, text: str) -> None:
        tab_id = self.selected_chatgpt_tab_id
        if tab_id is None:
            self.window.status_label.setText("Select a ChatGPT window first")
            return

        logger.info(
            "Send requested "
            f"tab_id={tab_id!r} characters={len(text)}"
        )
        self.window.send_button.setEnabled(False)
        self.window.status_label.setText("Sending to ChatGPT…")
        self.browser_monitor.request_send(tab_id, text)

    def _handle_clear_requested(self) -> None:
        tab_id = self.selected_chatgpt_tab_id
        if tab_id is None:
            self.window.status_label.setText(
                "Text cleared locally; no ChatGPT window selected"
            )
            return

        logger.info(f"Clearing ChatGPT input tab_id={tab_id!r}")
        self.window.status_label.setText("Clearing ChatGPT input…")
        self.browser_monitor.request_clear(tab_id)

    def _on_clear_finished(self, success: bool, message: str) -> None:
        del success
        self.window.status_label.setText(message)

    def _select_chatgpt_tab(self, tab_id: str) -> None:
        self.selected_chatgpt_tab_id = tab_id
        logger.info(f"Selected ChatGPT tab id={tab_id}")

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
