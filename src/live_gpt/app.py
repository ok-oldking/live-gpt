from __future__ import annotations

import os
import sys
from pathlib import Path

from PySide6.QtCore import QPoint, QSize, QTimer, Signal, Qt
from PySide6.QtGui import (
    QAction,
    QActionGroup,
    QCloseEvent,
    QIcon,
    QMouseEvent,
    QTextBlockFormat,
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

from .audio import (
    AudioDevice,
    AudioRecorder,
    list_playback_devices,
    list_recording_devices,
)
from .browser import (
    BrowserMonitor,
    discover_cdp_endpoint,
    open_remote_debugging_settings,
)
from .logger import Logger, config_logger, shutdown_logger
from .speech import SpeechTranscriber


ASSET_DIRECTORY = Path(__file__).resolve().parent / "assets"
ICON_PATH = ASSET_DIRECTORY / "app-icon.ico"
MICROPHONE_ICON_PATH = ASSET_DIRECTORY / "microphone.svg"
SETTINGS_ICON_PATH = ASSET_DIRECTORY / "settings.svg"
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
        self._set_reading_style(False)
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
        self._set_reading_style(True)
        self.clear()

    def update_reading_subtitle(self, subtitle: str) -> None:
        if self.toPlainText() == subtitle:
            return
        self.setPlainText(subtitle)
        cursor = self.textCursor()
        cursor.select(QTextCursor.SelectionType.Document)
        block_format = QTextBlockFormat()
        block_format.setAlignment(Qt.AlignmentFlag.AlignCenter)
        cursor.mergeBlockFormat(block_format)
        cursor.clearSelection()
        cursor.movePosition(QTextCursor.MoveOperation.Start)
        self.setTextCursor(cursor)

    def finish_reading(self) -> None:
        self._set_reading_style(False)
        self.setPlainText(self._full_response_text)
        self._response_complete = True

    def _set_reading_style(self, reading: bool) -> None:
        self.setProperty("readingMode", reading)
        style = self.style()
        style.unpolish(self)
        style.polish(self)
        self.update()

    def begin_composing(self) -> None:
        self._response_mode = False
        self._response_complete = False
        self._full_response_text = ""
        self._set_reading_style(False)
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
    settings_requested = Signal()
    recording_requested = Signal()
    recording_stop_requested = Signal()
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

        self.settings_button = QPushButton()
        self.settings_button.setObjectName("settingsButton")
        self._configure_icon_button(
            self.settings_button,
            SETTINGS_ICON_PATH,
            "Settings",
        )
        self.settings_button.clicked.connect(self.settings_requested.emit)
        title_layout.addWidget(self.settings_button)

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

        self.status_label = QLabel("Preparing speech model...")
        self.status_label.setObjectName("overlayStatus")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.transcript_area = TranscriptEditor()
        self.transcript_area.setObjectName("transcriptArea")
        self.transcript_area.setPlaceholderText(
            "Recognized speech will appear here in real time"
        )

        self.microphone_button = QPushButton()
        self.microphone_button.setObjectName("microphoneButton")
        self.microphone_button.setProperty("recordingState", "idle")
        self.microphone_button.setIcon(QIcon(str(MICROPHONE_ICON_PATH)))
        self.microphone_button.setIconSize(QSize(26, 26))
        self.microphone_button.setFixedSize(56, 56)
        self.microphone_button.setAccessibleName("Hold to record")
        self.microphone_button.setEnabled(False)
        self.microphone_button.setToolTip(
            "Press and hold to record from the default microphone"
        )
        self.microphone_button.pressed.connect(self.recording_requested.emit)
        self.microphone_button.released.connect(
            self.recording_stop_requested.emit
        )

        self.send_button = self.transcript_area.send_button
        self.send_button.clicked.connect(self._request_send)

        self.clear_button = self.transcript_area.clear_button
        self.clear_button.clicked.connect(self.clear_transcript)

        panel_layout.addLayout(title_layout)
        recording_layout = QHBoxLayout()
        recording_layout.setSpacing(16)
        recording_layout.addWidget(self.transcript_area, 1)

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
            QPlainTextEdit#transcriptArea[readingMode="true"] {
                padding: 14px;
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
            QPushButton#settingsButton,
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
            "idle": "Press and hold the microphone to record",
            "recording": "Recording... release to stop",
            "saved": message or "Recording saved",
            "error": "Microphone unavailable",
        }
        self.status_label.setText(labels[state])
        self.microphone_button.setProperty("recordingState", state)
        self.microphone_button.setAccessibleName(labels[state])
        self.microphone_button.setToolTip(message or labels[state])
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
        self.microphone_button.setVisible(False)
        self.status_label.setText(message)

    def set_reading_subtitle(self, subtitle: str) -> None:
        self.transcript_area.update_reading_subtitle(subtitle)
        self.status_label.setText("Reading aloud…")

    def finish_reading(self, success: bool, message: str) -> None:
        del success
        self.transcript_area.finish_reading()
        self.microphone_button.setVisible(True)
        self.status_label.setText(message)

    def _chatgpt_tab_changed(self, index: int) -> None:
        tab_id = self.chatgpt_tab_combo.itemData(index)
        if tab_id:
            self.chatgpt_tab_selected.emit(str(tab_id))

    def clear_transcript(self) -> None:
        self.transcript_area.clear()
        self.status_label.setText("Text cleared")

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
    def __init__(
        self,
        application: QApplication,
        recorder: AudioRecorder | None = None,
        transcriber: SpeechTranscriber | None = None,
    ) -> None:
        logger.debug("Creating tray controller")
        self.application = application
        self.recorder = recorder or AudioRecorder()
        self.transcriber = transcriber or SpeechTranscriber()
        self.playback_device: int | None = None
        self.recording_devices: list[AudioDevice] = []
        self.playback_devices: list[AudioDevice] = []
        self.icon = QIcon(str(ICON_PATH))
        self.window = OverlayWindow()
        self.browser_monitor = BrowserMonitor()
        self.selected_chatgpt_tab_id: str | None = None
        self.settings_menu = QMenu(self.window)

        self.application.setWindowIcon(self.icon)
        self.window.setWindowIcon(self.icon)
        self.window.exit_requested.connect(self._exit_application)
        self.window.hide_requested.connect(self.hide_window)
        self.window.settings_requested.connect(self.show_settings)
        self.window.recording_requested.connect(self.start_recording)
        self.window.recording_stop_requested.connect(self.stop_recording)
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
        self.transcriber.transcript_changed.connect(self.window.set_transcript)
        self.transcriber.ready_changed.connect(self._on_transcriber_ready)
        self.transcriber.status_changed.connect(self._on_transcriber_status)
        self.transcriber.failed.connect(self._on_transcriber_error)
        self.recorder.audio_chunk_callback = self.transcriber.feed_audio
        self.recorder.recording_started_callback = (
            self.transcriber.start_session
        )
        self.recorder.recording_stopped_callback = (
            self.transcriber.finish_session
        )

        self._populate_settings_menu()
        self.transcriber.prepare()
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

    def show_settings(self) -> None:
        logger.info("Showing audio device settings")
        self._populate_settings_menu()
        button = self.window.settings_button
        menu_size = self.settings_menu.sizeHint()
        menu_position = button.mapToGlobal(
            QPoint(
                button.width() - menu_size.width(),
                -menu_size.height() - 6,
            )
        )
        self.settings_menu.popup(menu_position)

    def _populate_settings_menu(self) -> None:
        self.settings_menu.clear()

        try:
            recording_devices = list_recording_devices()
            playback_devices = list_playback_devices()
        except Exception as error:
            logger.error("Unable to list audio devices", error)
            unavailable = self.settings_menu.addAction(
                "Audio devices unavailable"
            )
            unavailable.setEnabled(False)
            return

        self.recording_devices = recording_devices
        self.playback_devices = playback_devices

        if self.recorder.input_device is None:
            default_recording_device = next(
                (
                    device
                    for device in recording_devices
                    if device.is_system_default
                ),
                None,
            )
            if default_recording_device is not None:
                self.recorder.input_device = default_recording_device.index
                logger.info(
                    "Using system default recording device "
                    f"index={default_recording_device.index} "
                    f"name={default_recording_device.name!r}"
                )

        if self.playback_device is None:
            default_playback_device = next(
                (
                    device
                    for device in playback_devices
                    if device.is_system_default
                ),
                None,
            )
            if default_playback_device is not None:
                self.playback_device = default_playback_device.index
                logger.info(
                    "Using system default playback device "
                    f"index={default_playback_device.index} "
                    f"name={default_playback_device.name!r}"
                )

        recording_menu = self.settings_menu.addMenu("Recording device")
        self._add_device_actions(
            recording_menu,
            recording_devices,
            self.recorder.input_device,
            self._select_recording_device,
        )

        playback_menu = self.settings_menu.addMenu("Playback device")
        self._add_device_actions(
            playback_menu,
            playback_devices,
            self.playback_device,
            self._select_playback_device,
        )

    def _add_device_actions(
        self,
        menu: QMenu,
        devices: list[AudioDevice],
        selected_device: int | None,
        selection_handler,
    ) -> None:
        action_group = QActionGroup(menu)
        action_group.setExclusive(True)

        for device in devices:
            default_label = " (System Default)" if device.is_system_default else ""
            label = f"{device.name}{default_label}".replace("&", "&&")
            action = menu.addAction(label)
            action.setCheckable(True)
            action.setChecked(device.index == selected_device)
            action.triggered.connect(
                lambda checked, index=device.index: (
                    checked and selection_handler(index)
                )
            )
            action_group.addAction(action)

        if not devices:
            unavailable = menu.addAction("No devices found")
            unavailable.setEnabled(False)

    def _select_recording_device(self, device_index: int | None) -> None:
        if self.recorder.is_recording:
            logger.warning("Cannot change recording device while recording")
            return
        self.recorder.input_device = device_index
        logger.info(
            "Selected recording device "
            f"{self._describe_device(self.recording_devices, device_index)}"
        )

    def _select_playback_device(self, device_index: int | None) -> None:
        self.playback_device = device_index
        logger.info(
            "Selected playback device "
            f"{self._describe_device(self.playback_devices, device_index)}"
        )

    @staticmethod
    def _describe_device(
        devices: list[AudioDevice],
        device_index: int | None,
    ) -> str:
        device = next(
            (device for device in devices if device.index == device_index),
            None,
        )
        if device is None:
            return f"index={device_index} name=<unknown>"
        return f"index={device.index} name={device.name!r}"

    def start_recording(self) -> None:
        selected_device = self._describe_device(
            self.recording_devices,
            self.recorder.input_device,
        )
        logger.info(f"Microphone button pressed selected_device={selected_device}")
        if not self.transcriber.is_ready:
            logger.warning("Recording ignored because speech model is not ready")
            self.window.status_label.setText("Speech model is still preparing...")
            return
        if self.window.transcript_area.is_showing_response:
            self.window.transcript_area.begin_composing()
        try:
            self.recorder.start()
        except Exception as error:
            self.transcriber.finish_session()
            logger.error(
                "Unable to start microphone recording "
                f"selected_device={selected_device}",
                error,
            )
            self.window.set_microphone_state("error", str(error))
            QTimer.singleShot(2_000, self._reset_microphone_state)
            return

        self.window.set_microphone_state("recording")

    def stop_recording(self) -> None:
        if not self.recorder.is_recording:
            return

        logger.info("Microphone button released")
        try:
            result = self.recorder.stop()
        except Exception as error:
            self.transcriber.finish_session()
            logger.error("Unable to stop microphone recording", error)
            self.window.set_microphone_state("error", str(error))
            QTimer.singleShot(2_000, self._reset_microphone_state)
            return

        if result is None:
            self.window.set_microphone_state("error", "No audio was captured")
            QTimer.singleShot(2_000, self._reset_microphone_state)
            return

        self.window.set_microphone_state(
            "saved",
            f"Saved {result.duration_seconds:.1f}s recording",
        )
        self.window.microphone_button.setToolTip(str(result.path))
        QTimer.singleShot(1_500, self._reset_microphone_state)

    def _reset_microphone_state(self) -> None:
        if not self.recorder.is_recording:
            self.window.set_microphone_state("idle")

    def _on_transcriber_ready(self, ready: bool) -> None:
        self.window.microphone_button.setEnabled(ready)
        if ready:
            self.window.set_microphone_state("idle")

    def _on_transcriber_status(self, message: str) -> None:
        if not self.recorder.is_recording:
            self.window.status_label.setText(message)

    def _on_transcriber_error(self, message: str) -> None:
        logger.error(f"Speech recognition error: {message}")
        self.window.microphone_button.setEnabled(False)
        self.window.set_microphone_state("error", message)
        self.window.status_label.setText("Speech recognition unavailable")

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
        if self.recorder.is_recording:
            self.stop_recording()
        self.browser_monitor.request_stop()
        if not self.browser_monitor.wait(17_000):
            logger.warning("Browser monitor did not stop before application exit")
        self.transcriber.close()
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
