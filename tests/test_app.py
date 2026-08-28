from __future__ import annotations

import os
import unittest
from unittest.mock import Mock, patch


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from live_gpt.app import (  # noqa: E402
    OverlayWindow,
    TranscriptEditor,
    TrayController,
)


class TranscriptEditorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.application = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.editor = TranscriptEditor()
        self.editor.show()
        self.editor.setFocus()

    def tearDown(self) -> None:
        self.editor.close()

    def test_enter_clicks_send(self) -> None:
        sent: list[bool] = []
        self.editor.send_button.clicked.connect(lambda: sent.append(True))
        self.editor.setPlainText("Send this")

        QTest.keyClick(self.editor, Qt.Key.Key_Return)

        self.assertEqual(sent, [True])
        self.assertEqual(self.editor.toPlainText(), "Send this")

    def test_shift_enter_inserts_newline(self) -> None:
        sent: list[bool] = []
        self.editor.send_button.clicked.connect(lambda: sent.append(True))
        self.editor.setPlainText("First line")
        cursor = self.editor.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.editor.setTextCursor(cursor)

        QTest.keyClick(
            self.editor,
            Qt.Key.Key_Return,
            Qt.KeyboardModifier.ShiftModifier,
        )

        self.assertEqual(sent, [])
        self.assertEqual(self.editor.toPlainText(), "First line\n")

    def test_reading_restores_full_response(self) -> None:
        full_response = "First sentence. Second sentence. Final sentence."
        self.editor.begin_response()
        self.editor.update_response(full_response)

        self.editor.begin_reading()

        self.assertEqual(self.editor.toPlainText(), "")

        self.editor.finish_reading()

        self.assertEqual(self.editor.toPlainText(), full_response)


class TrayControllerBrowserTests(unittest.TestCase):
    def test_clear_queues_selected_browser_composer(self) -> None:
        controller = TrayController.__new__(TrayController)
        controller.window = Mock()
        controller.browser_monitor = Mock()
        controller.selected_chatgpt_tab_id = "selected-tab"

        controller._handle_clear_requested()

        controller.browser_monitor.request_clear.assert_called_once_with(
            "selected-tab"
        )
        controller.window.status_label.setText.assert_called_once_with(
            "Clearing ChatGPT input…"
        )

    def test_reading_uses_two_dedicated_subtitle_labels(self) -> None:
        window = OverlayWindow()
        try:
            window.transcript_area.begin_response()
            window.transcript_area.update_response("The complete response")

            window.begin_reading("Preparing Read aloud…")
            window.set_reading_subtitle("Current subtitle\nNext subtitle")

            self.assertTrue(window.transcript_area.isHidden())
            self.assertFalse(window.subtitle_panel.isHidden())
            self.assertEqual(
                window.subtitle_line_one.text(),
                "Current subtitle",
            )
            self.assertEqual(
                window.subtitle_line_two.text(),
                "Next subtitle",
            )

            window.finish_reading(True, "Read aloud complete")

            self.assertFalse(window.transcript_area.isHidden())
            self.assertTrue(window.subtitle_panel.isHidden())
            self.assertEqual(
                window.transcript_area.toPlainText(),
                "The complete response",
            )
        finally:
            window.close()

    def test_microphone_press_and_release_queue_browser_dictation(self) -> None:
        controller = TrayController.__new__(TrayController)
        controller.window = Mock()
        controller.window.transcript_area.is_showing_response = False
        controller.browser_monitor = Mock()
        controller.selected_chatgpt_tab_id = "selected-tab"
        controller.dictation_tab_id = None

        controller.start_dictation()
        controller.finish_dictation()

        controller.browser_monitor.request_start_dictation.assert_called_once_with(
            "selected-tab"
        )
        controller.browser_monitor.request_finish_dictation.assert_called_once_with(
            "selected-tab"
        )
        self.assertIsNone(controller.dictation_tab_id)

    def test_finished_dictation_populates_app_input(self) -> None:
        controller = TrayController.__new__(TrayController)
        controller.window = Mock()

        controller._on_dictation_finished(
            True,
            "Text recognized by ChatGPT",
            "Dictation copied from ChatGPT",
        )

        controller.window.set_transcript.assert_called_once_with(
            "Text recognized by ChatGPT"
        )
        controller.window.set_microphone_state.assert_called_once_with(
            "saved",
            "Dictation copied from ChatGPT",
        )

    @patch("live_gpt.app.open_remote_debugging_settings")
    @patch("live_gpt.app.discover_cdp_endpoint")
    def test_live_endpoint_retries_without_opening_settings(
        self,
        discover_endpoint: Mock,
        open_settings: Mock,
    ) -> None:
        discover_endpoint.return_value = "ws://127.0.0.1:9222/devtools/id"
        controller = TrayController.__new__(TrayController)
        controller.window = Mock()
        controller.browser_monitor = Mock()

        controller._open_remote_debugging_settings()

        open_settings.assert_not_called()
        controller.browser_monitor.request_retry_connection.assert_called_once()
        controller.window.set_browser_status.assert_called_once_with(
            "Retrying connection… approve it in the browser"
        )


if __name__ == "__main__":
    unittest.main()
