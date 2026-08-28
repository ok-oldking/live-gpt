from __future__ import annotations

import os
import unittest
from unittest.mock import Mock, patch


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from live_gpt.app import TranscriptEditor, TrayController  # noqa: E402


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

    def test_reading_subtitle_restores_full_response(self) -> None:
        full_response = "First sentence. Second sentence. Final sentence."
        self.editor.begin_response()
        self.editor.update_response(full_response)

        self.editor.begin_reading()
        self.editor.update_reading_subtitle("Second sentence.")

        self.assertEqual(self.editor.toPlainText(), "Second sentence.")
        self.assertTrue(self.editor.property("readingMode"))

        self.editor.finish_reading()

        self.assertEqual(self.editor.toPlainText(), full_response)
        self.assertFalse(self.editor.property("readingMode"))


class TrayControllerBrowserTests(unittest.TestCase):
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
