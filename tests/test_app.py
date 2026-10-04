from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEvent, QPoint, QRect, Qt  # noqa: E402
from PySide6.QtGui import QKeySequence, QPalette  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication,
    QPushButton,
    QSystemTrayIcon,
)

from live_gpt.app import (  # noqa: E402
    HotkeyConfigDialog,
    OverlayWindow,
    TranscriptEditor,
    TrayController,
    _LocalSpeechThread,
    _QueuedLocalSpeechThread,
)
from live_gpt.config import Config  # noqa: E402
from live_gpt.localization import resolve_language  # noqa: E402
from live_gpt.screen_capture import CaptureSource  # noqa: E402


class LocalSpeechThreadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.application = QApplication.instance() or QApplication([])

    def test_streaming_generation_overlaps_chunked_output_stream(self) -> None:
        writes: list[object] = []

        class FakeOutputStream:
            def __init__(self, **kwargs: object) -> None:
                self.kwargs = kwargs

            def __enter__(self) -> "FakeOutputStream":
                return self

            def __exit__(self, *_args: object) -> None:
                return None

            def write(self, samples: object) -> None:
                writes.append(samples)

        manager = Mock()
        manager.synthesize_stream.return_value = (
            ([0.1, -0.1], 24_000, "First."),
            ([0.2, -0.2], 24_000, "Second."),
        )
        worker = _LocalSpeechThread(
            manager,
            "test-model",
            "First. Second.",
            "Ryan",
        )
        started: list[str] = []
        progress: list[object] = []
        completed: list[tuple[bool, str]] = []
        worker.started.connect(started.append)
        worker.progress.connect(progress.append)
        worker.completed.connect(lambda ok, message: completed.append((ok, message)))

        with patch.dict(
            "sys.modules",
            {"sounddevice": SimpleNamespace(OutputStream=FakeOutputStream)},
        ):
            worker.run()

        manager.synthesize_stream.assert_called_once_with(
            "test-model",
            "First. Second.",
            "Ryan",
            "auto",
        )
        manager.synthesize.assert_not_called()
        self.assertEqual(len(writes), 2)
        self.assertEqual(started, ["Playing with streaming local TTS…"])
        self.assertEqual(progress[-1], {"text": "First. Second.", "fraction": 1.0})
        self.assertTrue(completed[0][0])
        self.assertIn("First audio in", completed[0][1])

    def test_streaming_audio_trims_silence_and_fades_chunk_edges(self) -> None:
        import numpy as np

        samples = np.concatenate(
            (
                np.zeros(1000, dtype=np.float32),
                np.ones(2000, dtype=np.float32),
                np.zeros(1000, dtype=np.float32),
            )
        )

        prepared = _LocalSpeechThread._prepare_streaming_waveform(
            np,
            samples,
            1000,
        )

        self.assertEqual(prepared.shape[1], 1)
        self.assertLess(len(prepared), len(samples))
        self.assertEqual(float(prepared[0, 0]), 0.0)
        self.assertEqual(float(prepared[-1, 0]), 0.0)

    def test_continuous_native_stream_preserves_chunk_boundaries(self) -> None:
        import numpy as np

        writes: list[object] = []

        class FakeOutputStream:
            def __init__(self, **_kwargs: object) -> None:
                pass

            def __enter__(self) -> "FakeOutputStream":
                return self

            def __exit__(self, *_args: object) -> None:
                return None

            def write(self, samples: object) -> None:
                writes.append(samples)

        manager = Mock()
        manager.continuous_audio_stream = True
        manager.stream_prebuffer_seconds = 0.0
        manager.synthesize_stream.return_value = (
            ([0.0, 1.0, 0.0], 1, ""),
            ([], 1, "test"),
        )
        worker = _LocalSpeechThread(manager, "test-model", "test")

        with patch.dict(
            "sys.modules",
            {"sounddevice": SimpleNamespace(OutputStream=FakeOutputStream)},
        ):
            worker.run()

        np.testing.assert_array_equal(
            writes[0],
            np.asarray([[0.0], [1.0], [0.0]], dtype=np.float32),
        )
        self.assertEqual(len(writes), 1)

    def test_queued_speech_generates_ahead_and_reuses_output_stream(self) -> None:
        import numpy as np

        writes: list[object] = []
        second_generation_started = threading.Event()
        streams: list[object] = []

        class FakeOutputStream:
            def __init__(self, **_kwargs: object) -> None:
                streams.append(self)

            def start(self) -> None:
                pass

            def write(self, samples: object) -> None:
                if not writes:
                    self.assert_generation_is_ahead()
                writes.append(samples)

            @staticmethod
            def assert_generation_is_ahead() -> None:
                if not second_generation_started.wait(1):
                    raise AssertionError("second sentence was not generated ahead")

            def stop(self) -> None:
                pass

            def close(self) -> None:
                pass

        manager = Mock()
        manager.display_name = "Test TTS"
        manager.continuous_audio_stream = True

        def synthesize(_model, sentence, _speaker, _language):
            if sentence == "Second sentence.":
                second_generation_started.set()
            return ((np.ones(20, dtype=np.float32), 24000, sentence),)

        manager.synthesize_stream.side_effect = synthesize
        worker = _QueuedLocalSpeechThread(manager, "model", "speaker", "en")
        worker.enqueue("First sentence.", "First sentence. Second sentence.")
        worker.enqueue("Second sentence.", "First sentence. Second sentence.")
        worker.finish_queue()

        with patch.dict(
            "sys.modules",
            {"sounddevice": SimpleNamespace(OutputStream=FakeOutputStream)},
        ):
            worker.run()

        self.assertEqual(len(streams), 1)
        self.assertEqual(len(writes), 2)
        self.assertEqual(
            [call.args[1] for call in manager.synthesize_stream.call_args_list],
            ["First sentence.", "Second sentence."],
        )

    def test_response_sentence_splitter_waits_for_complete_sentence(self) -> None:
        sentences, consumed = TrayController._completed_response_sentences(
            "First sentence. Second sentence is still", final=False
        )
        self.assertEqual(sentences, ["First sentence."])
        self.assertEqual(
            "First sentence. Second sentence is still"[consumed:],
            "Second sentence is still",
        )

        sentences, consumed = TrayController._completed_response_sentences(
            "Second sentence is done", final=True
        )
        self.assertEqual(sentences, ["Second sentence is done"])
        self.assertEqual(consumed, len("Second sentence is done"))

    def test_response_rewrite_queues_sentences_after_the_spoken_position(self) -> None:
        controller = TrayController.__new__(TrayController)
        controller._local_voice_longest_text = "Original first. Partial"
        controller._local_voice_queued_sentences = ["Original first."]
        worker = _QueuedLocalSpeechThread(Mock(), "model", "speaker", "en")
        controller._local_speech_thread = worker

        controller._update_local_voice(
            "Rewritten first. Rewritten second. Final remainder",
            True,
        )

        queued = worker._sentences.get_nowait()
        remainder = worker._sentences.get_nowait()
        finished = worker._sentences.get_nowait()
        self.assertEqual(queued, "Rewritten second.")
        self.assertEqual(remainder, "Final remainder")
        self.assertIs(finished, worker._FINISHED)

    def test_temporary_shorter_response_snapshot_is_ignored(self) -> None:
        controller = TrayController.__new__(TrayController)
        controller._local_voice_longest_text = "First sentence. Partial response"
        controller._local_voice_queued_sentences = ["First sentence."]
        worker = _QueuedLocalSpeechThread(Mock(), "model", "speaker", "en")
        controller._local_speech_thread = worker

        controller._update_local_voice("short", False)

        self.assertTrue(worker._sentences.empty())


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

    def test_auto_send_button_is_removed(self) -> None:
        self.assertFalse(hasattr(self.editor, "auto_send_button"))

    def test_click_response_preserves_text_and_allows_selection_and_clear(self) -> None:
        self.editor.begin_response()
        self.editor.update_response("Keep this response")
        self.editor.finish_response()
        QTest.mouseClick(self.editor.viewport(), Qt.MouseButton.LeftButton, pos=QPoint(8, 8))
        self.assertEqual(self.editor.toPlainText(), "Keep this response")
        self.assertFalse(self.editor.isReadOnly())
        self.editor.selectAll()
        self.assertEqual(self.editor.textCursor().selectedText(), "Keep this response")
        QTest.keyClick(self.editor, Qt.Key.Key_Backspace)
        self.assertEqual(self.editor.toPlainText(), "")

    @patch("live_gpt.app.QDesktopServices.openUrl")
    def test_plain_and_named_response_links_open_browser(self, open_url) -> None:
        for text, links, url in (
            ("https://example.com", (), "https://example.com"),
            ("Read docs", (("docs", "https://example.com/docs"),), "https://example.com/docs"),
            ("[docs](https://example.com/docs)", (), "https://example.com/docs"),
        ):
            self.editor.setPlainText(text)
            self.editor.response_links = links
            cursor = self.editor.textCursor()
            cursor.setPosition(6 if text == "Read docs" else 2)
            self.editor.setTextCursor(cursor)
            QApplication.processEvents()
            point = self.editor.cursorRect(cursor).center()
            QTest.mouseClick(self.editor.viewport(), Qt.MouseButton.LeftButton, pos=point)
            self.assertEqual(open_url.call_args.args[0].toString(), url)



class SettingsDialogTests(unittest.TestCase):
    def test_sovits_pair_blocks_incomplete_settings_and_saves_both(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = Config(root / "config.json")
            ckpt, pth = root / "voice.ckpt", root / "voice.pth"
            ckpt.touch()
            pth.touch()
            dialog = HotkeyConfigDialog("Right Alt", playing_backend="sovits", config=config)
            try:
                dialog.sovits_ckpt_edit.setText(str(ckpt))
                self.assertFalse(dialog.voice_play_button.isEnabled())
                self.assertFalse(dialog.playing_check_button.isEnabled())
                self.assertEqual(config["sovits_ckpt_path"], "")
                with patch("live_gpt.app._LocalSpeechThread") as worker:
                    dialog._start_voice_play_test()
                    worker.assert_not_called()
                dialog.sovits_pth_edit.setText(str(pth))
                self.assertTrue(dialog.voice_play_button.isEnabled())
                saved = Config(config.path)
                self.assertEqual(saved["sovits_ckpt_path"], str(ckpt))
                self.assertEqual(saved["sovits_pth_path"], str(pth))
                provider = dialog._voice_provider("tts")
                self.assertEqual(provider.ckpt_path, str(ckpt.resolve()))
                self.assertEqual(provider.pth_path, str(pth.resolve()))
                dialog.sovits_ckpt_edit.clear()
                self.assertFalse(dialog.voice_play_button.isEnabled())
                dialog.sovits_pth_edit.clear()
                self.assertTrue(dialog.voice_play_button.isEnabled())
                self.assertEqual(config["sovits_ckpt_path"], "")
                self.assertEqual(config["sovits_pth_path"], "")
                dialog._voice_status_checked["tts"] = True
                dialog._set_voice_busy(False)
                self.assertTrue(dialog._voice_status_checked["tts"])
            finally:
                dialog.close()

    @classmethod
    def setUpClass(cls) -> None:
        cls.application = QApplication.instance() or QApplication([])

    def test_recording_language_filters_models_and_preserves_supported_selection(self) -> None:
        dialog = HotkeyConfigDialog("Right Alt")
        try:
            for language, expected_count in (("auto", 1), ("en", 6), ("zh", 5)):
                dialog.stt_language_combo.setCurrentIndex(dialog.stt_language_combo.findData(language))
                self.assertEqual(dialog.stt_model_combo.count(), expected_count)
                keys = [dialog.stt_model_combo.itemData(i) for i in range(expected_count)]
                self.assertIn("zh_sense_voice_small_int8", keys)
                self.assertEqual(dialog.stt_model(), "zh_sense_voice_small_int8")
            dialog.stt_model_combo.setCurrentIndex(0)
            dialog.stt_language_combo.setCurrentIndex(dialog.stt_language_combo.findData("en"))
            self.assertNotIn("zh_zipformer_ctc_int8_2025_07_03", [
                dialog.stt_model_combo.itemData(i) for i in range(dialog.stt_model_combo.count())
            ])
        finally:
            dialog.close()

    def test_settings_use_navigation_and_live_language_selection(self) -> None:
        dialog = HotkeyConfigDialog(
            QKeySequence("CapsLock"),
            QKeySequence("Ctrl+S"),
            QKeySequence("Ctrl+D"),
        )
        try:
            self.assertEqual(dialog.windowTitle(), "Live GPT settings")
            self.assertEqual(
                dialog.hotkey_section.objectName(),
                "settingsPage",
            )
            self.assertEqual(
                dialog.language_section.objectName(),
                "settingsPage",
            )
            self.assertTrue(
                dialog.windowFlags() & Qt.WindowType.FramelessWindowHint
            )
            self.assertFalse(dialog.close_button.icon().isNull())
            self.assertFalse(dialog.shortcuts_nav_button.icon().isNull())
            self.assertFalse(dialog.language_nav_button.icon().isNull())
            self.assertFalse(dialog.recording_nav_button.icon().isNull())
            self.assertFalse(dialog.playing_nav_button.icon().isNull())
            self.assertEqual(
                dialog.hold_without_screenshot_edit.keySequence(),
                "Right Ctrl",
            )
            self.assertTrue(dialog.language_combo.isEnabled())
            self.assertEqual(dialog.language(), resolve_language())
            self.assertEqual(dialog.language_combo.count(), 2)

            dialog.language_nav_button.click()
            self.assertEqual(dialog.settings_pages.currentIndex(), 1)
            dialog.language_combo.setCurrentIndex(dialog.language_combo.findData("zh"))
            self.assertEqual(
                dialog.language_combo.currentText(),
                "简体中文",
            )
            self.assertEqual(dialog.windowTitle(), "Live GPT 设置")
            self.assertEqual(dialog.recording_nav_button.text(), "录音")
            dialog.language_combo.setCurrentIndex(dialog.language_combo.findData("en"))

            dialog.shortcuts_nav_button.click()
            self.assertEqual(dialog.settings_pages.currentIndex(), 0)

            dialog.recording_nav_button.click()
            self.assertEqual(dialog.settings_pages.currentIndex(), 2)
            self.assertEqual(dialog.recording_backend(), "web")
            self.assertEqual(dialog.stt_model_combo.count(), 5)
            self.assertIn("[Offline]", dialog.stt_model_combo.itemText(0))
            streaming_index = dialog.stt_model_combo.findData(
                "zh_streaming_zipformer_small_ctc_int8_2025_04_01"
            )
            self.assertIn(
                "[Streaming]", dialog.stt_model_combo.itemText(streaming_index)
            )
            dialog._voice_record_partial("实时转写")
            self.assertEqual(dialog.stt_test_result.text(), "实时转写")
            self.assertTrue(dialog.recording_sherpa_card.isHidden())

            dialog.playing_nav_button.click()
            self.assertEqual(dialog.settings_pages.currentIndex(), 3)
            self.assertEqual(dialog.playing_backend(), "web")
            self.assertEqual(dialog.playing_backend_combo.count(), 2)
            self.assertEqual(dialog.voice_record_button.text(), "Record microphone")
            self.assertEqual(dialog.voice_play_button.text(), "Play text")
            self.assertTrue(dialog.playing_local_card.isHidden())
            self.assertTrue(dialog.recording_install_log.isHidden())
            self.assertTrue(dialog.playing_install_log.isHidden())
            self.assertEqual(dialog.recording_pypi_mirror_combo.count(), 3)
            self.assertEqual(dialog.playing_pypi_mirror_combo.count(), 3)
            aliyun = dialog.recording_pypi_mirror_combo.findData("ali")
            dialog.recording_pypi_mirror_combo.setCurrentIndex(aliyun)
            self.assertEqual(dialog.pypi_mirror(), "ali")
            self.assertEqual(
                dialog.playing_pypi_mirror_combo.currentData(), "ali"
            )

            dialog._voice_operation_type = "tts"
            dialog._voice_operation_log("Downloading package  12%\n")
            self.assertFalse(dialog.playing_install_log.isHidden())
            self.assertEqual(
                dialog.playing_install_log.text(),
                "Downloading package 12%",
            )
            dialog._voice_operation_log(
                "Downloading torch-2.11.0-cp312-cp312-win_amd64.whl (2.6 GB)"
            )
            first_progress = "━━━━━━━━━━━━━━━━━━━━──────────────────── 1.0/2.6 GB"
            latest_progress = (
                "━━━━━━━━━━━━━━━━━━━━──────────────────── "
                "1.3/2.6 GB 28.5 MB/s eta 0:00:46"
            )
            dialog._voice_operation_log(first_progress)
            dialog._voice_operation_log(latest_progress)
            self.assertEqual(
                dialog.playing_install_log.text(),
                "Downloading torch-2.11.0-cp312-cp312-win_amd64.whl (2.6 GB)\n"
                + latest_progress,
            )
            worker = Mock()
            dialog._voice_worker = worker
            dialog._voice_operation_type = "tts"
            dialog._voice_operation_action = "install"
            dialog.playing_cancel_button.show()
            dialog.playing_cancel_button.setEnabled(True)
            dialog._cancel_voice_operation()
            worker.cancel_operation.assert_called_once_with()
            self.assertFalse(dialog.playing_cancel_button.isEnabled())
            dialog._voice_worker = None
        finally:
            dialog.close()

        chinese_dialog = HotkeyConfigDialog(
            QKeySequence("CapsLock"),
            QKeySequence("Ctrl+S"),
            QKeySequence("Ctrl+D"),
            language="zh",
        )
        try:
            self.assertEqual(chinese_dialog.language(), "zh")
            self.assertEqual(
                chinese_dialog.language_combo.currentText(),
                "简体中文",
            )
        finally:
            chinese_dialog.close()

        sovits_dialog = HotkeyConfigDialog(
            QKeySequence("CapsLock"),
            QKeySequence("Ctrl+S"),
            QKeySequence("Ctrl+D"),
            playing_backend="sovits",
            sovits_text_lang="en",
            sovits_prompt_lang="zh",
            sovits_ref_audio_path="E:/voices/reference.wav",
            sovits_prompt_text="参考音频文本",
        )
        try:
            self.assertEqual(
                sovits_dialog.sovits_reference_title.text(),
                "Reference voice",
            )
            self.assertEqual(
                sovits_dialog.sovits_prompt_lang_label.text(),
                "Reference language",
            )
            self.assertEqual(
                sovits_dialog.sovits_output_title.text(),
                "Generated speech",
            )
            self.assertEqual(
                sovits_dialog.sovits_text_lang_label.text(),
                "Output language",
            )
            layout = sovits_dialog.playing_local_card.layout()
            self.assertLess(
                layout.indexOf(sovits_dialog.sovits_reference_title),
                layout.indexOf(sovits_dialog.sovits_output_title),
            )
            self.assertFalse(sovits_dialog.sovits_reference_title.isHidden())
            self.assertFalse(sovits_dialog.sovits_output_title.isHidden())
            self.assertEqual(sovits_dialog.sovits_prompt_lang(), "zh")
            self.assertEqual(sovits_dialog.sovits_text_lang(), "en")
        finally:
            sovits_dialog.close()

    def test_settings_save_every_valid_change_without_save_button(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            config = Config(path)
            dialog = HotkeyConfigDialog(
                QKeySequence("CapsLock"),
                QKeySequence("Ctrl+S"),
                QKeySequence("Ctrl+D"),
                config=config,
            )
            try:
                self.assertFalse(hasattr(dialog, "settings_buttons"))
                self.assertFalse(hasattr(dialog, "hotkey_enabled_switches"))
                self.assertFalse(hasattr(dialog, "send_edit"))
                dialog.hold_microphone_edit.clear_button.click()
                dialog.hold_without_screenshot_edit.clear_button.click()
                saved_config = Config(path)
                self.assertEqual(saved_config["hotkey_hold"], "")
                self.assertEqual(saved_config["hotkey_hold_without_screenshot"], "")
                self.assertEqual(dialog.sequences(), {"hold": "", "hold_without_screenshot": ""})
                controller = TrayController.__new__(TrayController)
                controller.config = saved_config
                self.assertEqual(controller._load_hotkey_sequences(), dialog.sequences())
                self.assertFalse(any(controller._hotkey_enabled_states().values()))
                dialog.language_combo.setCurrentIndex(dialog.language_combo.findData("zh"))
                dialog.recording_backend_combo.setCurrentIndex(
                    dialog.recording_backend_combo.findData("sherpa")
                )
                dialog.playing_backend_combo.setCurrentIndex(
                    dialog.playing_backend_combo.findData("sovits")
                )
                dialog.playing_pypi_mirror_combo.setCurrentIndex(
                    dialog.playing_pypi_mirror_combo.findData("ali")
                )
                self.assertEqual(
                    dialog.recording_pypi_mirror_combo.currentData(),
                    "ali",
                )
                dialog.stt_language_combo.setCurrentIndex(dialog.stt_language_combo.findData("en"))
                dialog.stt_model_combo.setCurrentIndex(
                    dialog.stt_model_combo.findData("en_moonshine_tiny_int8")
                )
                dialog.hold_without_screenshot_edit.setKeySequence("Right Ctrl+S")

                saved = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(saved["language"], "zh")
                self.assertEqual(saved["recording_backend"], "sherpa")
                self.assertEqual(saved["playing_backend"], "sovits")
                self.assertEqual(saved["pypi_mirror"], "ali")
                self.assertEqual(saved["stt_language"], "en")
                self.assertEqual(saved["stt_model"], "en_moonshine_tiny_int8")
            finally:
                dialog.close()


class TrayControllerBrowserTests(unittest.TestCase):
    def test_stt_model_preloads_in_background_when_selected(self) -> None:
        controller = TrayController.__new__(TrayController)
        controller.config = {
            "recording_backend": "sherpa",
            "stt_model": "en_moonshine_tiny_int8",
        }
        controller.stt_manager = Mock()
        controller._stt_preload_thread = None

        controller._start_stt_preload()
        controller._stt_preload_thread.join(timeout=2)

        controller.stt_manager.preload.assert_called_once_with(
            "en_moonshine_tiny_int8", "en"
        )

    def test_sovits_model_preloads_in_background_when_selected(self) -> None:
        controller = TrayController.__new__(TrayController)
        controller.config = {
            "playing_backend": "sovits",
            "sovits_installation": "test-model",
            "sovits_prompt_text": "", "sovits_prompt_lang": "auto",
            "sovits_ref_audio_path": "", "sovits_text_lang": "auto",
            "sovits_ckpt_path": "", "sovits_pth_path": "",
        }
        controller.sovits_manager = Mock()
        controller.sovits_manager.dependency_status.return_value = (
            True,
            "runtime verified",
        )
        controller._local_tts_preload_thread = None

        controller._start_local_tts_preload()
        controller._local_tts_preload_thread.join(timeout=2)

        controller.sovits_manager.preload.assert_called_once_with(
            "test-model"
        )

    def test_invalid_sovits_runtime_is_not_preloaded_or_locked(self) -> None:
        controller = TrayController.__new__(TrayController)
        controller.config = {
            "playing_backend": "sovits",
            "sovits_installation": "test-model",
            "sovits_prompt_text": "", "sovits_prompt_lang": "auto",
            "sovits_ref_audio_path": "", "sovits_text_lang": "auto",
            "sovits_ckpt_path": "", "sovits_pth_path": "",
        }
        controller.sovits_manager = Mock()
        controller.sovits_manager.dependency_status.return_value = (
            False,
            "torch is not installed",
        )
        controller._local_tts_preload_thread = None

        controller._start_local_tts_preload()
        controller._local_tts_preload_thread.join(timeout=2)

        controller.sovits_manager.preload.assert_not_called()

    def test_overlay_is_fifty_percent_wider(self) -> None:
        window = OverlayWindow()
        try:
            self.assertEqual(window.width(), 1140)
            self.assertEqual(window.height(), 240)
            self.assertFalse(window.configure_button.icon().isNull())
            window.show()
            QApplication.processEvents()
            self.assertLess(
                window.configure_button.geometry().left(),
                window.lock_button.geometry().left(),
            )
            self.assertLess(
                window.lock_button.geometry().left(),
                window.auto_hide_button.geometry().left(),
            )
        finally:
            window.close()

    def test_overlay_can_resize_and_lock_its_geometry(self) -> None:
        window = OverlayWindow()
        try:
            window.show()
            QApplication.processEvents()

            window.resize(1280, 320)
            QApplication.processEvents()
            self.assertEqual(window.size().width(), 1280)
            self.assertEqual(window.size().height(), 320)

            window.lock_button.click()
            locked_size = window.size()
            window.resize(900, 200)
            QApplication.processEvents()

            self.assertTrue(window.lock_button.isChecked())
            self.assertEqual(window.size(), locked_size)

            window.lock_button.click()
            window.resize(900, 200)
            QApplication.processEvents()

            self.assertFalse(window.lock_button.isChecked())
            self.assertEqual(window.size().width(), 900)
            self.assertEqual(window.size().height(), 200)
        finally:
            window.close()

    def test_every_overlay_border_and_corner_is_resizable(self) -> None:
        window = OverlayWindow()
        try:
            window.show()
            window.setGeometry(100, 100, 1000, 300)
            QApplication.processEvents()
            width = window._resize_surface.width()
            height = window._resize_surface.height()
            self.assertEqual(
                window._resize_edges_at(QPoint(1, height // 2)),
                Qt.Edge.LeftEdge,
            )
            self.assertEqual(
                window._resize_edges_at(QPoint(15, height // 2)),
                Qt.Edge.LeftEdge,
            )
            self.assertEqual(
                window._resize_edges_at(QPoint(width - 2, height // 2)),
                Qt.Edge.RightEdge,
            )
            self.assertEqual(
                window._resize_edges_at(QPoint(width // 2, 1)),
                Qt.Edge.TopEdge,
            )
            self.assertEqual(
                window._resize_edges_at(QPoint(width // 2, height - 2)),
                Qt.Edge.BottomEdge,
            )
            self.assertEqual(
                window._resize_edges_at(QPoint(1, 1)),
                Qt.Edge.TopEdge | Qt.Edge.LeftEdge,
            )

            window._begin_border_resize(
                Qt.Edge.TopEdge | Qt.Edge.LeftEdge,
                QPoint(100, 100),
            )
            window._update_border_resize(QPoint(80, 70))

            self.assertEqual(window.geometry(), QRect(80, 70, 1020, 330))
            window._end_border_resize()
        finally:
            window.close()

    def test_chrome_is_transparent_outside_and_visible_while_resizing(self) -> None:
        window = OverlayWindow()
        try:
            window.set_chatgpt_tabs([{"id": "tab", "title": "ChatGPT", "url": "https://chatgpt.com"}])
            window._set_chrome_visible(False)
            self.assertEqual(window._title_opacity.opacity(), 0.0)
            self.assertEqual(window._microphone_opacity.opacity(), 0.0)
            self.assertFalse(window.panel.property("chromeVisible"))
            self.assertFalse(
                window._resize_surface.property("chromeVisible")
            )

            window._set_chrome_visible(True)

            self.assertEqual(window._title_opacity.opacity(), 1.0)
            self.assertEqual(window._microphone_opacity.opacity(), 1.0)
            self.assertTrue(window.panel.property("chromeVisible"))
            self.assertTrue(
                window._resize_surface.property("chromeVisible")
            )
            window.show()
            QApplication.processEvents()
            corner = window.grab().toImage().pixelColor(1, 1)
            self.assertEqual(corner.alpha(), 0)

            window._begin_border_resize(Qt.Edge.LeftEdge, QPoint(0, 0))
            window._set_chrome_visible(False)

            self.assertEqual(window._title_opacity.opacity(), 1.0)
            self.assertTrue(window.panel.property("chromeVisible"))
        finally:
            window.close()

    def test_status_uses_input_hint_and_errors_are_red(self) -> None:
        window = OverlayWindow()
        try:
            self.assertFalse(hasattr(window, "status_label"))

            window.set_status("Connection failed", error=True)

            self.assertEqual(
                window.transcript_area.placeholderText(),
                "Connection failed",
            )
            self.assertEqual(
                window.transcript_area.palette()
                .color(QPalette.ColorRole.PlaceholderText)
                .name(),
                "#ff667a",
            )
        finally:
            window.close()

    @patch("live_gpt.app.capture_webp", return_value=b"screenshot")
    def test_send_captures_selected_source(self, capture: Mock) -> None:
        source = CaptureSource(
            key="window:123",
            label="Test window",
            kind="window",
            left=0,
            top=0,
            width=1200,
            height=800,
            hwnd=123,
        )
        controller = TrayController.__new__(TrayController)
        controller.window = Mock()
        controller.browser_monitor = Mock()
        controller.selected_chatgpt_tab_id = "selected-tab"

        controller._handle_send_requested("Explain this", source)

        capture.assert_called_once_with(source)
        controller.browser_monitor.request_send.assert_called_once_with(
            "selected-tab",
            "Explain this",
            b"screenshot",
        )
        controller.window.begin_response_display.assert_called_once_with(
            "Explain this"
        )

    def test_clear_queues_selected_browser_composer(self) -> None:
        controller = TrayController.__new__(TrayController)
        controller.window = Mock()
        controller.browser_monitor = Mock()
        controller.selected_chatgpt_tab_id = "selected-tab"

        controller._handle_clear_requested()

        controller.browser_monitor.request_clear.assert_called_once_with(
            "selected-tab"
        )
        controller.window.set_status.assert_called_once_with(
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
                " ".join(
                    filter(
                        None,
                        (
                            window.subtitle_line_one.text(),
                            window.subtitle_line_two.text(),
                        ),
                    )
                ),
                "Current subtitle Next subtitle",
            )

            window.finish_reading(True, "Read aloud complete")

            self.assertTrue(window.transcript_area.isHidden())
            self.assertFalse(window.subtitle_panel.isHidden())

            window.dismiss_subtitle_mode()

            self.assertFalse(window.transcript_area.isHidden())
            self.assertTrue(window.subtitle_panel.isHidden())
            self.assertEqual(window.transcript_area.toPlainText(), "Current subtitle\nNext subtitle")
        finally:
            window.close()

    @patch("live_gpt.app.QCursor.pos", return_value=QPoint(-10000, -10000))
    def test_subtitle_hover_expands_and_mouse_leave_collapses(self, _cursor) -> None:
        window = OverlayWindow()
        try:
            window.show()
            window.begin_response_display("Question")
            response = " ".join(
                f"complete-response-word-{index}" for index in range(80)
            )
            window.set_response_update("Writing…", response)
            QApplication.processEvents()
            collapsed_height = window.height()
            collapsed_panel_height = window.subtitle_panel.height()
            available = window.screen().availableGeometry()
            content_width = max(window.subtitle_panel.width() - 40, 1)
            metrics = window.subtitle_full_text.fontMetrics()
            text_height = metrics.boundingRect(
                QRect(0, 0, content_width, 16_777_215),
                Qt.TextFlag.TextWordWrap,
                response,
            ).height()
            expected_height = min(
                max(
                    collapsed_height,
                    collapsed_height
                    - collapsed_panel_height
                    + max(text_height, metrics.lineSpacing())
                    + 32,
                ),
                max(available.height() - 40, collapsed_height),
            )

            window._expand_subtitle()
            QApplication.processEvents()

            self.assertTrue(window.subtitle_line_one.isHidden())
            self.assertFalse(window.subtitle_full_text.isHidden())
            self.assertTrue(window._subtitle_outside_timer.isActive())
            self.assertEqual(
                window.subtitle_full_text.toPlainText(),
                response,
            )
            self.assertEqual(window.height(), expected_height)

            with patch(
                "live_gpt.app.QCursor.pos",
                return_value=QPoint(-10_000, -10_000),
            ):
                window._collapse_subtitle_if_outside()

            self.assertFalse(window._subtitle_expanded)
            self.assertFalse(window._subtitle_outside_timer.isActive())
            self.assertEqual(window.height(), collapsed_height)
            self.assertFalse(window.subtitle_line_one.isHidden())
            self.assertTrue(window.subtitle_full_text.isHidden())
            self.assertEqual(window.subtitle_line_one.text(), window._subtitle_lines()[0])
            self.assertEqual(window.subtitle_line_two.text(), "Writing…")
        finally:
            window.close()

    @patch("live_gpt.app.QCursor.pos", return_value=QPoint(-10000, -10000))
    def test_dragged_subtitle_keeps_new_position_after_update_and_collapse(self, _cursor) -> None:
        window = OverlayWindow()
        try:
            window.show()
            window.move(40, 300)
            window.begin_response_display("Question")
            window.set_response_update("Writing…", "Reply text. " * 40)
            QApplication.processEvents()
            base = QRect(window.geometry())
            window._expand_subtitle()
            self.assertEqual(window._subtitle_collapsed_geometry, base)
            before_drag = window.pos()
            saved = []
            window.geometry_changed.connect(saved.append)
            window.pet.begin_drag(window.frameGeometry().topLeft() + QPoint(20, 20))
            window.move(before_drag + QPoint(30, 40))
            delta = window.pos() - before_drag
            expected = base.translated(delta)
            self.assertEqual(window._subtitle_collapsed_geometry, expected)
            self.assertEqual(saved[-1], expected)
            during_drag = QRect(window.geometry())
            window.set_response_update("Writing…", "More reply text. " * 60)
            window._collapse_subtitle_if_outside()
            self.assertTrue(window._subtitle_expanded)
            self.assertEqual(window.geometry(), during_drag)
            window.pet._finish_drag()
            window._fit_expanded_subtitle_height()
            self.assertEqual(window._subtitle_collapsed_geometry, expected)
            window._collapse_subtitle()
            self.assertEqual(window.geometry(), expected)
        finally:
            window.close()

    def test_expanded_subtitle_polling_catches_missed_leave_event(self) -> None:
        window = OverlayWindow()
        try:
            window.show()
            window.begin_response_display("Question")
            window.set_response_update("Writing…", "Complete response")
            window._expand_subtitle()
            QApplication.processEvents()

            options_position = window.configure_button.mapToGlobal(
                window.configure_button.rect().center()
            )
            self.assertFalse(window.subtitle_panel.rect().contains(
                window.subtitle_panel.mapFromGlobal(options_position)
            ))
            with patch(
                "live_gpt.app.QCursor.pos",
                return_value=options_position,
            ):
                window._subtitle_outside_timer.timeout.emit()

            self.assertTrue(window._subtitle_expanded)
            self.assertTrue(window._subtitle_outside_timer.isActive())

            with patch(
                "live_gpt.app.QCursor.pos",
                return_value=QPoint(-10_000, -10_000),
            ):
                window._subtitle_outside_timer.timeout.emit()

            self.assertFalse(window._subtitle_expanded)
            self.assertFalse(window._subtitle_outside_timer.isActive())
            self.assertFalse(window.subtitle_line_one.isHidden())
            self.assertTrue(window.subtitle_full_text.isHidden())
        finally:
            window.close()

    @patch("live_gpt.app.QCursor.pos", return_value=QPoint(-10000, -10000))
    def test_tall_response_stays_expanded_at_bottom_left_hover_margin(self, cursor) -> None:
        window = OverlayWindow()
        try:
            window.show()
            window.setGeometry(100, 300, 900, 240)
            window.begin_response_display("Question")
            window.set_response_update("Writing…", "Long reply text. " * 200)
            QApplication.processEvents()
            collapsed = QRect(window.geometry())
            position = window.frameGeometry().bottomLeft() + QPoint(-5, 5)
            self.assertFalse(window.frameGeometry().contains(position))
            self.assertTrue(window._pointer_hover_bounds().contains(position))
            cursor.return_value = position
            window._track_pointer(position)
            QApplication.processEvents()
            expanded = QRect(window.geometry())
            self.assertGreater(expanded.height(), collapsed.height())

            for _ in range(5):
                window._subtitle_outside_timer.timeout.emit()
                self.assertTrue(window._subtitle_expanded)
                self.assertEqual(window.geometry(), expanded)
                window._track_pointer(position)
                QApplication.processEvents()
                self.assertTrue(window._subtitle_expanded)
                self.assertEqual(window.geometry(), expanded)

            cursor.return_value = QPoint(-10000, -10000)
            window._subtitle_outside_timer.timeout.emit()
            self.assertFalse(window._subtitle_expanded)
            self.assertEqual(window.geometry(), collapsed)
        finally:
            window.close()

    def test_privilege_warning_notifies_and_reveals_overlay(self) -> None:
        application = QApplication.instance() or QApplication([])
        controller = TrayController.__new__(TrayController)
        controller.window = OverlayWindow()
        controller.tray_icon = Mock()
        try:
            controller.window.hide()
            controller._warn_hotkey_privileges()
            self.assertFalse(controller.window.isHidden())
            self.assertFalse(controller._hotkey_privilege_notice.isHidden())
            self.assertFalse(controller.window._can_auto_hide())
            controller.tray_icon.showMessage.assert_called_once()
            self.assertIn("run it as administrator", controller._hotkey_privilege_notice.text())
            controller._hotkey_privilege_notice.accept()
        finally:
            controller.window.close()

    def test_response_does_not_repeat_status_in_preview_line(self) -> None:
        window = OverlayWindow()
        try:
            window.begin_response_display("Question")
            window.set_response_update("已搜索 6 个网站", "已搜索 6 个网站")
            self.assertEqual(window.subtitle_line_one.text(), "")
            self.assertEqual(window.subtitle_line_two.text(), "已搜索 6 个网站")
            window.set_response_update("正在阅读来源", "Actual reply")
            self.assertEqual(window.subtitle_line_one.text(), "Actual reply")
            self.assertEqual(window.subtitle_line_two.text(), "正在阅读来源")
        finally:
            window.close()

    def test_response_shows_streaming_text_and_status_until_read_aloud(self) -> None:
        window = OverlayWindow()
        try:
            window.show()
            window.begin_response_display("Sent question")
            QApplication.processEvents()
            response = " ".join(
                f"preview-word-{index}" for index in range(100)
            )

            window.set_response_update("Searching websites…", response)
            lines = window._subtitle_lines()

            self.assertGreater(len(lines), 2)
            self.assertEqual(window.subtitle_line_one.text(), lines[0])
            self.assertEqual(
                window.subtitle_line_two.text(),
                "Searching websites…",
            )

            window.begin_reading("Preparing Read aloud…")

            self.assertEqual(window._subtitle_line_index, 0)
            self.assertEqual(window.subtitle_line_one.text(), lines[0])
            self.assertEqual(window.subtitle_line_two.text(), lines[1])

            reading_lines = (
                window.subtitle_line_one.text(),
                window.subtitle_line_two.text(),
            )
            window.finish_reading(True, "Read aloud complete")
            window.resize(window.width(), window.height() + 10)
            QApplication.processEvents()

            self.assertEqual(
                (
                    window.subtitle_line_one.text(),
                    window.subtitle_line_two.text(),
                ),
                reading_lines,
            )
        finally:
            window.close()

    def test_subtitle_expands_on_overlay_hover(self) -> None:
        window = OverlayWindow()
        try:
            window.show()
            window.begin_reading("Reading aloud…")
            window.set_reading_subtitle(
                {
                    "text": "First rendered subtitle line. Second rendered line.",
                    "fraction": 0.0,
                }
            )

            QApplication.processEvents()

            window._track_pointer(window.frameGeometry().center())
            self.assertTrue(window._subtitle_expanded)
            self.assertTrue(window.subtitle_line_one.isHidden())
            self.assertTrue(window.subtitle_line_two.isHidden())
            self.assertFalse(window.subtitle_full_text.isHidden())
        finally:
            window.close()

    def test_expanded_subtitle_is_stable_during_playback_progress(self) -> None:
        window = OverlayWindow()
        try:
            window.show()
            window.begin_reading("Reading aloud…")
            response = " ".join(
                f"stable-subtitle-word-{index}" for index in range(1_000)
            )
            window.set_reading_subtitle(
                {"text": response, "fraction": 0.1}
            )
            window._expand_subtitle()
            QApplication.processEvents()

            scrollbar = window.subtitle_full_text.verticalScrollBar()
            self.assertGreater(scrollbar.maximum(), 0)
            scrollbar.setValue(scrollbar.maximum() // 3)
            scroll_position = scrollbar.value()
            geometry = window.geometry()
            revision = window.subtitle_full_text.document().revision()

            window.set_reading_subtitle(
                {"text": response, "fraction": 0.8}
            )
            QApplication.processEvents()

            self.assertEqual(scrollbar.value(), scroll_position)
            self.assertEqual(window.geometry(), geometry)
            self.assertEqual(
                window.subtitle_full_text.document().revision(),
                revision,
            )
        finally:
            window.close()

    def test_reading_subtitle_collapses_to_two_lines_after_mouse_leaves(
        self,
    ) -> None:
        window = OverlayWindow()
        try:
            window.show()
            window.begin_reading("Reading aloud…")
            response = " ".join(
                f"current-playback-word-{index}" for index in range(80)
            )
            window.set_reading_subtitle(
                {
                    "text": response,
                    "fraction": 0.0,
                }
            )
            window._expand_subtitle()
            QApplication.processEvents()
            self.assertTrue(window._subtitle_expanded)

            lines = window._subtitle_lines()
            weights = [max(len(line), 12) for line in lines]
            after_first_line = (weights[0] + 0.1) / sum(weights)
            window.set_reading_subtitle(
                {"text": response, "fraction": after_first_line}
            )

            with patch(
                "live_gpt.app.QCursor.pos",
                return_value=QPoint(-10_000, -10_000),
            ):
                current_lines = (
                    window.subtitle_line_one.text(),
                    window.subtitle_line_two.text(),
                )
                window._collapse_subtitle_if_outside()

            self.assertFalse(window._subtitle_expanded)
            self.assertFalse(window.subtitle_line_one.isHidden())
            self.assertFalse(window.subtitle_line_two.isHidden())
            self.assertTrue(window.subtitle_full_text.isHidden())
            self.assertEqual(
                (
                    window.subtitle_line_one.text(),
                    window.subtitle_line_two.text(),
                ),
                current_lines,
            )
        finally:
            window.close()

    def test_subtitle_lines_use_the_available_panel_width(self) -> None:
        window = OverlayWindow()
        try:
            window.show()
            window.begin_reading("Reading aloud…")
            QApplication.processEvents()
            subtitle = " ".join(
                f"subtitle-word-{index}" for index in range(60)
            )

            window.set_reading_subtitle(subtitle)

            available = window.subtitle_panel.width() - 32
            metrics = window.subtitle_line_one.fontMetrics()
            first_width = metrics.horizontalAdvance(
                window.subtitle_line_one.text()
            )
            second_width = metrics.horizontalAdvance(
                window.subtitle_line_two.text()
            )
            self.assertGreater(first_width, available * 0.7)
            self.assertGreater(second_width, available * 0.7)
            self.assertLessEqual(first_width, available)
            self.assertLessEqual(second_width, available)
        finally:
            window.close()

    def test_subtitle_wraps_cjk_and_long_unbroken_text(self) -> None:
        window = OverlayWindow()
        try:
            window.show()
            window.begin_reading("Reading aloud…")
            QApplication.processEvents()
            available = window.subtitle_panel.width() - 32
            metrics = window.subtitle_line_one.fontMetrics()

            for subtitle in ("这是一个没有空格的中文字幕句子" * 12, "x" * 300):
                window.set_reading_subtitle(subtitle)
                lines = window._subtitle_lines()

                self.assertGreater(len(lines), 2)
                self.assertTrue(
                    all(
                        metrics.horizontalAdvance(line) <= available
                        for line in lines
                    )
                )
                self.assertEqual("".join(lines), subtitle)
        finally:
            window.close()

    def test_subtitle_progress_rolls_exactly_one_rendered_line(self) -> None:
        window = OverlayWindow()
        try:
            window.show()
            window.begin_reading("Reading aloud…")
            QApplication.processEvents()
            text = " ".join(f"spoken-word-{index}" for index in range(80))

            window.set_reading_subtitle({"text": text, "fraction": 0.0})
            previous_second_line = window.subtitle_line_two.text()
            lines = window._subtitle_lines()
            weights = [max(len(line), 12) for line in lines]
            after_first_line = (weights[0] + 0.1) / sum(weights)

            window.set_reading_subtitle(
                {"text": text, "fraction": after_first_line}
            )

            self.assertTrue(previous_second_line)
            self.assertEqual(
                window.subtitle_line_one.text(),
                previous_second_line,
            )
            self.assertEqual(window._subtitle_line_index, 1)
        finally:
            window.close()

    def test_capture_selector_keeps_no_screenshot_first(self) -> None:
        window = OverlayWindow()
        source = CaptureSource(
            "display:1",
            "Screenshot desktop",
            "display",
            0,
            0,
            1920,
            1080,
        )
        try:
            window.set_capture_sources([source])

            self.assertEqual(window.capture_source_combo.itemText(0), "No screenshot")
            self.assertIsNone(window.capture_source_combo.itemData(0))
            self.assertEqual(window.capture_source_combo.itemData(1), source)
        finally:
            window.close()

    def test_capture_selector_restores_and_updates_preference(self) -> None:
        window = OverlayWindow()
        first = CaptureSource(
            "display:1",
            "Screenshot desktop 1",
            "display",
            0,
            0,
            1920,
            1080,
        )
        second = CaptureSource(
            "display:2",
            "Screenshot desktop 2",
            "display",
            1920,
            0,
            1920,
            1080,
        )
        preferences: list[str] = []
        window.capture_source_selected.connect(preferences.append)
        try:
            window.set_preferred_capture_source(second.key)
            window.set_capture_sources([first, second])

            self.assertEqual(window.capture_source_combo.currentData(), second)
            self.assertEqual(preferences, [])

            window.capture_source_combo.setCurrentIndex(1)
            self.assertEqual(preferences, [first.key])
        finally:
            window.close()

    def test_chatgpt_selector_restores_preferred_url(self) -> None:
        window = OverlayWindow()
        selected_tabs: list[str] = []
        preferences: list[str] = []
        window.chatgpt_tab_selected.connect(selected_tabs.append)
        window.chatgpt_preference_changed.connect(preferences.append)
        try:
            window.set_preferred_chatgpt_window(
                "https://chatgpt.com/c/second"
            )
            window.set_chatgpt_tabs(
                [
                    {
                        "id": "first",
                        "title": "First",
                        "url": "https://chatgpt.com/c/first",
                    },
                    {
                        "id": "second",
                        "title": "Second",
                        "url": "https://chatgpt.com/c/second",
                    },
                ]
            )

            self.assertEqual(window.chatgpt_tab_combo.currentData(), "second")
            self.assertEqual(selected_tabs, ["second"])
            self.assertEqual(preferences, [])

            window.chatgpt_tab_combo.setCurrentIndex(0)
            self.assertEqual(
                preferences,
                ["https://chatgpt.com/c/first"],
            )
        finally:
            window.close()

    def test_selected_screenshot_adds_text_only_send_action(self) -> None:
        window = OverlayWindow()
        source = CaptureSource(
            "display:1",
            "Screenshot desktop",
            "display",
            0,
            0,
            1920,
            1080,
        )
        requests: list[tuple[str, object]] = []
        window.send_requested.connect(
            lambda text, selected: requests.append((text, selected))
        )
        try:
            window.set_chatgpt_tabs(
                [
                    {
                        "id": "tab",
                        "title": "ChatGPT",
                        "url": "https://chatgpt.com",
                    }
                ]
            )
            window.set_capture_sources([source])
            window.capture_source_combo.setCurrentIndex(1)
            window.set_transcript("Explain this")

            self.assertFalse(
                window.send_without_screenshot_button.isHidden()
            )
            self.assertEqual(
                window.send_without_screenshot_button.text(),
                "No Screenshot",
            )
            self.assertFalse(
                window.send_without_screenshot_button.icon().isNull()
            )
            self.assertEqual(window.send_button.text(), "With Screenshot")
            self.assertLess(
                window.clear_button.geometry().left(),
                window.send_without_screenshot_button.geometry().left(),
            )
            window.send_without_screenshot_button.click()

            self.assertEqual(requests, [("Explain this", None)])
        finally:
            window.close()


    def test_selected_screenshot_can_be_sent_without_text(self) -> None:
        window = OverlayWindow()
        source = CaptureSource(
            "display:1",
            "Screenshot desktop",
            "display",
            0,
            0,
            1920,
            1080,
        )
        requests: list[tuple[str, object]] = []
        window.send_requested.connect(
            lambda text, selected: requests.append((text, selected))
        )
        try:
            window.set_chatgpt_tabs(
                [{"id": "tab", "title": "ChatGPT", "url": "https://chatgpt.com"}]
            )
            window.set_capture_sources([source])
            window.capture_source_combo.setCurrentIndex(1)

            self.assertEqual(window.transcript_area.toPlainText(), "")
            self.assertFalse(window.send_button.isHidden())
            self.assertEqual(window.send_button.text(), "With Screenshot")
            self.assertTrue(
                window.send_without_screenshot_button.isHidden()
            )

            window.send_button.click()

            self.assertEqual(requests, [("", source)])
        finally:
            window.close()

    def test_every_overlay_button_has_a_tooltip(self) -> None:
        window = OverlayWindow()
        try:
            buttons = window.findChildren(QPushButton)
            self.assertTrue(buttons)
            self.assertEqual(
                [button.objectName() for button in buttons if not button.toolTip()],
                [],
            )
        finally:
            window.close()

    @patch("live_gpt.app.QCursor.pos", return_value=QPoint(-10000, -10000))
    def test_auto_hide_reveals_for_activity_and_hides_afterwards(self, _cursor) -> None:
        window = OverlayWindow()
        try:
            window.set_chatgpt_tabs([{"id": "tab", "title": "ChatGPT", "url": "https://chatgpt.com"}])
            window.show()
            window.auto_hide_button.click()
            QApplication.processEvents()

            self.assertTrue(window.auto_hide_enabled)
            self.assertTrue(window.isHidden())

            window.begin_dictation_waiting()
            QApplication.processEvents()
            self.assertFalse(window.isHidden())

            window.end_dictation_display()
            QApplication.processEvents()
            self.assertFalse(window.isHidden())
            self.assertFalse(window._auto_hide_timer.isActive())

            window.begin_response_display()
            window.set_response_update("Writing…", "Incoming reply")
            QApplication.processEvents()
            self.assertFalse(window.isHidden())

            window.begin_reading("Reading aloud…")
            window.finish_reading(True, "Read aloud complete")
            self.assertTrue(window._auto_hide_timer.isActive())
            self.assertEqual(window._auto_hide_timer.interval(), 5_000)

            window._hide_for_auto_hide()
            self.assertTrue(window.isHidden())
        finally:
            window.close()

    def test_response_updates_do_not_override_active_local_subtitles(self) -> None:
        window = OverlayWindow()
        try:
            window.begin_response_display()
            window.set_response_update(
                "ChatGPT is responding…", "First sentence. Second sentence"
            )
            window.begin_reading("Playing with streaming GPT-SoVITS…")
            window.set_reading_subtitle(
                {
                    "text": "First sentence. Second sentence",
                    "spoken_characters": 15,
                    "fraction": 0.45,
                }
            )

            longer = "First sentence. Second sentence is still growing."
            window.set_response_update("ChatGPT is responding…", longer)
            window.set_response_finished(True, "Reply complete · Local voice queued")

            self.assertTrue(window._subtitle_reading_active)
            self.assertEqual(window.transcript_area.placeholderText(), "Reading aloud…")
            self.assertAlmostEqual(window._reading_fraction, 15 / len(longer))
            self.assertNotEqual(
                window.subtitle_line_one.text(), "Reply complete · Local voice queued"
            )
            self.assertFalse(window._auto_hide_timer.isActive())
        finally:
            window.close()

    def test_auto_hide_waits_while_unsent_dictation_is_present(self) -> None:
        window = OverlayWindow()
        try:
            window.set_chatgpt_tabs([{"id": "tab", "title": "ChatGPT", "url": "https://chatgpt.com"}])
            window.show()
            window.set_transcript("Unsent dictated text")

            window.auto_hide_button.click()
            QApplication.processEvents()

            self.assertTrue(window.auto_hide_enabled)
            self.assertFalse(window.isHidden())
            self.assertFalse(window._auto_hide_timer.isActive())

            window.clear_transcript()
            window.schedule_auto_hide()
            QApplication.processEvents()

            self.assertTrue(window.isHidden())
        finally:
            window.close()

    def test_normal_send_keeps_selected_screenshot(self) -> None:
        window = OverlayWindow()
        source = CaptureSource(
            "display:1",
            "Screenshot desktop",
            "display",
            0,
            0,
            1920,
            1080,
        )
        requests: list[tuple[str, object]] = []
        window.send_requested.connect(
            lambda text, selected: requests.append((text, selected))
        )
        try:
            window.set_chatgpt_tabs(
                [{"id": "tab", "title": "ChatGPT", "url": "https://chatgpt.com"}]
            )
            window.set_capture_sources([source])
            window.capture_source_combo.setCurrentIndex(1)
            window.set_transcript("Explain this")

            window.send_button.click()

            self.assertEqual(requests, [("Explain this", source)])
        finally:
            window.close()

    def test_overlay_send_restores_focus_but_hotkey_send_does_not(self) -> None:
        window = OverlayWindow()
        focus_restorer = Mock()
        focus_restorer.restore_previous.return_value = True
        window._focus_restorer = focus_restorer
        try:
            window.set_chatgpt_tabs(
                [{"id": "tab", "title": "ChatGPT", "url": "https://chatgpt.com"}]
            )
            window.set_transcript("Send from overlay")

            window.send_button.click()
            QApplication.processEvents()

            focus_restorer.restore_previous.assert_called_once_with()

            focus_restorer.reset_mock()
            window.set_transcript("Send from global hotkey")
            window.request_send_from_hotkey(True)
            QApplication.processEvents()

            focus_restorer.restore_previous.assert_not_called()
        finally:
            window.close()

    def test_dictation_states_replace_input_area(self) -> None:
        window = OverlayWindow()
        try:
            window.begin_dictation_waiting()

            self.assertTrue(window.transcript_area.isHidden())
            self.assertFalse(window.dictation_panel.isHidden())
            self.assertEqual(
                window.dictation_state_label.text(),
                "Waiting for the browser to start listening…",
            )

            window.set_dictation_listening()
            self.assertEqual(window.dictation_state_label.text(), "Listening…")

            window.set_dictation_partial("实时转写")
            self.assertEqual(
                window.dictation_state_label.text(),
                "Listening…\n\n实时转写",
            )

            window.end_dictation_display()
            self.assertFalse(window.transcript_area.isHidden())
            self.assertTrue(window.dictation_panel.isHidden())
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
        controller._on_dictation_started(
            True,
            "Browser dictation is listening",
        )
        controller._dictation_pressed_since = time.monotonic() - 0.6
        controller._dictation_listening_since = time.monotonic() - 0.6
        controller.finish_dictation()

        controller.browser_monitor.request_start_dictation.assert_called_once_with(
            "selected-tab"
        )
        controller.window.clear_transcript.assert_called_once_with()
        controller.browser_monitor.request_finish_dictation.assert_called_once_with(
            "selected-tab"
        )
        controller.window.begin_dictation_waiting.assert_called_once_with()
        controller.window.set_dictation_listening.assert_called_once_with()
        controller.window.set_dictation_finishing.assert_called_once_with()
        self.assertEqual(controller._dictation_state, "finishing")

    def test_local_streaming_partial_updates_visible_dictation_text(self) -> None:
        controller = TrayController.__new__(TrayController)
        controller.window = Mock()
        controller._dictation_state = "listening"

        controller._on_local_dictation_partial("live words")

        controller.window.set_dictation_partial.assert_called_once_with(
            "live words",
            finishing=False,
        )

    @patch("live_gpt.app.capture_webp", return_value=b"hold-screenshot")
    def test_screenshot_is_uploaded_after_half_second_hold(
        self,
        capture: Mock,
    ) -> None:
        release_source = CaptureSource(
            key="window:release",
            label="Window at release",
            kind="window",
            left=0,
            top=0,
            width=1200,
            height=800,
            hwnd=123,
        )
        later_source = CaptureSource(
            key="window:later",
            label="Window selected later",
            kind="window",
            left=0,
            top=0,
            width=1000,
            height=700,
            hwnd=456,
        )
        controller = TrayController.__new__(TrayController)
        controller.window = Mock()
        controller.window.capture_source_combo.currentData.return_value = (
            release_source
        )
        controller.browser_monitor = Mock()
        controller.selected_chatgpt_tab_id = "selected-tab"
        controller.dictation_tab_id = "selected-tab"
        controller._dictation_state = "listening"
        controller._dictation_input_held = True
        controller._dictation_pressed_since = time.monotonic() - 0.6
        controller._dictation_listening_since = time.monotonic() - 0.6
        controller._dictation_press_generation = 1
        controller._dictation_attachment_tab_id = None
        controller._pending_dictation_capture = None

        controller._upload_dictation_screenshot_after_hold(1)

        capture.assert_called_once_with(release_source)
        controller.browser_monitor.request_replace_attachment.assert_called_once_with(
            "selected-tab",
            b"hold-screenshot",
        )
        controller.finish_dictation()

        controller.browser_monitor.request_finish_dictation.assert_called_once_with(
            "selected-tab"
        )

        controller._handle_send_requested("Dictated text", later_source)

        capture.assert_called_once_with(release_source)
        controller.browser_monitor.request_send.assert_called_once_with(
            "selected-tab",
            "Dictated text",
            None,
            preserve_attachments=True,
        )

    def test_new_recording_removes_the_previous_pending_screenshot(self) -> None:
        controller = TrayController.__new__(TrayController)
        controller.window = Mock()
        controller.window.transcript_area.is_showing_response = False
        controller.browser_monitor = Mock()
        controller.selected_chatgpt_tab_id = "selected-tab"
        controller.dictation_tab_id = None
        controller._dictation_state = "idle"
        controller._dictation_input_held = False
        controller._dictation_press_generation = 1
        controller._dictation_attachment_tab_id = "selected-tab"
        controller._pending_dictation_capture = Mock()

        controller.start_dictation()

        controller.browser_monitor.request_clear_attachments.assert_called_once_with(
            "selected-tab"
        )
        self.assertIsNone(controller._dictation_attachment_tab_id)

    def test_record_hotkey_stops_playback_and_retains_dictation(self) -> None:
        controller = TrayController.__new__(TrayController)
        controller.window = Mock()
        controller.window.transcript_area.is_showing_response = False
        controller.browser_monitor = Mock()
        controller.selected_chatgpt_tab_id = "selected-tab"
        controller.dictation_tab_id = None
        controller._dictation_state = "idle"
        controller._dictation_input_held = False
        controller._dictation_press_generation = 0
        controller._dictation_attachment_tab_id = None
        playback = Mock()
        controller._local_speech_thread = playback

        controller.start_dictation(
            include_screenshot=False,
        )

        playback.request_stop.assert_called_once_with()
        controller.browser_monitor.request_stop_reading.assert_called_once_with()
        controller.window.finish_reading.assert_called_once_with(
            False,
            "Playback stopped for recording",
        )

        controller._dictation_state = "finishing"
        controller._dictation_input_held = False
        controller._on_dictation_finished(True, "Dictated text", "Finished")

        controller.window.send_requested.emit.assert_not_called()

    def test_interrupted_reply_does_not_restart_voice_during_recording(self) -> None:
        controller = TrayController.__new__(TrayController)
        controller._local_voice_interrupted = True
        controller._local_speech_thread = None
        controller._local_voice_longest_text = ""
        controller._local_voice_queued_sentences = []

        controller._update_local_voice("Old reply continues.", False)

        self.assertIsNone(controller._local_speech_thread)
        self.assertEqual(controller._local_voice_longest_text, "")

    @patch("live_gpt.app.capture_webp")
    def test_record_without_screenshot_hotkey_never_captures(
        self,
        capture: Mock,
    ) -> None:
        controller = TrayController.__new__(TrayController)
        controller.window = Mock()
        controller.browser_monitor = Mock()
        controller.selected_chatgpt_tab_id = "selected-tab"
        controller._dictation_press_generation = 3
        controller._dictation_input_held = True
        controller._dictation_state = "listening"
        controller._dictation_include_screenshot = False

        controller._upload_dictation_screenshot_after_hold(3)

        capture.assert_not_called()
        self.assertIsNone(controller._pending_dictation_capture.source)

    @patch("live_gpt.app.capture_webp")
    def test_cancelled_hold_does_not_upload_a_screenshot(self, capture: Mock) -> None:
        controller = TrayController.__new__(TrayController)
        controller.window = Mock()
        controller.browser_monitor = Mock()
        controller.selected_chatgpt_tab_id = "selected-tab"
        controller._dictation_press_generation = 2
        controller._dictation_input_held = False
        controller._dictation_state = "cancelling"

        controller._upload_dictation_screenshot_after_hold(2)

        capture.assert_not_called()
        controller.browser_monitor.request_replace_attachment.assert_not_called()

    def test_short_mouse_dictation_is_cancelled(self) -> None:
        controller = TrayController.__new__(TrayController)
        controller.window = Mock()
        controller.window.transcript_area.is_showing_response = False
        controller.browser_monitor = Mock()
        controller.selected_chatgpt_tab_id = "selected-tab"
        controller.dictation_tab_id = None
        controller._dictation_state = "idle"
        controller._dictation_input_held = False
        controller._dictation_listening_since = None

        controller.start_dictation()
        controller._on_dictation_started(True, "Listening")
        controller.finish_dictation()

        controller.browser_monitor.request_cancel_dictation.assert_called_once_with(
            "selected-tab"
        )
        controller.browser_monitor.request_finish_dictation.assert_not_called()
        self.assertEqual(controller._dictation_state, "cancelling")
        controller.window.clear_transcript.assert_not_called()

    def test_release_before_browser_listens_cancels_when_ready(self) -> None:
        controller = TrayController.__new__(TrayController)
        controller.window = Mock()
        controller.window.transcript_area.is_showing_response = False
        controller.browser_monitor = Mock()
        controller.selected_chatgpt_tab_id = "selected-tab"
        controller.dictation_tab_id = None
        controller._dictation_state = "idle"
        controller._dictation_input_held = False
        controller._dictation_listening_since = None

        controller.start_dictation()
        controller.finish_dictation()
        controller._on_dictation_started(True, "Listening")

        controller.browser_monitor.request_cancel_dictation.assert_called_once_with(
            "selected-tab"
        )

    def test_press_during_cancel_restarts_only_if_still_held(self) -> None:
        controller = TrayController.__new__(TrayController)
        controller.window = Mock()
        controller.window.transcript_area.is_showing_response = False
        controller.browser_monitor = Mock()
        controller.selected_chatgpt_tab_id = "selected-tab"
        controller.dictation_tab_id = None
        controller._dictation_state = "idle"
        controller._dictation_input_held = False
        controller._dictation_listening_since = None

        controller.start_dictation()
        controller._on_dictation_started(True, "Listening")
        controller.finish_dictation()
        controller.start_dictation()
        controller._on_dictation_finished(
            True,
            "Original text",
            "Dictation cancelled",
        )

        self.assertEqual(
            controller.browser_monitor.request_start_dictation.call_count,
            2,
        )
        self.assertEqual(controller._dictation_state, "starting")

    def test_input_and_microphone_require_a_chatgpt_window(self) -> None:
        window = OverlayWindow()
        try:
            self.assertFalse(window.transcript_area.isEnabled())
            self.assertFalse(window.microphone_button.isEnabled())

            window.set_chatgpt_tabs(
                [{"id": "tab", "title": "ChatGPT", "url": "https://chatgpt.com"}]
            )

            self.assertTrue(window.transcript_area.isEnabled())
            self.assertTrue(window.microphone_button.isEnabled())

            window.set_chatgpt_tabs([])

            self.assertFalse(window.transcript_area.isEnabled())
            self.assertFalse(window.microphone_button.isEnabled())
        finally:
            window.close()

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
        controller.window.end_dictation_display.assert_called_once_with()
        controller.window.set_microphone_state.assert_called_once_with(
            "saved",
            "Dictation copied from ChatGPT",
        )
        controller.window.show_for_auto_hide.assert_called_once_with()
        controller.window.schedule_auto_hide.assert_not_called()

    def test_finished_dictation_waits_for_manual_send(self) -> None:
        controller = TrayController.__new__(TrayController)
        controller.window = Mock()
        controller._dictation_input_held = False

        controller._on_dictation_finished(
            True,
            "Text recognized by ChatGPT",
            "Dictation copied from ChatGPT",
        )

        controller.window.set_transcript.assert_called_once_with(
            "Text recognized by ChatGPT"
        )
        controller.window.send_requested.emit.assert_not_called()
        controller.window.show_for_auto_hide.assert_called_once_with()
        controller.window.request_send_from_hotkey.assert_not_called()

    def test_only_valid_hotkey_dictation_sends_automatically(self) -> None:
        for send_on_finish, screenshot, text, success, state, expected in (
            (True, True, "Hello", True, "finishing", True),
            (True, False, "Hello", True, "finishing", True),
            (False, True, "Hello", True, "finishing", False),
            (True, True, "A", True, "finishing", False),
            (True, True, "", True, "finishing", False),
            (True, True, "Hello", False, "finishing", False),
            (True, True, "Hello", True, "cancelling", False),
        ):
            with self.subTest(send_on_finish=send_on_finish, screenshot=screenshot, text=text, success=success, state=state):
                controller = TrayController.__new__(TrayController)
                controller.window = Mock()
                controller.browser_monitor = Mock()
                controller._dictation_send_on_finish = send_on_finish
                controller._dictation_include_screenshot = screenshot
                controller._dictation_input_held = False
                controller._dictation_state = state
                controller._on_dictation_finished(success, text, "Finished")
                if expected:
                    controller.window.request_send_from_hotkey.assert_called_once_with(screenshot)
                else:
                    controller.window.request_send_from_hotkey.assert_not_called()

    def test_one_character_voice_result_is_not_sent(self) -> None:
        controller = TrayController.__new__(TrayController)
        controller.window = Mock()
        controller._dictation_input_held = False

        controller._on_dictation_finished(
            True,
            "A",
            "Dictation copied from ChatGPT",
        )

        controller.window.set_transcript.assert_called_once_with("A")
        controller.window.send_requested.emit.assert_not_called()
        controller.window.show_for_auto_hide.assert_called_once_with()
        controller.window.set_microphone_state.assert_called_once_with(
            "saved",
            "Voice input must contain at least 2 characters to send",
        )
        self.assertEqual(controller._short_voice_text, "A")

    def test_two_character_voice_result_waits_for_manual_send(self) -> None:
        controller = TrayController.__new__(TrayController)
        controller.window = Mock()
        controller._dictation_input_held = False

        controller._on_dictation_finished(True, "OK", "Finished")

        controller.window.send_requested.emit.assert_not_called()
        self.assertIsNone(controller._short_voice_text)

    def test_unchanged_one_character_voice_result_cannot_be_sent(self) -> None:
        controller = TrayController.__new__(TrayController)
        controller.window = Mock()
        controller.browser_monitor = Mock()
        controller.selected_chatgpt_tab_id = "selected-tab"
        controller._short_voice_text = "A"

        controller._handle_send_requested("A", None)

        controller.browser_monitor.request_send.assert_not_called()
        controller.window.begin_response_display.assert_not_called()
        controller.window.set_status.assert_called_once_with(
            "Voice input must contain at least 2 characters to send",
            error=True,
        )

    def test_manually_typed_one_character_can_still_be_sent(self) -> None:
        controller = TrayController.__new__(TrayController)
        controller.window = Mock()
        controller.browser_monitor = Mock()
        controller.selected_chatgpt_tab_id = "selected-tab"

        controller._handle_send_requested("A", None)

        controller.browser_monitor.request_send.assert_called_once_with(
            "selected-tab",
            "A",
            None,
        )

    def test_tray_double_click_disables_auto_hide_before_showing(self) -> None:
        controller = TrayController.__new__(TrayController)
        controller.window = Mock()
        controller.show_window = Mock()

        controller._handle_activation(
            QSystemTrayIcon.ActivationReason.DoubleClick
        )

        controller.window.disable_auto_hide.assert_called_once_with()
        controller.show_window.assert_called_once_with()

    def test_empty_dictation_returns_to_auto_hidden_state(self) -> None:
        controller = TrayController.__new__(TrayController)
        controller.window = Mock()

        controller._on_dictation_finished(
            True,
            "",
            "No dictated text",
        )

        controller.window.show_for_auto_hide.assert_not_called()
        controller.window.schedule_auto_hide.assert_called_once_with()

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
