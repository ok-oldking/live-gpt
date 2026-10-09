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

from PySide6.QtCore import QEvent, QPoint, QRect, Qt, QUrl  # noqa: E402
from PySide6.QtGui import QKeySequence, QPalette  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication,
    QPlainTextEdit,
    QPushButton,
    QTextBrowser,
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

    def test_chinese_sentences_do_not_require_spaces_after_punctuation(self) -> None:
        text = (
            "打开 https://github.com/WowUp/WowUp/releases。"
            "下载 2.24.0-beta 的 Setup.exe！"
            "“装好了吗？”重新打开"
        )
        sentences, consumed = TrayController._completed_response_sentences(
            text, final=False
        )
        self.assertEqual(
            sentences,
            [
                "打开 https://github.com/WowUp/WowUp/releases。",
                "下载 2.24.0-beta 的 Setup.exe！",
                "“装好了吗？”",
            ],
        )
        self.assertEqual(text[consumed:], "重新打开")

    def test_response_rewrite_queues_changed_sentences(self) -> None:
        controller = TrayController.__new__(TrayController)
        controller._local_voice_longest_text = "Original first. Partial"
        controller._local_voice_queued_sentences = ["Original first."]
        worker = _QueuedLocalSpeechThread(Mock(), "model", "speaker", "en")
        controller._local_speech_thread = worker

        controller._update_local_voice(
            "Rewritten first. Rewritten second. Final remainder",
            True,
        )

        self.assertEqual(worker._sentences.get_nowait(), "Rewritten first.")
        self.assertEqual(worker._sentences.get_nowait(), "Rewritten second.")
        self.assertEqual(worker._sentences.get_nowait(), "Final remainder")
        self.assertIs(worker._sentences.get_nowait(), worker._FINISHED)
        self.assertEqual(
            controller._local_voice_queued_sentences,
            ["Rewritten first.", "Rewritten second.", "Final remainder"],
        )

    def test_stream_revisions_do_not_repeat_unchanged_sentences(self) -> None:
        controller = TrayController.__new__(TrayController)
        controller._local_voice_longest_text = ""
        controller._local_voice_queued_sentences = []
        worker = _QueuedLocalSpeechThread(Mock(), "model", "speaker", "en")
        controller._local_speech_thread = worker

        controller._update_local_voice("First.\n- Shared history. Last.", False)
        controller._update_local_voice("Revised first.\nShared history. Last. Extra.", False)
        controller._update_local_voice("First.\n- Shared history. Last. Extra. Final.", True)

        queued = []
        while not worker._sentences.empty():
            queued.append(worker._sentences.get_nowait())
        self.assertEqual(queued[:-1], [
            "First.", "- Shared history.", "Last.", "Revised first.", "Extra.", "Final.",
        ])
        self.assertIs(queued[-1], worker._FINISHED)

    def test_intentional_sentence_repetition_is_still_read(self) -> None:
        controller = TrayController.__new__(TrayController)
        controller._local_voice_longest_text = ""
        controller._local_voice_queued_sentences = []
        worker = _QueuedLocalSpeechThread(Mock(), "model", "speaker", "en")
        controller._local_speech_thread = worker
        controller._update_local_voice("Again.", False)
        controller._update_local_voice("Again. Again.", False)
        controller._update_local_voice("Again. Again.", True)
        self.assertEqual(worker._sentences.get_nowait(), "Again.")
        self.assertEqual(worker._sentences.get_nowait(), "Again.")
        self.assertIs(worker._sentences.get_nowait(), worker._FINISHED)
        self.assertTrue(worker._sentences.empty())

    def test_late_final_snapshot_does_not_restart_finished_playback(self) -> None:
        controller = TrayController.__new__(TrayController)
        controller._local_voice_longest_text = ""
        controller._local_voice_queued_sentences = []
        worker = _QueuedLocalSpeechThread(Mock(), "model", "speaker", "en")
        controller._local_speech_thread = worker
        controller._update_local_voice("The answer.", True)
        controller._local_speech_finished()
        with patch.object(controller, "_ensure_local_speech_worker") as create:
            controller._update_local_voice("The answer.", True)
        create.assert_not_called()

        controller.window = Mock()
        controller.browser_monitor = Mock()
        controller.selected_chatgpt_tab_id = "selected-tab"
        controller._handle_send_requested("Next question", None)
        next_worker = _QueuedLocalSpeechThread(Mock(), "model", "speaker", "en")
        controller._local_speech_thread = next_worker
        controller._update_local_voice("The answer.", True)
        self.assertEqual(next_worker._sentences.get_nowait(), "The answer.")

    def test_audio_without_word_timestamps_advances_subtitles_during_sentence(self) -> None:
        import numpy as np

        manager = Mock()
        manager.continuous_audio_stream = True
        manager.synthesize_stream.return_value = [
            (np.ones(1000, dtype=np.float32), 1000, ""),
            (np.ones(1000, dtype=np.float32), 1000, ""),
        ]
        text = "A long sentence with enough words to span several subtitle lines."
        worker = _QueuedLocalSpeechThread(manager, "model", "speaker", "en")
        worker.enqueue(text, text)
        worker.finish_queue()
        progress = []
        events = []
        worker.progress.connect(lambda update: progress.append(update))
        worker.completed.connect(lambda *_args: events.append("completed"))
        output = Mock()
        output.latency = 0.16
        output.stop.side_effect = lambda: events.append("drained")
        with patch.dict("sys.modules", {"sounddevice": SimpleNamespace(OutputStream=Mock(return_value=output))}):
            worker.run()

        intermediate = [update["spoken_characters"] for update in progress[:-1]]
        self.assertGreater(len(intermediate), 10)
        self.assertEqual(intermediate, sorted(intermediate))
        self.assertTrue(any(0 < character < len(text) for character in intermediate))
        self.assertEqual(progress[-1]["fraction"], 1.0)
        self.assertLess(events.index("drained"), events.index("completed"))
        self.assertLessEqual(max(len(call.args[0]) for call in output.write.call_args_list), 80)

    def test_replaced_intermediate_text_does_not_offset_final_subtitles(self) -> None:
        import numpy as np

        manager = Mock()
        manager.continuous_audio_stream = True
        manager.synthesize_stream.side_effect = lambda *_args: [
            (np.ones(20, dtype=np.float32), 24000, ""),
        ]
        worker = _QueuedLocalSpeechThread(manager, "model", "speaker", "en")
        worker.enqueue("An obsolete intermediate sentence with a lot of text.", "Intermediate.")
        final_text = "Final first.\n\nFinal second."
        worker.enqueue("Final first.", final_text)
        worker.enqueue("Final second.", final_text)
        worker.finish_queue()
        progress = []
        worker.progress.connect(lambda update: progress.append(update))
        output = Mock()
        with patch.dict("sys.modules", {"sounddevice": SimpleNamespace(OutputStream=Mock(return_value=output))}):
            worker.run()
        self.assertEqual(progress[0]["spoken_characters"], len("Final first."))
        self.assertTrue(all(update["spoken_characters"] <= len(final_text) for update in progress))
        self.assertEqual(progress[-1]["spoken_characters"], len(final_text))

    def test_repeated_sentence_audio_tracks_each_occurrence(self) -> None:
        import numpy as np

        manager = Mock()
        manager.continuous_audio_stream = True
        manager.synthesize_stream.side_effect = lambda *_args: [
            (np.ones(20, dtype=np.float32), 24000, ""),
        ]
        worker = _QueuedLocalSpeechThread(manager, "model", "speaker", "en")
        worker.enqueue("Echo.", "Echo.\n\nEcho.")
        worker.enqueue("Echo.", "Echo.\n\nEcho.")
        worker.finish_queue()
        progress = []
        worker.progress.connect(lambda update: progress.append(update))
        with patch.dict("sys.modules", {"sounddevice": SimpleNamespace(OutputStream=Mock())}):
            worker.run()
        self.assertEqual([update["spoken_characters"] for update in progress[:2]], [5, 12])

    def test_stopping_long_sentence_aborts_without_playing_remaining_blocks(self) -> None:
        import numpy as np

        manager = Mock()
        manager.continuous_audio_stream = True
        manager.synthesize_stream.side_effect = lambda *_args: [
            (np.ones(1000, dtype=np.float32), 1000, ""),
        ]
        worker = _QueuedLocalSpeechThread(manager, "model", "speaker", "en")
        for index in range(10):
            worker.enqueue(f"Sentence {index}.", "Sentence 0.")
        worker.finish_queue()
        completed = []
        progress = []
        worker.completed.connect(lambda ok, _message: completed.append(ok))
        worker.progress.connect(lambda update: progress.append(update))
        output = Mock()
        output.write.side_effect = lambda _samples: worker.request_stop()
        with patch.dict("sys.modules", {"sounddevice": SimpleNamespace(OutputStream=Mock(return_value=output))}):
            worker.run()
        self.assertEqual(output.write.call_count, 1)
        output.abort.assert_called_once()
        self.assertEqual(completed, [False])
        self.assertFalse(any(update["fraction"] == 1 for update in progress))

    def test_response_rewrite_preserves_unchanged_prefix(self) -> None:
        controller = TrayController.__new__(TrayController)
        controller._local_voice_longest_text = "First. Original second."
        controller._local_voice_queued_sentences = ["First.", "Original second."]
        worker = _QueuedLocalSpeechThread(Mock(), "model", "speaker", "en")
        controller._local_speech_thread = worker

        controller._update_local_voice("First. Replacement second.", False)
        controller._update_local_voice("First. Replacement second. Last", False)
        controller._update_local_voice("First. Replacement second. Last", True)

        self.assertEqual(worker._sentences.get_nowait(), "Replacement second.")
        self.assertEqual(worker._sentences.get_nowait(), "Last")
        self.assertIs(worker._sentences.get_nowait(), worker._FINISHED)
        self.assertTrue(worker._sentences.empty())

    def test_shorter_final_answer_is_queued_after_intermediate_reply(self) -> None:
        controller = TrayController.__new__(TrayController)
        controller._local_voice_longest_text = "I will look up the answer for you."
        controller._local_voice_queued_sentences = [controller._local_voice_longest_text]
        worker = _QueuedLocalSpeechThread(Mock(), "model", "speaker", "en")
        controller._local_speech_thread = worker

        controller._update_local_voice("The answer is 42.", True)

        self.assertEqual(worker._sentences.get_nowait(), "The answer is 42.")
        self.assertIs(worker._sentences.get_nowait(), worker._FINISHED)

    def test_final_chinese_answer_reaches_synthesis_after_intermediate_reply(self) -> None:
        import numpy as np

        manager = Mock()
        manager.display_name = "Test TTS"
        manager.continuous_audio_stream = True
        manager.synthesize_stream.side_effect = lambda _model, sentence, *_args: (
            (np.ones(20, dtype=np.float32), 24000, sentence),
        )
        controller = TrayController.__new__(TrayController)
        controller._local_voice_longest_text = ""
        controller._local_voice_queued_sentences = []
        worker = _QueuedLocalSpeechThread(manager, "model", "speaker", "zh")
        controller._local_speech_thread = worker
        intermediate = "我查一下 WowUp 切换测试版的具体位置。"
        final_sentences = [
            "最直接是去 WowUp 官方发布页，下载测试版安装包。",
            "官方说明测试版就在这里提供。",
            "找到 2.24.0-beta 或更新的测试版，展开下面的 Assets，"
            "下载 Windows 用的 Setup、以 .exe 结尾的文件。",
            "退出现在的 WowUp，运行安装包，装好再打开即可。",
            "之后到 Options → WoW Clients 重新扫描游戏，再选择 Forever 客户端。",
        ]
        final_text = "".join(final_sentences)
        completed: list[bool] = []
        worker.completed.connect(lambda ok, _message: completed.append(ok))
        controller._update_local_voice(intermediate, False)
        # The final response replaces the intermediate message and streams from
        # a shorter snapshot, as in the reported browser response.
        for length in (1, 20, 63, 121, len(final_text)):
            controller._update_local_voice(final_text[:length], False)
        controller._update_local_voice(final_text, True)

        output = Mock()
        with patch.dict(
            "sys.modules",
            {"sounddevice": SimpleNamespace(OutputStream=Mock(return_value=output))},
        ):
            worker.run()

        self.assertEqual(
            [call.args[1] for call in manager.synthesize_stream.call_args_list],
            [intermediate, *final_sentences],
        )
        self.assertEqual(output.write.call_count, 1 + len(final_sentences))
        self.assertEqual(completed, [True])

    def test_temporary_shorter_response_snapshot_is_ignored(self) -> None:
        controller = TrayController.__new__(TrayController)
        controller._local_voice_longest_text = "First sentence. Partial response"
        controller._local_voice_queued_sentences = ["First sentence."]
        worker = _QueuedLocalSpeechThread(Mock(), "model", "speaker", "en")
        controller._local_speech_thread = worker

        controller._update_local_voice("short", False)

        self.assertTrue(worker._sentences.empty())

    def test_local_thinking_stays_silent_while_reply_is_synthesized(self) -> None:
        import numpy as np
        from live_gpt.browser import BrowserMonitor, _ActiveResponse, _MonitorState, _ResponseSnapshot

        manager = Mock()
        manager.display_name = "Test TTS"
        manager.continuous_audio_stream = True
        manager.synthesize_stream.side_effect = lambda _model, sentence, *_args: (
            (np.ones(20, dtype=np.float32), 24000, sentence),
        )
        controller = TrayController.__new__(TrayController)
        controller.config = {"playing_backend": "sovits"}
        controller.window = Mock()
        controller._local_voice_longest_text = ""
        controller._local_voice_queued_sentences = []
        worker = _QueuedLocalSpeechThread(manager, "model", "speaker", "zh")
        controller._local_speech_thread = worker
        progress = []
        worker.progress.connect(progress.append)
        monitor = BrowserMonitor()
        monitor.set_use_browser_voice(False)
        monitor.response_changed.connect(controller._on_response_update)
        monitor.local_voice_updated.connect(controller._update_local_voice)
        state = _MonitorState(active_response=_ActiveResponse(
            page=Mock(), turn_marker_before="old", started_at=90,
        ))
        thinking = _ResponseSnapshot(False, True, False, "", "正在思考")
        reply = _ResponseSnapshot(True, False, True, "好了。", "Finishing reply…")
        with (
            patch.object(monitor, "_response_snapshot", side_effect=[thinking, thinking, reply, reply]),
            patch("live_gpt.browser.time.monotonic", return_value=100) as clock,
        ):
            monitor._poll_active_response(state)
            monitor._poll_active_response(state)
            monitor._poll_active_response(state)
            clock.return_value = 103
            monitor._poll_active_response(state)

        # Later thinking phases must remain silent, too.
        controller._on_response_update("正在思考…", "好了。")
        output = Mock()
        with patch.dict(
            "sys.modules",
            {"sounddevice": SimpleNamespace(OutputStream=Mock(return_value=output))},
        ):
            worker.run()
        self.assertEqual(
            [call.args[1] for call in manager.synthesize_stream.call_args_list],
            ["好了。"],
        )
        self.assertEqual(output.write.call_count, 1)
        self.assertEqual(controller._local_voice_queued_sentences, ["好了。"])
        self.assertTrue(progress)
        self.assertTrue(all(update["text"] == "好了。" for update in progress))
        self.assertTrue(all(update["spoken_characters"] == len("好了。") for update in progress))

    def test_thinking_does_not_start_local_voice_for_browser_playback_or_recording(self) -> None:
        for backend, interrupted in (("web", False), ("sovits", True), ("sovits", False)):
            with self.subTest(backend=backend, interrupted=interrupted):
                controller = TrayController.__new__(TrayController)
                controller.config = {"playing_backend": backend}
                controller.window = Mock()
                controller._local_voice_interrupted = interrupted
                with patch.object(controller, "_ensure_local_speech_worker") as start:
                    controller._on_response_update("正在思考", "")
                start.assert_not_called()
                controller.window.set_response_update.assert_called_once_with("正在思考", "")

    def test_thinking_label_variants_remain_silent_across_sends(self) -> None:
        for status in ("Thinking", "Thinking…", "Thinking...", "正在思考", "正在思考…", "思考中"):
            with self.subTest(status=status):
                controller = TrayController.__new__(TrayController)
                controller.config = {"playing_backend": "sovits"}
                controller.window = Mock()
                controller.browser_monitor = Mock()
                controller.selected_chatgpt_tab_id = "selected-tab"
                worker = _QueuedLocalSpeechThread(Mock(), "model", "speaker", "auto")
                controller._local_speech_thread = worker
                controller._on_response_update(status, "")
                controller._on_response_update(status, "")
                self.assertTrue(worker._sentences.empty())

                controller._handle_send_requested("Next question", None)
                controller._on_response_update(status, "")
                self.assertTrue(worker._sentences.empty())

    def test_thinking_failure_does_not_create_a_speech_worker(self) -> None:
        controller = TrayController.__new__(TrayController)
        controller.config = {"playing_backend": "sovits"}
        controller.window = Mock()
        controller._local_speech_thread = None
        controller._local_tts_configuration = Mock(return_value=(Mock(), "model", "speaker", "zh"))
        with patch.object(_QueuedLocalSpeechThread, "start") as start:
            controller._on_response_update("正在思考", "")
        start.assert_not_called()
        self.assertIsNone(controller._local_speech_thread)
        controller._on_response_finished(False, "Timed out")
        controller.window.set_response_finished.assert_called_once_with(False, "Timed out")

    def test_local_timeout_is_spoken_without_thinking_announcements(self) -> None:
        import numpy as np
        from live_gpt.localization import localization
        from live_gpt.browser import BrowserMonitor, _ActiveResponse, _MonitorState, _ResponseSnapshot

        original_language = localization.language
        self.addCleanup(localization.set_language, original_language)
        for language, error in (
            ("zh", "等待回复超时"),
            ("en", "Timed out while waiting for a reply"),
        ):
            with self.subTest(language=language):
                localization.set_language(language)
                manager = Mock(display_name="Test TTS", continuous_audio_stream=True)
                manager.synthesize_stream.side_effect = lambda _model, sentence, *_args: (
                    (np.ones(20, dtype=np.float32), 24000, sentence),
                )
                controller = TrayController.__new__(TrayController)
                controller.config = {"playing_backend": "sovits"}
                controller.window = Mock()
                worker = _QueuedLocalSpeechThread(manager, "model", "speaker", "auto")
                controller._local_speech_thread = worker
                controller._local_voice_longest_text = ""
                controller._local_voice_queued_sentences = []
                monitor = BrowserMonitor()
                monitor.set_use_browser_voice(False)
                monitor.response_changed.connect(controller._on_response_update)
                monitor.local_voice_announcement.connect(controller._speak_local_announcement)
                monitor.response_finished.connect(controller._on_response_finished)
                state = _MonitorState(active_response=_ActiveResponse(page=Mock(), turn_marker_before=None, started_at=0))
                with patch.object(monitor, "_response_snapshot", return_value=_ResponseSnapshot(False, True, False, "", "正在思考")), patch("live_gpt.browser.time.monotonic") as clock:
                    for moment in (0, 60, 120, 600):
                        clock.return_value = moment
                        monitor._poll_active_response(state)
                with patch.dict("sys.modules", {"sounddevice": SimpleNamespace(OutputStream=Mock(return_value=Mock()))}):
                    worker.run()
                self.assertEqual([call.args[1] for call in manager.synthesize_stream.call_args_list], [error])
                controller._on_local_speech_completed(True, "Playback complete")
                controller.window.finish_reading.assert_called_once_with(False, "Timed out while waiting for ChatGPT's reply")

    def test_local_send_error_is_spoken_without_hiding_preserved_prompt(self) -> None:
        from live_gpt.localization import localization
        original_language = localization.language
        self.addCleanup(localization.set_language, original_language)
        localization.set_language("zh")
        controller = TrayController.__new__(TrayController)
        controller.config = {"playing_backend": "sovits"}
        controller.window = Mock()
        controller._local_speech_thread = None
        controller._local_tts_configuration = Mock(return_value=(Mock(), "model", "speaker", "auto"))
        message = "Could not send to ChatGPT: Screenshot upload failed"
        with patch.object(_QueuedLocalSpeechThread, "start"):
            controller._on_send_finished(False, "Keep my prompt", message)
        worker = controller._local_speech_thread
        self.assertEqual(worker._sentences.get_nowait().text, "无法发送：截图上传失败")
        self.assertIs(worker._sentences.get_nowait(), worker._FINISHED)
        worker.started.emit("Playing")
        controller.window.begin_reading.assert_not_called()
        controller.window.set_send_result.assert_called_once_with(False, "Keep my prompt", message)

    def test_error_waiting_for_finished_audio_closes_its_new_queue(self) -> None:
        controller = TrayController.__new__(TrayController)
        controller.config = {"playing_backend": "sovits"}
        controller.window = Mock()
        worker = _QueuedLocalSpeechThread(Mock(), "model", "speaker", "auto")
        controller._local_speech_thread = worker
        worker.finish_queue()
        controller._local_tts_configuration = Mock(return_value=(Mock(), "model", "speaker", "auto"))
        message = "Timed out while waiting for ChatGPT's reply"
        controller._speak_local_announcement(message)
        controller._on_response_finished(False, message)
        with patch.object(_QueuedLocalSpeechThread, "start"):
            controller._local_speech_finished()
        worker = controller._local_speech_thread
        self.assertTrue(worker._queue_finished)
        self.assertIsNot(worker._sentences.get_nowait(), worker._FINISHED)
        self.assertIs(worker._sentences.get_nowait(), worker._FINISHED)


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



class ReplyDisplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.application = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.window = OverlayWindow()
        self.window.set_chatgpt_tabs([{"id": "tab", "title": "ChatGPT", "url": "https://chatgpt.com"}])
        self.window.show()
        self.window.begin_response_display("Question")
        self.window.set_response_update("Writing…", "Results\nRead these docs.\nName\tValue\nOne\t2")
        self.window.set_response_html('<h2>Results</h2><p>Read <strong>these</strong> '
                                      '<a href="https://example.com/docs">docs</a>.</p>'
                                      '<table border="1" cellpadding="6"><tr><th>Name</th><th>Value</th></tr>'
                                      '<tr><td>One</td><td>2</td></tr></table>')
        self.window._expand_subtitle()
        QApplication.processEvents()

    def tearDown(self) -> None:
        self.window.close()
        self.window.deleteLater()

    def test_reply_renders_html_and_selection_does_not_dismiss_it(self) -> None:
        display = self.window.subtitle_full_text
        self.assertIsInstance(display, QTextBrowser)
        self.assertNotIsInstance(display, QPlainTextEdit)
        self.assertTrue(display.isReadOnly())
        self.assertIsNotNone(display.document().find("Name").currentTable())
        self.assertGreater(display.document().find("these").charFormat().fontWeight(), 400)
        QTest.mouseClick(display.viewport(), Qt.MouseButton.LeftButton, pos=QPoint(20, 20))
        display.selectAll()
        self.assertTrue(display.textCursor().hasSelection())
        self.assertTrue(self.window._subtitle_mode_active)
        self.assertTrue(self.window.transcript_area.isHidden())
        self.assertTrue(self.window.reply_close_button.isVisible())

    @patch("live_gpt.app.webbrowser.open_new_tab")
    def test_reply_link_opens_new_tab_and_keeps_reply_visible(self, open_tab) -> None:
        display = self.window.subtitle_full_text
        cursor = display.document().find("docs")
        cursor.setPosition(cursor.selectionStart() + 1)
        point = display.cursorRect(cursor).center()
        QTest.mouseClick(display.viewport(), Qt.MouseButton.LeftButton, pos=point)
        open_tab.assert_called_once_with("https://example.com/docs")
        self.assertTrue(self.window._subtitle_mode_active)
        display.anchorClicked.emit(QUrl("javascript:alert(1)"))
        open_tab.assert_called_once()

    def test_close_returns_to_empty_input_and_late_updates_keep_draft(self) -> None:
        self.window.reply_close_button.click()
        self.assertFalse(self.window._subtitle_mode_active)
        self.assertTrue(self.window.subtitle_panel.isHidden())
        self.assertFalse(self.window.transcript_area.isHidden())
        self.assertFalse(self.window.transcript_area.isReadOnly())
        self.assertEqual(self.window.transcript_area.toPlainText(), "")
        self.window.transcript_area.setPlainText("Next question")
        self.window.set_response_update("Writing…", "Late reply")
        self.window.set_response_html("<p>Late reply</p>")
        self.window.set_reading_subtitle({"text": "Late reply", "fraction": 0.5})
        self.assertEqual(self.window.transcript_area.toPlainText(), "Next question")
        self.assertTrue(self.window.subtitle_panel.isHidden())

    def test_close_stops_playback_and_late_callbacks_cannot_restart_it(self) -> None:
        controller = TrayController.__new__(TrayController)
        controller.window = self.window
        controller.config = {"playing_backend": "sovits"}
        controller.browser_monitor = Mock()
        controller._pending_local_announcements = [("Old notice", False)]
        worker = _QueuedLocalSpeechThread(Mock(), "model", "speaker", "en")
        worker._output = Mock()
        controller._local_speech_thread = worker
        self.window.reply_closed.connect(controller._stop_reply_playback)
        self.window.begin_reading("Reading aloud…")
        self.window._expand_subtitle()
        self.window.reply_close_button.click()
        self.assertTrue(worker._cancel_event.is_set())
        worker._output.abort.assert_called_once()
        controller.browser_monitor.request_stop_reading.assert_called_once_with(message="Playback stopped")
        self.assertTrue(controller._local_voice_interrupted)
        self.assertEqual(controller._pending_local_announcements, [])
        self.assertFalse(self.window._pet_playing)
        self.assertFalse(self.window._pet_error)
        self.window.transcript_area.setPlainText("New draft")
        with patch.object(controller, "_ensure_local_speech_worker") as start:
            controller._update_local_voice("Late reply.", True)
            controller._speak_local_announcement("Old error", final=True)
            self.window.begin_reading("Late playback")
            controller._on_local_speech_completed(False, "Cancelled")
            controller._on_browser_reading_finished(False, "Playback stopped")
            controller._local_speech_finished()
        start.assert_not_called()
        self.assertEqual(self.window.transcript_area.toPlainText(), "New draft")
        self.assertFalse(self.window._subtitle_mode_active)
        self.assertFalse(self.window._pet_playing)
        self.assertFalse(self.window._pet_error)

    def test_new_reply_clears_previous_html(self) -> None:
        self.window.set_response_links((("Quest source", "https://example.com/source"),))
        self.window.begin_response_display("Next question")
        self.window.set_response_update("Writing…", "Next answer")
        self.window._expand_subtitle()
        self.assertEqual(self.window.subtitle_full_text.toPlainText(), "Next answer")
        self.assertNotIn("example.com/docs", self.window.subtitle_full_text.toHtml())
        self.assertNotIn("example.com/source", self.window.subtitle_full_text.toHtml())

    @patch("live_gpt.app.webbrowser.open_new_tab")
    def test_collected_sources_are_visible_clickable_and_not_in_subtitles(self, open_tab) -> None:
        display = self.window.subtitle_full_text
        reply_text = self.window._reading_full_text
        self.window.set_response_links((("Forever +1", "https://example.com/quest"),))
        QApplication.processEvents()
        cursor = display.document().find("Forever +1")
        self.assertFalse(cursor.isNull())
        self.assertEqual(cursor.charFormat().anchorHref(), "https://example.com/quest")
        cursor.setPosition(cursor.selectionStart() + 1)
        QTest.mouseClick(display.viewport(), Qt.MouseButton.LeftButton, pos=display.cursorRect(cursor).center())
        open_tab.assert_called_once_with("https://example.com/quest")
        self.assertEqual(self.window._reading_full_text, reply_text)
        self.assertNotIn("Forever", " ".join(self.window._subtitle_lines()))

    def test_playback_progress_preserves_html_and_scroll_position(self) -> None:
        self.window.set_response_html("<p>Paragraph</p>" * 100)
        QApplication.processEvents()
        display = self.window.subtitle_full_text
        scrollbar = display.verticalScrollBar()
        self.assertGreater(scrollbar.maximum(), 0)
        scrollbar.setValue(scrollbar.maximum() // 2)
        value, revision = scrollbar.value(), display.document().revision()
        self.window.set_reading_subtitle({"text": "Paragraph\n" * 100, "fraction": 0.2})
        self.assertEqual(display.document().revision(), revision)
        self.assertEqual(scrollbar.value(), value)
        self.assertTrue(self.window.reply_close_button.isVisible())

    def test_macro_copy_button_copies_only_exact_code_and_keeps_reply_open(self) -> None:
        macro = "#showtooltip 神圣打击\n/startattack\n/cast 神圣打击"
        self.window.set_response_html('<p>Macro:</p><pre data-live-gpt-language="Plain text"><code>'
                                      + macro + '</code></pre><p>Paste it into the game.</p>')
        QApplication.processEvents()
        display = self.window.subtitle_full_text
        self.assertIn(macro, display.toPlainText())
        self.assertEqual(len(display._copy_buttons), 1)
        self.assertFalse(display.document().find("Plain text").charFormat().fontUnderline())
        button = display._copy_buttons[0]
        self.assertEqual(button.text(), "")
        self.assertFalse(button.icon().isNull())
        icon_before_copy = button.icon().cacheKey()
        self.assertTrue(button.isVisible())
        self.assertTrue(display.viewport().rect().contains(button.geometry().center()))
        QApplication.clipboard().setText("Previous clipboard")
        QTest.mouseClick(button, Qt.MouseButton.LeftButton)
        self.assertEqual(QApplication.clipboard().text(), macro)
        self.assertNotEqual(button.icon().cacheKey(), icon_before_copy)
        self.assertTrue(self.window._subtitle_mode_active)
        self.assertTrue(self.window.transcript_area.isHidden())

    def test_multiple_code_blocks_copy_independently_and_updates_remove_old_buttons(self) -> None:
        self.window.set_response_html('<pre><code>First\n  indented</code></pre>'
                                      '<pre data-live-gpt-language="SQL"><code>SELECT 1;</code></pre>')
        QApplication.processEvents()
        display = self.window.subtitle_full_text
        self.assertEqual(len(display._copy_buttons), 2)
        display._copy_buttons[1].click()
        self.assertEqual(QApplication.clipboard().text(), "SELECT 1;")
        display._copy_buttons[0].click()
        self.assertEqual(QApplication.clipboard().text(), "First\n  indented")
        self.window.set_response_html('<pre><code>Updated</code></pre>')
        self.assertEqual(len(display._copy_buttons), 1)
        display._copy_buttons[0].click()
        self.assertEqual(QApplication.clipboard().text(), "Updated")
        self.window.begin_response_display("Next question")
        self.assertEqual(display._copy_buttons, [])

    def test_code_copy_button_moves_with_scrolling(self) -> None:
        self.window.set_response_html('<p>Introduction</p>' * 30 + '<pre><code>copy me</code></pre>'
                                      + '<p>Closing paragraph</p>' * 30)
        QApplication.processEvents()
        display = self.window.subtitle_full_text
        button = display._copy_buttons[0]
        cursor = display.document().find("copy me")
        display.setTextCursor(cursor)
        display.ensureCursorVisible()
        QApplication.processEvents()
        self.assertTrue(button.isVisible())
        old_y = button.y()
        scrollbar = display.verticalScrollBar()
        scrollbar.setValue(scrollbar.value() + 20)
        self.assertEqual(button.y(), old_y - 20)


class SettingsDialogTests(unittest.TestCase):
    def test_screenshot_cursor_option_defaults_on_and_saves_immediately(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = Config(Path(directory) / "config.json")
            dialog = HotkeyConfigDialog("Right Alt", config=config)
            try:
                dialog.screenshots_nav_button.click()
                self.assertEqual(dialog.settings_pages.currentIndex(), 5)
                self.assertTrue(dialog.capture_cursor_checkbox.isChecked())
                dialog.capture_cursor_checkbox.setChecked(False)
                self.assertFalse(Config(config.path)["capture_cursor"])
            finally:
                dialog.close()
            reopened = HotkeyConfigDialog("Right Alt", config=Config(config.path))
            try:
                self.assertFalse(reopened.capture_cursor_checkbox.isChecked())
            finally:
                reopened.close()

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
        controller.config = {"capture_cursor": True}
        controller.window = Mock()
        controller.browser_monitor = Mock()
        controller.selected_chatgpt_tab_id = "selected-tab"

        controller._handle_send_requested("Explain this", source)

        capture.assert_called_once_with(source, include_cursor=True)
        controller.browser_monitor.request_send.assert_called_once_with(
            "selected-tab",
            "Explain this",
            b"screenshot",
        )
        controller.window.begin_response_display.assert_called_once_with(
            "Explain this"
        )
        capture.reset_mock()
        controller.config["capture_cursor"] = False
        controller._handle_send_requested("Without cursor", source)
        capture.assert_called_once_with(source, include_cursor=False)

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

    def test_send_error_banner_remains_visible_above_preserved_prompt(self) -> None:
        window = OverlayWindow()
        message = "You’ve reached your 5-hour Work usage limit. Resets at 2:53 PM."
        try:
            window.show()
            window.begin_response_display("Unsent prompt")
            window.set_send_result(False, "Unsent prompt", message)
            QApplication.processEvents()
            self.assertEqual(window.transcript_area.toPlainText(), "Unsent prompt")
            self.assertTrue(window.error_banner.isVisible())
            self.assertEqual(window.error_banner.text(), message)
            self.assertTrue(window.error_banner.wordWrap())
            self.assertTrue(window._pet_error)
            window.set_status("Sent to ChatGPT")
            self.assertTrue(window.error_banner.isHidden())
        finally:
            window.close()

    def test_usage_limit_response_error_is_visible_after_subtitles_dismissed(self) -> None:
        window = OverlayWindow()
        try:
            window.begin_response_display("Question")
            window.dismiss_subtitle_mode()
            window.set_response_finished(False, "You’ve reached your usage limit")
            self.assertFalse(window.error_banner.isHidden())
            self.assertEqual(window.error_banner.text(), "You’ve reached your usage limit")
            self.assertFalse(window._pet_response_pending)
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

    def test_local_subtitles_follow_source_offsets_past_short_headings(self) -> None:
        window = OverlayWindow()
        try:
            window.show()
            window.begin_reading("Reading aloud…")
            QApplication.processEvents()
            text = "A\n\nB\n\n" + " ".join(f"spoken-word-{index}" for index in range(80))
            window.set_reading_subtitle({"text": text, "spoken_characters": 0})
            lines = window._subtitle_lines()
            target_index = 5
            position = text.index(lines[target_index])
            window.set_reading_subtitle({
                "text": text,
                "spoken_characters": position,
                "fraction": position / len(text),
            })
            self.assertEqual(window._subtitle_line_index, target_index)
            self.assertEqual(window.subtitle_line_one.text(), lines[target_index])
            window.resize(window.width() - 100, window.height())
            window._render_reading_subtitle(resized=True)
            self.assertIn(window.subtitle_line_one.text().split()[0], text[position:position + 100])
        finally:
            window.close()

    def test_local_subtitle_offsets_handle_cjk_and_collapsed_spaces(self) -> None:
        window = OverlayWindow()
        try:
            window.show()
            window.begin_reading("Reading aloud…")
            QApplication.processEvents()
            text = "短标题\n\n" + "一二三四五六七八九十" * 60
            window.set_reading_subtitle({"text": text, "spoken_characters": 0})
            lines = window._subtitle_lines()
            position = len("短标题\n\n") + sum(len(line) for line in lines[1:4])
            window.set_reading_subtitle({"text": text, "spoken_characters": position})
            self.assertEqual(window._subtitle_line_index, 4)
            self.assertEqual(window.subtitle_line_one.text(), lines[4])

            text = "A\n\nB\n\n" + "  \t".join(f"word-{index}" for index in range(80))
            window.set_reading_subtitle({"text": text, "spoken_characters": 0})
            lines = window._subtitle_lines()
            position = text.index(lines[5].split()[0])
            window.set_reading_subtitle({"text": text, "spoken_characters": position})
            self.assertEqual(window._subtitle_line_index, 5)
        finally:
            window.close()

    def test_capture_selector_defaults_to_desktop_without_no_screenshot_option(self) -> None:
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

            self.assertEqual(window.capture_source_combo.count(), 1)
            self.assertEqual(window.capture_source_combo.itemText(0), "Screenshot desktop")
            self.assertEqual(window.capture_source_combo.currentData(), source)
            self.assertTrue(window.transcript_area._screenshot_selected)
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

            window.capture_source_combo.setCurrentIndex(0)
            self.assertEqual(preferences, [first.key])
        finally:
            window.close()

    def test_capture_selector_restores_saved_window_or_falls_back_to_desktop(self) -> None:
        desktop = CaptureSource(
            "display:1", "Screenshot desktop", "display", 0, 0, 1920, 1080,
        )
        saved_window = CaptureSource(
            "window:1", "Saved window", "window", 0, 0, 1200, 900, hwnd=1,
        )
        for saved_key, expected in (
            ("", desktop),
            ("window:missing", desktop),
            (saved_window.key, saved_window),
        ):
            with self.subTest(saved_key=saved_key):
                window = OverlayWindow()
                preferences: list[str] = []
                window.capture_source_selected.connect(preferences.append)
                try:
                    window.set_preferred_capture_source(saved_key)
                    window.set_capture_sources([desktop, saved_window])

                    self.assertEqual(window.capture_source_combo.currentData(), expected)
                    self.assertEqual(preferences, [])
                finally:
                    window.close()

    def test_capture_selector_falls_back_to_desktop_when_selected_window_disappears(self) -> None:
        window = OverlayWindow()
        desktop = CaptureSource(
            "display:1", "Screenshot desktop", "display", 0, 0, 1920, 1080,
        )
        selected_window = CaptureSource(
            "window:1", "Selected window", "window", 0, 0, 1200, 900, hwnd=1,
        )
        try:
            window.set_preferred_capture_source(selected_window.key)
            window.set_capture_sources([desktop, selected_window])
            self.assertEqual(window.capture_source_combo.currentData(), selected_window)

            window.set_capture_sources([desktop])

            self.assertEqual(window.capture_source_combo.currentData(), desktop)
            self.assertTrue(window.transcript_area._screenshot_selected)
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
            window.capture_source_combo.setCurrentIndex(0)
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
            window.capture_source_combo.setCurrentIndex(0)

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
            window.capture_source_combo.setCurrentIndex(0)
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
    def test_screenshot_is_queued_with_browser_dictation_after_half_second_hold(
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
        controller.config = {"capture_cursor": False}
        controller.window.capture_source_combo.currentData.return_value = (
            release_source
        )
        controller.browser_monitor = Mock()
        controller.selected_chatgpt_tab_id = "selected-tab"
        controller.dictation_tab_id = "selected-tab"
        controller._dictation_state = "starting"
        controller._browser_dictation_start_requested = False
        controller._dictation_input_held = True
        controller._dictation_pressed_since = time.monotonic() - 0.6
        controller._dictation_listening_since = time.monotonic() - 0.6
        controller._dictation_press_generation = 1
        controller._dictation_attachment_tab_id = None
        controller._pending_dictation_capture = None

        controller._prepare_browser_dictation_after_hold(1)

        capture.assert_called_once_with(release_source, include_cursor=False)
        controller.browser_monitor.request_start_dictation.assert_called_once_with(
            "selected-tab",
            b"hold-screenshot",
        )
        controller.browser_monitor.request_replace_attachment.assert_not_called()
        controller._on_dictation_started(True, "Listening")
        controller.finish_dictation()

        controller.browser_monitor.request_finish_dictation.assert_called_once_with(
            "selected-tab"
        )

        controller._handle_send_requested("Dictated text", later_source)

        capture.assert_called_once_with(release_source, include_cursor=False)
        controller.browser_monitor.request_send.assert_called_once_with(
            "selected-tab",
            "Dictated text",
            b"hold-screenshot",
            preserve_attachments=True,
        )

    @patch("live_gpt.app.QTimer.singleShot")
    @patch("live_gpt.app.capture_webp", return_value=b"screenshot")
    def test_browser_microphone_waits_for_capture_and_cancelled_hold_starts_nothing(
        self, capture: Mock, single_shot: Mock,
    ) -> None:
        controller = TrayController.__new__(TrayController)
        controller.window = Mock()
        controller.browser_monitor = Mock()
        controller.config = {"capture_cursor": False}
        controller.selected_chatgpt_tab_id = "selected-tab"
        controller.dictation_tab_id = None
        controller.window.capture_source_combo.currentData.return_value = CaptureSource(
            key="display:1", label="Desktop", kind="display",
            left=0, top=0, width=1920, height=1080,
        )
        controller.start_dictation()
        callback = single_shot.call_args.args[1]
        self.assertGreater(single_shot.call_args.args[0], 0)
        capture.assert_not_called()
        controller.browser_monitor.request_start_dictation.assert_not_called()

        controller.finish_dictation()
        callback()
        capture.assert_not_called()
        controller.browser_monitor.request_start_dictation.assert_not_called()
        controller.browser_monitor.request_cancel_dictation.assert_not_called()
        controller.window.end_dictation_display.assert_called_once_with()
        self.assertEqual(controller._dictation_state, "idle")

        controller.start_dictation()
        callback = single_shot.call_args.args[1]
        # Tab selection can change during preparation; attach to the tab
        # this recording started on.
        controller.selected_chatgpt_tab_id = "different-tab"
        callback()
        callback()
        capture.assert_called_once()
        controller.browser_monitor.request_start_dictation.assert_called_once_with(
            "selected-tab", b"screenshot",
        )
        controller.browser_monitor.request_replace_attachment.assert_not_called()
        self.assertEqual(controller._dictation_attachment_tab_id, "selected-tab")

    @patch("live_gpt.app.QTimer.singleShot")
    @patch("live_gpt.app.capture_webp", side_effect=RuntimeError("Capture unavailable"))
    def test_capture_failure_returns_to_idle_without_starting_browser_dictation(
        self, capture: Mock, single_shot: Mock,
    ) -> None:
        controller = TrayController.__new__(TrayController)
        controller.window = Mock()
        controller.browser_monitor = Mock()
        controller.config = {"capture_cursor": False}
        controller.selected_chatgpt_tab_id = "selected-tab"
        controller.dictation_tab_id = None
        controller.window.capture_source_combo.currentData.return_value = CaptureSource(
            key="display:1", label="Desktop", kind="display",
            left=0, top=0, width=1920, height=1080,
        )
        controller.start_dictation()
        single_shot.call_args.args[1]()
        capture.assert_called_once()
        controller.browser_monitor.request_start_dictation.assert_not_called()
        controller.window.end_dictation_display.assert_called_once_with()
        self.assertEqual(controller._dictation_state, "idle")

    @patch("live_gpt.app.QTimer.singleShot")
    @patch("live_gpt.app.capture_webp")
    def test_browser_dictation_without_screenshot_starts_immediately(
        self, capture: Mock, single_shot: Mock,
    ) -> None:
        controller = TrayController.__new__(TrayController)
        controller.window = Mock()
        controller.browser_monitor = Mock()
        controller.selected_chatgpt_tab_id = "selected-tab"
        controller.dictation_tab_id = None
        controller.window.capture_source_combo.currentData.return_value = CaptureSource(
            key="display:1", label="Desktop", kind="display",
            left=0, top=0, width=1920, height=1080,
        )
        controller.start_dictation(include_screenshot=False)
        controller.browser_monitor.request_start_dictation.assert_called_once_with("selected-tab")
        capture.assert_not_called()
        single_shot.assert_not_called()
        self.assertIsNone(controller._pending_dictation_capture.screenshot)

    @patch("live_gpt.app.capture_webp", return_value=b"screenshot")
    def test_local_recording_still_uploads_screenshot_while_listening(self, capture: Mock) -> None:
        controller = TrayController.__new__(TrayController)
        controller.window = Mock()
        controller.browser_monitor = Mock()
        controller.config = {"capture_cursor": False}
        controller.selected_chatgpt_tab_id = "selected-tab"
        controller.dictation_tab_id = "local"
        controller._dictation_press_generation = 1
        controller._dictation_input_held = True
        controller._dictation_state = "listening"
        controller.window.capture_source_combo.currentData.return_value = CaptureSource(
            key="display:1", label="Desktop", kind="display",
            left=0, top=0, width=1920, height=1080,
        )
        controller._upload_dictation_screenshot_after_hold(1)
        capture.assert_called_once()
        controller.browser_monitor.request_replace_attachment.assert_called_once_with(
            "selected-tab", b"screenshot",
        )
        controller.browser_monitor.request_start_dictation.assert_not_called()

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

    def test_release_before_browser_listens_cancels_startup_immediately(self) -> None:
        controller = TrayController.__new__(TrayController)
        controller.window = Mock()
        controller.window.transcript_area.is_showing_response = False
        controller.browser_monitor = Mock()
        cancelled = threading.Event()
        controller.browser_monitor.request_start_dictation.return_value = cancelled
        controller.selected_chatgpt_tab_id = "selected-tab"
        controller.dictation_tab_id = None
        controller._dictation_state = "idle"
        controller._dictation_input_held = False
        controller._dictation_listening_since = None

        controller.start_dictation()
        controller.finish_dictation()

        self.assertTrue(cancelled.is_set())
        self.assertEqual(controller._dictation_state, "cancelling")
        controller.browser_monitor.request_cancel_dictation.assert_not_called()
        controller._on_dictation_finished(True, "", "Dictation cancelled")
        self.assertEqual(controller._dictation_state, "idle")
        controller.window.end_dictation_display.assert_called_once_with()

    def test_late_listening_signal_after_release_cancels_browser_session(self) -> None:
        controller = TrayController.__new__(TrayController)
        controller.window = Mock()
        controller.browser_monitor = Mock()
        cancelled = threading.Event()
        controller.browser_monitor.request_start_dictation.return_value = cancelled
        controller.selected_chatgpt_tab_id = "selected-tab"
        controller.dictation_tab_id = None
        controller.start_dictation(include_screenshot=False)
        controller.finish_dictation()
        controller._on_dictation_started(True, "Listening")
        controller.browser_monitor.request_cancel_dictation.assert_called_once_with("selected-tab")
        controller.window.set_dictation_listening.assert_not_called()
        controller._on_dictation_finished(True, "", "Dictation cancelled")
        self.assertEqual(controller._dictation_state, "idle")

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
            (True, True, "  A \n", True, "finishing", False),
            (True, True, "  OK \n", True, "finishing", True),
            (True, True, " 好 \n", True, "finishing", True),
            (True, False, "あ", True, "finishing", True),
            (True, True, "é", True, "finishing", True),
            (True, True, "ع", True, "finishing", True),
            (True, True, " \t\n　", True, "finishing", False),
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
                if success and state != "cancelling":
                    controller.window.set_transcript.assert_called_once_with(text.strip())
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

    def test_one_non_english_character_is_trimmed_and_can_be_sent_manually(self) -> None:
        controller = TrayController.__new__(TrayController)
        controller.window = Mock()
        controller.browser_monitor = Mock()
        controller.selected_chatgpt_tab_id = "selected-tab"
        controller._dictation_input_held = False

        controller._on_dictation_finished(True, "　 好 \n", "Finished")

        controller.window.set_transcript.assert_called_once_with("好")
        controller.window.set_microphone_state.assert_called_once_with("saved", "Finished")
        controller.window.request_send_from_hotkey.assert_not_called()
        self.assertIsNone(controller._short_voice_text)
        controller._handle_send_requested("　 好 \n", None)
        controller.browser_monitor.request_send.assert_called_once_with("selected-tab", "好", None)
        controller.window.begin_response_display.assert_called_once_with("好")

    def test_send_trims_surrounding_whitespace_and_keeps_internal_spacing(self) -> None:
        controller = TrayController.__new__(TrayController)
        controller.window = Mock()
        controller.browser_monitor = Mock()
        controller.selected_chatgpt_tab_id = "selected-tab"

        controller._handle_send_requested(" \nHello  world\nSecond line\t ", None)

        controller.browser_monitor.request_send.assert_called_once_with(
            "selected-tab", "Hello  world\nSecond line", None,
        )

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

    def test_tray_clicks_disable_auto_hide_and_reveal_controls(self) -> None:
        for reason in (
            QSystemTrayIcon.ActivationReason.Trigger,
            QSystemTrayIcon.ActivationReason.DoubleClick,
        ):
            controller = TrayController.__new__(TrayController)
            controller.window = Mock()
            controller.show_window = Mock()
            controller._handle_activation(reason)
            controller.window.disable_auto_hide.assert_called_once_with()
            controller.show_window.assert_called_once_with()
            controller.window.reveal_controls.assert_called_once_with()

    @patch("live_gpt.app.QCursor.pos", return_value=QPoint(-10000, -10000))
    def test_tray_reveal_recovers_offscreen_window_and_survives_pointer_ticks(self, _cursor) -> None:
        window = OverlayWindow()
        controller = TrayController.__new__(TrayController)
        controller.window = window
        bounds = QRect(0, 0, 3413, 1440)
        screen = Mock()
        screen.availableGeometry.return_value = bounds
        try:
            window.set_chatgpt_tabs([{"id": "tab", "title": "ChatGPT", "url": "https://chatgpt.com"}])
            window.setGeometry(3246, 1808, 1619, 306)
            window.auto_hide_button.setChecked(True)
            window.hide()
            with patch("live_gpt.app.QApplication.screens", return_value=[screen]), \
                    patch("live_gpt.app.QApplication.primaryScreen", return_value=screen):
                controller._handle_activation(QSystemTrayIcon.ActivationReason.Trigger)
            self.assertFalse(window.isHidden())
            self.assertFalse(window.auto_hide_enabled)
            self.assertTrue(bounds.contains(window.geometry()))
            for _ in range(5):
                window._track_pointer(QPoint(-10000, -10000))
                self.assertTrue(window._chrome_visible)
                self.assertEqual(window._title_opacity.opacity(), 1)
            window._track_pointer(window.frameGeometry().center())
            window._track_pointer(QPoint(-10000, -10000))
            self.assertFalse(window._chrome_visible)
        finally:
            window.close()

    def test_screen_recovery_handles_locked_sizes_and_collapsed_geometry(self) -> None:
        window = OverlayWindow()
        screen = Mock()
        bounds = QRect(-1280, 0, 1280, 720)
        screen.availableGeometry.return_value = bounds
        try:
            window.setGeometry(3246, 1808, 1619, 900)
            window.lock_button.setChecked(True)
            window._input_collapsed_geometry = QRect(3246, 1808, 1619, 306)
            with patch("live_gpt.app.QApplication.screens", return_value=[screen]), \
                    patch("live_gpt.app.QApplication.primaryScreen", return_value=screen):
                window.ensure_on_screen()
            self.assertTrue(bounds.contains(window.geometry()))
            self.assertTrue(window._position_locked)
            self.assertTrue(bounds.contains(window._input_collapsed_geometry))
            window._collapse_hover_input()
            self.assertTrue(bounds.contains(window.geometry()))
        finally:
            window.close()

    def test_screen_recovery_preserves_visible_secondary_monitor_position(self) -> None:
        window = OverlayWindow()
        primary, secondary = Mock(), Mock()
        primary.availableGeometry.return_value = QRect(0, 0, 1920, 1080)
        secondary.availableGeometry.return_value = QRect(-1920, 0, 1920, 1080)
        try:
            expected = QRect(-1700, 700, 1140, 240)
            window.setGeometry(expected)
            with patch("live_gpt.app.QApplication.screens", return_value=[primary, secondary]), \
                    patch("live_gpt.app.QApplication.primaryScreen", return_value=primary):
                window.ensure_on_screen()
            self.assertEqual(window.geometry(), expected)
        finally:
            window.close()

    def test_empty_dictation_returns_to_auto_hidden_state(self) -> None:
        controller = TrayController.__new__(TrayController)
        controller.window = Mock()
        controller.browser_monitor = Mock()
        controller._dictation_state = "finishing"
        controller.dictation_tab_id = "selected-tab"
        controller._dictation_input_held = False
        controller._dictation_send_on_finish = True
        controller._dictation_attachment_tab_id = "selected-tab"
        controller._pending_dictation_capture = Mock()

        controller._on_dictation_finished(
            True,
            "",
            "Dictation copied from ChatGPT",
        )

        controller.window.show_for_auto_hide.assert_not_called()
        controller.window.schedule_auto_hide.assert_called_once_with()
        controller.window.end_dictation_display.assert_called_once_with()
        controller.window.set_transcript.assert_called_once_with("")
        controller.window.set_microphone_state.assert_called_once_with("idle")
        controller.window.request_send_from_hotkey.assert_not_called()
        controller.browser_monitor.request_clear_attachments.assert_called_once_with("selected-tab")
        self.assertEqual(controller._dictation_state, "idle")
        self.assertIsNone(controller.dictation_tab_id)
        self.assertIsNone(controller._pending_dictation_capture)

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
