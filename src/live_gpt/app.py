from __future__ import annotations

import os
import queue
import re
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import (
    QEvent,
    QPoint,
    QRect,
    QSettings,
    QSize,
    QThread,
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
    QButtonGroup,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QKeySequenceEdit,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QProgressBar,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QSystemTrayIcon,
    QVBoxLayout,
    QWidget,
)

from .browser import (
    BrowserMonitor,
    discover_cdp_endpoint,
    open_remote_debugging_settings,
)
from .config import Config, DEFAULT_CONFIG
from .logger import Logger, config_logger, shutdown_logger
from .hotkeys import GlobalHotkeyMonitor, HotkeyBinding
from .screen_capture import CaptureSource, capture_webp, list_capture_sources
from .window_focus import ForegroundWindowRestorer
from .voice.base import ModelProvider, TextToSpeechProvider
from .voice.dependencies import OperationCancelled, PYPI_MIRRORS
from .voice.cosyvoice_tts import (
    COSYVOICE_TTS_MODELS,
    CosyVoiceTtsProvider,
)
from .voice.qwen_tts import (
    MODEL_DOWNLOAD_SOURCES,
    QwenTtsProvider,
    TTS_LANGUAGES,
    TTS_MODELS,
)
from .voice.sherpa_stt import LocalDictationSession, SherpaSttProvider, STT_MODELS
from .voice.sovits_tts import SOVITS_LANGUAGES, SovitsTtsProvider


ASSET_DIRECTORY = Path(__file__).resolve().parent / "assets"
ICON_PATH = ASSET_DIRECTORY / "app-icon.ico"
MICROPHONE_ICON_PATH = ASSET_DIRECTORY / "microphone.svg"
AUTO_HIDE_ICON_PATH = ASSET_DIRECTORY / "auto-hide.svg"
EXIT_ICON_PATH = ASSET_DIRECTORY / "exit.svg"
SEND_ICON_PATH = ASSET_DIRECTORY / "send.svg"
SPEAKER_ICON_PATH = ASSET_DIRECTORY / "speaker.svg"
CHECK_ICON_PATH = ASSET_DIRECTORY / "check.svg"
SETTINGS_ICON_PATH = ASSET_DIRECTORY / "settings.svg"
SHORTCUTS_ICON_PATH = ASSET_DIRECTORY / "shortcuts.svg"
LANGUAGE_ICON_PATH = ASSET_DIRECTORY / "language.svg"
LOCK_ICON_PATH = ASSET_DIRECTORY / "lock.svg"
UNLOCK_ICON_PATH = ASSET_DIRECTORY / "unlock.svg"
logger = Logger.get_logger(__name__)

DEFAULT_HOLD_MIC_HOTKEY = str(DEFAULT_CONFIG["hotkey_hold"])
DEFAULT_HOLD_WITHOUT_SCREENSHOT_HOTKEY = str(
    DEFAULT_CONFIG["hotkey_hold_without_screenshot"]
)
DEFAULT_SEND_HOTKEY = str(DEFAULT_CONFIG["hotkey_send"])
DEFAULT_SEND_WITHOUT_SCREENSHOT_HOTKEY = str(
    DEFAULT_CONFIG["hotkey_send_without_screenshot"]
)
HOTKEY_CONFIG_KEYS = {
    "hold": "hotkey_hold",
    "hold_without_screenshot": "hotkey_hold_without_screenshot",
    "send": "hotkey_send",
    "send_without_screenshot": "hotkey_send_without_screenshot",
}
LEGACY_HOTKEY_SETTING_KEYS = {
    "hold": "hotkeys/hold_microphone",
    "send": "hotkeys/send",
    "send_without_screenshot": "hotkeys/send_without_screenshot",
}


@dataclass(frozen=True)
class _PendingDictationCapture:
    source: CaptureSource | None
    screenshot: bytes | None
    error: str | None = None
    preuploaded: bool = False


class _VoiceOperationThread(QThread):
    progress = Signal(str, object)
    log_line = Signal(str)
    completed = Signal(bool, str, object)

    def __init__(
        self,
        manager: ModelProvider,
        action: str,
        model_type: str = "",
        model_key: str = "",
        mirror: str = "default",
        model_source: str = "huggingface",
    ) -> None:
        super().__init__()
        self.manager = manager
        self.action = action
        self.model_type = model_type
        self.model_key = model_key
        self.mirror = mirror
        self.model_source = model_source
        self._cancel_event = threading.Event()

    def cancel_operation(self) -> None:
        self._cancel_event.set()

    def run(self) -> None:
        try:
            result: object = None
            if self.action == "status":
                if self.model_type == "tts" and isinstance(
                    self.manager, (QwenTtsProvider, CosyVoiceTtsProvider)
                ):
                    dependency_ok, dependency_message = (
                        self.manager.dependency_status(self.model_source)
                    )
                else:
                    dependency_ok, dependency_message = (
                        self.manager.dependency_status()
                    )
                model_ok, model_message = self.manager.model_status(
                    self.model_type, self.model_key
                )
                if (
                    self.model_type == "tts"
                    and isinstance(
                        self.manager,
                        (QwenTtsProvider, CosyVoiceTtsProvider, SovitsTtsProvider),
                    )
                    and dependency_ok
                    and model_ok
                ):
                    self.progress.emit(
                        "Preloading the local TTS model on the GPU…", None
                    )
                    model_message = (
                        f"{model_message} · {self.manager.preload(self.model_key)}"
                    )
                result = {
                    "dependency_ok": dependency_ok,
                    "dependency_message": dependency_message,
                    "model_ok": model_ok,
                    "model_message": model_message,
                }
                message = dependency_message
            elif self.action == "install":
                if self.model_type == "tts" and isinstance(
                    self.manager, (QwenTtsProvider, CosyVoiceTtsProvider)
                ):
                    message = self.manager.install_dependencies(
                        self.progress.emit,
                        self.log_line.emit,
                        self.mirror,
                        self._cancel_event,
                        source=self.model_source,
                    )
                else:
                    message = self.manager.install_dependencies(
                        self.progress.emit,
                        self.log_line.emit,
                        self.mirror,
                        self._cancel_event,
                    )
            elif self.action == "download":
                if self.model_type == "tts" and isinstance(
                    self.manager, (QwenTtsProvider, CosyVoiceTtsProvider)
                ):
                    self.manager.ensure_download_client(
                        self.progress.emit,
                        self.log_line.emit,
                        self.mirror,
                        self._cancel_event,
                        source=self.model_source,
                    )
                    message = self.manager.download_model(
                        self.model_type,
                        self.model_key,
                        self.progress.emit,
                        self.log_line.emit,
                        self._cancel_event,
                        source=self.model_source,
                    )
                else:
                    message = self.manager.download_model(
                        self.model_type,
                        self.model_key,
                        self.progress.emit,
                        self.log_line.emit,
                        self._cancel_event,
                    )
            else:
                raise RuntimeError(f"Unknown voice operation {self.action!r}")
            self.completed.emit(True, message, result)
        except OperationCancelled as error:
            logger.info(f"Voice operation {self.action!r} cancelled")
            self.log_line.emit(str(error))
            self.completed.emit(False, str(error), {"cancelled": True})
        except Exception as error:
            logger.error(f"Voice operation {self.action!r} failed", error)
            if self.action in ("install", "download"):
                self.log_line.emit(f"ERROR: {error}")
            self.completed.emit(False, str(error), None)


class _LocalDictationThread(QThread):
    listening = Signal()
    partial_text = Signal(str)
    completed = Signal(bool, str, str)

    def __init__(self, session: LocalDictationSession) -> None:
        super().__init__()
        self.session = session

    def stop_recording(self, *, cancel: bool = False) -> None:
        self.session.stop(cancel=cancel)

    def run(self) -> None:
        success, text, message = self.session.run(
            self.listening.emit,
            self.partial_text.emit,
        )
        self.completed.emit(success, text, message)


class _LocalSpeechThread(QThread):
    started = Signal(str)
    progress = Signal(object)
    completed = Signal(bool, str)

    def __init__(
        self,
        manager: TextToSpeechProvider,
        tts_model: str,
        text: str,
        speaker: str = "Vivian",
        language: str = "Auto",
    ) -> None:
        super().__init__()
        self.manager = manager
        self.tts_model = tts_model
        self.text = text
        self.speaker = speaker
        self.language = language
        self._cancel_event = threading.Event()

    def request_stop(self) -> None:
        self._cancel_event.set()
        try:
            import sounddevice as sd

            sd.stop()
        except Exception:
            pass

    def run(self) -> None:
        try:
            import sounddevice as sd

            provider_name = getattr(self.manager, "display_name", "Qwen3-TTS")
            if not isinstance(provider_name, str):
                provider_name = "Qwen3-TTS"
            synthesis_started = time.perf_counter()
            stream_synthesis = getattr(self.manager, "synthesize_stream", None)
            if callable(stream_synthesis):
                self._run_streaming(sd, stream_synthesis, synthesis_started)
                return
            samples, sample_rate = self.manager.synthesize(
                self.tts_model, self.text, self.speaker, self.language
            )
            latency_ms = (time.perf_counter() - synthesis_started) * 1000
            audio_seconds = len(samples) / sample_rate
            self.started.emit(f"Playing with {provider_name}…")
            self.progress.emit({"text": self.text, "fraction": 0.0})
            sd.play(samples, sample_rate, blocking=True)
            if self._cancel_event.is_set():
                self.completed.emit(False, "Playback stopped for recording")
                return
            self.progress.emit({"text": self.text, "fraction": 1.0})
            self.completed.emit(
                True,
                f"Generated in {latency_ms:.0f} ms · audio {audio_seconds:.1f} s",
            )
        except Exception as error:
            if self._cancel_event.is_set():
                self.completed.emit(False, "Playback stopped for recording")
                return
            logger.error("Local voice playback failed", error)
            self.completed.emit(False, f"Local voice playback failed: {error}")

    def _run_streaming(
        self,
        sd: object,
        stream_synthesis: object,
        synthesis_started: float,
    ) -> None:
        import numpy as np

        continuous_stream = (
            getattr(self.manager, "continuous_audio_stream", False) is True
        )
        configured_prebuffer = getattr(
            self.manager, "stream_prebuffer_seconds", 0.0
        )
        prebuffer_seconds = (
            max(0.0, float(configured_prebuffer))
            if isinstance(configured_prebuffer, (int, float))
            else 0.0
        )
        generated: queue.Queue[object] = queue.Queue(
            maxsize=16 if continuous_stream else 2
        )
        finished = object()
        generation_finished_at = [synthesis_started]

        def produce() -> None:
            try:
                for chunk in stream_synthesis(
                    self.tts_model,
                    self.text,
                    self.speaker,
                    self.language,
                ):
                    if self._cancel_event.is_set():
                        break
                    generated.put(chunk)
            except Exception as error:
                generated.put(error)
            finally:
                generation_finished_at[0] = time.perf_counter()
                generated.put(finished)

        producer = threading.Thread(
            target=produce,
            name="local-tts-streaming-generator",
            daemon=True,
        )
        producer.start()
        first = generated.get()
        if isinstance(first, Exception):
            generated.get()
            raise first
        if first is finished:
            raise RuntimeError("The local TTS model generated no audio")
        first_generated_seconds = time.perf_counter() - synthesis_started
        first_samples, sample_rate, first_text = first
        buffered: deque[object] = deque((first,))
        first_audio_seconds = len(first_samples) / max(1, sample_rate)
        buffered_audio_seconds = first_audio_seconds
        # A continuous native stream cannot sound smooth if generation is
        # slower than playback. In that case finish buffering before opening
        # the audio device; otherwise retain a short jitter cushion.
        buffer_to_end = (
            continuous_stream
            and first_audio_seconds > 0
            and first_generated_seconds / first_audio_seconds >= 0.85
        )
        if continuous_stream:
            first_chunk_rtf = first_generated_seconds / max(
                first_audio_seconds, 0.001
            )
            logger.info(
                "Continuous TTS buffering "
                f"first_chunk_rtf={first_chunk_rtf:.2f} "
                f"strategy={'complete' if buffer_to_end else 'lookahead'} "
                f"target_seconds={prebuffer_seconds:.1f}"
            )
        while buffer_to_end or buffered_audio_seconds < prebuffer_seconds:
            item = generated.get()
            buffered.append(item)
            if item is finished or isinstance(item, Exception):
                break
            samples, chunk_rate, _chunk_text = item
            if chunk_rate != sample_rate:
                raise RuntimeError(
                    "The local TTS model changed sample rate during streaming"
                )
            buffered_audio_seconds += len(samples) / max(1, sample_rate)
        first_latency_ms = (time.perf_counter() - synthesis_started) * 1000
        provider_name = getattr(self.manager, "display_name", "Qwen3-TTS")
        if not isinstance(provider_name, str):
            provider_name = "Qwen3-TTS"
        self.started.emit(f"Playing with streaming {provider_name}…")
        self.progress.emit({"text": self.text, "fraction": 0.0})
        spoken_characters = 0
        audio_seconds = 0.0
        pending: object = buffered.popleft()
        pending_error: Exception | None = None
        with sd.OutputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
        ) as output:
            while pending is not finished:
                if self._cancel_event.is_set():
                    self.completed.emit(False, "Playback stopped for recording")
                    return
                if isinstance(pending, Exception):
                    pending_error = pending
                else:
                    samples, chunk_rate, chunk_text = pending
                    if chunk_rate != sample_rate:
                        raise RuntimeError(
                            "The local TTS model changed sample rate during streaming"
                        )
                    if continuous_stream:
                        waveform = np.asarray(
                            samples, dtype=np.float32
                        ).reshape(-1, 1)
                    else:
                        waveform = self._prepare_streaming_waveform(
                            np,
                            samples,
                            sample_rate,
                        )
                    if len(waveform):
                        output.write(waveform)
                        audio_seconds += len(waveform) / sample_rate
                    spoken_characters += len(chunk_text)
                    self.progress.emit(
                        {
                            "text": self.text,
                            "fraction": min(
                                spoken_characters / max(1, len(self.text)),
                                0.99,
                            ),
                        }
                    )
                pending = buffered.popleft() if buffered else generated.get()
        producer.join(timeout=1)
        if pending_error is not None:
            raise pending_error
        self.progress.emit({"text": self.text, "fraction": 1.0})
        generation_seconds = generation_finished_at[0] - synthesis_started
        self.completed.emit(
            True,
            f"First audio in {first_latency_ms:.0f} ms · generated in "
            f"{generation_seconds:.1f} s · audio {audio_seconds:.1f} s",
        )

    @staticmethod
    def _prepare_streaming_waveform(
        np: object,
        samples: object,
        sample_rate: int,
    ) -> object:
        """Remove generated edge silence and soften joins between chunks."""
        waveform = np.asarray(samples, dtype=np.float32).reshape(-1)
        if len(waveform) == 0:
            return waveform.reshape(-1, 1)
        peak = float(np.max(np.abs(waveform)))
        if peak > 0:
            audible = np.flatnonzero(np.abs(waveform) >= max(peak * 0.015, 1e-4))
            if len(audible):
                padding = int(sample_rate * 0.025)
                start = max(0, int(audible[0]) - padding)
                end = min(len(waveform), int(audible[-1]) + padding + 1)
                waveform = waveform[start:end].copy()
        fade_samples = min(int(sample_rate * 0.008), len(waveform) // 4)
        if fade_samples:
            waveform[:fade_samples] *= np.linspace(
                0.0,
                1.0,
                fade_samples,
                dtype=np.float32,
            )
            waveform[-fade_samples:] *= np.linspace(
                1.0,
                0.0,
                fade_samples,
                dtype=np.float32,
            )
        return waveform.reshape(-1, 1)


class _QueuedLocalSpeechThread(QThread):
    """Synthesize growing response sentences through one audio output stream."""

    started = Signal(str)
    progress = Signal(object)
    completed = Signal(bool, str)

    _FINISHED = object()
    _SENTENCE_DONE = object()

    def __init__(
        self,
        manager: TextToSpeechProvider,
        tts_model: str,
        speaker: str,
        language: str,
    ) -> None:
        super().__init__()
        self.manager = manager
        self.tts_model = tts_model
        self.speaker = speaker
        self.language = language
        self._sentences: queue.Queue[object] = queue.Queue()
        self._full_text = ""
        self._cancel_event = threading.Event()
        self._output: object | None = None

    def enqueue(self, sentence: str, full_text: str) -> None:
        self._full_text = full_text
        if sentence.strip():
            self._sentences.put(sentence.strip())

    def update_full_text(self, full_text: str) -> None:
        self._full_text = full_text

    def finish_queue(self) -> None:
        self._sentences.put(self._FINISHED)

    def request_stop(self) -> None:
        self._cancel_event.set()
        self._sentences.put(self._FINISHED)
        output = self._output
        if output is not None:
            try:
                output.abort()
            except Exception:
                pass

    def run(self) -> None:
        output = None
        sample_rate = 0
        audio_seconds = 0.0
        spoken_characters = 0
        started_at = time.perf_counter()
        first_audio_at: float | None = None
        try:
            import numpy as np
            import sounddevice as sd

            continuous = (
                getattr(self.manager, "continuous_audio_stream", False) is True
            )
            generated: queue.Queue[object] = queue.Queue(maxsize=16)

            def produce() -> None:
                sentence_index = 0
                try:
                    while True:
                        if self._cancel_event.is_set():
                            generated.put(self._FINISHED)
                            return
                        item = self._sentences.get()
                        if item is self._FINISHED:
                            logger.debug(
                                "Local TTS sentence queue complete "
                                f"sentences={sentence_index}"
                            )
                            generated.put(self._FINISHED)
                            return
                        sentence = str(item)
                        sentence_index += 1
                        logger.debug(
                            "Sending sentence to local TTS "
                            f"index={sentence_index} "
                            f"characters={len(sentence)} "
                            f"text={sentence!r}"
                        )
                        produced = False
                        for chunk in self.manager.synthesize_stream(
                            self.tts_model,
                            sentence,
                            self.speaker,
                            self.language,
                        ):
                            if self._cancel_event.is_set():
                                generated.put(self._FINISHED)
                                return
                            generated.put((chunk, sentence))
                            produced = True
                        if not produced:
                            raise RuntimeError(
                                "The local TTS model generated no audio"
                            )
                        generated.put((self._SENTENCE_DONE, sentence))
                except Exception as error:
                    generated.put(error)
                    generated.put(self._FINISHED)

            producer = threading.Thread(
                target=produce,
                name="queued-local-tts-generator",
                daemon=True,
            )
            producer.start()
            progress_sentence = ""
            sentence_characters_reported = 0
            while True:
                item = generated.get()
                if self._cancel_event.is_set():
                    self.completed.emit(False, "Playback stopped for recording")
                    return
                if item is self._FINISHED:
                    break
                if isinstance(item, Exception):
                    raise item
                if item[0] is self._SENTENCE_DONE:
                    sentence = item[1]
                    already_reported = (
                        sentence_characters_reported
                        if sentence == progress_sentence
                        else 0
                    )
                    spoken_characters += max(0, len(sentence) - already_reported)
                    progress_sentence = ""
                    sentence_characters_reported = 0
                    self.progress.emit(
                        {
                            "text": self._full_text,
                            "spoken_characters": spoken_characters,
                            "fraction": min(
                                spoken_characters / max(1, len(self._full_text)),
                                0.99,
                            ),
                        }
                    )
                    continue
                chunk, sentence = item
                samples, chunk_rate, chunk_text = chunk
                if output is None:
                    sample_rate = chunk_rate
                    output = sd.OutputStream(
                        samplerate=sample_rate,
                        channels=1,
                        dtype="float32",
                    )
                    self._output = output
                    output.start()
                    first_audio_at = time.perf_counter()
                    provider = getattr(self.manager, "display_name", "local TTS")
                    self.started.emit(f"Playing with streaming {provider}…")
                elif chunk_rate != sample_rate:
                    raise RuntimeError(
                        "The local TTS model changed sample rate while queued"
                    )
                if continuous:
                    waveform = np.asarray(samples, dtype=np.float32).reshape(-1, 1)
                else:
                    waveform = _LocalSpeechThread._prepare_streaming_waveform(
                        np, samples, sample_rate
                    )
                if len(waveform):
                    output.write(waveform)
                    audio_seconds += len(waveform) / sample_rate
                if chunk_text:
                    if sentence != progress_sentence:
                        progress_sentence = sentence
                        sentence_characters_reported = 0
                    added = min(
                        len(str(chunk_text)),
                        len(sentence) - sentence_characters_reported,
                    )
                    sentence_characters_reported += max(0, added)
                    spoken_characters += max(0, added)
                    self.progress.emit(
                        {
                            "text": self._full_text,
                            "spoken_characters": spoken_characters,
                            "fraction": min(
                                spoken_characters / max(1, len(self._full_text)),
                                0.99,
                            ),
                        }
                    )
            producer.join(timeout=1)
            if output is None or first_audio_at is None:
                raise RuntimeError("The local TTS queue received no speech")
            self.progress.emit(
                {
                    "text": self._full_text,
                    "spoken_characters": len(self._full_text),
                    "fraction": 1.0,
                }
            )
            self.completed.emit(
                True,
                f"First audio in {(first_audio_at - started_at) * 1000:.0f} ms · "
                f"audio {audio_seconds:.1f} s",
            )
        except Exception as error:
            if self._cancel_event.is_set():
                self.completed.emit(False, "Playback stopped for recording")
                return
            logger.error("Queued local voice playback failed", error)
            self.completed.emit(False, f"Local voice playback failed: {error}")
        finally:
            if output is not None:
                try:
                    output.stop()
                except Exception:
                    pass
                try:
                    output.close()
                except Exception:
                    pass
            self._output = None


class HotkeyConfigDialog(QDialog):
    """Edit Live GPT settings, including pass-through global shortcuts."""

    def __init__(
        self,
        hold_microphone: QKeySequence,
        send: QKeySequence,
        send_without_screenshot: QKeySequence,
        parent: QWidget | None = None,
        *,
        hold_without_screenshot: QKeySequence | None = None,
        language: str = "en",
        recording_backend: str = "web",
        playing_backend: str = "web",
        stt_model: str = "zh_zipformer_ctc_int8_2025_07_03",
        tts_model: str = "qwen3_tts_0_6b_custom_voice",
        tts_speaker: str = "Vivian",
        tts_language: str = "Auto",
        pypi_mirror: str = "default",
        qwen_model_source: str = "huggingface",
        cosyvoice_model_source: str = "huggingface",
        cosyvoice_model: str = "fun_cosyvoice3_0_5b_2512",
        cosyvoice_prompt_audio: str = "",
        cosyvoice_prompt_text: str = "",
        sovits_installation: str = "",
        sovits_text_lang: str = "auto",
        sovits_ref_audio_path: str = "",
        sovits_prompt_text: str = "",
        sovits_prompt_lang: str = "auto",
        config: Config | None = None,
        stt_manager: SherpaSttProvider | None = None,
        tts_manager: QwenTtsProvider | None = None,
        cosyvoice_manager: CosyVoiceTtsProvider | None = None,
        sovits_manager: SovitsTtsProvider | None = None,
    ) -> None:
        super().__init__(parent)
        self._title_drag_offset: QPoint | None = None
        self.stt_manager = stt_manager or SherpaSttProvider()
        self.tts_manager = tts_manager or QwenTtsProvider()
        self.cosyvoice_manager = cosyvoice_manager or CosyVoiceTtsProvider()
        self.sovits_manager = sovits_manager or SovitsTtsProvider()
        self.config = config
        self._voice_worker: _VoiceOperationThread | None = None
        self._voice_record_thread: _LocalDictationThread | None = None
        self._voice_play_thread: _LocalSpeechThread | None = None
        self._voice_status_checked = {"stt": False, "tts": False}
        self._voice_operation_type = "stt"
        self._voice_operation_action = ""
        self._voice_install_log_lines: dict[str, list[str]] = {
            "stt": [],
            "tts": [],
        }
        self.setObjectName("settingsDialog")
        self.setWindowTitle("Live GPT settings")
        self.setWindowFlags(
            Qt.WindowType.Dialog | Qt.WindowType.FramelessWindowHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setMinimumSize(860, 680)
        self.resize(940, 820)

        self.hold_microphone_edit = self._sequence_edit(hold_microphone)
        self.hold_without_screenshot_edit = self._sequence_edit(
            hold_without_screenshot or QKeySequence("Shift")
        )
        self.send_edit = self._sequence_edit(send)
        self.send_without_screenshot_edit = self._sequence_edit(
            send_without_screenshot
        )

        self.settings_shell = QFrame()
        self.settings_shell.setObjectName("settingsShell")
        shell_layout = QVBoxLayout(self.settings_shell)
        shell_layout.setContentsMargins(0, 0, 0, 0)
        shell_layout.setSpacing(0)

        self.title_bar = QFrame()
        self.title_bar.setObjectName("settingsTitleBar")
        self.title_bar.setFixedHeight(56)
        title_layout = QHBoxLayout(self.title_bar)
        title_layout.setContentsMargins(20, 0, 12, 0)
        title_layout.setSpacing(10)
        app_title = QLabel("Live GPT")
        app_title.setObjectName("settingsAppTitle")
        title_separator = QLabel("/")
        title_separator.setObjectName("settingsTitleSeparator")
        window_title = QLabel("Settings")
        window_title.setObjectName("settingsWindowTitle")
        title_layout.addWidget(app_title)
        title_layout.addWidget(title_separator)
        title_layout.addWidget(window_title)
        title_layout.addStretch()

        self.close_button = QPushButton()
        self.close_button.setObjectName("settingsCloseButton")
        self.close_button.setIcon(QIcon(str(EXIT_ICON_PATH)))
        self.close_button.setIconSize(QSize(16, 16))
        self.close_button.setFixedSize(36, 36)
        self.close_button.setAccessibleName("Close settings")
        self.close_button.setToolTip("Close settings; changes are saved automatically")
        self.close_button.clicked.connect(self.reject)
        title_layout.addWidget(self.close_button)
        shell_layout.addWidget(self.title_bar)

        body = QWidget()
        body.setObjectName("settingsBody")
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(0)

        navigation = QFrame()
        navigation.setObjectName("settingsNavigation")
        navigation.setFixedWidth(190)
        navigation_layout = QVBoxLayout(navigation)
        navigation_layout.setContentsMargins(14, 24, 14, 18)
        navigation_layout.setSpacing(8)
        navigation_label = QLabel("SETTINGS")
        navigation_label.setObjectName("settingsNavigationLabel")
        navigation_layout.addWidget(navigation_label)
        navigation_layout.addSpacing(8)

        self.shortcuts_nav_button = self._navigation_button(
            "Shortcuts",
            SHORTCUTS_ICON_PATH,
        )
        self.language_nav_button = self._navigation_button(
            "Language",
            LANGUAGE_ICON_PATH,
        )
        self.recording_nav_button = self._navigation_button(
            "Recording",
            MICROPHONE_ICON_PATH,
        )
        self.playing_nav_button = self._navigation_button(
            "Playing",
            SPEAKER_ICON_PATH,
        )
        self.navigation_group = QButtonGroup(self)
        self.navigation_group.setExclusive(True)
        self.navigation_group.addButton(self.shortcuts_nav_button, 0)
        self.navigation_group.addButton(self.language_nav_button, 1)
        self.navigation_group.addButton(self.recording_nav_button, 2)
        self.navigation_group.addButton(self.playing_nav_button, 3)
        self.shortcuts_nav_button.setChecked(True)
        navigation_layout.addWidget(self.shortcuts_nav_button)
        navigation_layout.addWidget(self.language_nav_button)
        navigation_layout.addWidget(self.recording_nav_button)
        navigation_layout.addWidget(self.playing_nav_button)
        navigation_layout.addStretch()
        body_layout.addWidget(navigation)

        content = QWidget()
        content.setObjectName("settingsContent")
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(28, 24, 28, 22)
        content_layout.setSpacing(16)

        self.settings_pages = QStackedWidget()
        self.settings_pages.setObjectName("settingsPages")

        self.hotkey_section = QWidget()
        self.hotkey_section.setObjectName("settingsPage")
        hotkey_page_layout = QVBoxLayout(self.hotkey_section)
        hotkey_page_layout.setContentsMargins(0, 0, 0, 0)
        hotkey_page_layout.setSpacing(16)
        hotkey_title = QLabel("Keyboard shortcuts")
        hotkey_title.setObjectName("settingsPageTitle")
        hotkey_description = QLabel(
            "Control Live GPT without leaving the app you are using."
        )
        hotkey_description.setObjectName("settingsPageDescription")
        hotkey_description.setWordWrap(True)
        hotkey_page_layout.addWidget(hotkey_title)
        hotkey_page_layout.addWidget(hotkey_description)

        hotkey_card = QFrame()
        hotkey_card.setObjectName("settingsCard")
        hotkey_card_layout = QVBoxLayout(hotkey_card)
        hotkey_card_layout.setContentsMargins(20, 20, 20, 20)
        hotkey_card_layout.setSpacing(14)
        card_title = QLabel("Global shortcuts")
        card_title.setObjectName("settingsCardTitle")
        hotkey_card_layout.addWidget(card_title)

        form = QFormLayout()
        form.setContentsMargins(0, 4, 0, 0)
        form.setHorizontalSpacing(22)
        form.setVerticalSpacing(12)
        form.addRow(
            "Record and Send with Screenshot",
            self.hold_microphone_edit,
        )
        form.addRow(
            "Record and Send without Screenshot",
            self.hold_without_screenshot_edit,
        )
        form.addRow("Send with screenshot", self.send_edit)
        form.addRow(
            "Send without screenshot",
            self.send_without_screenshot_edit,
        )
        hotkey_card_layout.addLayout(form)

        note = QLabel(
            "These shortcuts work globally and are still passed to the "
            "foreground program."
        )
        note.setObjectName("settingsNote")
        note.setWordWrap(True)
        hotkey_card_layout.addWidget(note)
        hotkey_page_layout.addWidget(hotkey_card)
        hotkey_page_layout.addStretch()

        self.language_section = QWidget()
        self.language_section.setObjectName("settingsPage")
        language_page_layout = QVBoxLayout(self.language_section)
        language_page_layout.setContentsMargins(0, 0, 0, 0)
        language_page_layout.setSpacing(16)
        language_title = QLabel("Interface language")
        language_title.setObjectName("settingsPageTitle")
        language_description = QLabel(
            "Choose the language used for menus, labels, and messages."
        )
        language_description.setObjectName("settingsPageDescription")
        language_description.setWordWrap(True)
        language_page_layout.addWidget(language_title)
        language_page_layout.addWidget(language_description)

        language_card = QFrame()
        language_card.setObjectName("settingsCard")
        language_card_layout = QVBoxLayout(language_card)
        language_card_layout.setContentsMargins(20, 20, 20, 20)
        language_card_layout.setSpacing(14)
        language_card_title = QLabel("Display language")
        language_card_title.setObjectName("settingsCardTitle")
        language_card_layout.addWidget(language_card_title)

        self.language_combo = QComboBox()
        self.language_combo.setObjectName("languageCombo")
        self.language_combo.setAccessibleName("Interface language")
        self.language_combo.addItems(("English", "中文 (Chinese)"))
        self.language_combo.setCurrentIndex(1 if language == "zh" else 0)
        self.language_combo.setToolTip(
            "Choose an interface language preview"
        )
        language_card_layout.addWidget(self.language_combo)
        self.language_status = QLabel(
            "Language selection is available in settings, but translations "
            "are not applied yet."
        )
        self.language_status.setObjectName("settingsNote")
        self.language_status.setWordWrap(True)
        language_card_layout.addWidget(self.language_status)
        language_page_layout.addWidget(language_card)
        language_page_layout.addStretch()

        self.recording_section = QWidget()
        self.recording_section.setObjectName("settingsPage")
        recording_layout = QVBoxLayout(self.recording_section)
        recording_layout.setContentsMargins(0, 0, 0, 0)
        recording_layout.setSpacing(12)
        recording_title = QLabel("Recording")
        recording_title.setObjectName("settingsPageTitle")
        recording_description = QLabel(
            "Configure microphone transcription independently from voice playback."
        )
        recording_description.setObjectName("settingsPageDescription")
        recording_description.setWordWrap(True)
        recording_layout.addWidget(recording_title)
        recording_layout.addWidget(recording_description)

        recording_backend_card = QFrame()
        recording_backend_card.setObjectName("settingsCard")
        recording_backend_layout = QVBoxLayout(recording_backend_card)
        recording_backend_layout.setContentsMargins(18, 16, 18, 16)
        recording_backend_layout.setSpacing(10)
        recording_backend_title = QLabel("Recording engine")
        recording_backend_title.setObjectName("settingsCardTitle")
        recording_backend_layout.addWidget(recording_backend_title)
        self.recording_backend_combo = QComboBox()
        self.recording_backend_combo.setObjectName("voiceCombo")
        self.recording_backend_combo.addItem("Web built-in (browser)", "web")
        self.recording_backend_combo.addItem("Sherpa-ONNX (local)", "sherpa")
        recording_backend_index = self.recording_backend_combo.findData(
            recording_backend
        )
        self.recording_backend_combo.setCurrentIndex(
            max(recording_backend_index, 0)
        )
        recording_backend_layout.addWidget(self.recording_backend_combo)
        recording_layout.addWidget(recording_backend_card)

        self.recording_sherpa_card = QFrame()
        self.recording_sherpa_card.setObjectName("settingsCard")
        recording_sherpa_layout = QVBoxLayout(self.recording_sherpa_card)
        recording_sherpa_layout.setContentsMargins(18, 16, 18, 16)
        recording_sherpa_layout.setSpacing(10)
        recording_sherpa_title = QLabel("Local speech to text")
        recording_sherpa_title.setObjectName("settingsCardTitle")
        recording_sherpa_layout.addWidget(recording_sherpa_title)
        recording_runtime_row = QHBoxLayout()
        recording_runtime_row.setSpacing(8)
        self.recording_check_button = QPushButton("Check")
        self.recording_install_button = QPushButton("Install / Repair runtime")
        for button in (self.recording_check_button, self.recording_install_button):
            button.setObjectName("voiceActionButton")
            recording_runtime_row.addWidget(button)
        self.recording_pypi_mirror_combo = QComboBox()
        self.recording_pypi_mirror_combo.setObjectName("voiceCombo")
        self.recording_pypi_mirror_combo.setToolTip(
            "PyPI mirror used when installing or repairing dependencies"
        )
        for mirror in PYPI_MIRRORS.values():
            self.recording_pypi_mirror_combo.addItem(
                f"PyPI: {mirror.label}", mirror.key
            )
        recording_mirror_index = self.recording_pypi_mirror_combo.findData(
            pypi_mirror
        )
        self.recording_pypi_mirror_combo.setCurrentIndex(
            max(recording_mirror_index, 0)
        )
        recording_runtime_row.addWidget(self.recording_pypi_mirror_combo)
        self.recording_cancel_button = QPushButton("Cancel install / download")
        self.recording_cancel_button.setObjectName("voiceCancelButton")
        self.recording_cancel_button.setToolTip(
            "Stop the active dependency installation or model download"
        )
        self.recording_cancel_button.hide()
        recording_runtime_row.addWidget(self.recording_cancel_button)
        recording_runtime_row.addStretch()
        recording_sherpa_layout.addLayout(recording_runtime_row)

        self.stt_model_combo = QComboBox()
        self.stt_model_combo.setObjectName("voiceCombo")
        for model in STT_MODELS.values():
            size_mb = round(model.asset.size / 1024 / 1024)
            self.stt_model_combo.addItem(
                f"[{model.mode}] {model.language} · {model.label} · {size_mb} MB",
                model.key,
            )
        stt_index = self.stt_model_combo.findData(stt_model)
        self.stt_model_combo.setCurrentIndex(max(stt_index, 0))
        recording_sherpa_layout.addWidget(self.stt_model_combo)
        self.stt_model_description = QLabel()
        self.stt_model_description.setObjectName("settingsNote")
        self.stt_model_description.setWordWrap(True)
        recording_sherpa_layout.addWidget(self.stt_model_description)
        self.stt_test_result = QLineEdit()
        self.stt_test_result.setObjectName("voiceTestText")
        self.stt_test_result.setReadOnly(True)
        self.stt_test_result.setPlaceholderText(
            "Live transcription appears here while streaming"
        )
        recording_sherpa_layout.addWidget(self.stt_test_result)
        stt_action_row = QHBoxLayout()
        stt_action_row.setSpacing(8)
        self.stt_download_button = QPushButton("Download STT model")
        self.voice_record_button = QPushButton("Record microphone")
        for button in (self.stt_download_button, self.voice_record_button):
            button.setObjectName("voiceActionButton")
            stt_action_row.addWidget(button)
        stt_action_row.addStretch()
        recording_sherpa_layout.addLayout(stt_action_row)
        self.recording_progress = QProgressBar()
        self.recording_progress.setObjectName("voiceProgress")
        self.recording_progress.setRange(0, 100)
        self.recording_progress.hide()
        recording_sherpa_layout.addWidget(self.recording_progress)
        self.recording_status = QLabel(
            "Select Check to validate the runtime and selected recording model."
        )
        self.recording_status.setObjectName("voiceStatus")
        self.recording_status.setWordWrap(True)
        recording_sherpa_layout.addWidget(self.recording_status)
        self.recording_install_log = QLabel()
        self.recording_install_log.setObjectName("voiceInstallLog")
        self.recording_install_log.setWordWrap(False)
        self.recording_install_log.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )
        self.recording_install_log.hide()
        recording_sherpa_layout.addWidget(self.recording_install_log)
        recording_layout.addWidget(self.recording_sherpa_card)
        recording_layout.addStretch()

        self.playing_section = QWidget()
        self.playing_section.setObjectName("settingsPage")
        playing_layout = QVBoxLayout(self.playing_section)
        playing_layout.setContentsMargins(0, 0, 0, 0)
        playing_layout.setSpacing(12)
        playing_title = QLabel("Playing")
        playing_title.setObjectName("settingsPageTitle")
        playing_description = QLabel(
            "Configure reply speech independently from microphone transcription."
        )
        playing_description.setObjectName("settingsPageDescription")
        playing_description.setWordWrap(True)
        playing_layout.addWidget(playing_title)
        playing_layout.addWidget(playing_description)

        playing_backend_card = QFrame()
        playing_backend_card.setObjectName("settingsCard")
        playing_backend_layout = QVBoxLayout(playing_backend_card)
        playing_backend_layout.setContentsMargins(18, 16, 18, 16)
        playing_backend_layout.setSpacing(10)
        playing_backend_title = QLabel("Playback engine")
        playing_backend_title.setObjectName("settingsCardTitle")
        playing_backend_layout.addWidget(playing_backend_title)
        self.playing_backend_combo = QComboBox()
        self.playing_backend_combo.setObjectName("voiceCombo")
        self.playing_backend_combo.addItem("Web built-in (browser)", "web")
        self.playing_backend_combo.addItem(
            "Qwen3-TTS (local · NVIDIA GPU required)", "qwen"
        )
        self.playing_backend_combo.addItem(
            "CosyVoice 3 (local · zero-shot voice cloning)", "cosyvoice"
        )
        self.playing_backend_combo.addItem(
            "GPT-SoVITS (existing local installation)", "sovits"
        )
        playing_backend_index = self.playing_backend_combo.findData(
            playing_backend
        )
        self.playing_backend_combo.setCurrentIndex(max(playing_backend_index, 0))
        playing_backend_layout.addWidget(self.playing_backend_combo)
        playing_layout.addWidget(playing_backend_card)

        self.playing_qwen_card = QFrame()
        self.playing_qwen_card.setObjectName("settingsCard")
        playing_qwen_layout = QVBoxLayout(self.playing_qwen_card)
        playing_qwen_layout.setContentsMargins(18, 16, 18, 16)
        playing_qwen_layout.setSpacing(10)
        self.playing_local_title = QLabel("Local Qwen3 text to speech")
        self.playing_local_title.setObjectName("settingsCardTitle")
        playing_qwen_layout.addWidget(self.playing_local_title)
        playing_runtime_row = QHBoxLayout()
        playing_runtime_row.setSpacing(8)
        self.playing_check_button = QPushButton("Check")
        self.playing_install_button = QPushButton("Install / Repair runtime")
        for button in (self.playing_check_button, self.playing_install_button):
            button.setObjectName("voiceActionButton")
            playing_runtime_row.addWidget(button)
        self.playing_pypi_mirror_combo = QComboBox()
        self.playing_pypi_mirror_combo.setObjectName("voiceCombo")
        self.playing_pypi_mirror_combo.setToolTip(
            "PyPI mirror used when installing or repairing dependencies"
        )
        for mirror in PYPI_MIRRORS.values():
            self.playing_pypi_mirror_combo.addItem(
                f"PyPI: {mirror.label}", mirror.key
            )
        playing_mirror_index = self.playing_pypi_mirror_combo.findData(
            pypi_mirror
        )
        self.playing_pypi_mirror_combo.setCurrentIndex(
            max(playing_mirror_index, 0)
        )
        playing_runtime_row.addWidget(self.playing_pypi_mirror_combo)
        self.playing_cancel_button = QPushButton("Cancel install / download")
        self.playing_cancel_button.setObjectName("voiceCancelButton")
        self.playing_cancel_button.setToolTip(
            "Stop the active dependency installation or model download"
        )
        self.playing_cancel_button.hide()
        playing_runtime_row.addWidget(self.playing_cancel_button)
        playing_runtime_row.addStretch()
        playing_qwen_layout.addLayout(playing_runtime_row)

        self.tts_model_combo = QComboBox()
        self.tts_model_combo.setObjectName("voiceCombo")
        for model in TTS_MODELS.values():
            self.tts_model_combo.addItem(
                f"[Local] {model.label} · {model.download_size / 1024**3:.2f} GB",
                model.key,
            )
        tts_index = self.tts_model_combo.findData(tts_model)
        self.tts_model_combo.setCurrentIndex(max(tts_index, 0))
        playing_qwen_layout.addWidget(self.tts_model_combo)

        self.cosyvoice_model_combo = QComboBox()
        self.cosyvoice_model_combo.setObjectName("voiceCombo")
        for model in COSYVOICE_TTS_MODELS.values():
            self.cosyvoice_model_combo.addItem(
                f"[Bi-streaming] {model.label} · "
                f"~{model.download_size / 1024**3:.2f} GB",
                model.key,
            )
        cosyvoice_index = self.cosyvoice_model_combo.findData(cosyvoice_model)
        self.cosyvoice_model_combo.setCurrentIndex(max(cosyvoice_index, 0))

        tts_download_row = QHBoxLayout()
        tts_download_row.setSpacing(8)
        self.qwen_model_source_combo = QComboBox()
        self.qwen_model_source_combo.setObjectName("voiceCombo")
        self.qwen_model_source_combo.setToolTip(
            "Service used to download the selected Qwen model"
        )
        for source_key, source_label in MODEL_DOWNLOAD_SOURCES.items():
            self.qwen_model_source_combo.addItem(source_label, source_key)
        source_index = self.qwen_model_source_combo.findData(qwen_model_source)
        self.qwen_model_source_combo.setCurrentIndex(max(source_index, 0))
        self.cosyvoice_model_source_combo = QComboBox()
        self.cosyvoice_model_source_combo.setObjectName("voiceCombo")
        self.cosyvoice_model_source_combo.setToolTip(
            "Service used to download the selected CosyVoice model"
        )
        for source_key, source_label in MODEL_DOWNLOAD_SOURCES.items():
            self.cosyvoice_model_source_combo.addItem(source_label, source_key)
        cosyvoice_source_index = self.cosyvoice_model_source_combo.findData(
            cosyvoice_model_source
        )
        self.cosyvoice_model_source_combo.setCurrentIndex(
            max(cosyvoice_source_index, 0)
        )
        self.tts_download_button = QPushButton("Download TTS model")
        self.tts_download_button.setObjectName("voiceActionButton")
        tts_download_row.addWidget(self.qwen_model_source_combo)
        tts_download_row.addWidget(self.tts_download_button)
        tts_download_row.addStretch()
        playing_qwen_layout.addLayout(tts_download_row)

        self.tts_model_description = QLabel()
        self.tts_model_description.setObjectName("settingsNote")
        self.tts_model_description.setWordWrap(True)
        playing_qwen_layout.addWidget(self.tts_model_description)

        playing_qwen_layout.addWidget(self.cosyvoice_model_combo)
        cosyvoice_download_row = QHBoxLayout()
        cosyvoice_download_row.setSpacing(8)
        self.cosyvoice_download_button = QPushButton("Download TTS model")
        self.cosyvoice_download_button.setObjectName("voiceActionButton")
        cosyvoice_download_row.addWidget(self.cosyvoice_model_source_combo)
        cosyvoice_download_row.addWidget(self.cosyvoice_download_button)
        cosyvoice_download_row.addStretch()
        playing_qwen_layout.addLayout(cosyvoice_download_row)

        self.cosyvoice_model_description = QLabel()
        self.cosyvoice_model_description.setObjectName("settingsNote")
        self.cosyvoice_model_description.setWordWrap(True)
        playing_qwen_layout.addWidget(self.cosyvoice_model_description)

        speaker_row = QHBoxLayout()
        speaker_row.setSpacing(8)
        self.tts_speaker_label = QLabel("Speaker")
        self.tts_speaker_combo = QComboBox()
        self.tts_speaker_combo.setObjectName("voiceCombo")
        speaker_row.addWidget(self.tts_speaker_label)
        speaker_row.addWidget(self.tts_speaker_combo, 1)
        self.tts_language_label = QLabel("Language")
        self.tts_language_combo = QComboBox()
        self.tts_language_combo.setObjectName("voiceCombo")
        self.tts_language_combo.setToolTip(
            "Choose Qwen's synthesis language, or Auto for model detection"
        )
        for language_name in TTS_LANGUAGES:
            self.tts_language_combo.addItem(language_name, language_name)
        tts_language_index = self.tts_language_combo.findData(tts_language)
        self.tts_language_combo.setCurrentIndex(max(tts_language_index, 0))
        speaker_row.addWidget(self.tts_language_label)
        speaker_row.addWidget(self.tts_language_combo, 1)
        playing_qwen_layout.addLayout(speaker_row)

        self.cosyvoice_prompt_audio_label = QLabel("Reference voice")
        self.cosyvoice_prompt_audio_edit = QLineEdit(cosyvoice_prompt_audio)
        self.cosyvoice_prompt_audio_edit.setObjectName("voiceTestText")
        self.cosyvoice_prompt_audio_edit.setPlaceholderText(
            "WAV path (blank uses the official CosyVoice sample)"
        )
        self.cosyvoice_prompt_browse_button = QPushButton("Browse…")
        self.cosyvoice_prompt_browse_button.setObjectName("voiceActionButton")
        cosyvoice_audio_row = QHBoxLayout()
        cosyvoice_audio_row.setSpacing(8)
        cosyvoice_audio_row.addWidget(self.cosyvoice_prompt_audio_label)
        cosyvoice_audio_row.addWidget(self.cosyvoice_prompt_audio_edit, 1)
        cosyvoice_audio_row.addWidget(self.cosyvoice_prompt_browse_button)
        playing_qwen_layout.addLayout(cosyvoice_audio_row)

        self.cosyvoice_prompt_text_label = QLabel("Reference transcript")
        self.cosyvoice_prompt_text_edit = QLineEdit(cosyvoice_prompt_text)
        self.cosyvoice_prompt_text_edit.setObjectName("voiceTestText")
        self.cosyvoice_prompt_text_edit.setPlaceholderText(
            "Exact transcript (blank uses the official sample transcript)"
        )
        cosyvoice_text_row = QHBoxLayout()
        cosyvoice_text_row.setSpacing(8)
        cosyvoice_text_row.addWidget(self.cosyvoice_prompt_text_label)
        cosyvoice_text_row.addWidget(self.cosyvoice_prompt_text_edit, 1)
        playing_qwen_layout.addLayout(cosyvoice_text_row)

        self.sovits_installation_label = QLabel("Installation folder")
        self.sovits_installation_edit = QLineEdit(sovits_installation)
        self.sovits_installation_edit.setObjectName("voiceTestText")
        self.sovits_installation_edit.setPlaceholderText(
            "Folder containing GPT_SoVITS and runtime/python.exe"
        )
        self.sovits_installation_browse_button = QPushButton("Browse…")
        self.sovits_installation_browse_button.setObjectName("voiceActionButton")
        sovits_installation_row = QHBoxLayout()
        sovits_installation_row.setSpacing(8)
        sovits_installation_row.addWidget(self.sovits_installation_label)
        sovits_installation_row.addWidget(self.sovits_installation_edit, 1)
        sovits_installation_row.addWidget(self.sovits_installation_browse_button)
        playing_qwen_layout.addLayout(sovits_installation_row)

        self.sovits_reference_title = QLabel("Reference voice")
        self.sovits_reference_title.setObjectName("voiceFieldGroupTitle")
        self.sovits_reference_description = QLabel(
            "These settings describe the voice sample GPT-SoVITS should imitate."
        )
        self.sovits_reference_description.setObjectName("settingsNote")
        self.sovits_reference_description.setWordWrap(True)
        playing_qwen_layout.addWidget(self.sovits_reference_title)
        playing_qwen_layout.addWidget(self.sovits_reference_description)

        self.sovits_text_lang_label = QLabel("Output language")
        self.sovits_text_lang_combo = QComboBox()
        self.sovits_text_lang_combo.setObjectName("voiceCombo")
        self.sovits_text_lang_combo.setToolTip(
            "Language of the text that GPT-SoVITS will generate"
        )
        self.sovits_prompt_lang_label = QLabel("Reference language")
        self.sovits_prompt_lang_combo = QComboBox()
        self.sovits_prompt_lang_combo.setObjectName("voiceCombo")
        self.sovits_prompt_lang_combo.setToolTip(
            "Language spoken in the reference audio and transcript"
        )
        for language_code in SOVITS_LANGUAGES:
            label = language_code.replace("all_", "all ").replace("_", " ")
            self.sovits_text_lang_combo.addItem(label, language_code)
            self.sovits_prompt_lang_combo.addItem(label, language_code)
        self.sovits_text_lang_combo.setCurrentIndex(
            max(self.sovits_text_lang_combo.findData(sovits_text_lang), 0)
        )
        self.sovits_prompt_lang_combo.setCurrentIndex(
            max(self.sovits_prompt_lang_combo.findData(sovits_prompt_lang), 0)
        )
        self.sovits_ref_audio_label = QLabel("Reference audio")
        self.sovits_ref_audio_edit = QLineEdit(sovits_ref_audio_path)
        self.sovits_ref_audio_edit.setObjectName("voiceTestText")
        self.sovits_ref_audio_edit.setPlaceholderText(
            "Required for synthesis; cached while unchanged"
        )
        self.sovits_ref_audio_browse_button = QPushButton("Browse…")
        self.sovits_ref_audio_browse_button.setObjectName("voiceActionButton")
        sovits_ref_row = QHBoxLayout()
        sovits_ref_row.setSpacing(8)
        sovits_ref_row.addWidget(self.sovits_ref_audio_label)
        sovits_ref_row.addWidget(self.sovits_ref_audio_edit, 1)
        sovits_ref_row.addWidget(self.sovits_ref_audio_browse_button)
        playing_qwen_layout.addLayout(sovits_ref_row)

        self.sovits_prompt_text_label = QLabel("Reference transcript")
        self.sovits_prompt_text_edit = QLineEdit(sovits_prompt_text)
        self.sovits_prompt_text_edit.setObjectName("voiceTestText")
        self.sovits_prompt_text_edit.setPlaceholderText("Optional exact transcript")
        sovits_prompt_row = QHBoxLayout()
        sovits_prompt_row.setSpacing(8)
        sovits_prompt_row.addWidget(self.sovits_prompt_text_label)
        sovits_prompt_row.addWidget(self.sovits_prompt_text_edit, 1)
        playing_qwen_layout.addLayout(sovits_prompt_row)

        sovits_reference_language_row = QHBoxLayout()
        sovits_reference_language_row.setSpacing(8)
        sovits_reference_language_row.addWidget(self.sovits_prompt_lang_label)
        sovits_reference_language_row.addWidget(self.sovits_prompt_lang_combo, 1)
        playing_qwen_layout.addLayout(sovits_reference_language_row)

        self.sovits_output_title = QLabel("Generated speech")
        self.sovits_output_title.setObjectName("voiceFieldGroupTitle")
        self.sovits_output_description = QLabel(
            "Choose the language of ChatGPT replies sent to speech synthesis."
        )
        self.sovits_output_description.setObjectName("settingsNote")
        self.sovits_output_description.setWordWrap(True)
        playing_qwen_layout.addWidget(self.sovits_output_title)
        playing_qwen_layout.addWidget(self.sovits_output_description)
        sovits_output_language_row = QHBoxLayout()
        sovits_output_language_row.setSpacing(8)
        sovits_output_language_row.addWidget(self.sovits_text_lang_label)
        sovits_output_language_row.addWidget(self.sovits_text_lang_combo, 1)
        playing_qwen_layout.addLayout(sovits_output_language_row)

        self.voice_test_text = QLineEdit()
        self.voice_test_text.setObjectName("voiceTestText")
        self.voice_test_text.setPlaceholderText("Text to synthesize")
        tts_action_row = QHBoxLayout()
        tts_action_row.setSpacing(8)
        self.voice_play_button = QPushButton("Play text")
        self.voice_play_button.setObjectName("voiceActionButton")
        tts_action_row.addWidget(self.voice_play_button)
        tts_action_row.addStretch()
        playing_qwen_layout.addWidget(self.voice_test_text)
        playing_qwen_layout.addLayout(tts_action_row)
        self.playing_progress = QProgressBar()
        self.playing_progress.setObjectName("voiceProgress")
        self.playing_progress.setRange(0, 100)
        self.playing_progress.hide()
        playing_qwen_layout.addWidget(self.playing_progress)
        self.playing_status = QLabel(
            "Select Check to validate the runtime and selected playback model."
        )
        self.playing_status.setObjectName("voiceStatus")
        self.playing_status.setWordWrap(True)
        playing_qwen_layout.addWidget(self.playing_status)
        self.playing_install_log = QLabel()
        self.playing_install_log.setObjectName("voiceInstallLog")
        self.playing_install_log.setWordWrap(False)
        self.playing_install_log.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )
        self.playing_install_log.hide()
        playing_qwen_layout.addWidget(self.playing_install_log)
        playing_layout.addWidget(self.playing_qwen_card)
        playing_layout.addStretch()

        self.recording_scroll = QScrollArea()
        self.recording_scroll.setObjectName("settingsVoiceScroll")
        self.recording_scroll.setWidgetResizable(True)
        self.recording_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.recording_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.recording_scroll.setWidget(self.recording_section)
        self.playing_scroll = QScrollArea()
        self.playing_scroll.setObjectName("settingsVoiceScroll")
        self.playing_scroll.setWidgetResizable(True)
        self.playing_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.playing_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.playing_scroll.setWidget(self.playing_section)
        self.settings_pages.addWidget(self.hotkey_section)
        self.settings_pages.addWidget(self.language_section)
        self.settings_pages.addWidget(self.recording_scroll)
        self.settings_pages.addWidget(self.playing_scroll)
        self.shortcuts_nav_button.clicked.connect(
            lambda checked: checked and self.settings_pages.setCurrentIndex(0)
        )
        self.language_nav_button.clicked.connect(
            lambda checked: checked and self.settings_pages.setCurrentIndex(1)
        )
        self.recording_nav_button.clicked.connect(self._show_recording_page)
        self.playing_nav_button.clicked.connect(self._show_playing_page)
        self.recording_backend_combo.currentIndexChanged.connect(
            self._sync_recording_controls
        )
        self.playing_backend_combo.currentIndexChanged.connect(
            self._sync_playing_controls
        )
        self.recording_pypi_mirror_combo.currentIndexChanged.connect(
            lambda: self._sync_pypi_mirror(
                self.recording_pypi_mirror_combo,
                self.playing_pypi_mirror_combo,
            )
        )
        self.playing_pypi_mirror_combo.currentIndexChanged.connect(
            lambda: self._sync_pypi_mirror(
                self.playing_pypi_mirror_combo,
                self.recording_pypi_mirror_combo,
            )
        )
        self.recording_cancel_button.clicked.connect(self._cancel_voice_operation)
        self.playing_cancel_button.clicked.connect(self._cancel_voice_operation)
        self.stt_model_combo.currentIndexChanged.connect(self._stt_model_changed)
        self.tts_model_combo.currentIndexChanged.connect(self._tts_model_changed)
        self.cosyvoice_model_combo.currentIndexChanged.connect(
            self._cosyvoice_model_changed
        )
        self.qwen_model_source_combo.currentIndexChanged.connect(
            self._qwen_model_source_changed
        )
        self.cosyvoice_model_source_combo.currentIndexChanged.connect(
            self._cosyvoice_model_source_changed
        )
        self.cosyvoice_prompt_browse_button.clicked.connect(
            self._browse_cosyvoice_prompt_audio
        )
        self.sovits_installation_browse_button.clicked.connect(
            self._browse_sovits_installation
        )
        self.sovits_ref_audio_browse_button.clicked.connect(
            self._browse_sovits_ref_audio
        )
        self.recording_check_button.clicked.connect(
            lambda: self._start_voice_operation("status", "stt")
        )
        self.recording_install_button.clicked.connect(
            lambda: self._start_voice_operation("install", "stt")
        )
        self.playing_check_button.clicked.connect(
            lambda: self._start_voice_operation("status", "tts")
        )
        self.playing_install_button.clicked.connect(
            lambda: self._start_voice_operation("install", "tts")
        )
        self.stt_download_button.clicked.connect(
            lambda: self._start_voice_operation("download", "stt")
        )
        self.tts_download_button.clicked.connect(
            lambda: self._start_voice_operation("download", "tts")
        )
        self.cosyvoice_download_button.clicked.connect(
            lambda: self._start_voice_operation("download", "tts")
        )
        self.voice_record_button.clicked.connect(self._toggle_voice_record_test)
        self.voice_play_button.clicked.connect(self._start_voice_play_test)
        self._pending_tts_speaker = tts_speaker
        self._stt_model_changed()
        self._tts_model_changed()
        self._cosyvoice_model_changed()
        self._sync_recording_controls()
        self._sync_playing_controls()
        self._connect_auto_save()
        content_layout.addWidget(self.settings_pages, 1)
        body_layout.addWidget(content, 1)
        shell_layout.addWidget(body, 1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.settings_shell)

        self._title_drag_widgets = (
            self.title_bar,
            app_title,
            title_separator,
            window_title,
        )
        for widget in self._title_drag_widgets:
            widget.installEventFilter(self)

        self.setStyleSheet(
            """
            QDialog#settingsDialog {
                background: transparent;
            }
            QFrame#settingsShell {
                color: #f5f7ff;
                background-color: #0c1430;
                border: 1px solid rgba(130, 165, 230, 75);
                border-radius: 14px;
            }
            QFrame#settingsTitleBar {
                background-color: #10182e;
                border: none;
                border-bottom: 1px solid rgba(130, 165, 230, 45);
                border-top-left-radius: 14px;
                border-top-right-radius: 14px;
            }
            QLabel#settingsAppTitle {
                color: #f5f7ff;
                background: transparent;
                font-size: 15px;
                font-weight: 700;
            }
            QLabel#settingsTitleSeparator,
            QLabel#settingsWindowTitle {
                color: #8795b8;
                background: transparent;
                font-size: 14px;
            }
            QPushButton#settingsCloseButton {
                background: transparent;
                border: none;
                border-radius: 8px;
            }
            QPushButton#settingsCloseButton:hover {
                background-color: rgba(239, 68, 88, 190);
            }
            QWidget#settingsBody,
            QWidget#settingsContent,
            QWidget#settingsPage,
            QScrollArea#settingsVoiceScroll,
            QScrollArea#settingsVoiceScroll QWidget#qt_scrollarea_viewport,
            QStackedWidget#settingsPages {
                background: transparent;
                border: none;
            }
            QFrame#settingsNavigation {
                background-color: rgba(8, 14, 34, 120);
                border: none;
                border-right: 1px solid rgba(130, 165, 230, 45);
                border-bottom-left-radius: 14px;
            }
            QLabel#settingsNavigationLabel {
                color: #7180a4;
                background: transparent;
                border: none;
                padding-left: 10px;
                font-size: 10px;
                font-weight: 700;
            }
            QPushButton#settingsNavigationButton {
                min-height: 42px;
                padding: 0 12px;
                color: #cbd4ed;
                background: transparent;
                border: 1px solid transparent;
                border-radius: 9px;
                text-align: left;
                font-size: 14px;
            }
            QPushButton#settingsNavigationButton:hover {
                color: #f5f7ff;
                background-color: rgba(70, 88, 140, 80);
            }
            QPushButton#settingsNavigationButton:checked {
                color: #f5f7ff;
                background-color: rgba(38, 112, 145, 115);
                border-color: rgba(76, 201, 240, 80);
            }
            QLabel#settingsPageTitle {
                color: #f5f7ff;
                background: transparent;
                font-size: 24px;
                font-weight: 700;
            }
            QLabel#settingsPageDescription {
                color: #aeb9d5;
                background: transparent;
                font-size: 14px;
            }
            QFrame#settingsCard {
                color: #f5f7ff;
                background-color: #121c3a;
                border: 1px solid rgba(130, 165, 230, 70);
                border-radius: 12px;
            }
            QFrame#settingsCard QLabel {
                color: #e9edff;
                background: transparent;
                border: none;
            }
            QLabel#settingsCardTitle {
                color: #f5f7ff;
                font-size: 16px;
                font-weight: 700;
            }
            QLabel#settingsNote {
                color: #8795b8;
                font-size: 12px;
            }
            QLabel#voiceFieldGroupTitle {
                color: #dce7ff;
                font-size: 13px;
                font-weight: 600;
                padding-top: 7px;
            }
            QKeySequenceEdit {
                min-width: 230px;
                background: transparent;
                border: none;
            }
            QKeySequenceEdit QLineEdit,
            QComboBox#languageCombo,
            QComboBox#voiceCombo,
            QLineEdit#voiceTestText {
                min-height: 34px;
                min-width: 230px;
                padding: 0 10px;
                color: #f5f7ff;
                background-color: rgba(5, 10, 28, 165);
                border: 1px solid rgba(130, 165, 230, 85);
                border-radius: 8px;
                selection-background-color: rgba(76, 201, 240, 130);
            }
            QKeySequenceEdit QLineEdit:focus {
                border-color: rgba(76, 201, 240, 190);
            }
            QComboBox#languageCombo QAbstractItemView,
            QComboBox#voiceCombo QAbstractItemView {
                color: #f5f7ff;
                background-color: #182342;
                border: 1px solid rgba(130, 165, 230, 85);
                selection-background-color: rgb(38, 112, 145);
            }
            QScrollArea#settingsVoiceScroll QScrollBar:vertical {
                width: 10px;
                margin: 0;
                background: rgba(5, 10, 28, 130);
                border: none;
            }
            QScrollArea#settingsVoiceScroll QScrollBar::handle:vertical {
                min-height: 36px;
                background: rgba(76, 201, 240, 120);
                border-radius: 5px;
            }
            QScrollArea#settingsVoiceScroll QScrollBar::add-line:vertical,
            QScrollArea#settingsVoiceScroll QScrollBar::sub-line:vertical {
                height: 0;
            }
            QPushButton#voiceActionButton {
                min-height: 32px;
                padding: 0 12px;
                color: #f5f7ff;
                background-color: rgba(38, 112, 145, 125);
                border: 1px solid rgba(76, 201, 240, 95);
                border-radius: 8px;
            }
            QPushButton#voiceActionButton:hover {
                background-color: rgba(44, 140, 175, 170);
            }
            QPushButton#voiceActionButton:disabled {
                color: #6f7893;
                background-color: rgba(40, 50, 80, 80);
                border-color: rgba(100, 115, 150, 45);
            }
            QPushButton#voiceCancelButton {
                min-height: 32px;
                padding: 0 12px;
                color: #ffd8de;
                background-color: rgba(135, 45, 65, 135);
                border: 1px solid rgba(255, 120, 145, 105);
                border-radius: 8px;
            }
            QPushButton#voiceCancelButton:hover {
                background-color: rgba(175, 50, 75, 180);
            }
            QPushButton#voiceCancelButton:disabled {
                color: #7f6670;
                background-color: rgba(70, 38, 48, 80);
                border-color: rgba(120, 75, 85, 45);
            }
            QProgressBar#voiceProgress {
                min-height: 8px;
                max-height: 8px;
                color: transparent;
                background-color: rgba(5, 10, 28, 165);
                border: none;
                border-radius: 4px;
            }
            QProgressBar#voiceProgress::chunk {
                background-color: rgb(76, 201, 240);
                border-radius: 4px;
            }
            QLabel#voiceStatus {
                color: #9eacce;
                font-size: 12px;
            }
            QLabel#voiceStatus[error="true"] {
                color: #ff8d9b;
            }
            QLabel#voiceInstallLog {
                min-height: 18px;
                color: #7182aa;
                font-size: 11px;
                font-family: Consolas, "Courier New", monospace;
            }
            """
        )

    def _show_recording_page(self, checked: bool) -> None:
        if checked:
            self.settings_pages.setCurrentIndex(2)
            self._check_voice_page_when_needed("stt")

    def _show_playing_page(self, checked: bool) -> None:
        if checked:
            self.settings_pages.setCurrentIndex(3)
            self._check_voice_page_when_needed("tts")

    def _check_voice_page_when_needed(self, model_type: str) -> None:
        if (
            self._is_local_voice_backend(model_type)
            and not self._voice_status_checked[model_type]
        ):
            self._voice_status_checked[model_type] = True
            QTimer.singleShot(
                0,
                lambda: self._start_voice_operation("status", model_type),
            )

    def _sync_recording_controls(self) -> None:
        enabled = self.recording_backend() == "sherpa"
        self.recording_sherpa_card.setVisible(enabled)
        if enabled and self.settings_pages.currentIndex() == 2:
            self._check_voice_page_when_needed("stt")

    def _sync_playing_controls(self) -> None:
        backend = self.playing_backend()
        local_enabled = backend in ("qwen", "cosyvoice", "sovits")
        qwen_enabled = backend == "qwen"
        cosyvoice_enabled = backend == "cosyvoice"
        sovits_enabled = backend == "sovits"
        self.playing_qwen_card.setVisible(local_enabled)
        title = "Local Qwen3 text to speech"
        if cosyvoice_enabled:
            title = "Local CosyVoice 3 text to speech"
        elif sovits_enabled:
            title = "GPT-SoVITS text to speech"
        self.playing_local_title.setText(title)
        self.playing_install_button.setVisible(not sovits_enabled)
        self.playing_pypi_mirror_combo.setVisible(not sovits_enabled)
        for widget in (
            self.tts_model_combo,
            self.qwen_model_source_combo,
            self.tts_download_button,
            self.tts_model_description,
            self.tts_speaker_label,
            self.tts_speaker_combo,
            self.tts_language_label,
            self.tts_language_combo,
        ):
            widget.setVisible(qwen_enabled)
        for widget in (
            self.cosyvoice_model_combo,
            self.cosyvoice_model_source_combo,
            self.cosyvoice_download_button,
            self.cosyvoice_model_description,
            self.cosyvoice_prompt_audio_label,
            self.cosyvoice_prompt_audio_edit,
            self.cosyvoice_prompt_browse_button,
            self.cosyvoice_prompt_text_label,
            self.cosyvoice_prompt_text_edit,
        ):
            widget.setVisible(cosyvoice_enabled)
        for widget in (
            self.sovits_installation_label,
            self.sovits_installation_edit,
            self.sovits_installation_browse_button,
            self.sovits_reference_title,
            self.sovits_reference_description,
            self.sovits_text_lang_label,
            self.sovits_text_lang_combo,
            self.sovits_prompt_lang_label,
            self.sovits_prompt_lang_combo,
            self.sovits_ref_audio_label,
            self.sovits_ref_audio_edit,
            self.sovits_ref_audio_browse_button,
            self.sovits_prompt_text_label,
            self.sovits_prompt_text_edit,
            self.sovits_output_title,
            self.sovits_output_description,
        ):
            widget.setVisible(sovits_enabled)
        if qwen_enabled:
            self.voice_test_text.setText(TTS_MODELS[self.tts_model()].test_text)
        elif cosyvoice_enabled:
            self.voice_test_text.setText(
                COSYVOICE_TTS_MODELS[self.cosyvoice_model()].test_text
            )
        elif sovits_enabled:
            self.voice_test_text.setText("Hello from GPT-SoVITS.")
        self._voice_status_checked["tts"] = False
        if local_enabled and self.settings_pages.currentIndex() == 3:
            self._check_voice_page_when_needed("tts")

    def _connect_auto_save(self) -> None:
        self.language_combo.currentIndexChanged.connect(
            lambda: self._save_setting("language", self.language())
        )
        self.recording_backend_combo.currentIndexChanged.connect(
            lambda: self._save_setting(
                "recording_backend", self.recording_backend()
            )
        )
        self.playing_backend_combo.currentIndexChanged.connect(
            lambda: self._save_setting("playing_backend", self.playing_backend())
        )
        self.recording_pypi_mirror_combo.currentIndexChanged.connect(
            lambda: self._save_setting("pypi_mirror", self.pypi_mirror())
        )
        self.playing_pypi_mirror_combo.currentIndexChanged.connect(
            lambda: self._save_setting("pypi_mirror", self.pypi_mirror())
        )
        self.stt_model_combo.currentIndexChanged.connect(
            lambda: self._save_setting("stt_model", self.stt_model())
        )
        self.tts_model_combo.currentIndexChanged.connect(
            lambda: self._save_setting("tts_model", self.tts_model())
        )
        self.tts_speaker_combo.currentIndexChanged.connect(
            lambda: self._save_setting("tts_speaker", self.tts_speaker())
        )
        self.tts_language_combo.currentIndexChanged.connect(
            lambda: self._save_setting("tts_language", self.tts_language())
        )
        self.qwen_model_source_combo.currentIndexChanged.connect(
            lambda: self._save_setting(
                "qwen_model_source", self.qwen_model_source()
            )
        )
        self.cosyvoice_model_combo.currentIndexChanged.connect(
            lambda: self._save_setting("cosyvoice_model", self.cosyvoice_model())
        )
        self.cosyvoice_model_source_combo.currentIndexChanged.connect(
            lambda: self._save_setting(
                "cosyvoice_model_source", self.cosyvoice_model_source()
            )
        )
        self.cosyvoice_prompt_audio_edit.textChanged.connect(
            lambda text: self._save_setting("cosyvoice_prompt_audio", text)
        )
        self.cosyvoice_prompt_text_edit.textChanged.connect(
            lambda text: self._save_setting("cosyvoice_prompt_text", text)
        )
        self.sovits_installation_edit.textChanged.connect(
            lambda text: self._save_setting("sovits_installation", text)
        )
        self.sovits_text_lang_combo.currentIndexChanged.connect(
            lambda: self._save_setting("sovits_text_lang", self.sovits_text_lang())
        )
        self.sovits_ref_audio_edit.textChanged.connect(
            lambda text: self._save_setting("sovits_ref_audio_path", text)
        )
        self.sovits_prompt_text_edit.textChanged.connect(
            lambda text: self._save_setting("sovits_prompt_text", text)
        )
        self.sovits_prompt_lang_combo.currentIndexChanged.connect(
            lambda: self._save_setting("sovits_prompt_lang", self.sovits_prompt_lang())
        )
        for editor in (
            self.hold_microphone_edit,
            self.hold_without_screenshot_edit,
            self.send_edit,
            self.send_without_screenshot_edit,
        ):
            editor.keySequenceChanged.connect(
                lambda _sequence: self._save_hotkeys()
            )

    def _save_setting(self, key: str, value: object) -> None:
        if self.config is not None:
            self.config[key] = value

    def _save_hotkeys(self) -> None:
        if self.config is None:
            return
        try:
            self.bindings()
        except ValueError:
            return
        updates = {
            HOTKEY_CONFIG_KEYS[name]: sequence.toString(
                QKeySequence.SequenceFormat.PortableText
            )
            for name, sequence in self.sequences().items()
        }
        if not all(updates.values()):
            return
        self.config.update(updates)

    @staticmethod
    def _sync_pypi_mirror(source: QComboBox, target: QComboBox) -> None:
        index = target.findData(source.currentData())
        if index < 0 or index == target.currentIndex():
            return
        target.blockSignals(True)
        try:
            target.setCurrentIndex(index)
        finally:
            target.blockSignals(False)

    def _stt_model_changed(self) -> None:
        model = STT_MODELS[self.stt_model()]
        self.stt_model_description.setText(model.description)
        self.stt_test_result.clear()
        self._voice_status_checked["stt"] = False

    def _tts_model_changed(self) -> None:
        model = TTS_MODELS[self.tts_model()]
        self.tts_model_description.setText(
            f"{model.description}\nStored in {self.tts_manager.model_root.resolve()}"
        )
        selected_speaker = getattr(self, "_pending_tts_speaker", "Vivian")
        self.tts_speaker_combo.clear()
        for speaker in model.speakers:
            self.tts_speaker_combo.addItem(
                f"{speaker.label} · {speaker.description}", speaker.key
            )
        speaker_index = self.tts_speaker_combo.findData(selected_speaker)
        self.tts_speaker_combo.setCurrentIndex(max(speaker_index, 0))
        self._pending_tts_speaker = "Vivian"
        self.tts_speaker_combo.setEnabled(len(model.speakers) > 1)
        self.voice_test_text.setText(model.test_text)
        self._voice_status_checked["tts"] = False

    def _qwen_model_source_changed(self) -> None:
        self._voice_status_checked["tts"] = False
        self.playing_status.setProperty("error", False)
        self.playing_status.setText(
            f"{MODEL_DOWNLOAD_SOURCES[self.qwen_model_source()]} selected for "
            "Qwen model downloads. Select Check to validate its client."
        )
        self._refresh_voice_status_style(self.playing_status)

    def _cosyvoice_model_changed(self) -> None:
        model = COSYVOICE_TTS_MODELS[self.cosyvoice_model()]
        self.cosyvoice_model_description.setText(
            f"{model.description}\nStored in "
            f"{self.cosyvoice_manager.model_root.resolve()}"
        )
        if self.playing_backend() == "cosyvoice":
            self.voice_test_text.setText(model.test_text)
        self._voice_status_checked["tts"] = False

    def _cosyvoice_model_source_changed(self) -> None:
        self._voice_status_checked["tts"] = False
        if self.playing_backend() != "cosyvoice":
            return
        self.playing_status.setProperty("error", False)
        self.playing_status.setText(
            f"{MODEL_DOWNLOAD_SOURCES[self.cosyvoice_model_source()]} selected "
            "for CosyVoice model downloads. Select Check to validate it."
        )
        self._refresh_voice_status_style(self.playing_status)

    def _browse_cosyvoice_prompt_audio(self) -> None:
        selected, _filter = QFileDialog.getOpenFileName(
            self,
            "Choose CosyVoice reference audio",
            self.cosyvoice_prompt_audio_edit.text(),
            "Wave audio (*.wav);;Audio files (*.wav *.flac *.mp3);;All files (*)",
        )
        if selected:
            self.cosyvoice_prompt_audio_edit.setText(selected)

    def _browse_sovits_installation(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "Choose GPT-SoVITS installation",
            self.sovits_installation_edit.text(),
        )
        if selected:
            self.sovits_installation_edit.setText(selected)

    def _browse_sovits_ref_audio(self) -> None:
        selected, _filter = QFileDialog.getOpenFileName(
            self,
            "Choose GPT-SoVITS reference audio",
            self.sovits_ref_audio_edit.text(),
            "Audio files (*.wav *.flac *.mp3);;All files (*)",
        )
        if selected:
            self.sovits_ref_audio_edit.setText(selected)

    def _voice_backend(self, model_type: str) -> str:
        return (
            self.recording_backend()
            if model_type == "stt"
            else self.playing_backend()
        )

    def _is_local_voice_backend(self, model_type: str) -> bool:
        backend = self._voice_backend(model_type)
        return backend == "sherpa" if model_type == "stt" else backend in (
            "qwen",
            "cosyvoice",
            "sovits",
        )

    def _voice_provider(self, model_type: str) -> ModelProvider:
        if model_type == "stt":
            return self.stt_manager
        if self.playing_backend() == "sovits":
            self.sovits_manager.configure(
                self.sovits_prompt_text(), self.sovits_prompt_lang()
            )
            self.sovits_manager.model_status("tts", self.sovits_installation())
            return self.sovits_manager
        return (
            self.cosyvoice_manager
            if self.playing_backend() == "cosyvoice"
            else self.tts_manager
        )

    def _active_tts_model(self) -> str:
        if self.playing_backend() == "sovits":
            return self.sovits_installation()
        return (
            self.cosyvoice_model()
            if self.playing_backend() == "cosyvoice"
            else self.tts_model()
        )

    def _active_tts_source(self) -> str:
        if self.playing_backend() == "sovits":
            return "existing"
        return (
            self.cosyvoice_model_source()
            if self.playing_backend() == "cosyvoice"
            else self.qwen_model_source()
        )

    def _voice_widgets(self, model_type: str) -> tuple[QLabel, QProgressBar]:
        if model_type == "stt":
            return self.recording_status, self.recording_progress
        return self.playing_status, self.playing_progress

    def _voice_install_log_widget(self, model_type: str) -> QLabel:
        return (
            self.recording_install_log
            if model_type == "stt"
            else self.playing_install_log
        )

    def _voice_cancel_button(self, model_type: str) -> QPushButton:
        return (
            self.recording_cancel_button
            if model_type == "stt"
            else self.playing_cancel_button
        )

    def _start_voice_operation(
        self, action: str, model_type: str = ""
    ) -> None:
        if self._voice_worker is not None or self._voice_test_running():
            return
        if not self._is_local_voice_backend(model_type):
            return
        if action == "status":
            self._voice_status_checked[model_type] = True
        self._set_voice_busy(True)
        self._voice_operation_type = model_type
        self._voice_operation_action = action
        status, progress = self._voice_widgets(model_type)
        install_log = self._voice_install_log_widget(model_type)
        cancel_button = self._voice_cancel_button(model_type)
        for button in (self.recording_cancel_button, self.playing_cancel_button):
            button.hide()
            button.setEnabled(False)
        cancel_button.setVisible(action in ("install", "download"))
        cancel_button.setEnabled(cancel_button.isVisible())
        progress.show()
        progress.setRange(0, 0)
        install_log.setVisible(action in ("install", "download"))
        if install_log.isVisible():
            self._voice_install_log_lines[model_type].clear()
            install_log.setText("Waiting for installer output…")
            install_log.setToolTip("")
        status.setProperty("error", False)
        status.setText(
            {
                "status": "Checking versions and file integrity…",
                "install": "Installing the local voice runtime…",
                "download": "Preparing verified model download…",
            }[action]
        )
        self._refresh_voice_status_style(status)
        worker = _VoiceOperationThread(
            self._voice_provider(model_type),
            action,
            model_type,
            self.stt_model() if model_type == "stt" else self._active_tts_model(),
            self.pypi_mirror(),
            self._active_tts_source(),
        )
        self._voice_worker = worker
        worker.progress.connect(self._voice_operation_progress)
        worker.log_line.connect(self._voice_operation_log)
        worker.completed.connect(self._voice_operation_completed)
        worker.finished.connect(self._voice_operation_thread_finished)
        worker.start()

    def _cancel_voice_operation(self) -> None:
        worker = self._voice_worker
        if worker is None or self._voice_operation_action not in (
            "install",
            "download",
        ):
            return
        button = self._voice_cancel_button(self._voice_operation_type)
        button.setEnabled(False)
        status, progress = self._voice_widgets(self._voice_operation_type)
        status.setText("Cancelling the active install or download…")
        progress.setRange(0, 0)
        self._voice_operation_log("Cancellation requested…")
        worker.cancel_operation()

    def _voice_operation_progress(
        self,
        message: str,
        percent: object,
    ) -> None:
        status, progress = self._voice_widgets(self._voice_operation_type)
        status.setText(message)
        if isinstance(percent, int):
            progress.setRange(0, 100)
            progress.setValue(percent)
        else:
            progress.setRange(0, 0)

    def _voice_operation_log(self, line: str) -> None:
        compact = " ".join(str(line).split())
        if not compact:
            return
        lines = self._voice_install_log_lines[self._voice_operation_type]
        is_progress = compact.startswith(("━", "─"))
        previous_is_progress = bool(lines) and lines[-1].startswith(("━", "─"))
        if is_progress and previous_is_progress:
            lines[-1] = compact
        else:
            lines.append(compact)
            del lines[:-2]
        label = self._voice_install_log_widget(self._voice_operation_type)
        visible_lines = [
            item if len(item) <= 160 else f"{item[:157]}…"
            for item in lines
        ]
        label.setText("\n".join(visible_lines))
        label.setToolTip("\n".join(lines))
        label.show()

    def _voice_operation_completed(
        self,
        success: bool,
        message: str,
        result: object,
    ) -> None:
        cancelled = isinstance(result, dict) and bool(result.get("cancelled"))
        if isinstance(result, dict) and "dependency_ok" in result:
            success = bool(result["dependency_ok"] and result["model_ok"])
            message = (
                f"Runtime: {result['dependency_message']}\n"
                f"Model: {result['model_message']}"
            )
        if not success and not cancelled:
            logger.error(f"Voice setup check failed: {message}")
        status, progress = self._voice_widgets(self._voice_operation_type)
        status.setProperty("error", not success and not cancelled)
        status.setText(message)
        self._refresh_voice_status_style(status)
        progress.setRange(0, 100)
        progress.setValue(100 if success else 0)
        self._voice_cancel_button(self._voice_operation_type).setEnabled(False)
        self._set_voice_busy(False)

    def _voice_operation_thread_finished(self) -> None:
        worker = self._voice_worker
        self._voice_worker = None
        self._voice_operation_action = ""
        for button in (self.recording_cancel_button, self.playing_cancel_button):
            button.hide()
            button.setEnabled(False)
        if worker is not None:
            worker.deleteLater()
        current = self.settings_pages.currentIndex()
        if current == 2:
            self._check_voice_page_when_needed("stt")
        elif current == 3:
            self._check_voice_page_when_needed("tts")

    def _set_voice_busy(self, busy: bool) -> None:
        for button in (
            self.recording_check_button,
            self.recording_install_button,
            self.playing_check_button,
            self.playing_install_button,
            self.stt_download_button,
            self.tts_download_button,
            self.cosyvoice_download_button,
            self.cosyvoice_prompt_browse_button,
            self.voice_record_button,
            self.voice_play_button,
            self.close_button,
        ):
            button.setEnabled(not busy)
        self.recording_backend_combo.setEnabled(not busy)
        self.playing_backend_combo.setEnabled(not busy)
        self.recording_pypi_mirror_combo.setEnabled(not busy)
        self.playing_pypi_mirror_combo.setEnabled(not busy)
        self.stt_model_combo.setEnabled(not busy)
        self.tts_model_combo.setEnabled(not busy)
        self.cosyvoice_model_combo.setEnabled(not busy)
        self.qwen_model_source_combo.setEnabled(not busy)
        self.cosyvoice_model_source_combo.setEnabled(not busy)
        self.tts_speaker_combo.setEnabled(
            not busy and len(TTS_MODELS[self.tts_model()].speakers) > 1
        )
        self.tts_language_combo.setEnabled(not busy)
        self.cosyvoice_prompt_audio_edit.setEnabled(not busy)
        self.cosyvoice_prompt_text_edit.setEnabled(not busy)
        self.sovits_installation_edit.setEnabled(not busy)
        self.sovits_text_lang_combo.setEnabled(not busy)
        self.sovits_ref_audio_edit.setEnabled(not busy)
        self.sovits_prompt_text_edit.setEnabled(not busy)
        self.sovits_prompt_lang_combo.setEnabled(not busy)
        self.voice_test_text.setEnabled(not busy)

    def _voice_test_running(self) -> bool:
        return (
            self._voice_record_thread is not None
            or self._voice_play_thread is not None
        )

    def _toggle_voice_record_test(self) -> None:
        if self._voice_record_thread is not None:
            self.voice_record_button.setText("Transcribing…")
            self.voice_record_button.setEnabled(False)
            self._voice_record_thread.stop_recording()
            return
        if self._voice_worker is not None or self._voice_play_thread is not None:
            return
        self._set_voice_busy(True)
        self.voice_record_button.setEnabled(True)
        self.voice_record_button.setText("Starting microphone…")
        self.stt_test_result.clear()
        self.recording_status.setProperty("error", False)
        self.recording_status.setText("Starting the microphone…")
        self._refresh_voice_status_style(self.recording_status)
        worker = _LocalDictationThread(
            LocalDictationSession(self.stt_manager, self.stt_model())
        )
        self._voice_record_thread = worker
        worker.listening.connect(self._voice_record_listening)
        worker.partial_text.connect(self._voice_record_partial)
        worker.completed.connect(self._voice_record_completed)
        worker.finished.connect(self._voice_record_finished)
        worker.start()

    def _voice_record_listening(self) -> None:
        streaming = STT_MODELS[self.stt_model()].mode == "Streaming"
        self.voice_record_button.setText(
            "Stop recording" if streaming else "Stop & transcribe"
        )
        self.recording_status.setText(
            "Streaming transcription… text updates live."
            if streaming
            else "Recording… select Stop when finished."
        )

    def _voice_record_partial(self, text: str) -> None:
        self.stt_test_result.setText(text)

    def _voice_record_completed(
        self, success: bool, text: str, message: str
    ) -> None:
        if success and text.strip():
            self.stt_test_result.setText(text)
            message = f"{message} · {text}"
        if not success:
            logger.error(f"Voice record test failed: {message}")
        self.recording_status.setProperty("error", not success)
        self.recording_status.setText(message)
        self._refresh_voice_status_style(self.recording_status)
        self._set_voice_busy(False)

    def _voice_record_finished(self) -> None:
        worker = self._voice_record_thread
        self._voice_record_thread = None
        self.voice_record_button.setText("Record microphone")
        self._set_voice_busy(False)
        if worker is not None:
            worker.deleteLater()

    def _start_voice_play_test(self) -> None:
        if self._voice_worker is not None or self._voice_test_running():
            return
        text = self.voice_test_text.text().strip()
        if not text:
            if self.playing_backend() == "cosyvoice":
                text = COSYVOICE_TTS_MODELS[self.cosyvoice_model()].test_text
            elif self.playing_backend() == "sovits":
                text = "Hello from GPT-SoVITS."
            else:
                text = TTS_MODELS[self.tts_model()].test_text
            self.voice_test_text.setText(text)
        self._set_voice_busy(True)
        self.playing_progress.show()
        self.playing_progress.setRange(0, 0)
        self.playing_status.setProperty("error", False)
        self.playing_status.setText("Generating speech…")
        self._refresh_voice_status_style(self.playing_status)
        worker = _LocalSpeechThread(
            self._voice_provider("tts"),
            self._active_tts_model(),
            text,
            (
                self.cosyvoice_prompt_audio()
                if self.playing_backend() == "cosyvoice"
                else (
                    self.sovits_ref_audio_path()
                    if self.playing_backend() == "sovits"
                    else self.tts_speaker()
                )
            ),
            (
                self.cosyvoice_prompt_text()
                if self.playing_backend() == "cosyvoice"
                else (
                    self.sovits_text_lang()
                    if self.playing_backend() == "sovits"
                    else self.tts_language()
                )
            ),
        )
        self._voice_play_thread = worker
        worker.started.connect(self.playing_status.setText)
        worker.completed.connect(self._voice_play_completed)
        worker.finished.connect(self._voice_play_finished)
        worker.start()

    def _voice_play_completed(self, success: bool, message: str) -> None:
        if not success:
            logger.error(f"Voice playback test failed: {message}")
        self.playing_status.setProperty("error", not success)
        self.playing_status.setText(message)
        self._refresh_voice_status_style(self.playing_status)
        self.playing_progress.setRange(0, 100)
        self.playing_progress.setValue(100 if success else 0)
        self._set_voice_busy(False)

    def _voice_play_finished(self) -> None:
        worker = self._voice_play_thread
        self._voice_play_thread = None
        self._set_voice_busy(False)
        if worker is not None:
            worker.deleteLater()

    @staticmethod
    def _refresh_voice_status_style(status: QLabel) -> None:
        status.style().unpolish(status)
        status.style().polish(status)

    @staticmethod
    def _navigation_button(text: str, icon_path: Path) -> QPushButton:
        button = QPushButton(text)
        button.setObjectName("settingsNavigationButton")
        button.setCheckable(True)
        button.setIcon(QIcon(str(icon_path)))
        button.setIconSize(QSize(19, 19))
        return button

    def eventFilter(self, watched: object, event: QEvent) -> bool:
        if watched in self._title_drag_widgets:
            if (
                event.type() == QEvent.Type.MouseButtonPress
                and isinstance(event, QMouseEvent)
                and event.button() == Qt.MouseButton.LeftButton
            ):
                self._title_drag_offset = (
                    event.globalPosition().toPoint()
                    - self.frameGeometry().topLeft()
                )
                return True
            if (
                event.type() == QEvent.Type.MouseMove
                and isinstance(event, QMouseEvent)
                and self._title_drag_offset is not None
                and event.buttons() & Qt.MouseButton.LeftButton
            ):
                self.move(
                    event.globalPosition().toPoint()
                    - self._title_drag_offset
                )
                return True
            if event.type() == QEvent.Type.MouseButtonRelease:
                self._title_drag_offset = None
                return True
        return super().eventFilter(watched, event)

    @staticmethod
    def _sequence_edit(sequence: QKeySequence) -> QKeySequenceEdit:
        edit = QKeySequenceEdit(sequence)
        edit.setMaximumSequenceLength(1)
        return edit

    def sequences(self) -> dict[str, QKeySequence]:
        return {
            "hold": self.hold_microphone_edit.keySequence(),
            "hold_without_screenshot": (
                self.hold_without_screenshot_edit.keySequence()
            ),
            "send": self.send_edit.keySequence(),
            "send_without_screenshot": (
                self.send_without_screenshot_edit.keySequence()
            ),
        }

    def language(self) -> str:
        return "zh" if self.language_combo.currentIndex() == 1 else "en"

    def recording_backend(self) -> str:
        return str(self.recording_backend_combo.currentData() or "web")

    def playing_backend(self) -> str:
        return str(self.playing_backend_combo.currentData() or "web")

    def pypi_mirror(self) -> str:
        return str(
            self.recording_pypi_mirror_combo.currentData() or "default"
        )

    def qwen_model_source(self) -> str:
        return str(
            self.qwen_model_source_combo.currentData() or "huggingface"
        )

    def cosyvoice_model_source(self) -> str:
        return str(
            self.cosyvoice_model_source_combo.currentData() or "huggingface"
        )

    def stt_model(self) -> str:
        return str(
            self.stt_model_combo.currentData()
            or "zh_zipformer_ctc_int8_2025_07_03"
        )

    def tts_model(self) -> str:
        return str(
            self.tts_model_combo.currentData() or "qwen3_tts_0_6b_custom_voice"
        )

    def tts_speaker(self) -> str:
        return str(self.tts_speaker_combo.currentData() or "Vivian")

    def tts_language(self) -> str:
        return str(self.tts_language_combo.currentData() or "Auto")

    def cosyvoice_model(self) -> str:
        return str(
            self.cosyvoice_model_combo.currentData()
            or "fun_cosyvoice3_0_5b_2512"
        )

    def cosyvoice_prompt_audio(self) -> str:
        return self.cosyvoice_prompt_audio_edit.text().strip()

    def cosyvoice_prompt_text(self) -> str:
        return self.cosyvoice_prompt_text_edit.text().strip()

    def sovits_installation(self) -> str:
        return self.sovits_installation_edit.text().strip()

    def sovits_text_lang(self) -> str:
        return str(self.sovits_text_lang_combo.currentData() or "auto")

    def sovits_ref_audio_path(self) -> str:
        return self.sovits_ref_audio_edit.text().strip()

    def sovits_prompt_text(self) -> str:
        return self.sovits_prompt_text_edit.text().strip()

    def sovits_prompt_lang(self) -> str:
        return str(self.sovits_prompt_lang_combo.currentData() or "auto")

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
        if self._voice_worker is not None or self._voice_test_running():
            QMessageBox.information(
                self,
                "Voice setup is running",
                "Wait for the current voice setup operation to finish.",
            )
            return
        try:
            self.bindings()
        except ValueError as error:
            QMessageBox.warning(self, "Invalid hotkey", str(error))
            return
        super().accept()

    def reject(self) -> None:
        if self._voice_worker is not None:
            QMessageBox.information(
                self,
                "Voice setup is running",
                "Wait for the current voice setup operation to finish.",
            )
            return
        super().reject()


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
            "No Screenshot",
            self,
        )
        self.send_without_screenshot_button.setObjectName(
            "sendWithoutScreenshotButton"
        )
        self.send_without_screenshot_button.setIcon(
            QIcon(str(SEND_ICON_PATH))
        )
        self.send_without_screenshot_button.setIconSize(QSize(16, 16))
        self.send_without_screenshot_button.setFixedSize(132, 32)
        self.send_without_screenshot_button.setAccessibleName(
            "Send without screenshot"
        )
        self.send_without_screenshot_button.setToolTip(
            "Send the text without the selected screenshot"
        )

        self.auto_send_button = QPushButton("Auto Send", self)
        self.auto_send_button.setObjectName("autoSendButton")
        self.auto_send_button.setCheckable(True)
        self.auto_send_button.setFixedSize(104, 32)
        self.auto_send_button.setAccessibleName(
            "Automatically send dictated text"
        )
        self.auto_send_button.setToolTip(
            "Automatically send dictated text using the selected "
            "screenshot option"
        )
        self.auto_send_button.toggled.connect(
            self._sync_auto_send_icon
        )
        self._sync_auto_send_icon(False)

        self.textChanged.connect(self._sync_action_visibility)
        self._sync_action_visibility()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._position_action_buttons()

    def _sync_auto_send_icon(self, enabled: bool) -> None:
        icon = QIcon(str(CHECK_ICON_PATH)) if enabled else QIcon()
        self.auto_send_button.setIcon(icon)
        self.auto_send_button.setIconSize(QSize(16, 16))

    def _position_action_buttons(self) -> None:
        margin = 8
        spacing = 6
        y = self.height() - self.send_button.height() - margin
        right = self.width() - margin
        for button in (
            self.send_button,
            self.send_without_screenshot_button,
            self.auto_send_button,
            self.clear_button,
        ):
            if button.isHidden():
                continue
            right -= button.width()
            button.move(right, y)
            button.raise_()
            right -= spacing

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
            if self.toPlainText().strip() or self._screenshot_selected:
                self.send_button.click()
            event.accept()
            return
        super().keyPressEvent(event)

    @property
    def is_showing_response(self) -> bool:
        return self._response_mode

    @property
    def is_response_complete(self) -> bool:
        return self._response_complete

    @property
    def auto_send_enabled(self) -> bool:
        return self.auto_send_button.isChecked()

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
        if selected:
            self.send_button.setText("With Screenshot")
            self.send_button.setFixedSize(144, 32)
            self.send_button.setAccessibleName("Send with screenshot")
            self.send_button.setToolTip(
                "Send the message with the selected screenshot"
            )
        else:
            self.send_button.setText("")
            self.send_button.setFixedSize(32, 32)
            self.send_button.setAccessibleName("Send")
            self.send_button.setToolTip("Send the message")
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
        can_send = (
            not self._response_mode
            and (has_text or self._screenshot_selected)
        )
        self.clear_button.setVisible(has_text)
        self.send_button.setVisible(can_send)
        self.send_without_screenshot_button.setVisible(
            has_text and self._screenshot_selected
        )
        self.auto_send_button.setVisible(not self._response_mode)
        visible_buttons = [
            button
            for button in (
                self.clear_button,
                self.auto_send_button,
                self.send_without_screenshot_button,
                self.send_button,
            )
            if not button.isHidden()
        ]
        right_margin = sum(button.width() for button in visible_buttons)
        if visible_buttons:
            right_margin += 8 + 6 * (len(visible_buttons) - 1)
        self.setViewportMargins(0, 0, right_margin, 0)
        self._position_action_buttons()


class OverlayWindow(QMainWindow):
    exit_requested = Signal()
    dictation_requested = Signal()
    dictation_finish_requested = Signal()
    clear_requested = Signal()
    send_requested = Signal(str, object)
    configure_requested = Signal()
    open_remote_debugging_requested = Signal()
    chatgpt_tab_selected = Signal(str)
    chatgpt_connection_changed = Signal(bool)
    chatgpt_preference_changed = Signal(str)
    capture_source_selected = Signal(str)
    geometry_changed = Signal(object)

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
        self._auto_hide_enabled = False
        self._preferred_capture_source_key = ""
        self._preferred_chatgpt_url = ""
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
        self._auto_hide_timer = QTimer(self)
        self._auto_hide_timer.setSingleShot(True)
        self._auto_hide_timer.timeout.connect(self._hide_for_auto_hide)

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
            "Open settings",
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

        self.auto_hide_button = QPushButton()
        self.auto_hide_button.setObjectName("autoHideButton")
        self.auto_hide_button.setCheckable(True)
        self._configure_icon_button(
            self.auto_hide_button,
            AUTO_HIDE_ICON_PATH,
            "Enable auto-hide",
        )
        self.auto_hide_button.toggled.connect(
            self._set_auto_hide_enabled
        )
        title_layout.addWidget(self.auto_hide_button)

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
        self._reading_spoken_characters = 0
        self._subtitle_line_index = -1
        self._subtitle_mode_active = False
        self._subtitle_dismissed = False
        self._subtitle_expanded = False
        self._subtitle_reading_active = False
        self._subtitle_reading_started = False
        self._subtitle_status_text = ""
        self._sent_message_text = ""
        self._subtitle_collapsed_geometry: QRect | None = None
        self._subtitle_hover_origin: QPoint | None = None
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
        self.auto_send_button = self.transcript_area.auto_send_button

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
            QPushButton#autoHideButton,
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
            QPushButton#autoSendButton,
            QPushButton#sendWithoutScreenshotButton {
                min-width: 32px;
                min-height: 32px;
                max-height: 32px;
                padding: 0;
                border-radius: 8px;
            }
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
            QPushButton#autoHideButton:checked {
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
                min-width: 132px;
                max-width: 132px;
                background-color: rgba(38, 112, 145, 190);
            }
            QPushButton#sendWithoutScreenshotButton:hover {
                background-color: rgba(48, 145, 185, 220);
            }
            QPushButton#autoSendButton {
                min-width: 104px;
                max-width: 104px;
            }
            QPushButton#autoSendButton:checked {
                background-color: rgba(35, 155, 116, 190);
                border-color: rgba(130, 255, 195, 190);
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
            if selected_index < 0 and self._preferred_chatgpt_url:
                for index in range(self.chatgpt_tab_combo.count()):
                    url = self.chatgpt_tab_combo.itemData(
                        index,
                        Qt.ItemDataRole.ToolTipRole,
                    )
                    if url == self._preferred_chatgpt_url:
                        selected_index = index
                        break
            self.chatgpt_tab_combo.setCurrentIndex(
                selected_index if selected_index >= 0 else 0
            )
        self.chatgpt_tab_combo.blockSignals(False)
        self.chatgpt_connection_changed.emit(bool(tabs))
        if tabs:
            self._activate_chatgpt_tab(
                self.chatgpt_tab_combo.currentIndex(),
                save_preference=not bool(self._preferred_chatgpt_url),
            )

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
        selected_key = (
            selected.key
            if isinstance(selected, CaptureSource)
            else self._preferred_capture_source_key
        )
        self.capture_source_combo.blockSignals(True)
        self.capture_source_combo.clear()
        self.capture_source_combo.addItem("No screenshot", None)
        for source in sources:
            self.capture_source_combo.addItem(source.label, source)
        if selected_key:
            for index in range(1, self.capture_source_combo.count()):
                source = self.capture_source_combo.itemData(index)
                if isinstance(source, CaptureSource) and source.key == selected_key:
                    self.capture_source_combo.setCurrentIndex(index)
                    break
        self.capture_source_combo.blockSignals(False)
        self._apply_capture_source(
            self.capture_source_combo.currentIndex(),
            save_preference=False,
        )

    def set_preferred_capture_source(self, source_key: str) -> None:
        self._preferred_capture_source_key = source_key

    def set_preferred_chatgpt_window(self, url: str) -> None:
        self._preferred_chatgpt_url = url

    def set_send_result(
        self,
        success: bool,
        sent_text: str,
        message: str,
    ) -> None:
        self.send_button.setEnabled(True)
        self.send_without_screenshot_button.setEnabled(True)
        if not success:
            self.show_for_auto_hide()
            self.dismiss_subtitle_mode()
            self.set_transcript(sent_text)
            self.microphone_button.setVisible(True)
            self.set_status(message, error=True)
            self.schedule_auto_hide(5_000)
            return

        del message
        if not self._subtitle_mode_active:
            self.begin_response_display(sent_text)
        self._set_subtitle_status("Waiting for ChatGPT…")
        self.set_status("Waiting for ChatGPT…")

    def begin_response_display(
        self,
        sent_text: str = "",
        message: str = "Sending to ChatGPT…",
    ) -> None:
        self._collapse_subtitle()
        self._subtitle_mode_active = True
        self._subtitle_dismissed = False
        self._subtitle_reading_active = False
        self._subtitle_reading_started = False
        self._subtitle_hover_origin = None
        self._sent_message_text = " ".join(sent_text.split())
        self._reading_full_text = ""
        self._reading_fraction = 0.0
        self._reading_spoken_characters = 0
        self._subtitle_line_index = -1
        self.transcript_area.begin_response()
        self.transcript_area.hide()
        self.dictation_panel.hide()
        self.subtitle_panel.show()
        self.microphone_button.setVisible(True)
        self._set_subtitle_status(message)
        self.set_status(message)

    def set_response_update(self, status: str, text: str) -> None:
        self.show_for_auto_hide()
        if self._subtitle_dismissed:
            self.set_status(status)
            return
        self.transcript_area.update_response(text)
        if not self._subtitle_mode_active:
            self.begin_response_display(message=status)
        self._reading_full_text = text
        if self._subtitle_reading_active:
            self._reading_fraction = min(
                self._reading_spoken_characters / max(1, len(text)), 0.99
            )
            self._render_reading_subtitle(resized=True)
            self._update_expanded_subtitle()
            self.set_status("Reading aloud…")
            return
        self.set_status(status)
        self._reading_fraction = 0.0
        self._set_subtitle_status(status)
        if text:
            self._update_expanded_subtitle()

    def set_response_finished(self, success: bool, message: str) -> None:
        if self._subtitle_dismissed:
            return
        self.show_for_auto_hide()
        self.transcript_area.finish_response()
        self.microphone_button.setVisible(True)
        if self._subtitle_reading_active:
            self.set_status("Reading aloud…")
            return
        self.set_status(message, error=not success)
        if self._subtitle_mode_active:
            self._set_subtitle_status(message)
        self.schedule_auto_hide(5_000)

    def begin_reading(self, message: str) -> None:
        if self._subtitle_dismissed:
            return
        self.show_for_auto_hide()
        self.transcript_area.begin_reading()
        self._subtitle_reading_active = True
        self._subtitle_reading_started = True
        self._subtitle_hover_origin = None
        self._reading_fraction = 0.0
        self._reading_spoken_characters = 0
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
            spoken_characters = update.get("spoken_characters")
            if isinstance(spoken_characters, int):
                self._reading_spoken_characters = max(0, spoken_characters)
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
            self._reading_spoken_characters = 0
        if self._subtitle_dismissed:
            return
        self._render_reading_subtitle()
        self._update_expanded_subtitle()
        self.set_status("Reading aloud…")

    def _subtitle_lines(self) -> list[str]:
        text = self._reading_full_text.strip()
        if not text:
            return []

        available_width = max(self.subtitle_panel.width() - 32, 1)
        metrics = self.subtitle_line_one.fontMetrics()
        lines: list[str] = []
        for raw_paragraph in text.splitlines() or [text]:
            paragraph = " ".join(raw_paragraph.split())
            if not paragraph:
                continue
            current = ""
            for word in paragraph.split(" "):
                prefix = f"{current} " if current else ""
                candidate = prefix
                whole_word_fits = True
                for character in word:
                    if (
                        metrics.horizontalAdvance(candidate + character)
                        > available_width
                    ):
                        whole_word_fits = False
                        break
                    candidate += character
                if whole_word_fits:
                    current = candidate
                    continue
                if current:
                    lines.append(current)
                    current = ""
                for character in word:
                    if (
                        current
                        and metrics.horizontalAdvance(current + character)
                        > available_width
                    ):
                        lines.append(current)
                        current = character
                    else:
                        current += character
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
    ) -> None:
        lines = self._subtitle_lines()
        if not lines:
            self.subtitle_line_one.clear()
            self.subtitle_line_two.clear()
            self._subtitle_line_index = -1
            return

        target_index = self._subtitle_index_at_progress(lines)
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
        self._subtitle_reading_active = False
        self.microphone_button.setVisible(True)
        self.set_status(message, error=not success)
        self.schedule_auto_hide(5_000)

    def _set_subtitle_status(self, message: str) -> None:
        self._subtitle_status_text = message
        if self._subtitle_mode_active and not self._subtitle_reading_started:
            self.subtitle_line_one.setText(self._sent_message_text)
            self.subtitle_line_two.setText(message)
        else:
            self.subtitle_line_one.setText(message)
            self.subtitle_line_two.clear()
        full_text = self._reading_full_text or message
        if (
            not self._subtitle_expanded
            and self.subtitle_full_text.toPlainText() != full_text
        ):
            self.subtitle_full_text.setPlainText(full_text)

    def _update_expanded_subtitle(self) -> None:
        if not self._subtitle_expanded:
            return
        text = self._reading_full_text or self._subtitle_status_text
        if self.subtitle_full_text.toPlainText() == text:
            return
        scrollbar = self.subtitle_full_text.verticalScrollBar()
        previous_value = scrollbar.value()
        was_at_bottom = previous_value >= scrollbar.maximum() - 1
        self.subtitle_full_text.setPlainText(text)
        if was_at_bottom:
            scrollbar.setValue(scrollbar.maximum())
        else:
            scrollbar.setValue(min(previous_value, scrollbar.maximum()))
        self._fit_expanded_subtitle_height()

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
        self._fit_expanded_subtitle_height()

    def _fit_expanded_subtitle_height(self) -> None:
        if (
            not self._subtitle_expanded
            or self._subtitle_collapsed_geometry is None
        ):
            return
        available = self.screen().availableGeometry()
        collapsed = self._subtitle_collapsed_geometry
        content_width = max(self.subtitle_panel.width() - 40, 1)
        text = self._reading_full_text or self._subtitle_status_text
        metrics = self.subtitle_full_text.fontMetrics()
        text_height = metrics.boundingRect(
            QRect(0, 0, content_width, 16_777_215),
            Qt.TextFlag.TextWordWrap,
            text,
        ).height()
        panel_height = max(text_height, metrics.lineSpacing()) + 32
        fixed_chrome_height = max(
            collapsed.height() - self.subtitle_panel.height(),
            0,
        )
        target_height = min(
            max(collapsed.height(), fixed_chrome_height + panel_height),
            max(available.height() - 40, collapsed.height()),
        )
        if self._position_locked:
            self.setMinimumSize(760, 180)
            self.setMaximumSize(16_777_215, 16_777_215)
        geometry = QRect(collapsed)
        geometry.setTop(
            max(available.top(), collapsed.bottom() - target_height + 1)
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
        if self._subtitle_collapsed_geometry is not None:
            self.setGeometry(self._subtitle_collapsed_geometry)
            if self._position_locked:
                self.setFixedSize(self._subtitle_collapsed_geometry.size())
        self._subtitle_collapsed_geometry = None
        if self._subtitle_reading_started and self._reading_full_text:
            self._render_reading_subtitle(resized=True)
        else:
            self._set_subtitle_status(self._subtitle_status_text)

    def _collapse_subtitle_if_outside(self) -> None:
        if not self._subtitle_expanded:
            return
        position = self.subtitle_panel.mapFromGlobal(QCursor.pos())
        if not self.subtitle_panel.rect().contains(position):
            self._collapse_subtitle()
            self.schedule_auto_hide(5_000)

    def dismiss_subtitle_mode(self) -> bool:
        if not self._subtitle_mode_active:
            return False
        self._collapse_subtitle()
        self._subtitle_mode_active = False
        self._subtitle_dismissed = True
        self._subtitle_reading_active = False
        self._subtitle_reading_started = False
        self.subtitle_panel.hide()
        self.dictation_panel.hide()
        self.transcript_area.begin_composing()
        self.transcript_area.show()
        self.microphone_button.setVisible(True)
        self.set_status("Hold the microphone or enter a message")
        self.schedule_auto_hide()
        return True

    def begin_dictation_waiting(self) -> None:
        self.dismiss_subtitle_mode()
        self.show_for_auto_hide()
        self.subtitle_panel.hide()
        self.transcript_area.hide()
        self.dictation_state_label.setText(
            "Waiting for the browser to start listening…"
        )
        self.dictation_panel.show()

    def set_dictation_listening(self) -> None:
        self.dictation_state_label.setText("Listening…")

    def set_dictation_partial(
        self,
        text: str,
        *,
        finishing: bool = False,
    ) -> None:
        if text.strip():
            state = "Finishing dictation…" if finishing else "Listening…"
            self.dictation_state_label.setText(f"{state}\n\n{text}")

    def set_dictation_finishing(self) -> None:
        self.dictation_state_label.setText("Finishing dictation…")

    def set_dictation_cancelling(self) -> None:
        self.dictation_state_label.setText("Cancelling short dictation…")

    def end_dictation_display(self) -> None:
        self.dictation_panel.hide()
        self.transcript_area.show()

    def _chatgpt_tab_changed(self, index: int) -> None:
        self._activate_chatgpt_tab(index, save_preference=True)

    def _activate_chatgpt_tab(
        self,
        index: int,
        *,
        save_preference: bool,
    ) -> None:
        tab_id = self.chatgpt_tab_combo.itemData(index)
        if tab_id:
            self.chatgpt_tab_selected.emit(str(tab_id))
            if save_preference:
                url = self.chatgpt_tab_combo.itemData(
                    index,
                    Qt.ItemDataRole.ToolTipRole,
                )
                self._preferred_chatgpt_url = str(url or "")
                self.chatgpt_preference_changed.emit(
                    self._preferred_chatgpt_url
                )

    def _capture_source_changed(self, index: int) -> None:
        self._apply_capture_source(index, save_preference=True)

    def _apply_capture_source(
        self,
        index: int,
        *,
        save_preference: bool,
    ) -> None:
        source = self.capture_source_combo.itemData(index)
        self.transcript_area.set_screenshot_selected(
            source is not None
        )
        if save_preference:
            source_key = (
                source.key if isinstance(source, CaptureSource) else ""
            )
            self._preferred_capture_source_key = source_key
            self.capture_source_selected.emit(source_key)

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
        capture_source = (
            self.capture_source_combo.currentData()
            if include_screenshot
            else None
        )
        if not text and capture_source is None:
            self.set_status(
                "Enter text or select a screenshot before sending",
                error=True,
            )
            return
        self.send_requested.emit(text, capture_source)
        if restore_focus:
            QTimer.singleShot(0, self._restore_previous_focus)
        self.schedule_auto_hide()

    def _request_send_without_screenshot(self) -> None:
        self._request_send(include_screenshot=False)

    def request_send_from_hotkey(self, include_screenshot: bool) -> None:
        self._request_send(
            include_screenshot=include_screenshot,
            restore_focus=False,
        )

    def request_auto_send(self) -> None:
        self._request_send(include_screenshot=True, restore_focus=False)

    def remember_foreground_app(self) -> None:
        self._focus_restorer.remember_foreground()

    def _restore_previous_focus(self) -> None:
        if self._focus_restorer.restore_previous():
            logger.debug("Restored focus to the previous application")
        else:
            logger.debug("No external application was available to restore")

    @property
    def auto_hide_enabled(self) -> bool:
        return self._auto_hide_enabled

    @property
    def auto_send_enabled(self) -> bool:
        return self.transcript_area.auto_send_enabled

    def disable_auto_hide(self) -> None:
        if self.auto_hide_button.isChecked():
            self.auto_hide_button.setChecked(False)
        else:
            self._set_auto_hide_enabled(False)

    def _set_auto_hide_enabled(self, enabled: bool) -> None:
        self._auto_hide_enabled = enabled
        label = "Disable auto-hide" if enabled else "Enable auto-hide"
        self.auto_hide_button.setAccessibleName(label)
        self.auto_hide_button.setToolTip(
            f"{label}; the overlay appears for dictation and ChatGPT replies"
        )
        if enabled:
            self.schedule_auto_hide()
        else:
            self._auto_hide_timer.stop()

    def _can_auto_hide(self) -> bool:
        if not self._auto_hide_enabled:
            return False
        if not self.dictation_panel.isHidden():
            return False
        if (
            not self.transcript_area.is_showing_response
            and bool(self.transcript_area.toPlainText().strip())
        ):
            return False
        if self._subtitle_mode_active and (
            not self.transcript_area.is_response_complete
            or self._subtitle_reading_active
            or self._subtitle_expanded
        ):
            return False
        return True

    def show_for_auto_hide(self) -> None:
        if not self._auto_hide_enabled:
            return
        self._auto_hide_timer.stop()
        if self.isHidden():
            self.showNormal()
        self.raise_()

    def schedule_auto_hide(self, delay_ms: int = 0) -> None:
        if self._can_auto_hide():
            self._auto_hide_timer.start(max(delay_ms, 0))
        else:
            self._auto_hide_timer.stop()

    def _hide_for_auto_hide(self) -> None:
        if self._can_auto_hide():
            self.hide()

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
            if self._subtitle_reading_started and self._reading_full_text:
                self._render_reading_subtitle(resized=True)
            elif self._subtitle_mode_active:
                self._set_subtitle_status(self._subtitle_status_text)
        if not getattr(self, "_subtitle_mode_active", False):
            self.geometry_changed.emit(QRect(self.geometry()))

    def moveEvent(self, event) -> None:  # noqa: N802
        super().moveEvent(event)
        if not getattr(self, "_subtitle_mode_active", False):
            self.geometry_changed.emit(QRect(self.geometry()))

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
                self._subtitle_hover_origin = (
                    event.globalPosition().toPoint()
                    if hasattr(event, "globalPosition")
                    else QCursor.pos()
                )
            elif event_type == QEvent.Type.MouseMove:
                position = (
                    event.globalPosition().toPoint()
                    if hasattr(event, "globalPosition")
                    else QCursor.pos()
                )
                if self._subtitle_hover_origin is None:
                    self._subtitle_hover_origin = position
                elif (
                    position - self._subtitle_hover_origin
                ).manhattanLength() >= 4:
                    QTimer.singleShot(0, self._expand_subtitle)
            elif event_type == QEvent.Type.Leave:
                QTimer.singleShot(
                    0,
                    self._collapse_subtitle_if_outside,
                )
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
        self.config = Config()
        self.stt_manager = SherpaSttProvider()
        self.tts_manager = QwenTtsProvider()
        self.cosyvoice_manager = CosyVoiceTtsProvider()
        self.sovits_manager = SovitsTtsProvider()
        self._local_dictation_thread: _LocalDictationThread | None = None
        self._local_speech_thread: (
            _LocalSpeechThread | _QueuedLocalSpeechThread | None
        ) = None
        self._local_voice_longest_text = ""
        self._local_voice_queued_sentences: list[str] = []
        self._local_voice_interrupted = False
        self._stt_preload_thread: threading.Thread | None = None
        self._qwen_preload_thread: threading.Thread | None = None
        self._migrate_legacy_hotkeys()
        self.window = OverlayWindow()
        self.window.set_preferred_capture_source(
            str(self.config["capture_source"])
        )
        self.window.set_preferred_chatgpt_window(
            str(self.config["chatgpt_window"])
        )
        self.window.auto_send_button.setChecked(
            bool(self.config["auto_send"])
        )
        self.window.auto_hide_button.setChecked(
            bool(self.config["auto_hide"])
        )
        self.browser_monitor = BrowserMonitor()
        self.browser_monitor.set_use_browser_voice(
            self.config["playing_backend"] == "web"
        )
        self.selected_chatgpt_tab_id: str | None = None
        self.dictation_tab_id: str | None = None
        self._dictation_state = "idle"
        self._dictation_input_held = False
        self._dictation_force_auto_send = False
        self._dictation_include_screenshot = True
        self._dictation_pressed_since: float | None = None
        self._dictation_listening_since: float | None = None
        self._dictation_press_generation = 0
        self._dictation_attachment_tab_id: str | None = None
        self._pending_dictation_capture: _PendingDictationCapture | None = None
        self._hotkey_sequences = self._load_hotkey_sequences()
        bindings = self._bindings_for_sequences(self._hotkey_sequences)
        self.hotkey_monitor = GlobalHotkeyMonitor(
            bindings["hold"],
            bindings["send"],
            bindings["send_without_screenshot"],
            self.window,
            hold_without_screenshot=bindings["hold_without_screenshot"],
        )

        self.application.setWindowIcon(self.icon)
        self.window.setWindowIcon(self.icon)
        self.window.exit_requested.connect(self._exit_application)
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
        self.window.chatgpt_preference_changed.connect(
            self._save_chatgpt_preference
        )
        self.window.capture_source_selected.connect(
            self._save_capture_source_preference
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
            self._on_browser_reading_finished
        )
        self.browser_monitor.local_voice_updated.connect(
            self._update_local_voice
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
        self.hotkey_monitor.hold_pressed.connect(
            lambda: self.start_dictation(
                force_auto_send=True,
                include_screenshot=True,
                initial_hold_seconds=self.hotkey_monitor.hold_delay_seconds,
            )
        )
        self.hotkey_monitor.hold_released.connect(self.finish_dictation)
        self.hotkey_monitor.hold_without_screenshot_pressed.connect(
            lambda: self.start_dictation(
                force_auto_send=True,
                include_screenshot=False,
                initial_hold_seconds=self.hotkey_monitor.hold_delay_seconds,
            )
        )
        self.hotkey_monitor.hold_without_screenshot_released.connect(
            self.finish_dictation
        )
        self.hotkey_monitor.send_pressed.connect(
            lambda: self.window.request_send_from_hotkey(True)
        )
        self.hotkey_monitor.send_without_screenshot_pressed.connect(
            lambda: self.window.request_send_from_hotkey(False)
        )

        self._position_overlay()
        self._restore_window_geometry()
        self.window.lock_button.setChecked(
            bool(self.config["window_locked"])
        )
        self.window.auto_send_button.toggled.connect(
            lambda enabled: self.config.__setitem__("auto_send", enabled)
        )
        self.window.auto_hide_button.toggled.connect(
            lambda enabled: self.config.__setitem__("auto_hide", enabled)
        )
        self.window.lock_button.toggled.connect(
            lambda enabled: self.config.__setitem__("window_locked", enabled)
        )
        self._geometry_save_timer = QTimer(self.window)
        self._geometry_save_timer.setSingleShot(True)
        self._geometry_save_timer.timeout.connect(self._save_window_geometry)
        self.window.geometry_changed.connect(
            self._schedule_window_geometry_save
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
        self.show_window()
        self._start_stt_preload()
        self._start_qwen_preload()

    @staticmethod
    def _bindings_for_sequences(
        sequences: dict[str, QKeySequence],
    ) -> dict[str, HotkeyBinding]:
        bindings = {
            name: HotkeyBinding.from_sequence(sequence)
            for name, sequence in sequences.items()
        }
        if len({binding.text.casefold() for binding in bindings.values()}) != 4:
            raise ValueError("Each action must use a different hotkey")
        return bindings

    def _migrate_legacy_hotkeys(self) -> None:
        if self.config.file_existed:
            return
        legacy_settings = QSettings("Live GPT", "Live GPT")
        migrated: dict[str, str] = {}
        for name, legacy_key in LEGACY_HOTKEY_SETTING_KEYS.items():
            value = legacy_settings.value(legacy_key)
            if value is not None and str(value).strip():
                migrated[HOTKEY_CONFIG_KEYS[name]] = str(value)
        if migrated:
            self.config.update(migrated)
            logger.info("Migrated legacy hotkeys to JSON configuration")

    def _load_hotkey_sequences(self) -> dict[str, QKeySequence]:
        defaults = {
            "hold": DEFAULT_HOLD_MIC_HOTKEY,
            "hold_without_screenshot": (
                DEFAULT_HOLD_WITHOUT_SCREENSHOT_HOTKEY
            ),
            "send": DEFAULT_SEND_HOTKEY,
            "send_without_screenshot": (
                DEFAULT_SEND_WITHOUT_SCREENSHOT_HOTKEY
            ),
        }
        sequences = {
            name: QKeySequence(
                str(self.config[HOTKEY_CONFIG_KEYS[name]])
            )
            for name in defaults
        }
        try:
            self._bindings_for_sequences(sequences)
        except ValueError as error:
            logger.warning(f"Invalid saved hotkey configuration: {error}")
            sequences = {
                name: QKeySequence(default)
                for name, default in defaults.items()
            }
            self.config.update(
                {
                    HOTKEY_CONFIG_KEYS[name]: sequence.toString(
                        QKeySequence.SequenceFormat.PortableText
                    )
                    for name, sequence in sequences.items()
                }
            )
        return sequences

    def _open_configuration(self) -> None:
        self.hotkey_monitor.stop()
        try:
            dialog = HotkeyConfigDialog(
                self._hotkey_sequences["hold"],
                self._hotkey_sequences["send"],
                self._hotkey_sequences["send_without_screenshot"],
                self.window,
                hold_without_screenshot=self._hotkey_sequences[
                    "hold_without_screenshot"
                ],
                language=str(self.config["language"]),
                recording_backend=str(self.config["recording_backend"]),
                playing_backend=str(self.config["playing_backend"]),
                stt_model=str(self.config["stt_model"]),
                tts_model=str(self.config["tts_model"]),
                tts_speaker=str(self.config["tts_speaker"]),
                tts_language=str(self.config["tts_language"]),
                pypi_mirror=str(self.config["pypi_mirror"]),
                qwen_model_source=str(self.config["qwen_model_source"]),
                cosyvoice_model_source=str(
                    self.config["cosyvoice_model_source"]
                ),
                cosyvoice_model=str(self.config["cosyvoice_model"]),
                cosyvoice_prompt_audio=str(
                    self.config["cosyvoice_prompt_audio"]
                ),
                cosyvoice_prompt_text=str(
                    self.config["cosyvoice_prompt_text"]
                ),
                sovits_installation=str(self.config["sovits_installation"]),
                sovits_text_lang=str(self.config["sovits_text_lang"]),
                sovits_ref_audio_path=str(self.config["sovits_ref_audio_path"]),
                sovits_prompt_text=str(self.config["sovits_prompt_text"]),
                sovits_prompt_lang=str(self.config["sovits_prompt_lang"]),
                config=self.config,
                stt_manager=self.stt_manager,
                tts_manager=self.tts_manager,
                cosyvoice_manager=self.cosyvoice_manager,
                sovits_manager=self.sovits_manager,
            )
            dialog.exec()

            sequences = self._load_hotkey_sequences()
            bindings = self._bindings_for_sequences(sequences)
            self._hotkey_sequences = sequences
            self.browser_monitor.set_use_browser_voice(
                self.config["playing_backend"] == "web"
            )
            self._start_stt_preload()
            self._start_qwen_preload()
            self.hotkey_monitor.update_bindings(
                bindings["hold"],
                bindings["send"],
                bindings["send_without_screenshot"],
                hold_without_screenshot=bindings["hold_without_screenshot"],
            )
            self.window.set_status("Settings updated")
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
            self.window.disable_auto_hide()
            self.show_window()

    def show_window(self) -> None:
        logger.info("Showing the overlay window")
        self.window.remember_foreground_app()
        self.window.showNormal()
        self.window.raise_()
        self.window.activateWindow()

    def _local_tts_configuration(
        self,
    ) -> tuple[TextToSpeechProvider, str, str, str] | None:
        backend = str(self.config["playing_backend"])
        if backend not in ("qwen", "cosyvoice", "sovits"):
            return None
        if backend == "cosyvoice":
            manager: TextToSpeechProvider = self.cosyvoice_manager
            model_key = str(self.config["cosyvoice_model"])
            speaker = str(self.config["cosyvoice_prompt_audio"])
            language = str(self.config["cosyvoice_prompt_text"])
        elif backend == "sovits":
            manager = self.sovits_manager
            manager.configure(
                str(self.config["sovits_prompt_text"]),
                str(self.config["sovits_prompt_lang"]),
            )
            model_key = str(self.config["sovits_installation"])
            speaker = str(self.config["sovits_ref_audio_path"])
            language = str(self.config["sovits_text_lang"])
        else:
            manager = self.tts_manager
            model_key = str(self.config["tts_model"])
            speaker = str(self.config["tts_speaker"])
            language = str(self.config["tts_language"])
        return manager, model_key, speaker, language

    @staticmethod
    def _completed_response_sentences(
        text: str,
        *,
        final: bool,
    ) -> tuple[list[str], int]:
        sentences: list[str] = []
        start = 0
        consumed = 0
        for match in re.finditer(r'[.!?。！？]+["\'”’）)\]]*', text):
            punctuation_start = match.start()
            if (
                text[punctuation_start] == "."
                and punctuation_start > 0
                and match.end() < len(text)
                and text[punctuation_start - 1].isdigit()
                and text[match.end()].isdigit()
            ):
                continue
            if match.end() < len(text) and not text[match.end()].isspace():
                continue
            end = match.end()
            sentence = text[start:end].strip()
            if sentence:
                sentences.append(sentence)
            while end < len(text) and text[end].isspace():
                end += 1
            start = end
            consumed = end
        if final:
            remainder = text[start:].strip()
            if remainder:
                sentences.append(remainder)
            consumed = len(text)
        return sentences, consumed

    def _update_local_voice(self, text: str, final: bool) -> None:
        if getattr(self, "_local_voice_interrupted", False):
            return
        text = text.strip()
        if not text:
            return
        if not final and len(text) < len(self._local_voice_longest_text):
            logger.debug(
                "Ignoring a temporary shorter ChatGPT response snapshot "
                f"characters={len(text)} "
                f"longest={len(self._local_voice_longest_text)}"
            )
            return
        if len(text) >= len(self._local_voice_longest_text) or final:
            self._local_voice_longest_text = text
        sentences, _consumed = self._completed_response_sentences(
            text, final=final
        )
        queued_count = len(self._local_voice_queued_sentences)
        new_sentences = sentences[queued_count:]
        if new_sentences and self._local_speech_thread is None:
            configuration = self._local_tts_configuration()
            if configuration is None:
                return
            manager, model_key, speaker, language = configuration
            worker = _QueuedLocalSpeechThread(
                manager, model_key, speaker, language
            )
            self._local_speech_thread = worker
            worker.started.connect(self.window.begin_reading)
            worker.progress.connect(self.window.set_reading_subtitle)
            worker.completed.connect(self._on_local_speech_completed)
            worker.finished.connect(self._local_speech_finished)
            worker.start()

        worker = self._local_speech_thread
        if isinstance(worker, _QueuedLocalSpeechThread):
            worker.update_full_text(text)
            for sentence in new_sentences:
                worker.enqueue(sentence, text)
            self._local_voice_queued_sentences.extend(new_sentences)
            if final:
                worker.finish_queue()

    def _play_local_voice(self, text: str) -> None:
        """Compatibility entry point for a complete local-voice response."""
        self._update_local_voice(text, True)

    def _start_qwen_preload(self) -> None:
        """Warm the selected local model without delaying the settings or UI."""
        backend = str(self.config["playing_backend"])
        if backend not in ("qwen", "cosyvoice", "sovits"):
            return
        existing = getattr(self, "_qwen_preload_thread", None)
        if existing is not None and existing.is_alive():
            return
        if backend == "cosyvoice":
            manager = self.cosyvoice_manager
            model_key = str(self.config["cosyvoice_model"])
            model_source = str(
                self.config.get("cosyvoice_model_source", "huggingface")
            )
            provider_name = "CosyVoice"
        elif backend == "sovits":
            manager = self.sovits_manager
            manager.configure(
                str(self.config["sovits_prompt_text"]),
                str(self.config["sovits_prompt_lang"]),
            )
            model_key = str(self.config["sovits_installation"])
            model_source = "existing"
            provider_name = "GPT-SoVITS"
        else:
            manager = self.tts_manager
            model_key = str(self.config["tts_model"])
            model_source = str(
                self.config.get("qwen_model_source", "huggingface")
            )
            provider_name = "Qwen"

        def preload() -> None:
            try:
                if isinstance(manager, SovitsTtsProvider):
                    manager.model_status("tts", model_key)
                    runtime_ok, runtime_message = manager.dependency_status()
                else:
                    runtime_ok, runtime_message = manager.dependency_status(
                        model_source
                    )
                if not runtime_ok:
                    logger.warning(
                        f"{provider_name} background preload skipped until "
                        "runtime repair: "
                        f"{runtime_message}"
                    )
                    return
                manager.preload(model_key)
            except Exception as error:
                logger.warning(
                    f"{provider_name} background preload skipped: {error}"
                )

        worker = threading.Thread(
            target=preload,
            name="local-tts-background-preload",
            daemon=True,
        )
        self._qwen_preload_thread = worker
        worker.start()

    def _start_stt_preload(self) -> None:
        """Warm local transcription before the first microphone press."""
        if str(self.config["recording_backend"]) != "sherpa":
            return
        existing = getattr(self, "_stt_preload_thread", None)
        if existing is not None and existing.is_alive():
            return
        model_key = str(self.config["stt_model"])

        def preload() -> None:
            try:
                message = self.stt_manager.preload(model_key)
                logger.info(f"Sherpa background preload complete: {message}")
            except Exception as error:
                logger.warning(
                    f"Sherpa background preload skipped: {error}"
                )

        worker = threading.Thread(
            target=preload,
            name="local-stt-background-preload",
            daemon=True,
        )
        self._stt_preload_thread = worker
        worker.start()

    def _local_speech_finished(self) -> None:
        worker = self._local_speech_thread
        self._local_speech_thread = None
        self._local_voice_longest_text = ""
        self._local_voice_queued_sentences = []
        if worker is not None:
            worker.deleteLater()

    def _on_local_speech_completed(self, success: bool, message: str) -> None:
        if getattr(self, "_dictation_state", "idle") == "idle":
            self.window.finish_reading(success, message)

    def _on_browser_reading_finished(self, success: bool, message: str) -> None:
        if getattr(self, "_dictation_state", "idle") == "idle":
            self.window.finish_reading(success, message)

    def start_dictation(
        self,
        *,
        force_auto_send: bool = False,
        include_screenshot: bool = True,
        initial_hold_seconds: float = 0.0,
    ) -> None:
        self._stop_playback_for_recording()
        self._dictation_force_auto_send = force_auto_send
        self._dictation_include_screenshot = include_screenshot
        self._dictation_input_held = True
        self._dictation_pressed_since = (
            time.monotonic() - max(0.0, initial_hold_seconds)
        )
        self._dictation_press_generation = (
            getattr(self, "_dictation_press_generation", 0) + 1
        )
        self._discard_preuploaded_dictation_screenshot()
        state = getattr(self, "_dictation_state", "idle")
        if state != "idle":
            return
        self._pending_dictation_capture = None

        self.window.show_for_auto_hide()
        dismissed = self.window.dismiss_subtitle_mode()
        if (
            not dismissed
            and self.window.transcript_area.is_showing_response
        ):
            self.window.transcript_area.begin_composing()

        if (
            getattr(self, "config", {}).get("recording_backend", "web")
            == "sherpa"
        ):
            self._schedule_dictation_screenshot_upload()
            self._start_local_dictation()
            return

        tab_id = self.selected_chatgpt_tab_id
        if tab_id is None:
            self._dictation_input_held = False
            self.window.set_microphone_state(
                "error",
                "Select a ChatGPT window first",
            )
            self.window.schedule_auto_hide(5_000)
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
        self._schedule_dictation_screenshot_upload()
        self.browser_monitor.request_start_dictation(tab_id)

    def _stop_playback_for_recording(self) -> None:
        self.browser_monitor.request_stop_reading()
        worker = getattr(self, "_local_speech_thread", None)
        if worker is not None:
            self._local_voice_interrupted = True
            worker.request_stop()
            self.window.finish_reading(False, "Playback stopped for recording")

    def _schedule_dictation_screenshot_upload(self) -> None:
        generation = self._dictation_press_generation
        pressed_since = self._dictation_pressed_since or time.monotonic()
        remaining_ms = max(
            0,
            round((0.5 - (time.monotonic() - pressed_since)) * 1_000),
        )
        QTimer.singleShot(
            remaining_ms,
            lambda: self._upload_dictation_screenshot_after_hold(generation),
        )

    def _upload_dictation_screenshot_after_hold(self, generation: int) -> None:
        if (
            generation != getattr(self, "_dictation_press_generation", 0)
            or not getattr(self, "_dictation_input_held", False)
            or getattr(self, "_dictation_state", "idle")
            not in ("starting", "listening")
            or not (
                self.window.auto_send_enabled is True
                or getattr(self, "_dictation_force_auto_send", False)
            )
        ):
            return

        if not getattr(self, "_dictation_include_screenshot", True):
            self._pending_dictation_capture = _PendingDictationCapture(
                source=None,
                screenshot=None,
            )
            return

        selected = self.window.capture_source_combo.currentData()
        source = selected if isinstance(selected, CaptureSource) else None
        if source is None:
            self._pending_dictation_capture = _PendingDictationCapture(
                source=None,
                screenshot=None,
            )
            return

        tab_id = self.selected_chatgpt_tab_id
        if tab_id is None:
            return
        try:
            screenshot = capture_webp(source)
        except Exception as error:
            logger.error("Unable to capture dictation screenshot", error)
            self._pending_dictation_capture = _PendingDictationCapture(
                source=source,
                screenshot=None,
                error=str(error),
            )
            return

        self._pending_dictation_capture = _PendingDictationCapture(
            source=source,
            screenshot=None,
            preuploaded=True,
        )
        self._dictation_attachment_tab_id = tab_id
        logger.info(
            "Captured and queued dictation screenshot after 0.5-second hold "
            f"source={source.key!r} tab_id={tab_id!r}"
        )
        self.browser_monitor.request_replace_attachment(tab_id, screenshot)

    def _discard_preuploaded_dictation_screenshot(self) -> None:
        tab_id = getattr(self, "_dictation_attachment_tab_id", None)
        if tab_id is not None:
            logger.info(
                "Removing the previous pending dictation screenshot "
                f"tab_id={tab_id!r}"
            )
            self.browser_monitor.request_clear_attachments(tab_id)
        self._dictation_attachment_tab_id = None
        self._pending_dictation_capture = None

    def _start_local_dictation(self) -> None:
        if self._local_dictation_thread is not None:
            return
        stt_model = str(self.config["stt_model"])
        logger.info(f"Starting local dictation model={stt_model!r}")
        self.dictation_tab_id = "local"
        self._dictation_state = "starting"
        self._dictation_listening_since = None
        self.window.begin_dictation_waiting()
        self.window.set_microphone_state(
            "recording",
            "Starting the local microphone…",
        )
        session = LocalDictationSession(self.stt_manager, stt_model)
        worker = _LocalDictationThread(session)
        self._local_dictation_thread = worker
        worker.listening.connect(
            lambda: self._on_dictation_started(
                True,
                "Sherpa-ONNX is listening",
            )
        )
        worker.partial_text.connect(self._on_local_dictation_partial)
        worker.completed.connect(self._on_dictation_finished)
        worker.finished.connect(self._local_dictation_finished)
        worker.start()

    def _local_dictation_finished(self) -> None:
        worker = self._local_dictation_thread
        self._local_dictation_thread = None
        if worker is not None:
            worker.deleteLater()

    def _on_local_dictation_partial(self, text: str) -> None:
        state = getattr(self, "_dictation_state", "idle")
        if state in (
            "listening",
            "finishing",
        ):
            self.window.set_dictation_partial(
                text,
                finishing=state == "finishing",
            )

    def finish_dictation(self) -> None:
        self._dictation_input_held = False
        state = getattr(self, "_dictation_state", "idle")
        tab_id = self.dictation_tab_id
        if tab_id is None or state == "idle":
            return

        if state == "starting":
            self._discard_preuploaded_dictation_screenshot()
            self.window.set_dictation_cancelling()
            local_thread = getattr(self, "_local_dictation_thread", None)
            if local_thread is not None:
                self._dictation_state = "cancelling"
                local_thread.stop_recording(cancel=True)
            return
        if state != "listening":
            return

        pressed_since = getattr(self, "_dictation_pressed_since", None)
        pressed_since = (
            pressed_since
            or self._dictation_listening_since
            or time.monotonic()
        )
        if time.monotonic() - pressed_since < 0.5:
            self._cancel_dictation(tab_id)
            return

        self._dictation_state = "finishing"
        local = tab_id == "local"
        logger.info(
            f"Finishing {'local' if local else 'ChatGPT'} dictation "
            f"tab_id={tab_id!r}"
        )
        self.window.set_dictation_finishing()
        self.window.set_microphone_state(
            "recording",
            "Transcribing local audio…" if local else "Finishing ChatGPT dictation…",
        )
        if local and self._local_dictation_thread is not None:
            self._local_dictation_thread.stop_recording()
        else:
            self.browser_monitor.request_finish_dictation(tab_id)

    def _cancel_dictation(self, tab_id: str) -> None:
        self._dictation_state = "cancelling"
        self._discard_preuploaded_dictation_screenshot()
        logger.info(f"Cancelling short ChatGPT dictation tab_id={tab_id!r}")
        self.window.set_dictation_cancelling()
        self.window.set_microphone_state(
            "recording",
            "Dictation was too short; cancelling…",
        )
        if tab_id == "local" and self._local_dictation_thread is not None:
            self._local_dictation_thread.stop_recording(cancel=True)
        else:
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
        self._dictation_pressed_since = None
        self._dictation_listening_since = None
        self.dictation_tab_id = None
        self._discard_preuploaded_dictation_screenshot()
        self.window.end_dictation_display()
        self.window.set_microphone_state("error", message)
        self.window.schedule_auto_hide(5_000)

    def _on_dictation_finished(
        self,
        success: bool,
        text: str,
        message: str,
    ) -> None:
        was_cancelled = getattr(self, "_dictation_state", "idle") == "cancelling"
        restart = getattr(self, "_dictation_input_held", False)
        pressed_since = getattr(self, "_dictation_pressed_since", None)
        restart_held_seconds = (
            max(0.0, time.monotonic() - pressed_since)
            if restart and pressed_since is not None
            else 0.0
        )
        self._dictation_state = "idle"
        self._dictation_pressed_since = None
        self._dictation_listening_since = None
        self.dictation_tab_id = None
        self.window.end_dictation_display()
        if not success:
            self._dictation_input_held = False
            self._discard_preuploaded_dictation_screenshot()
            self.window.set_microphone_state("error", message)
            self.window.schedule_auto_hide(5_000)
            return

        if was_cancelled:
            self._discard_preuploaded_dictation_screenshot()
            self.window.set_microphone_state("idle", message)
        else:
            self.window.set_transcript(text)
            self.window.set_microphone_state("saved", message)
        auto_sent = bool(
            not was_cancelled
            and not restart
            and text.strip()
            and (
                self.window.auto_send_enabled
                or getattr(self, "_dictation_force_auto_send", False)
            )
        )
        if auto_sent:
            self._dictation_input_held = False
            self.window.request_auto_send()
            self._pending_dictation_capture = None
        elif text.strip():
            self._discard_preuploaded_dictation_screenshot()
            self.window.show_for_auto_hide()
        else:
            self._discard_preuploaded_dictation_screenshot()
            self.window.schedule_auto_hide()
        if restart and not auto_sent:
            force_auto_send = getattr(self, "_dictation_force_auto_send", False)
            include_screenshot = getattr(
                self,
                "_dictation_include_screenshot",
                True,
            )
            if (
                getattr(self, "config", {}).get("recording_backend", "web")
                == "sherpa"
            ):
                QTimer.singleShot(
                    0,
                    lambda: self.start_dictation(
                        force_auto_send=force_auto_send,
                        include_screenshot=include_screenshot,
                        initial_hold_seconds=restart_held_seconds,
                    ),
                )
            else:
                self.start_dictation(
                    force_auto_send=force_auto_send,
                    include_screenshot=include_screenshot,
                    initial_hold_seconds=restart_held_seconds,
                )

    def _handle_send_requested(
        self,
        text: str,
        capture_source: CaptureSource | None,
    ) -> None:
        pending_capture = getattr(
            self,
            "_pending_dictation_capture",
            None,
        )
        self._pending_dictation_capture = None
        tab_id = self.selected_chatgpt_tab_id
        if tab_id is None:
            self.window.set_status(
                "Select a ChatGPT window first",
                error=True,
            )
            return

        self.window.send_button.setEnabled(False)
        self.window.send_without_screenshot_button.setEnabled(False)
        if pending_capture is not None:
            capture_source = pending_capture.source
            screenshot = pending_capture.screenshot
            if pending_capture.error is not None:
                self.window.send_button.setEnabled(True)
                self.window.send_without_screenshot_button.setEnabled(True)
                self.window.set_status(
                    "Could not capture screenshot after the microphone hold: "
                    f"{pending_capture.error}",
                    error=True,
                )
                return
        else:
            screenshot = None

        if capture_source is not None and pending_capture is None:
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
        self._local_voice_interrupted = False
        preserve_attachments = bool(
            pending_capture is not None and pending_capture.preuploaded
        )
        if preserve_attachments:
            self.browser_monitor.request_send(
                tab_id,
                text,
                screenshot,
                preserve_attachments=True,
            )
        else:
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

    def _save_chatgpt_preference(self, url: str) -> None:
        self.config["chatgpt_window"] = url

    def _save_capture_source_preference(self, source_key: str) -> None:
        self.config["capture_source"] = source_key

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

    def _restore_window_geometry(self) -> None:
        geometry = self.config["window_geometry"]
        if not geometry:
            return
        restored = QRect(*geometry)
        if not any(
            screen.availableGeometry().intersects(restored)
            for screen in self.application.screens()
        ):
            logger.warning("Saved overlay geometry is outside visible screens")
            self.config["window_geometry"] = []
            return
        self.window.setGeometry(restored)

    def _schedule_window_geometry_save(self, geometry: QRect) -> None:
        del geometry
        self._geometry_save_timer.start(250)

    def _save_window_geometry(self) -> None:
        geometry = (
            self.window._subtitle_collapsed_geometry
            or self.window.geometry()
        )
        self.config["window_geometry"] = [
            geometry.x(),
            geometry.y(),
            geometry.width(),
            geometry.height(),
        ]

    def _exit_application(self, checked: bool = False) -> None:
        del checked
        logger.info("Exit requested")
        self._save_window_geometry()
        self.capture_refresh_timer.stop()
        self.hotkey_monitor.stop()
        if self._local_dictation_thread is not None:
            self._local_dictation_thread.stop_recording(cancel=True)
            self._local_dictation_thread.wait(5_000)
        if self._local_speech_thread is not None:
            if isinstance(self._local_speech_thread, _QueuedLocalSpeechThread):
                self._local_speech_thread.finish_queue()
            try:
                import sounddevice as sd

                sd.stop()
            except Exception:
                pass
            self._local_speech_thread.wait(5_000)
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
