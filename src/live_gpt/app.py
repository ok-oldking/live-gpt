from __future__ import annotations

import os
import queue
import re
import sys
import threading
import time
import webbrowser
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
    QUrl,
    Signal,
    Qt,
)
from PySide6.QtGui import (
    QAction,
    QCloseEvent,
    QColor,
    QCursor,
    QDesktopServices,
    QIcon,
    QKeySequence,
    QMouseEvent,
    QPalette,
    QRegion,
    QTextCursor,
)
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
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
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QProgressBar,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QSpinBox,
    QSystemTrayIcon,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from .browser import (
    BrowserMonitor,
    discover_cdp_endpoint,
    open_remote_debugging_settings,
)
from .config import Config, DEFAULT_CONFIG
from .localization import UiTranslations, localization, resolve_language, translate_message, tr
from .pet import PetWidget, available_pets, default_pet_path
from .pet_download import download_pet, pet_source
from .logger import Logger, config_logger, shutdown_logger
from .response_content import reply_code_blocks, reply_html_with_links
from .hotkeys import GlobalHotkeyMonitor, HotkeyBinding, HotkeyEdit, shortcut_text
from .screen_capture import CaptureSource, capture_webp, list_capture_sources
from .window_focus import ForegroundWindowRestorer
from .voice.base import ModelProvider, TextToSpeechProvider
from .voice.dependencies import OperationCancelled, PYPI_MIRRORS
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
COPY_ICON_PATH = ASSET_DIRECTORY / "copy.svg"
SETTINGS_ICON_PATH = ASSET_DIRECTORY / "settings.svg"
SHORTCUTS_ICON_PATH = ASSET_DIRECTORY / "shortcuts.svg"
SCREENSHOT_ICON_PATH = ASSET_DIRECTORY / "screenshot.svg"
LANGUAGE_ICON_PATH = ASSET_DIRECTORY / "language.svg"
LOCK_ICON_PATH = ASSET_DIRECTORY / "lock.svg"
UNLOCK_ICON_PATH = ASSET_DIRECTORY / "unlock.svg"
PET_ICON_PATH = ASSET_DIRECTORY / "pet.svg"
COMBOBOX_STYLE = """
QComboBox, QComboBox#voiceCombo, QComboBox#languageCombo,
QComboBox#chatgptTabCombo, QComboBox#captureSourceCombo { padding-right: 32px; }
QComboBox::drop-down {
    subcontrol-origin: border;
    subcontrol-position: top right;
    width: 28px;
    border: none;
    background: transparent;
}
QComboBox::down-arrow {
    image: url("%s");
    width: 16px;
    height: 16px;
}
""" % (ASSET_DIRECTORY / "chevron-down.svg").as_posix()
logger = Logger.get_logger(__name__)

DEFAULT_HOLD_MIC_HOTKEY = str(DEFAULT_CONFIG["hotkey_hold"])
DEFAULT_HOLD_WITHOUT_SCREENSHOT_HOTKEY = str(
    DEFAULT_CONFIG["hotkey_hold_without_screenshot"]
)
MIN_ENGLISH_VOICE_SEND_CHARACTERS = 2
HOTKEY_CONFIG_KEYS = {
    "hold": "hotkey_hold",
    "hold_without_screenshot": "hotkey_hold_without_screenshot",
}
LEGACY_HOTKEY_SETTING_KEYS = {
    "hold": "hotkeys/hold_microphone",
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
    ) -> None:
        super().__init__()
        self.manager = manager
        self.action = action
        self.model_type = model_type
        self.model_key = model_key
        self.mirror = mirror
        self._cancel_event = threading.Event()

    def cancel_operation(self) -> None:
        self._cancel_event.set()

    def run(self) -> None:
        try:
            result: object = None
            if self.action == "status":
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
                        SovitsTtsProvider,
                    )
                    and dependency_ok
                    and model_ok
                ):
                    self.progress.emit(
                        tr("Preloading the local TTS model on the GPU…"), None
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
                message = self.manager.install_dependencies(
                    self.progress.emit,
                    self.log_line.emit,
                    self.mirror,
                    self._cancel_event,
                )
            elif self.action == "download":
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
        speaker: str = "",
        language: str = "auto",
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

            provider_name = getattr(self.manager, "display_name", "local TTS")
            if not isinstance(provider_name, str):
                provider_name = "local TTS"
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
                self.completed.emit(False, tr("Playback stopped for recording"))
                return
            self.progress.emit({"text": self.text, "fraction": 1.0})
            self.completed.emit(
                True,
                f"Generated in {latency_ms:.0f} ms · audio {audio_seconds:.1f} s",
            )
        except Exception as error:
            if self._cancel_event.is_set():
                self.completed.emit(False, tr("Playback stopped for recording"))
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
        provider_name = getattr(self.manager, "display_name", "local TTS")
        if not isinstance(provider_name, str):
            provider_name = "local TTS"
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
                    self.completed.emit(False, tr("Playback stopped for recording"))
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


@dataclass(frozen=True)
class _LocalSpeechAnnouncement:
    text: str


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
        self._queue_finished = False
        self._cancel_event = threading.Event()
        self._output: object | None = None

    def enqueue(self, sentence: str, full_text: str) -> None:
        self._full_text = full_text
        if sentence.strip():
            self._sentences.put(sentence.strip())

    def update_full_text(self, full_text: str) -> None:
        self._full_text = full_text

    def enqueue_announcement(self, text: str) -> None:
        if text.strip() and not self._queue_finished:
            self._sentences.put(_LocalSpeechAnnouncement(text.strip()))

    def finish_queue(self) -> None:
        if not self._queue_finished:
            self._queue_finished = True
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
                        announcement = isinstance(item, _LocalSpeechAnnouncement)
                        sentence = item.text if announcement else str(item)
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
                            generated.put((chunk, sentence, announcement))
                            produced = True
                        if not produced:
                            raise RuntimeError(
                                "The local TTS model generated no audio"
                            )
                        generated.put((self._SENTENCE_DONE, sentence, announcement))
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
                    self.completed.emit(False, tr("Playback stopped for recording"))
                    return
                if item is self._FINISHED:
                    break
                if isinstance(item, Exception):
                    raise item
                if item[0] is self._SENTENCE_DONE:
                    if item[2]:
                        continue
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
                chunk, sentence, announcement = item
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
                if chunk_text and not announcement:
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
                self.completed.emit(False, tr("Playback stopped for recording"))
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


class _PetDownloadThread(QThread):
    completed = Signal(bool, str)

    def __init__(self, url: str, parent: QWidget) -> None:
        super().__init__(parent)
        self.url = url

    def run(self) -> None:
        try:
            self.completed.emit(True, str(download_pet(self.url)))
        except Exception as error:
            self.completed.emit(False, str(error))


class HotkeyConfigDialog(QDialog):
    """Edit Live GPT settings, including pass-through global shortcuts."""
    pet_settings_changed = Signal(str, str, int)

    def __init__(
        self,
        hold_microphone: QKeySequence | str,
        send: QKeySequence | None = None,
        send_without_screenshot: QKeySequence | None = None,
        parent: QWidget | None = None,
        *,
        hold_without_screenshot: QKeySequence | str | None = None,
        language: str = "",
        recording_backend: str = "web",
        playing_backend: str = "web",
        stt_language: str | None = None,
        stt_model: str = "zh_zipformer_ctc_int8_2025_07_03",
        pypi_mirror: str = "default",
        sovits_installation: str = "",
        sovits_ckpt_path: str = "",
        sovits_pth_path: str = "",
        sovits_text_lang: str = "auto",
        sovits_ref_audio_path: str = "",
        sovits_prompt_text: str = "",
        sovits_prompt_lang: str = "auto",
        config: Config | None = None,
        stt_manager: SherpaSttProvider | None = None,
        sovits_manager: SovitsTtsProvider | None = None,
    ) -> None:
        super().__init__(parent)
        language = resolve_language(str(config["language"]) if config is not None else language)
        localization.set_language(language)
        self._title_drag_offset: QPoint | None = None
        self.stt_manager = stt_manager or SherpaSttProvider()
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
        self.setWindowTitle(tr("Live GPT settings"))
        self.setWindowFlags(
            Qt.WindowType.Dialog | Qt.WindowType.FramelessWindowHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setMinimumSize(860, 680)
        self.resize(940, 820)

        self.hold_microphone_edit = self._sequence_edit(hold_microphone)
        self.hold_without_screenshot_edit = self._sequence_edit(
            hold_without_screenshot if hold_without_screenshot is not None else "Right Ctrl"
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
        window_title = QLabel(tr("Settings"))
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
        self.close_button.setAccessibleName(tr("Close settings"))
        self.close_button.setToolTip(tr("Close settings; changes are saved automatically"))
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
        navigation_label = QLabel(tr("SETTINGS"))
        navigation_label.setObjectName("settingsNavigationLabel")
        navigation_layout.addWidget(navigation_label)
        navigation_layout.addSpacing(8)

        self.shortcuts_nav_button = self._navigation_button(
            tr("Shortcuts"),
            SHORTCUTS_ICON_PATH,
        )
        self.language_nav_button = self._navigation_button(
            tr("Language"),
            LANGUAGE_ICON_PATH,
        )
        self.recording_nav_button = self._navigation_button(
            tr("Recording"),
            MICROPHONE_ICON_PATH,
        )
        self.playing_nav_button = self._navigation_button(
            tr("Playing"),
            SPEAKER_ICON_PATH,
        )
        self.navigation_group = QButtonGroup(self)
        self.navigation_group.setExclusive(True)
        self.navigation_group.addButton(self.shortcuts_nav_button, 0)
        self.navigation_group.addButton(self.language_nav_button, 1)
        self.navigation_group.addButton(self.recording_nav_button, 2)
        self.navigation_group.addButton(self.playing_nav_button, 3)
        self.pet_nav_button = self._navigation_button(tr("Pet"), PET_ICON_PATH)
        self.navigation_group.addButton(self.pet_nav_button, 4)
        self.screenshots_nav_button = self._navigation_button(
            tr("Screenshots"), SCREENSHOT_ICON_PATH,
        )
        self.navigation_group.addButton(self.screenshots_nav_button, 5)
        self.shortcuts_nav_button.setChecked(True)
        navigation_layout.addWidget(self.shortcuts_nav_button)
        navigation_layout.addWidget(self.screenshots_nav_button)
        navigation_layout.addWidget(self.language_nav_button)
        navigation_layout.addWidget(self.recording_nav_button)
        navigation_layout.addWidget(self.playing_nav_button)
        navigation_layout.addWidget(self.pet_nav_button)
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
        hotkey_title = QLabel(tr("Keyboard shortcuts"))
        hotkey_title.setObjectName("settingsPageTitle")
        hotkey_description = QLabel(
            tr("Control Live GPT without leaving the app you are using.")
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
        card_title = QLabel(tr("Global shortcuts"))
        card_title.setObjectName("settingsCardTitle")
        hotkey_card_layout.addWidget(card_title)

        form = QFormLayout()
        form.setContentsMargins(0, 4, 0, 0)
        form.setHorizontalSpacing(22)
        form.setVerticalSpacing(12)
        for name, label, editor in (
            ("hold", tr("Record and Send with Screenshot"), self.hold_microphone_edit),
            ("hold_without_screenshot", tr("Record and Send without Screenshot"),
             self.hold_without_screenshot_edit),
        ):
            form.addRow(label, editor)
        hotkey_card_layout.addLayout(form)

        note = QLabel(
            tr("These shortcuts work globally and are still passed to the "
            "foreground program.")
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
        language_title = QLabel(tr("Interface language"))
        language_title.setObjectName("settingsPageTitle")
        language_description = QLabel(
            tr("Choose the language used for menus, labels, and messages.")
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
        language_card_title = QLabel(tr("Display language"))
        language_card_title.setObjectName("settingsCardTitle")
        language_card_layout.addWidget(language_card_title)

        self.language_combo = QComboBox()
        self.language_combo.setObjectName("languageCombo")
        self.language_combo.setAccessibleName(tr("Interface language"))
        self.language_combo.addItem("English", "en")
        self.language_combo.addItem("简体中文", "zh")
        self.language_combo.setCurrentIndex(max(self.language_combo.findData(language), 0))
        self.language_combo.setToolTip(
            tr("Choose an interface language")
        )
        language_card_layout.addWidget(self.language_combo)
        self.language_status = QLabel(
            tr("Changes apply immediately and are saved automatically.")
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
        recording_title = QLabel(tr("Recording"))
        recording_title.setObjectName("settingsPageTitle")
        recording_description = QLabel(
            tr("Configure microphone transcription independently from voice playback.")
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
        recording_backend_title = QLabel(tr("Recording engine"))
        recording_backend_title.setObjectName("settingsCardTitle")
        recording_backend_layout.addWidget(recording_backend_title)
        self.recording_backend_combo = QComboBox()
        self.recording_backend_combo.setObjectName("voiceCombo")
        self.recording_backend_combo.addItem(tr("Web built-in (browser)"), "web")
        self.recording_backend_combo.addItem(tr("Sherpa-ONNX (local)"), "sherpa")
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
        recording_sherpa_title = QLabel(tr("Local speech to text"))
        recording_sherpa_title.setObjectName("settingsCardTitle")
        recording_sherpa_layout.addWidget(recording_sherpa_title)
        recording_runtime_row = QHBoxLayout()
        recording_runtime_row.setSpacing(8)
        self.recording_check_button = QPushButton(tr("Check"))
        self.recording_install_button = QPushButton(tr("Install / Repair runtime"))
        for button in (self.recording_check_button, self.recording_install_button):
            button.setObjectName("voiceActionButton")
            recording_runtime_row.addWidget(button)
        self.recording_pypi_mirror_combo = QComboBox()
        self.recording_pypi_mirror_combo.setObjectName("voiceCombo")
        self.recording_pypi_mirror_combo.setToolTip(
            tr("PyPI mirror used when installing or repairing dependencies")
        )
        for mirror in PYPI_MIRRORS.values():
            self.recording_pypi_mirror_combo.addItem(
                tr("PyPI: {mirror}", mirror=tr(mirror.label)), mirror.key
            )
        recording_mirror_index = self.recording_pypi_mirror_combo.findData(
            pypi_mirror
        )
        self.recording_pypi_mirror_combo.setCurrentIndex(
            max(recording_mirror_index, 0)
        )
        recording_runtime_row.addWidget(self.recording_pypi_mirror_combo)
        self.recording_cancel_button = QPushButton(tr("Cancel install / download"))
        self.recording_cancel_button.setObjectName("voiceCancelButton")
        self.recording_cancel_button.setToolTip(
            tr("Stop the active dependency installation or model download")
        )
        self.recording_cancel_button.hide()
        recording_runtime_row.addWidget(self.recording_cancel_button)
        recording_runtime_row.addStretch()
        recording_sherpa_layout.addLayout(recording_runtime_row)

        recording_sherpa_layout.addWidget(QLabel(tr("Recording language")))
        self.stt_language_combo = QComboBox()
        self.stt_language_combo.setObjectName("voiceCombo")
        for label, code in ((tr("Auto (multiple languages)"), "auto"), (tr("Chinese (zh)"), "zh"), (tr("English (en)"), "en")):
            self.stt_language_combo.addItem(label, code)
        self.stt_language_combo.setCurrentIndex(self.stt_language_combo.findData(
            stt_language or STT_MODELS[stt_model].supported_languages[0]
        ))
        recording_sherpa_layout.addWidget(self.stt_language_combo)
        recording_sherpa_layout.addWidget(QLabel(tr("Recording model")))
        self.stt_model_combo = QComboBox()
        self.stt_model_combo.setObjectName("voiceCombo")
        self._populate_stt_models(stt_model)
        recording_sherpa_layout.addWidget(self.stt_model_combo)
        self.stt_model_description = QLabel()
        self.stt_model_description.setObjectName("settingsNote")
        self.stt_model_description.setWordWrap(True)
        recording_sherpa_layout.addWidget(self.stt_model_description)
        self.stt_test_result = QLineEdit()
        self.stt_test_result.setObjectName("voiceTestText")
        self.stt_test_result.setReadOnly(True)
        self.stt_test_result.setPlaceholderText(
            tr("Live transcription appears here while streaming")
        )
        recording_sherpa_layout.addWidget(self.stt_test_result)
        stt_action_row = QHBoxLayout()
        stt_action_row.setSpacing(8)
        self.stt_download_button = QPushButton(tr("Download STT model"))
        self.voice_record_button = QPushButton(tr("Record microphone"))
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
            tr("Select Check to validate the runtime and selected recording model.")
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
        playing_title = QLabel(tr("Playing"))
        playing_title.setObjectName("settingsPageTitle")
        playing_description = QLabel(
            tr("Configure reply speech independently from microphone transcription.")
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
        playing_backend_title = QLabel(tr("Playback engine"))
        playing_backend_title.setObjectName("settingsCardTitle")
        playing_backend_layout.addWidget(playing_backend_title)
        self.playing_backend_combo = QComboBox()
        self.playing_backend_combo.setObjectName("voiceCombo")
        self.playing_backend_combo.addItem(tr("Web built-in (browser)"), "web")
        self.playing_backend_combo.addItem(
            tr("GPT-SoVITS (existing local installation)"), "sovits"
        )
        playing_backend_index = self.playing_backend_combo.findData(
            playing_backend
        )
        self.playing_backend_combo.setCurrentIndex(max(playing_backend_index, 0))
        playing_backend_layout.addWidget(self.playing_backend_combo)
        playing_layout.addWidget(playing_backend_card)

        self.playing_local_card = QFrame()
        self.playing_local_card.setObjectName("settingsCard")
        playing_local_layout = QVBoxLayout(self.playing_local_card)
        playing_local_layout.setContentsMargins(18, 16, 18, 16)
        playing_local_layout.setSpacing(10)
        self.playing_local_title = QLabel(tr("GPT-SoVITS text to speech"))
        self.playing_local_title.setObjectName("settingsCardTitle")
        playing_local_layout.addWidget(self.playing_local_title)
        playing_runtime_row = QHBoxLayout()
        playing_runtime_row.setSpacing(8)
        self.playing_check_button = QPushButton(tr("Check"))
        self.playing_install_button = QPushButton(tr("Install / Repair runtime"))
        for button in (self.playing_check_button, self.playing_install_button):
            button.setObjectName("voiceActionButton")
            playing_runtime_row.addWidget(button)
        self.playing_pypi_mirror_combo = QComboBox()
        self.playing_pypi_mirror_combo.setObjectName("voiceCombo")
        self.playing_pypi_mirror_combo.setToolTip(
            tr("PyPI mirror used when installing or repairing dependencies")
        )
        for mirror in PYPI_MIRRORS.values():
            self.playing_pypi_mirror_combo.addItem(
                tr("PyPI: {mirror}", mirror=tr(mirror.label)), mirror.key
            )
        playing_mirror_index = self.playing_pypi_mirror_combo.findData(
            pypi_mirror
        )
        self.playing_pypi_mirror_combo.setCurrentIndex(
            max(playing_mirror_index, 0)
        )
        playing_runtime_row.addWidget(self.playing_pypi_mirror_combo)
        self.playing_cancel_button = QPushButton(tr("Cancel install / download"))
        self.playing_cancel_button.setObjectName("voiceCancelButton")
        self.playing_cancel_button.setToolTip(
            tr("Stop the active dependency installation or model download")
        )
        self.playing_cancel_button.hide()
        playing_runtime_row.addWidget(self.playing_cancel_button)
        playing_runtime_row.addStretch()
        playing_local_layout.addLayout(playing_runtime_row)

        self.sovits_weight_widgets = []
        for suffix, label, value in (
            ("ckpt", tr("GPT model (.ckpt)"), sovits_ckpt_path),
            ("pth", tr("SoVITS model (.pth)"), sovits_pth_path),
        ):
            row = QHBoxLayout()
            edit = QLineEdit(value)
            edit.setObjectName("voiceTestText")
            edit.setAccessibleName(label)
            edit.setPlaceholderText(tr("Optional — select both model files or leave both blank"))
            button = QPushButton(tr("Browse…"))
            button.setObjectName("voiceActionButton")
            button.clicked.connect(lambda _checked=False, ext=suffix: self._browse_sovits_weight(ext))
            setattr(self, f"sovits_{suffix}_edit", edit)
            self.sovits_weight_widgets.extend((edit, button))
            row.addWidget(QLabel(label))
            row.addWidget(edit, 1)
            row.addWidget(button)
            playing_local_layout.addLayout(row)
        self.sovits_weights_status = QLabel()
        self.sovits_weights_status.setObjectName("settingsNote")
        self.sovits_weights_status.setWordWrap(True)
        playing_local_layout.addWidget(self.sovits_weights_status)

        self.sovits_installation_label = QLabel(tr("Installation folder"))
        self.sovits_installation_edit = QLineEdit(sovits_installation)
        self.sovits_installation_edit.setObjectName("voiceTestText")
        self.sovits_installation_edit.setPlaceholderText(
            tr("Folder containing GPT_SoVITS and runtime/python.exe")
        )
        self.sovits_installation_browse_button = QPushButton(tr("Browse…"))
        self.sovits_installation_browse_button.setObjectName("voiceActionButton")
        sovits_installation_row = QHBoxLayout()
        sovits_installation_row.setSpacing(8)
        sovits_installation_row.addWidget(self.sovits_installation_label)
        sovits_installation_row.addWidget(self.sovits_installation_edit, 1)
        sovits_installation_row.addWidget(self.sovits_installation_browse_button)
        playing_local_layout.addLayout(sovits_installation_row)

        self.sovits_reference_title = QLabel(tr("Reference voice"))
        self.sovits_reference_title.setObjectName("voiceFieldGroupTitle")
        self.sovits_reference_description = QLabel(
            tr("These settings describe the voice sample GPT-SoVITS should imitate.")
        )
        self.sovits_reference_description.setObjectName("settingsNote")
        self.sovits_reference_description.setWordWrap(True)
        playing_local_layout.addWidget(self.sovits_reference_title)
        playing_local_layout.addWidget(self.sovits_reference_description)

        self.sovits_text_lang_label = QLabel(tr("Output language"))
        self.sovits_text_lang_combo = QComboBox()
        self.sovits_text_lang_combo.setObjectName("voiceCombo")
        self.sovits_text_lang_combo.setToolTip(
            tr("Language of the text that GPT-SoVITS will generate")
        )
        self.sovits_prompt_lang_label = QLabel(tr("Reference language"))
        self.sovits_prompt_lang_combo = QComboBox()
        self.sovits_prompt_lang_combo.setObjectName("voiceCombo")
        self.sovits_prompt_lang_combo.setToolTip(
            tr("Language spoken in the reference audio and transcript")
        )
        sovits_language_labels = {
            "auto": "Auto", "auto_yue": "Auto (Cantonese)",
            "zh": "Chinese", "en": "English", "ja": "Japanese",
            "yue": "Cantonese", "ko": "Korean",
            "all_zh": "Chinese only", "all_ja": "Japanese only",
            "all_yue": "Cantonese only", "all_ko": "Korean only",
        }
        for language_code in SOVITS_LANGUAGES:
            label = tr(sovits_language_labels[language_code])
            self.sovits_text_lang_combo.addItem(label, language_code)
            self.sovits_prompt_lang_combo.addItem(label, language_code)
        self.sovits_text_lang_combo.setCurrentIndex(
            max(self.sovits_text_lang_combo.findData(sovits_text_lang), 0)
        )
        self.sovits_prompt_lang_combo.setCurrentIndex(
            max(self.sovits_prompt_lang_combo.findData(sovits_prompt_lang), 0)
        )
        self.sovits_ref_audio_label = QLabel(tr("Reference audio"))
        self.sovits_ref_audio_edit = QLineEdit(sovits_ref_audio_path)
        self.sovits_ref_audio_edit.setObjectName("voiceTestText")
        self.sovits_ref_audio_edit.setPlaceholderText(
            tr("Required for synthesis; cached while unchanged")
        )
        self.sovits_ref_audio_browse_button = QPushButton(tr("Browse…"))
        self.sovits_ref_audio_browse_button.setObjectName("voiceActionButton")
        sovits_ref_row = QHBoxLayout()
        sovits_ref_row.setSpacing(8)
        sovits_ref_row.addWidget(self.sovits_ref_audio_label)
        sovits_ref_row.addWidget(self.sovits_ref_audio_edit, 1)
        sovits_ref_row.addWidget(self.sovits_ref_audio_browse_button)
        playing_local_layout.addLayout(sovits_ref_row)

        self.sovits_prompt_text_label = QLabel(tr("Reference transcript"))
        self.sovits_prompt_text_edit = QLineEdit(sovits_prompt_text)
        self.sovits_prompt_text_edit.setObjectName("voiceTestText")
        self.sovits_prompt_text_edit.setPlaceholderText(tr("Optional exact transcript"))
        sovits_prompt_row = QHBoxLayout()
        sovits_prompt_row.setSpacing(8)
        sovits_prompt_row.addWidget(self.sovits_prompt_text_label)
        sovits_prompt_row.addWidget(self.sovits_prompt_text_edit, 1)
        playing_local_layout.addLayout(sovits_prompt_row)

        sovits_reference_language_row = QHBoxLayout()
        sovits_reference_language_row.setSpacing(8)
        sovits_reference_language_row.addWidget(self.sovits_prompt_lang_label)
        sovits_reference_language_row.addWidget(self.sovits_prompt_lang_combo, 1)
        playing_local_layout.addLayout(sovits_reference_language_row)

        self.sovits_output_title = QLabel(tr("Generated speech"))
        self.sovits_output_title.setObjectName("voiceFieldGroupTitle")
        self.sovits_output_description = QLabel(
            tr("Choose the language of ChatGPT replies sent to speech synthesis.")
        )
        self.sovits_output_description.setObjectName("settingsNote")
        self.sovits_output_description.setWordWrap(True)
        playing_local_layout.addWidget(self.sovits_output_title)
        playing_local_layout.addWidget(self.sovits_output_description)
        sovits_output_language_row = QHBoxLayout()
        sovits_output_language_row.setSpacing(8)
        sovits_output_language_row.addWidget(self.sovits_text_lang_label)
        sovits_output_language_row.addWidget(self.sovits_text_lang_combo, 1)
        playing_local_layout.addLayout(sovits_output_language_row)

        self.voice_test_text = QLineEdit()
        self.voice_test_text.setObjectName("voiceTestText")
        self.voice_test_text.setPlaceholderText(tr("Text to synthesize"))
        tts_action_row = QHBoxLayout()
        tts_action_row.setSpacing(8)
        self.voice_play_button = QPushButton(tr("Play text"))
        self.voice_play_button.setObjectName("voiceActionButton")
        tts_action_row.addWidget(self.voice_play_button)
        tts_action_row.addStretch()
        playing_local_layout.addWidget(self.voice_test_text)
        playing_local_layout.addLayout(tts_action_row)
        self.playing_progress = QProgressBar()
        self.playing_progress.setObjectName("voiceProgress")
        self.playing_progress.setRange(0, 100)
        self.playing_progress.hide()
        playing_local_layout.addWidget(self.playing_progress)
        self.playing_status = QLabel(
            tr("Select Check to validate the runtime and selected playback model.")
        )
        self.playing_status.setObjectName("voiceStatus")
        self.playing_status.setWordWrap(True)
        playing_local_layout.addWidget(self.playing_status)
        self.playing_install_log = QLabel()
        self.playing_install_log.setObjectName("voiceInstallLog")
        self.playing_install_log.setWordWrap(False)
        self.playing_install_log.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )
        self.playing_install_log.hide()
        playing_local_layout.addWidget(self.playing_install_log)
        playing_layout.addWidget(self.playing_local_card)
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
        self._build_pet_page()
        self._build_screenshots_page()
        self.screenshots_nav_button.clicked.connect(
            lambda checked: checked and self.settings_pages.setCurrentIndex(5)
        )
        self.pet_nav_button.clicked.connect(
            lambda checked: checked and self.settings_pages.setCurrentIndex(4)
        )
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
        self.stt_language_combo.currentIndexChanged.connect(self._stt_language_changed)
        self.stt_model_combo.currentIndexChanged.connect(self._stt_model_changed)
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
        self.voice_record_button.clicked.connect(self._toggle_voice_record_test)
        self.voice_play_button.clicked.connect(self._start_voice_play_test)
        self._stt_model_changed()
        self._sync_recording_controls()
        self._sync_playing_controls()
        self._connect_auto_save()
        self.sovits_ckpt_edit.textChanged.connect(self._sovits_weights_changed)
        self.sovits_pth_edit.textChanged.connect(self._sovits_weights_changed)
        self._sovits_weights_changed()
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
            QFrame#settingsCard QCheckBox {
                color: #f5f7ff;
                background: transparent;
                border: none;
                spacing: 10px;
                font-size: 14px;
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
            QLineEdit#hotkeyEdit,
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
            QLineEdit#hotkeyEdit:focus,
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

        self.setStyleSheet(self.styleSheet() + COMBOBOX_STYLE)
        self._translations = UiTranslations(self)
        localization.changed.connect(self._retranslate_models)

    def _retranslate_models(self) -> None:
        self._populate_stt_models(self.stt_model())
        self._update_stt_description()

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
        enabled = self.playing_backend() == "sovits"
        self.playing_local_card.setVisible(enabled)
        self.playing_install_button.hide()
        self.playing_pypi_mirror_combo.hide()
        self.voice_test_text.setText("Hello from GPT-SoVITS.")
        self._voice_status_checked["tts"] = False
        if enabled and self.settings_pages.currentIndex() == 3:
            self._check_voice_page_when_needed("tts")

    def _connect_auto_save(self) -> None:
        self.capture_cursor_checkbox.toggled.connect(
            lambda enabled: self._save_setting("capture_cursor", enabled)
        )
        self.language_combo.currentIndexChanged.connect(
            self._language_changed
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
        ):
            editor.keySequenceChanged.connect(
                lambda _sequence: self._save_hotkeys()
            )

    def _build_screenshots_page(self) -> None:
        page = QWidget()
        page.setObjectName("settingsPage")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)
        title = QLabel(tr("Screenshots"))
        title.setObjectName("settingsPageTitle")
        layout.addWidget(title)
        card = QFrame()
        card.setObjectName("settingsCard")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(20, 20, 20, 20)
        card_layout.setSpacing(14)
        self.capture_cursor_checkbox = QCheckBox(
            tr("Capture mouse cursor in window screenshots")
        )
        self.capture_cursor_checkbox.setChecked(
            bool(self.config["capture_cursor"]) if self.config is not None else True
        )
        card_layout.addWidget(self.capture_cursor_checkbox)
        note = QLabel(tr(
            "Include the visible mouse cursor when it overlaps the selected window."
        ))
        note.setObjectName("settingsNote")
        note.setWordWrap(True)
        card_layout.addWidget(note)
        layout.addWidget(card)
        layout.addStretch()
        self.settings_pages.addWidget(page)

    def _build_pet_page(self) -> None:
        page = QWidget()
        page.setObjectName("settingsPage")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        title = QLabel(tr("Pet"))
        title.setObjectName("settingsPageTitle")
        layout.addWidget(title)
        description = QLabel(tr("Choose a pet or download one from GitHub. Changes are saved and applied immediately."))
        description.setObjectName("settingsPageDescription")
        description.setWordWrap(True)
        layout.addWidget(description)
        pet_card = QFrame()
        pet_card.setObjectName("settingsCard")
        pet_layout = QVBoxLayout(pet_card)
        pet_layout.setContentsMargins(18, 18, 18, 18)
        pet_title = QLabel(tr("Choose a pet"))
        pet_title.setObjectName("settingsCardTitle")
        pet_layout.addWidget(pet_title)
        self.pet_list = QListWidget()
        self.pet_list.setAccessibleName(tr("Available pets"))
        self.pet_list.setIconSize(QSize(48, 52))
        self.pet_list.setSpacing(4)
        self.pet_list.setStyleSheet("""
            QListWidget { background: transparent; color: #f5f7ff; selection-color: #f5f7ff; border: none; outline: none; }
            QListWidget::item { padding: 8px 12px; border: 1px solid transparent; border-radius: 8px; }
            QListWidget::item:hover { background: #202d4b; }
            QListWidget::item:selected,
            QListWidget::item:selected:!active { color: #f5f7ff; background: #20445b; border-color: #4cc9f0; }
        """)
        values = self.config if self.config is not None else DEFAULT_CONFIG
        self._refresh_pet_list(str(values["pet_path"] or default_pet_path()))
        pet_layout.addWidget(self.pet_list)
        self._pet_download_thread: _PetDownloadThread | None = None
        download_row = QHBoxLayout()
        self.pet_url_edit = QLineEdit()
        self.pet_url_edit.setObjectName("voiceTestText")
        self.pet_url_edit.setAccessibleName(tr("Pet GitHub folder URL"))
        self.pet_url_edit.setPlaceholderText("https://github.com/owner/repo/tree/main/pets/name")
        self.pet_url_edit.setToolTip(tr("Paste a GitHub pet folder URL using /tree/ or /blob/."))
        self.pet_download_button = QPushButton(tr("Download pet"))
        self.pet_download_button.setObjectName("voiceActionButton")
        download_row.addWidget(self.pet_url_edit, 1)
        download_row.addWidget(self.pet_download_button)
        pet_layout.addLayout(download_row)
        self.pet_download_tip = QLabel(
            tr('Find pets at <a href="https://github.com/legeling/awesome-codex-pet" '
            'style="color: #4cc9f0;">awesome-codex-pet</a>. '
            'Copy a pet folder link and paste it above.')
        )
        self.pet_download_tip.setObjectName("settingsNote")
        self.pet_download_tip.setOpenExternalLinks(True)
        self.pet_download_tip.setWordWrap(True)
        pet_layout.addWidget(self.pet_download_tip)
        self.pet_download_status = QLabel(tr("Download pet.json and spritesheet.webp to download/pets, then switch to the pet."))
        self.pet_download_status.setObjectName("settingsNote")
        self.pet_download_status.setWordWrap(True)
        pet_layout.addWidget(self.pet_download_status)
        self.pet_download_button.clicked.connect(self._start_pet_download)
        self.pet_url_edit.returnPressed.connect(self._start_pet_download)
        layout.addWidget(pet_card)
        layout.addSpacing(18)
        card = QFrame()
        card.setObjectName("settingsCard")
        self.pet_idle_form = form = QFormLayout(card)
        form.setContentsMargins(18, 18, 18, 18)
        form.setSpacing(16)
        idle_title = QLabel(tr("Idle animation"))
        idle_title.setObjectName("settingsCardTitle")
        form.addRow(idle_title)
        self.pet_idle_combo = QComboBox()
        self.pet_idle_combo.setObjectName("voiceCombo")
        for label, mode in ((tr("Always play idle animation"), "always"),
                            (tr("Do not play idle animation"), "never"),
                            (tr("Play idle animation for a duration"), "timed")):
            self.pet_idle_combo.addItem(label, mode)
        self.pet_idle_combo.setCurrentIndex(self.pet_idle_combo.findData(values["pet_idle_mode"]))
        form.addRow(tr("Playback"), self.pet_idle_combo)
        self.pet_idle_seconds = QSpinBox()
        self.pet_idle_seconds.setRange(1, 3600)
        self.pet_idle_seconds.setSuffix(tr(" seconds"))
        self.pet_idle_seconds.setValue(int(values["pet_idle_seconds"]))
        self.pet_idle_seconds.setEnabled(self.pet_idle_combo.currentData() == "timed")
        self.pet_idle_seconds.setStyleSheet("QSpinBox { color: #f5f7ff; background: #18213b; padding: 8px; border: 1px solid #536080; border-radius: 6px; }")
        form.addRow(tr("Duration"), self.pet_idle_seconds)
        form.setRowVisible(self.pet_idle_seconds, self.pet_idle_combo.currentData() == "timed")
        layout.addWidget(card)
        self.pet_settings_note = QLabel(
            tr("Idle stops on a still pose. The timer restarts when the pet returns to idle. "
            "Looking and activity animations continue to work.")
        )
        self.pet_settings_note.setObjectName("settingsPageDescription")
        self.pet_settings_note.setWordWrap(True)
        layout.addWidget(self.pet_settings_note)
        layout.addStretch()
        self.settings_pages.addWidget(page)
        self.pet_list.currentRowChanged.connect(self._save_pet_settings)
        self.pet_idle_combo.currentIndexChanged.connect(self._save_pet_settings)
        self.pet_idle_seconds.valueChanged.connect(self._save_pet_settings)

    def _refresh_pet_list(self, selected: str) -> None:
        self.pet_list.blockSignals(True)
        self.pet_list.clear()
        selected_path = Path(selected).resolve()
        if selected_path.name == "pet.json":
            selected_path = selected_path.parent
        pets = available_pets()
        if not any(Path(path).resolve() == selected_path for _, path in pets):
            pets.append((selected_path.name, str(selected_path)))
        for label, path in pets:
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, path)
            item.setToolTip(path)
            item.setSizeHint(QSize(200, 70))
            try:
                probe = PetWidget(path)
                item.setIcon(QIcon(probe.sheet.copy(0, 0, 192, 208)))
                probe.deleteLater()
            except (OSError, ValueError, TypeError, KeyError):
                item.setIcon(QIcon(str(PET_ICON_PATH)))
            self.pet_list.addItem(item)
            if Path(path).resolve() == selected_path:
                self.pet_list.setCurrentItem(item)
        # Fit the visible rows rather than expanding into unused card space.
        # Longer collections remain scrollable.
        self.pet_list.setFixedHeight(min(260, max(78, self.pet_list.count() * 78)))
        self.pet_list.blockSignals(False)

    def _start_pet_download(self) -> None:
        if self._pet_download_thread is not None:
            return
        url = self.pet_url_edit.text().strip()
        try:
            pet_source(url)
        except ValueError as error:
            self.pet_download_status.setText(str(error))
            return
        self.pet_download_button.setEnabled(False)
        self.pet_url_edit.setEnabled(False)
        self.pet_download_status.setText(tr("Downloading and checking the pet…"))
        worker = _PetDownloadThread(url, self)
        self._pet_download_thread = worker
        worker.completed.connect(self._pet_download_completed)
        worker.finished.connect(self._pet_download_finished)
        worker.start()

    def _pet_download_completed(self, success: bool, message: str) -> None:
        if not success:
            self.pet_download_status.setText(tr("Could not download pet: {message}", message=message))
            return
        self._refresh_pet_list(message)
        self._save_pet_settings()
        self.pet_download_status.setText(tr("Pet ready and selected: {name}", name=Path(message).name))

    def _pet_download_finished(self) -> None:
        worker = self._pet_download_thread
        self._pet_download_thread = None
        self.pet_download_button.setEnabled(True)
        self.pet_url_edit.setEnabled(True)
        if worker is not None:
            worker.deleteLater()

    def _save_pet_settings(self) -> None:
        item = self.pet_list.currentItem()
        if item is None:
            return
        path = str(item.data(Qt.ItemDataRole.UserRole))
        mode = str(self.pet_idle_combo.currentData())
        seconds = self.pet_idle_seconds.value()
        self.pet_idle_seconds.setEnabled(mode == "timed")
        self.pet_idle_form.setRowVisible(self.pet_idle_seconds, mode == "timed")
        try:
            probe = PetWidget(path)
            probe.deleteLater()
        except (OSError, ValueError, TypeError, KeyError) as error:
            self.pet_settings_note.setText(tr("Could not load this pet: {error}", error=error))
            return
        self.pet_settings_note.setText(tr("Saved. Idle stops on a still pose; looking and activity animations remain enabled."))
        if self.config is not None:
            self.config.update({"pet_path": path, "pet_idle_mode": mode, "pet_idle_seconds": seconds})
        self.pet_settings_changed.emit(path, mode, seconds)

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
            HOTKEY_CONFIG_KEYS[name]: shortcut_text(sequence)
            for name, sequence in self.sequences().items()
        }
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

    def _populate_stt_models(self, selected: str) -> None:
        self.stt_model_combo.blockSignals(True)
        self.stt_model_combo.clear()
        for model in STT_MODELS.values():
            if self.stt_language() in model.supported_languages:
                size_mb = round(model.asset.size / 1024 / 1024)
                self.stt_model_combo.addItem(
                    f"[{tr(model.mode)}] {model.label} · {size_mb} MB", model.key
                )
        self.stt_model_combo.setCurrentIndex(max(self.stt_model_combo.findData(selected), 0))
        self.stt_model_combo.blockSignals(False)

    def _stt_language_changed(self) -> None:
        self._populate_stt_models(self.stt_model())
        self._stt_model_changed()
        if self.config is not None:
            self.config.update({"stt_language": self.stt_language(), "stt_model": self.stt_model()})

    def stt_language(self) -> str:
        return str(self.stt_language_combo.currentData() or "auto")

    def _stt_model_changed(self) -> None:
        self._update_stt_description()
        self.stt_test_result.clear()
        self._voice_status_checked["stt"] = False

    def _update_stt_description(self) -> None:
        model = STT_MODELS[self.stt_model()]
        self.stt_model_description.setText(tr(
            "{mode} · {accuracy} · 5-sec compute {compute} · model {size} · RAM {ram} · {best_for}",
            mode=tr(model.mode), accuracy=tr(model.accuracy), compute=tr(model.compute),
            size=model.model_size, ram=model.ram, best_for=tr(model.best_for),
        ))

    def _browse_sovits_installation(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            tr("Choose GPT-SoVITS installation"),
            self.sovits_installation_edit.text(),
        )
        if selected:
            self.sovits_installation_edit.setText(selected)

    def sovits_ckpt_path(self) -> str:
        return self.sovits_ckpt_edit.text().strip()

    def sovits_pth_path(self) -> str:
        return self.sovits_pth_edit.text().strip()

    def _browse_sovits_weight(self, extension: str) -> None:
        edit = getattr(self, f"sovits_{extension}_edit")
        selected, _ = QFileDialog.getOpenFileName(
            self, tr("Choose .{extension} model", extension=extension), edit.text(),
            tr("Model files (*.{extension})", extension=extension),
        )
        if selected:
            edit.setText(selected)

    def _sovits_weights_changed(self) -> None:
        pair = (self.sovits_ckpt_path(), self.sovits_pth_path())
        if pair != getattr(self, "_last_weight_inputs", None):
            self._last_weight_inputs = pair
            self._voice_status_checked["tts"] = False
        try:
            SovitsTtsProvider.validate_weights(*pair)
        except ValueError as error:
            self.sovits_weights_status.setText(tr("{error} Changes are not saved.", error=translate_message(str(error))))
            self.playing_check_button.setEnabled(False)
            self.voice_play_button.setEnabled(False)
            return
        self.sovits_weights_status.setText(tr("Both paths are optional. Leave both blank to use the installation’s configured models."))
        busy = self._voice_worker is not None or self._voice_test_running()
        self.playing_check_button.setEnabled(not busy)
        self.voice_play_button.setEnabled(not busy)
        if self.config is not None:
            self.config.update(sovits_ckpt_path=self.sovits_ckpt_path(), sovits_pth_path=self.sovits_pth_path())

    def _browse_sovits_ref_audio(self) -> None:
        selected, _filter = QFileDialog.getOpenFileName(
            self,
            tr("Choose GPT-SoVITS reference audio"),
            self.sovits_ref_audio_edit.text(),
            tr("Audio files (*.wav *.flac *.mp3);;All files (*)"),
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
        return self._voice_backend(model_type) == ("sherpa" if model_type == "stt" else "sovits")

    def _voice_provider(self, model_type: str) -> ModelProvider:
        if model_type == "stt":
            return self.stt_manager
        self.sovits_manager.configure(
            self.sovits_prompt_text(), self.sovits_prompt_lang(),
            self.sovits_ckpt_path(), self.sovits_pth_path(),
        )
        self.sovits_manager.model_status("tts", self.sovits_installation())
        return self.sovits_manager

    def _active_tts_model(self) -> str:
        return self.sovits_installation()

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
        if model_type == "tts":
            try:
                SovitsTtsProvider.validate_weights(self.sovits_ckpt_path(), self.sovits_pth_path())
            except ValueError:
                self._sovits_weights_changed()
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
            install_log.setText(tr("Waiting for installer output…"))
            install_log.setToolTip("")
        status.setProperty("error", False)
        status.setText(
            {
                "status": tr("Checking versions and file integrity…"),
                "install": tr("Installing the local voice runtime…"),
                "download": tr("Preparing verified model download…"),
            }[action]
        )
        self._refresh_voice_status_style(status)
        worker = _VoiceOperationThread(
            self._voice_provider(model_type),
            action,
            model_type,
            self.stt_model() if model_type == "stt" else self._active_tts_model(),
            self.pypi_mirror(),
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
        status.setText(tr("Cancelling the active install or download…"))
        progress.setRange(0, 0)
        self._voice_operation_log(tr("Cancellation requested…"))
        worker.cancel_operation()

    def _voice_operation_progress(
        self,
        message: str,
        percent: object,
    ) -> None:
        status, progress = self._voice_widgets(self._voice_operation_type)
        status.setText(translate_message(message))
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
            message = tr("Runtime: {runtime}\nModel: {model}",
                         runtime=translate_message(result["dependency_message"]),
                         model=translate_message(result["model_message"]))
        if not success and not cancelled:
            logger.error(f"Voice setup check failed: {message}")
        status, progress = self._voice_widgets(self._voice_operation_type)
        status.setProperty("error", not success and not cancelled)
        status.setText(translate_message(message))
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
            self.voice_record_button,
            self.voice_play_button,
            self.close_button,
        ):
            button.setEnabled(not busy)
        self.recording_backend_combo.setEnabled(not busy)
        self.playing_backend_combo.setEnabled(not busy)
        self.recording_pypi_mirror_combo.setEnabled(not busy)
        self.playing_pypi_mirror_combo.setEnabled(not busy)
        self.stt_language_combo.setEnabled(not busy)
        self.stt_model_combo.setEnabled(not busy)
        self.sovits_installation_edit.setEnabled(not busy)
        self.sovits_text_lang_combo.setEnabled(not busy)
        self.sovits_ref_audio_edit.setEnabled(not busy)
        self.sovits_prompt_text_edit.setEnabled(not busy)
        self.sovits_prompt_lang_combo.setEnabled(not busy)
        for widget in self.sovits_weight_widgets:
            widget.setEnabled(not busy)
        if not busy:
            self._sovits_weights_changed()
        self.voice_test_text.setEnabled(not busy)

    def _voice_test_running(self) -> bool:
        return (
            self._voice_record_thread is not None
            or self._voice_play_thread is not None
        )

    def _toggle_voice_record_test(self) -> None:
        if self._voice_record_thread is not None:
            self.voice_record_button.setText(tr("Transcribing…"))
            self.voice_record_button.setEnabled(False)
            self._voice_record_thread.stop_recording()
            return
        if self._voice_worker is not None or self._voice_play_thread is not None:
            return
        self._set_voice_busy(True)
        self.voice_record_button.setEnabled(True)
        self.voice_record_button.setText(tr("Starting microphone…"))
        self.stt_test_result.clear()
        self.recording_status.setProperty("error", False)
        self.recording_status.setText(tr("Starting the microphone…"))
        self._refresh_voice_status_style(self.recording_status)
        worker = _LocalDictationThread(
            LocalDictationSession(self.stt_manager, self.stt_model(), self.stt_language())
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
            tr("Stop recording") if streaming else tr("Stop & transcribe")
        )
        self.recording_status.setText(
            tr("Streaming transcription… text updates live.")
            if streaming
            else tr("Recording… select Stop when finished.")
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
        self.recording_status.setText(translate_message(message))
        self._refresh_voice_status_style(self.recording_status)
        self._set_voice_busy(False)

    def _voice_record_finished(self) -> None:
        worker = self._voice_record_thread
        self._voice_record_thread = None
        self.voice_record_button.setText(tr("Record microphone"))
        self._set_voice_busy(False)
        if worker is not None:
            worker.deleteLater()

    def _start_voice_play_test(self) -> None:
        if self._voice_worker is not None or self._voice_test_running():
            return
        try:
            SovitsTtsProvider.validate_weights(self.sovits_ckpt_path(), self.sovits_pth_path())
        except ValueError:
            self._sovits_weights_changed()
            return
        text = self.voice_test_text.text().strip()
        if not text:
            text = "Hello from GPT-SoVITS."
            self.voice_test_text.setText(text)
        self._set_voice_busy(True)
        self.playing_progress.show()
        self.playing_progress.setRange(0, 0)
        self.playing_status.setProperty("error", False)
        self.playing_status.setText(tr("Generating speech…"))
        self._refresh_voice_status_style(self.playing_status)
        worker = _LocalSpeechThread(
            self._voice_provider("tts"),
            self._active_tts_model(),
            text,
            self.sovits_ref_audio_path(),
            self.sovits_text_lang(),
        )
        self._voice_play_thread = worker
        worker.started.connect(lambda message: self.playing_status.setText(translate_message(message)))
        worker.completed.connect(self._voice_play_completed)
        worker.finished.connect(self._voice_play_finished)
        worker.start()

    def _voice_play_completed(self, success: bool, message: str) -> None:
        if not success:
            logger.error(f"Voice playback test failed: {message}")
        self.playing_status.setProperty("error", not success)
        self.playing_status.setText(translate_message(message))
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
    def _sequence_edit(sequence: QKeySequence | str) -> HotkeyEdit:
        return HotkeyEdit(sequence)

    def sequences(self) -> dict[str, str]:
        return {
            "hold": self.hold_microphone_edit.keySequence(),
            "hold_without_screenshot": (
                self.hold_without_screenshot_edit.keySequence()
            ),
        }

    def _language_changed(self) -> None:
        self._save_setting("language", self.language())
        localization.set_language(self.language())

    def language(self) -> str:
        return str(self.language_combo.currentData() or localization.language)

    def recording_backend(self) -> str:
        return str(self.recording_backend_combo.currentData() or "web")

    def playing_backend(self) -> str:
        return str(self.playing_backend_combo.currentData() or "web")

    def pypi_mirror(self) -> str:
        return str(
            self.recording_pypi_mirror_combo.currentData() or "default"
        )

    def stt_model(self) -> str:
        return str(
            self.stt_model_combo.currentData()
            or "zh_zipformer_ctc_int8_2025_07_03"
        )

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
        texts = [binding.text.casefold() for binding in bindings.values() if binding.text]
        if len(set(texts)) != len(texts):
            raise ValueError(tr("Each action must use a different hotkey"))
        return bindings

    def accept(self) -> None:
        if self._pet_download_thread is not None:
            self.pet_download_status.setText(tr("Please wait for the pet download to finish."))
            return
        if self._voice_worker is not None or self._voice_test_running():
            QMessageBox.information(
                self,
                tr("Voice setup is running"),
                tr("Wait for the current voice setup operation to finish."),
            )
            return
        try:
            self.bindings()
        except ValueError as error:
            QMessageBox.warning(self, tr("Invalid hotkey"), str(error))
            return
        super().accept()

    def reject(self) -> None:
        if self._pet_download_thread is not None:
            self.pet_download_status.setText(tr("Please wait for the pet download to finish."))
            return
        if self._voice_worker is not None:
            QMessageBox.information(
                self,
                tr("Voice setup is running"),
                tr("Wait for the current voice setup operation to finish."),
            )
            return
        super().reject()


class LinkTextEdit(QPlainTextEdit):
    """Selectable plain text with clickable HTTP(S) and Markdown links."""

    def _link_at(self, position: QPoint) -> str | None:
        offset = self.cursorForPosition(position).position()
        text = self.toPlainText()
        for match in re.finditer(r"\[([^\]]+)\]\((https?://[^\s)]+)\)|https?://[^\s<>]+", text):
            if match.start() <= offset < match.end():
                return (match.group(2) or match.group()).rstrip(".,;:!?)\"'")
        for label, url in getattr(self, "response_links", ()):
            for match in re.finditer(re.escape(label), text):
                if match.start() <= offset < match.end():
                    return url
        return None

    def mousePressEvent(self, event) -> None:  # noqa: N802
        self._link_press = event.position().toPoint()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        super().mouseReleaseEvent(event)
        position = event.position().toPoint()
        if (event.button() == Qt.MouseButton.LeftButton
                and (position - getattr(self, "_link_press", position)).manhattanLength() < 4
                and not self.textCursor().hasSelection()):
            link = self._link_at(position)
            if link:
                QDesktopServices.openUrl(QUrl(link))


class TranscriptEditor(LinkTextEdit):
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
        self.clear_button.setAccessibleName(tr("Delete text"))
        self.clear_button.setToolTip(tr("Delete text"))

        self.send_button = QPushButton(self)
        self.send_button.setObjectName("sendButton")
        self.send_button.setIcon(QIcon(str(SEND_ICON_PATH)))
        self.send_button.setIconSize(QSize(16, 16))
        self.send_button.setFixedSize(32, 32)
        self.send_button.setAccessibleName(tr("Send"))
        self.send_button.setToolTip(tr("Send"))

        self.send_without_screenshot_button = QPushButton(
            tr("No Screenshot"),
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
            tr("Send without screenshot")
        )
        self.send_without_screenshot_button.setToolTip(
            tr("Send the text without the selected screenshot")
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
        right = self.width() - margin
        for button in (
            self.send_button,
            self.send_without_screenshot_button,
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
            self.begin_composing(preserve_text=True)
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

    def begin_response(self) -> None:
        self.response_links = ()
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

    def begin_composing(self, *, preserve_text: bool = False) -> None:
        text = self.toPlainText() if preserve_text else ""
        if not preserve_text:
            self.response_links = ()
        self._response_mode = False
        self._response_complete = False
        self._full_response_text = ""
        self.setReadOnly(False)
        self.setPlainText(text)
        self._sync_action_visibility()

    def set_screenshot_selected(self, selected: bool) -> None:
        self._screenshot_selected = selected
        if selected:
            self.send_button.setText(tr("With Screenshot"))
            self.send_button.setFixedSize(144, 32)
            self.send_button.setAccessibleName(tr("Send with screenshot"))
            self.send_button.setToolTip(
                tr("Send the message with the selected screenshot")
            )
        else:
            self.send_button.setText("")
            self.send_button.setFixedSize(32, 32)
            self.send_button.setAccessibleName(tr("Send"))
            self.send_button.setToolTip(tr("Send the message"))
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
        self.viewport().update()

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
        visible_buttons = [
            button
            for button in (
                self.clear_button,
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


class ReplyDisplay(QTextBrowser):
    """Read-only HTML reply with links opened in a new browser tab."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._code_blocks: list[tuple[str, str]] = []
        self._copy_buttons: list[QPushButton] = []
        self._code_cards: list[QFrame] = []
        self._copy_timers: list[QTimer] = []
        self._code_positions: dict[int, int] = {}
        self.setOpenLinks(False)
        self.setOpenExternalLinks(False)
        self.anchorClicked.connect(self._open_link)
        self.document().setDefaultStyleSheet("""
            p { margin-top: 0; margin-bottom: 12px; }
            a { color: #66dcff; text-decoration: underline; }
            th { background-color: #243352; font-weight: bold; }
            td, th { padding: 6px; }
            pre { white-space: pre-wrap; }
            code { font-family: Consolas, monospace; }
            blockquote { margin-left: 16px; color: #c1cbe0; }
        """)
        self.viewport().setStyleSheet("background: transparent;")
        self.verticalScrollBar().valueChanged.connect(self._position_copy_buttons)
        self.horizontalScrollBar().valueChanged.connect(self._position_copy_buttons)
        self._copy_position_timer = QTimer(self)
        self._copy_position_timer.setSingleShot(True)
        self._copy_position_timer.timeout.connect(self._position_copy_buttons)
        self.document().documentLayout().documentSizeChanged.connect(
            lambda _size: self._copy_position_timer.start(0)
        )
        localization.changed.connect(self._translate_copy_controls)

    def _translate_copy_controls(self) -> None:
        for button, timer in zip(self._copy_buttons, self._copy_timers):
            self._set_copy_feedback(button, timer.isActive())

    @staticmethod
    def _set_copy_feedback(button: QPushButton, copied: bool) -> None:
        button.setText("")
        button.setIcon(QIcon(str(CHECK_ICON_PATH if copied else COPY_ICON_PATH)))
        label = tr("Copied!") if copied else tr("Copy code")
        button.setToolTip(label)
        button.setAccessibleName(label)

    def setHtml(self, html: str) -> None:  # noqa: N802
        formatted, self._code_blocks = reply_code_blocks(html, tr("Plain text"))
        super().setHtml(formatted)
        while len(self._copy_buttons) > len(self._code_blocks):
            self._copy_timers.pop().stop()
            button = self._copy_buttons.pop()
            button.hide()
            button.deleteLater()
            card = self._code_cards.pop()
            card.hide()
            card.deleteLater()
        while len(self._copy_buttons) < len(self._code_blocks):
            index = len(self._copy_buttons)
            card = QFrame(self)
            card.setObjectName("replyCodeCard")
            card.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
            card.setStyleSheet("background-color: #17223b; border: none; border-radius: 18px;")
            card.lower()
            self._code_cards.append(card)
            button = QPushButton(self.viewport())
            button.setObjectName("replyCodeCopyButton")
            button.setFixedSize(36, 36)
            button.setIconSize(QSize(22, 22))
            button.setAccessibleName(tr("Copy code"))
            button.setToolTip(tr("Copy code"))
            button.setStyleSheet("""
                QPushButton { padding: 0; min-width: 0; min-height: 0;
                    color: #f5f7ff; background: transparent;
                    border: none; border-radius: 10px; }
                QPushButton:hover { background: #344768; }
            """)
            button.clicked.connect(lambda _checked=False, index=index: self._copy_code(index))
            timer = QTimer(button)
            timer.setSingleShot(True)
            timer.timeout.connect(lambda button=button: self._set_copy_feedback(button, False))
            self._copy_buttons.append(button)
            self._copy_timers.append(timer)
        for button, timer in zip(self._copy_buttons, self._copy_timers):
            timer.stop()
            self._set_copy_feedback(button, False)
        self._code_positions = {}
        block = self.document().begin()
        while block.isValid():
            iterator = block.begin()
            while not iterator.atEnd():
                fragment = iterator.fragment()
                if fragment.isValid():
                    for name in fragment.charFormat().anchorNames():
                        if name.startswith("live-gpt-code-"):
                            self._code_positions.setdefault(int(name.removeprefix("live-gpt-code-")), fragment.position())
                iterator += 1
            block = block.next()
        self._position_copy_buttons()

    def _copy_code(self, index: int) -> None:
        if 0 <= index < len(self._code_blocks):
            QApplication.clipboard().setText(self._code_blocks[index][1])
            self._set_copy_feedback(self._copy_buttons[index], True)
            self._copy_timers[index].start(1500)

    def _position_copy_buttons(self, *_args: object) -> None:
        for index, button in enumerate(getattr(self, "_copy_buttons", ())):
            card = self._code_cards[index]
            position = self._code_positions.get(index)
            if position is None:
                button.hide()
                card.hide()
                continue
            cursor = QTextCursor(self.document())
            cursor.setPosition(position)
            table = cursor.currentTable()
            if table is None:
                button.hide()
                card.hide()
                continue
            bounds = self.document().documentLayout().frameBoundingRect(table)
            header_bounds = self.cursorRect(cursor)
            header_format = table.cellAt(0, 0).format().toTableCellFormat()
            last_cell = table.cellAt(table.rows() - 1, 0)
            code_end = self.cursorRect(last_cell.lastCursorPosition())
            code_format = last_cell.format().toTableCellFormat()
            left = header_bounds.left() - int(header_format.leftPadding())
            top = header_bounds.top() - int(header_format.topPadding())
            height = code_end.bottom() + 1 + int(code_format.bottomPadding()) - top
            code_bounds = QRect(left, top, int(bounds.width()), height)
            viewport_origin = self.viewport().pos()
            card_bounds = code_bounds.translated(viewport_origin)
            card.setGeometry(card_bounds)
            visible_bounds = self.viewport().geometry().translated(-card.pos())
            card.setMask(QRegion(card.rect()).intersected(QRegion(visible_bounds)))
            card.setVisible(self.viewport().geometry().intersects(card_bounds))
            x = min(code_bounds.right() - button.width() - 16,
                    self.viewport().width() - button.width() - 4)
            y = self.cursorRect(cursor).top() - 8
            button.move(max(4, x), y)
            button.setVisible(y + button.height() > 0 and y < self.viewport().height())
            button.raise_()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._position_copy_buttons()

    @staticmethod
    def _open_link(url: QUrl) -> None:
        if url.scheme().lower() in ("http", "https"):
            webbrowser.open_new_tab(url.toString())


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

    def __init__(self, pet_path: str = "") -> None:
        # A hidden owner keeps the Windows overlay out of the taskbar without
        # using Qt.Tool, which broadcast window pickers exclude.
        taskbar_owner = (
            QWidget(None, Qt.WindowType.Tool) if sys.platform == "win32" else None
        )
        super().__init__(taskbar_owner, Qt.WindowType.Window)
        self._taskbar_owner = taskbar_owner
        logger.debug("Creating overlay window")
        self._pet_response_pending = False
        self._pet_playing = False
        self._pet_listening = False
        self._pet_error = False
        self._input_collapsed_geometry: QRect | None = None
        self._fitting_hover_input = False
        self._fitting_subtitle = False
        self._position_locked = False
        self._resize_edges = Qt.Edges()
        self._resize_start_global: QPoint | None = None
        self._resize_start_geometry: QRect | None = None
        self._chrome_visible = False
        self._reveal_until_hovered = False
        self._focus_restorer = ForegroundWindowRestorer()
        self._auto_hide_enabled = False
        self._browser_connected = False
        self._debug_connected = False
        self._preferred_capture_source_key = ""
        self._preferred_chatgpt_url = ""
        self.setWindowTitle("Live GPT")
        self.setMinimumSize(760, 180)
        # Broadcast window pickers filter out Qt.Tool / WS_EX_TOOLWINDOW.
        self.setWindowFlags(
            Qt.WindowType.Window
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
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
        self._playback_input_timer = QTimer(self)
        self._playback_input_timer.setSingleShot(True)
        self._playback_input_timer.setInterval(5_000)
        self._playback_input_timer.timeout.connect(self._finish_playback_input_delay)
        self._subtitle_outside_timer = QTimer(self)
        self._subtitle_outside_timer.setInterval(75)
        self._subtitle_outside_timer.timeout.connect(
            self._collapse_subtitle_if_outside
        )

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
        self.chatgpt_tab_combo.setAccessibleName(tr("ChatGPT window"))
        self.chatgpt_tab_combo.setMinimumWidth(220)
        self.chatgpt_tab_combo.setMaximumWidth(300)
        self.chatgpt_tab_combo.addItem(tr("Looking for ChatGPT windows…"))
        self.chatgpt_tab_combo.setEnabled(False)
        self.chatgpt_tab_combo.currentIndexChanged.connect(
            self._chatgpt_tab_changed
        )
        title_layout.addWidget(self.chatgpt_tab_combo, 1)

        self.capture_source_combo = QComboBox()
        self.capture_source_combo.setObjectName("captureSourceCombo")
        self.capture_source_combo.setAccessibleName(tr("Screenshot source"))
        self.capture_source_combo.setMinimumWidth(150)
        self.capture_source_combo.setMaximumWidth(220)
        self.capture_source_combo.setToolTip(
            tr("Choose a desktop or visible window to attach when sending")
        )
        self.capture_source_combo.currentIndexChanged.connect(
            self._capture_source_changed
        )
        title_layout.addWidget(self.capture_source_combo, 1)

        self.remote_debugging_button = QPushButton(tr("Enable Debugging"))
        self.remote_debugging_button.setObjectName("remoteDebuggingButton")
        self.remote_debugging_button.setAccessibleName(
            tr("Open remote debugging settings")
        )
        self.remote_debugging_button.setToolTip(
            tr("Open the browser's remote debugging settings")
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
            tr("Open settings"),
        )
        self.configure_button.clicked.connect(self.configure_requested.emit)
        title_layout.addWidget(self.configure_button)

        self.lock_button = QPushButton()
        self.lock_button.setObjectName("lockButton")
        self.lock_button.setCheckable(True)
        self._configure_icon_button(
            self.lock_button,
            UNLOCK_ICON_PATH,
            tr("Lock overlay position"),
        )
        self.lock_button.toggled.connect(self._set_position_locked)
        title_layout.addWidget(self.lock_button)

        self.auto_hide_button = QPushButton()
        self.auto_hide_button.setObjectName("autoHideButton")
        self.auto_hide_button.setCheckable(True)
        self._configure_icon_button(
            self.auto_hide_button,
            AUTO_HIDE_ICON_PATH,
            tr("Enable auto-hide"),
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
            tr("Exit"),
        )
        self.exit_button.clicked.connect(self.exit_requested.emit)
        title_layout.addWidget(self.exit_button)

        self.transcript_area = TranscriptEditor()
        self.transcript_area.setObjectName("transcriptArea")
        self.transcript_area.setEnabled(False)
        self.transcript_area.set_hint(tr("Looking for ChatGPT windows…"))

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
        self.subtitle_full_text = ReplyDisplay()
        self.subtitle_full_text.setObjectName("subtitleFullText")
        self.subtitle_full_text.setReadOnly(True)
        self.subtitle_full_text.setMouseTracking(True)
        self.subtitle_full_text.hide()
        self._response_html = ""
        self._response_links: object = ()
        self._displayed_reply_content: tuple[str, str] | None = None
        self.reply_close_button = QPushButton()
        self.reply_close_button.setObjectName("replyCloseButton")
        self.reply_close_button.setIcon(QIcon(str(EXIT_ICON_PATH)))
        self.reply_close_button.setIconSize(QSize(16, 16))
        self.reply_close_button.setFixedSize(32, 32)
        self.reply_close_button.setAccessibleName(tr("Close reply"))
        self.reply_close_button.setToolTip(tr("Close reply"))
        self.reply_close_button.clicked.connect(self._close_reply)
        self.reply_close_button.hide()
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
        subtitle_layout.addWidget(self.reply_close_button, 0, Qt.AlignmentFlag.AlignRight)
        subtitle_layout.addWidget(self.subtitle_full_text, 1)
        self.subtitle_panel.hide()
        self._subtitle_hover_widgets = (
            self.subtitle_panel,
            self.subtitle_line_one,
            self.subtitle_line_two,
            self.subtitle_full_text,
            self.subtitle_full_text.viewport(),
            self.reply_close_button,
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
        self.microphone_button.setAccessibleName(tr("Hold to dictate"))
        self.microphone_button.setEnabled(False)
        self.microphone_button.setToolTip(
            tr("Press and hold to use ChatGPT dictation")
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
        self.error_banner = QLabel()
        self.error_banner.setObjectName("errorBanner")
        self.error_banner.setTextFormat(Qt.TextFormat.PlainText)
        self.error_banner.setWordWrap(True)
        self.error_banner.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.error_banner.hide()
        panel_layout.addWidget(self.error_banner)
        recording_layout = QHBoxLayout()
        recording_layout.setSpacing(16)
        try:
            self.pet = PetWidget(pet_path)
        except (OSError, ValueError, TypeError, KeyError) as error:
            logger.warning(f"Unable to load selected pet; using default: {error}")
            self.pet = PetWidget()
        recording_layout.addWidget(self.pet, 0, Qt.AlignmentFlag.AlignVCenter)
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
        self._content_opacities = []
        for widget in (self.transcript_area, self.subtitle_panel, self.dictation_panel, self.error_banner):
            effect = QGraphicsOpacityEffect(widget)
            widget.setGraphicsEffect(effect)
            self._content_opacities.append(effect)
        self.pet.pointer_tracked.connect(self._track_pointer)

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
            QLabel#errorBanner {
                color: #ff8999;
                background-color: rgba(120, 30, 48, 110);
                border: 1px solid rgba(255, 102, 122, 120);
                border-radius: 10px;
                padding: 10px 14px;
                font-size: 14px;
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
            QTextBrowser#subtitleFullText {
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
            QPushButton#replyCloseButton,
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
        self.setStyleSheet(self.styleSheet() + COMBOBOX_STYLE)
        self._set_chrome_visible(False)
        self._translations = UiTranslations(self)
        localization.changed.connect(self._retranslate_status)

    def _retranslate_status(self) -> None:
        if self._subtitle_mode_active and not self._subtitle_reading_started:
            self._set_subtitle_status(self._subtitle_status_text)
        state, separator, text = self.dictation_state_label.text().partition("\n\n")
        self.dictation_state_label.setText(translate_message(state) + separator + text)
        blocked = self.capture_source_combo.blockSignals(True)
        for index in range(self.capture_source_combo.count()):
            source = self.capture_source_combo.itemData(index)
            if isinstance(source, CaptureSource) and source.kind == "display":
                self.capture_source_combo.setItemText(index, translate_message(source.label))
        self.capture_source_combo.blockSignals(blocked)

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
            "idle": tr("Press and hold the microphone to dictate"),
            "recording": tr("ChatGPT is listening… release to finish"),
            "saved": tr("Dictation copied from ChatGPT"),
            "error": tr("Browser dictation unavailable"),
        }
        label = translate_message(message) if message else labels[state]
        self._pet_listening = state == "recording"
        self.set_status(label, error=state == "error")
        self.microphone_button.setProperty("recordingState", state)
        self.microphone_button.setAccessibleName(label)
        self.microphone_button.setToolTip(label)
        style = self.microphone_button.style()
        style.unpolish(self.microphone_button)
        style.polish(self.microphone_button)
        self.microphone_button.update()
        self._track_pointer(QCursor.pos())

    def set_status(self, message: str, *, error: bool = False) -> None:
        message = translate_message(message)
        self.transcript_area.set_hint(message, error=error)
        self.error_banner.setText(message if error else "")
        self.error_banner.setVisible(error)
        self._pet_error = error
        self._refresh_pet()

    def _refresh_pet(self) -> None:
        state = (
            "failed" if self._pet_error else
            "waiting" if self._pet_listening else
            "review" if self._pet_playing else
            "running" if self._pet_response_pending else "idle"
        )
        self.pet.set_state(state)
        self._track_pointer(QCursor.pos())

    def set_transcript(self, text: str) -> None:
        self.transcript_area.setPlainText(text)
        cursor = self.transcript_area.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.transcript_area.setTextCursor(cursor)

    def set_chatgpt_tabs(self, tabs: list[dict[str, str]]) -> None:
        was_connected = self._browser_connected
        self._browser_connected = bool(tabs)
        selected_id = self.chatgpt_tab_combo.currentData()
        self.chatgpt_tab_combo.blockSignals(True)
        self.chatgpt_tab_combo.clear()
        if not tabs:
            self.chatgpt_tab_combo.addItem(tr("No ChatGPT tabs open"))
            self.chatgpt_tab_combo.setEnabled(False)
            self.remote_debugging_button.setVisible(not self._debug_connected)
            self.microphone_button.setEnabled(False)
            self.transcript_area.setEnabled(False)
            self._show_browser_connection_hint()
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
            self.set_status(tr("Hold the microphone or enter a message"))
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
        if not self._browser_connected:
            self._auto_hide_timer.stop()
            if was_connected and self.isHidden():
                self.showNormal()
        elif not was_connected:
            self.schedule_auto_hide()
        self._track_pointer(QCursor.pos())
        self.chatgpt_connection_changed.emit(bool(tabs))
        if tabs:
            self._activate_chatgpt_tab(
                self.chatgpt_tab_combo.currentIndex(),
                save_preference=not bool(self._preferred_chatgpt_url),
            )

    def set_debug_connection(self, connected: bool) -> None:
        self._debug_connected = connected
        if not self._browser_connected:
            self.remote_debugging_button.setVisible(not connected)
            self._show_browser_connection_hint()

    def _show_browser_connection_hint(self) -> None:
        if self._debug_connected:
            self.chatgpt_tab_combo.setItemText(0, tr("Debugger connected — open ChatGPT"))
        message = (
            tr("Debugger connected. No ChatGPT tab is open. Open chatgpt.com in this browser; "
            "Live GPT will detect the tab automatically.")
            if self._debug_connected else
            tr("Connect to a ChatGPT window to begin")
        )
        self.chatgpt_tab_combo.setToolTip(message)
        self.set_status(message)

    def set_browser_status(self, status: str) -> None:
        # Connection state takes precedence over queued setup/retry messages.
        # The explicit disconnect signal clears this state on a real disconnect.
        if self._debug_connected and not self._browser_connected:
            self._show_browser_connection_hint()
            return
        self.chatgpt_tab_combo.setToolTip(translate_message(status))
        status_lower = status.casefold()
        self.set_status(
            status,
            error=any(
                word in status_lower
                for word in ("not installed", "disconnected", "unable", "could not")
            ),
        )
        if not self.chatgpt_tab_combo.isEnabled():
            self.chatgpt_tab_combo.setItemText(0, translate_message(status))

    def set_response_links(self, links: object) -> None:
        self.transcript_area.response_links = links
        if self._subtitle_dismissed:
            return
        self._response_links = links
        self._update_expanded_subtitle()

    def set_response_html(self, reply_html: str) -> None:
        if self._subtitle_dismissed:
            return
        self._response_html = reply_html
        self._update_expanded_subtitle()

    def set_capture_sources(self, sources: list[CaptureSource]) -> None:
        selected = self.capture_source_combo.currentData()
        selected_key = (
            selected.key
            if isinstance(selected, CaptureSource)
            else self._preferred_capture_source_key
        )
        self.capture_source_combo.blockSignals(True)
        self.capture_source_combo.clear()
        selected_index = next(
            (index for index, source in enumerate(sources) if source.kind == "display"),
            0,
        )
        for source in sources:
            self.capture_source_combo.addItem(
                translate_message(source.label) if source.kind == "display" else source.label, source
            )
        if selected_key:
            for index in range(self.capture_source_combo.count()):
                source = self.capture_source_combo.itemData(index)
                if isinstance(source, CaptureSource) and source.key == selected_key:
                    selected_index = index
                    break
        self.capture_source_combo.setCurrentIndex(selected_index)
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
            self._pet_response_pending = False
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
        self._set_subtitle_status(tr("Waiting for ChatGPT…"))
        self.set_status(tr("Waiting for ChatGPT…"))

    def begin_response_display(
        self,
        sent_text: str = "",
        message: str = tr("Sending to ChatGPT…"),
    ) -> None:
        self._collapse_hover_input()
        self._playback_input_timer.stop()
        self._pet_response_pending = True
        self._pet_playing = False
        self._pet_listening = False
        self._collapse_subtitle()
        self._subtitle_mode_active = True
        self._subtitle_dismissed = False
        self._subtitle_reading_active = False
        self._subtitle_reading_started = False
        self._subtitle_hover_origin = None
        self._sent_message_text = " ".join(sent_text.split())
        self._reading_full_text = ""
        self._response_html = ""
        self._displayed_reply_content = None
        self._response_links = ()
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
            self.set_status(tr("Reading aloud…"))
            return
        self.set_status(status)
        self._reading_fraction = 0.0
        self._set_subtitle_status(status)
        if text:
            self._update_expanded_subtitle()

    def set_response_finished(self, success: bool, message: str) -> None:
        self._pet_response_pending = False
        self._pet_error = not success
        self._refresh_pet()
        if self._subtitle_dismissed:
            if not success:
                self.show_for_auto_hide()
                self.set_status(message, error=True)
            return
        self.show_for_auto_hide()
        self.transcript_area.finish_response()
        self.microphone_button.setVisible(True)
        if self._subtitle_reading_active and success:
            self.set_status(tr("Reading aloud…"))
            return
        self.set_status(message, error=not success)
        if self._subtitle_mode_active:
            self._set_subtitle_status(message)
        self.schedule_auto_hide(5_000)

    def begin_reading(self, message: str) -> None:
        self._playback_input_timer.stop()
        self._pet_playing = True
        self._pet_error = False
        self._refresh_pet()
        if self._subtitle_dismissed:
            return
        self.show_for_auto_hide()
        self.transcript_area.begin_reading()
        self._subtitle_reading_active = True
        self._subtitle_reading_started = True
        self._subtitle_status_text = message
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
        self._update_expanded_subtitle()
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
        self.set_status(tr("Reading aloud…"))

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
        self._pet_playing = False
        self._pet_response_pending = False
        self._pet_error = not success
        if not self._subtitle_dismissed:
            self._playback_input_timer.start()
        self._refresh_pet()
        if self._subtitle_dismissed:
            return
        self.transcript_area.finish_reading()
        self._subtitle_reading_active = False
        self.microphone_button.setVisible(True)
        self.set_status(message, error=not success)
        self.schedule_auto_hide(5_000)

    def _finish_playback_input_delay(self) -> None:
        self._playback_input_timer.stop()
        self._track_pointer(QCursor.pos())

    def _set_subtitle_status(self, message: str) -> None:
        message = translate_message(message)
        self._subtitle_status_text = message
        if self._subtitle_mode_active and not self._subtitle_reading_started:
            lines = self._subtitle_lines()
            preview = lines[0] if lines else self._sent_message_text
            self.subtitle_line_one.setText(
                "" if " ".join(preview.split()) == " ".join(message.split()) else preview
            )
            self.subtitle_line_two.setText(message)
        else:
            self.subtitle_line_one.setText(message)
            self.subtitle_line_two.clear()
        if not self._subtitle_expanded:
            self._render_full_reply()

    def _render_full_reply(self) -> bool:
        text = self._reading_full_text or self._subtitle_status_text
        reply_html = reply_html_with_links(self._response_html, text, self._response_links, tr("Sources"))
        content = (reply_html, "")
        if content == self._displayed_reply_content:
            return False
        self._displayed_reply_content = content
        self.subtitle_full_text.setHtml(reply_html)
        return True

    def _update_expanded_subtitle(self) -> None:
        if not self._subtitle_expanded:
            return
        scrollbar = self.subtitle_full_text.verticalScrollBar()
        previous_value = scrollbar.value()
        was_at_bottom = previous_value >= scrollbar.maximum() - 1
        if not self._render_full_reply():
            return
        if was_at_bottom and self._subtitle_reading_started:
            scrollbar.setValue(scrollbar.maximum())
        else:
            scrollbar.setValue(min(previous_value, scrollbar.maximum()))
        self._fit_expanded_subtitle_height()

    def _expand_subtitle(self) -> None:
        if (not self._subtitle_mode_active or self._subtitle_expanded
                or self.pet._drag_offset is not None):
            return
        self._subtitle_expanded = True
        self._subtitle_outside_timer.start()
        self._subtitle_collapsed_geometry = QRect(self.geometry())
        self._render_full_reply()
        self.subtitle_line_one.hide()
        self.subtitle_line_two.hide()
        self.subtitle_full_text.show()
        self.reply_close_button.show()
        self.subtitle_panel.layout().activate()
        self._fit_expanded_subtitle_height()

    def _fit_expanded_subtitle_height(self) -> None:
        if (
            not self._subtitle_expanded
            or self._subtitle_collapsed_geometry is None
            or self.pet._drag_offset is not None
        ):
            return
        available = self.screen().availableGeometry()
        collapsed = self._subtitle_collapsed_geometry
        content_width = max(self.subtitle_full_text.viewport().width(), 1)
        metrics = self.subtitle_full_text.fontMetrics()
        document = self.subtitle_full_text.document().clone()
        document.setTextWidth(content_width)
        text_height = int(document.size().height())
        document.deleteLater()
        panel_height = max(text_height, metrics.lineSpacing()) + 32 + self.reply_close_button.height()
        fixed_chrome_height = max(
            self.height() - self.subtitle_panel.height(),
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
        self._fitting_subtitle = True
        try:
            self.setGeometry(geometry)
        finally:
            self._fitting_subtitle = False

    def _collapse_subtitle(self) -> None:
        self._subtitle_outside_timer.stop()
        if not self._subtitle_expanded:
            return
        self._subtitle_expanded = False
        self.subtitle_full_text.hide()
        self.reply_close_button.hide()
        self.subtitle_line_one.show()
        self.subtitle_line_two.show()
        if self._subtitle_collapsed_geometry is not None:
            self._fitting_subtitle = True
            try:
                self.setGeometry(self._subtitle_collapsed_geometry)
                if self._position_locked:
                    self.setFixedSize(self._subtitle_collapsed_geometry.size())
            finally:
                self._fitting_subtitle = False
        self._subtitle_collapsed_geometry = None
        if self._subtitle_reading_started and self._reading_full_text:
            self._render_reading_subtitle(resized=True)
        else:
            self._set_subtitle_status(self._subtitle_status_text)

    def _collapse_subtitle_if_outside(self) -> None:
        if not self._subtitle_expanded or self.pet._drag_offset is not None:
            return
        if not self._pointer_hover_bounds().contains(QCursor.pos()):
            self._collapse_subtitle()
            self.schedule_auto_hide(5_000)

    def _close_reply(self) -> None:
        self.dismiss_subtitle_mode(preserve_text=False)
        self.transcript_area.setFocus()

    def dismiss_subtitle_mode(self, *, preserve_text: bool = True) -> bool:
        if not self._subtitle_mode_active:
            return False
        self._collapse_subtitle()
        self._subtitle_mode_active = False
        self._subtitle_dismissed = True
        self._subtitle_reading_active = False
        self._subtitle_reading_started = False
        self.subtitle_panel.hide()
        self.dictation_panel.hide()
        response_text = self._reading_full_text or self.transcript_area.toPlainText()
        self.transcript_area.begin_composing()
        if preserve_text:
            self.transcript_area.setPlainText(response_text)
        self.transcript_area.show()
        self.microphone_button.setVisible(True)
        self.set_status(tr("Hold the microphone or enter a message"))
        self.schedule_auto_hide()
        return True

    def begin_dictation_waiting(self) -> None:
        self._playback_input_timer.stop()
        self._pet_listening = True
        self._pet_playing = False
        self._pet_response_pending = False
        self._pet_error = False
        self._refresh_pet()
        self.dismiss_subtitle_mode()
        self.show_for_auto_hide()
        self.subtitle_panel.hide()
        self.transcript_area.hide()
        self.dictation_state_label.setText(
            tr("Waiting for the browser to start listening…")
        )
        self.dictation_panel.show()
        self._track_pointer(QCursor.pos())

    def prepare_for_dictation(self) -> None:
        self._playback_input_timer.stop()
        self.dismiss_subtitle_mode()
        self._subtitle_dismissed = True
        self._reading_full_text = ""
        self._pet_playing = False
        self._pet_response_pending = False
        self.clear_transcript()

    def set_dictation_listening(self) -> None:
        self._pet_listening = True
        self._pet_error = False
        self._refresh_pet()
        self.dictation_state_label.setText(tr("Listening…"))
        self._track_pointer(QCursor.pos())

    def set_dictation_partial(
        self,
        text: str,
        *,
        finishing: bool = False,
    ) -> None:
        if text.strip():
            state = tr("Finishing dictation…") if finishing else tr("Listening…")
            self.dictation_state_label.setText(f"{state}\n\n{text}")

    def set_dictation_finishing(self) -> None:
        self.dictation_state_label.setText(tr("Finishing dictation…"))

    def set_dictation_cancelling(self) -> None:
        self.dictation_state_label.setText(tr("Cancelling short dictation…"))

    def end_dictation_display(self) -> None:
        self._pet_listening = False
        self._refresh_pet()
        self.dictation_panel.hide()
        self.transcript_area.show()
        self._track_pointer(QCursor.pos())

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
        self.transcript_area.begin_composing()
        self.set_status(tr("Text cleared"))

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
            self.set_status(tr("Wait for the current response to finish"))
            return
        text = self.transcript_area.toPlainText().strip()
        capture_source = (
            self.capture_source_combo.currentData()
            if include_screenshot
            else None
        )
        if not text and capture_source is None:
            self.set_status(
                tr("Enter text or select a screenshot before sending"),
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

    def remember_foreground_app(self) -> None:
        self._focus_restorer.remember_foreground()

    def ensure_on_screen(self) -> None:
        """Recover geometry after monitor, resolution, or scaling changes."""
        screens = QApplication.screens()
        if not screens:
            return
        geometry = self.geometry()

        def overlap(screen) -> int:
            intersection = screen.availableGeometry().intersected(geometry)
            return max(0, intersection.width()) * max(0, intersection.height())

        screen = max(screens, key=overlap)
        if overlap(screen) == 0:
            screen = QApplication.primaryScreen() or screen
        bounds = screen.availableGeometry()
        minimum_width = 760 if self._position_locked else self.minimumWidth()
        minimum_height = 180 if self._position_locked else self.minimumHeight()

        def clamp(rect: QRect) -> QRect:
            result = QRect(rect)
            result.setWidth(max(minimum_width, min(rect.width(), bounds.width())))
            result.setHeight(max(minimum_height, min(rect.height(), bounds.height())))
            result.moveLeft(max(bounds.left(), min(
                rect.left(), bounds.right() - result.width() + 1,
            )))
            result.moveTop(max(bounds.top(), min(
                rect.top(), bounds.bottom() - result.height() + 1,
            )))
            return result

        recovered = clamp(geometry)
        # Hover/subtitle collapse must not restore a stale offscreen position.
        for attribute in ("_input_collapsed_geometry", "_subtitle_collapsed_geometry"):
            collapsed = getattr(self, attribute)
            if collapsed is not None:
                setattr(self, attribute, clamp(collapsed))
        if recovered == geometry:
            return
        logger.warning(f"Recovering offscreen overlay: {geometry.getRect()} -> {recovered.getRect()}")
        if self._position_locked:
            # A position lock must not prevent recovery on a smaller display.
            self.setFixedSize(recovered.size())
        self._fitting_hover_input = True
        try:
            self.setGeometry(recovered)
        finally:
            self._fitting_hover_input = False
        self.geometry_changed.emit(QRect(
            self._subtitle_collapsed_geometry or self._input_collapsed_geometry or recovered
        ))

    def reveal_controls(self) -> None:
        # A tray click happens outside the overlay. Keep controls visible until
        # the user reaches them instead of fading on the next pointer tick.
        self._reveal_until_hovered = True
        self._set_chrome_visible(True)

    def _restore_previous_focus(self) -> None:
        if self._focus_restorer.restore_previous():
            logger.debug("Restored focus to the previous application")
        else:
            logger.debug("No external application was available to restore")

    @property
    def auto_hide_enabled(self) -> bool:
        return self._auto_hide_enabled

    def disable_auto_hide(self) -> None:
        if self.auto_hide_button.isChecked():
            self.auto_hide_button.setChecked(False)
        else:
            self._set_auto_hide_enabled(False)

    def _set_auto_hide_enabled(self, enabled: bool) -> None:
        self._auto_hide_enabled = enabled
        label = tr("Disable auto-hide") if enabled else tr("Enable auto-hide")
        self.auto_hide_button.setAccessibleName(label)
        self.auto_hide_button.setToolTip(
            tr("{label}; the overlay appears for dictation and ChatGPT replies", label=label)
        )
        if enabled:
            self.schedule_auto_hide()
        else:
            self._auto_hide_timer.stop()

    def _can_auto_hide(self) -> bool:
        if any(not notice.isHidden() for notice in self.findChildren(QMessageBox)):
            return False
        if not self._browser_connected or not self._auto_hide_enabled:
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
        self.pet.set_locked(locked)
        self._end_border_resize()
        if locked:
            self.setFixedSize(self.size())
            icon_path = LOCK_ICON_PATH
            label = tr("Unlock overlay position")
        else:
            self.setMinimumSize(760, 180)
            self.setMaximumSize(16_777_215, 16_777_215)
            icon_path = UNLOCK_ICON_PATH
            label = tr("Lock overlay position")
        self.lock_button.setIcon(QIcon(str(icon_path)))
        self.lock_button.setAccessibleName(label)
        self.lock_button.setToolTip(label)

    def _set_chrome_visible(self, visible: bool) -> None:
        visible = (
            visible or self._reveal_until_hovered or not self._browser_connected
            or any(not notice.isHidden() for notice in self.findChildren(QMessageBox))
            or bool(self._resize_edges) or self.pet._drag_offset is not None
        )
        content_visible = (
            visible or self._pet_listening or self._pet_response_pending
            or self._pet_playing or self._playback_input_timer.isActive()
        )
        if (visible == self._chrome_visible
                and content_visible == getattr(self, "_content_visible", None)
                and hasattr(self, "_chrome_initialized")):
            return
        self._chrome_initialized = True
        self._chrome_visible = visible
        self._content_visible = content_visible
        opacity = 1.0 if visible else 0.0
        self._title_opacity.setOpacity(opacity)
        self._microphone_opacity.setOpacity(opacity)
        for effect in self._content_opacities:
            effect.setOpacity(1.0 if content_visible else 0.0)
            # Render visible editors directly: an enabled opacity effect can
            # retain stale viewport pixels when only placeholder text changes.
            effect.setEnabled(not content_visible)
        self.panel.setProperty("chromeVisible", visible)
        self._resize_surface.setProperty("chromeVisible", visible)
        for widget in (self.panel, self._resize_surface):
            style = widget.style()
            style.unpolish(widget)
            style.polish(widget)
            widget.update()

    def enterEvent(self, event) -> None:  # noqa: N802
        self._track_pointer(QCursor.pos())
        super().enterEvent(event)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if hasattr(self, "_reading_full_text"):
            if self._subtitle_reading_started and self._reading_full_text:
                self._render_reading_subtitle(resized=True)
            elif self._subtitle_mode_active:
                self._set_subtitle_status(self._subtitle_status_text)
        if not getattr(self, "_subtitle_mode_active", False) and self._input_collapsed_geometry is None:
            self.geometry_changed.emit(QRect(self.geometry()))

    def moveEvent(self, event) -> None:  # noqa: N802
        super().moveEvent(event)
        if self._fitting_hover_input or self._fitting_subtitle:
            return
        delta = event.pos() - event.oldPos()
        if self._subtitle_collapsed_geometry is not None:
            self._subtitle_collapsed_geometry.translate(delta)
        if self._input_collapsed_geometry is not None and not self._fitting_hover_input:
            self._input_collapsed_geometry.translate(delta)
        self.geometry_changed.emit(QRect(
            self._subtitle_collapsed_geometry or self._input_collapsed_geometry or self.geometry()
        ))

    def leaveEvent(self, event) -> None:  # noqa: N802
        QTimer.singleShot(0, self._hide_chrome_if_outside)
        super().leaveEvent(event)

    def _hide_chrome_if_outside(self) -> None:
        self._track_pointer(QCursor.pos())

    def _pointer_hover_bounds(self) -> QRect:
        bounds = self.frameGeometry()
        # Keep the original hover area reachable if expansion hits a screen
        # edge and moves the window. Expansion and collapse share this region.
        for collapsed in (
            self._subtitle_collapsed_geometry, self._input_collapsed_geometry,
        ):
            if collapsed is not None:
                bounds = bounds.united(collapsed)
        # Add 5% of the overlay dimensions on each side for border access.
        margin_x = max(1, round(bounds.width() * 0.05))
        margin_y = max(1, round(bounds.height() * 0.05))
        return bounds.adjusted(-margin_x, -margin_y, margin_x, margin_y)

    def _pointer_over_visible_content(self, position: QPoint) -> bool:
        widgets = [self.pet]
        if self._content_visible:
            widgets.extend((
                self.transcript_area, self.subtitle_panel, self.dictation_panel,
            ))
        return any(
            widget.isVisible()
            and widget.rect().contains(widget.mapFromGlobal(position))
            for widget in widgets
        )

    def _track_pointer(self, position: QPoint) -> None:
        # Reveal from visible content; once open, retain the full overlay and
        # its existing margin as the area that keeps controls visible.
        hovered = (
            self._pointer_hover_bounds().contains(position)
            if self._chrome_visible
            else self._pointer_over_visible_content(position)
        )
        if hovered:
            self._reveal_until_hovered = False
        self._set_chrome_visible(hovered)
        if (self._fitting_hover_input or self._fitting_subtitle
                or self._resize_edges or self.pet._drag_offset is not None):
            return
        if hovered:
            if self._subtitle_mode_active:
                self._expand_subtitle()
            else:
                self._fit_hover_input()
        else:
            self._collapse_hover_input()

    def _fit_hover_input(self) -> None:
        if self._input_collapsed_geometry is None:
            self._input_collapsed_geometry = QRect(self.geometry())
        if not self.dictation_panel.isHidden():
            widget = self.dictation_panel
            text = self.dictation_state_label.text()
            metrics = self.dictation_state_label.fontMetrics()
            width = max(self.dictation_state_label.width(), 1)
        else:
            widget = self.transcript_area
            text = widget.toPlainText() or widget.placeholderText()
            metrics = widget.fontMetrics()
            width = max(widget.viewport().width() - 12, 1)
        text_height = metrics.boundingRect(
            QRect(0, 0, width, 16_777_215), Qt.TextFlag.TextWordWrap, text + "\n"
        ).height()
        base = self._input_collapsed_geometry
        available = self.screen().availableGeometry()
        height = min(max(base.height(), self.height() - widget.height() + text_height + 28), available.height())
        target = QRect(base)
        target.setTop(max(available.top(), min(base.bottom() - height + 1, available.bottom() - height + 1)))
        target.setHeight(height)
        self._fitting_hover_input = True
        try:
            if self._position_locked:
                self.setMinimumSize(760, 180)
                self.setMaximumSize(16_777_215, 16_777_215)
            self.setGeometry(target)
        finally:
            self._fitting_hover_input = False

    def _collapse_hover_input(self) -> None:
        base = self._input_collapsed_geometry
        if base is None:
            return
        self._fitting_hover_input = True
        try:
            self.setGeometry(base)
            if self._position_locked:
                self.setFixedSize(base.size())
        finally:
            self._input_collapsed_geometry = None
            self._fitting_hover_input = False

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
                ).manhattanLength() >= 4 and self._chrome_visible:
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
                if self._subtitle_expanded:
                    return False
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
        self._input_collapsed_geometry = None
        self._resize_edges = edges
        self._resize_start_global = global_position
        self._resize_start_geometry = self.geometry()
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
        if was_resizing:
            self._track_pointer(QCursor.pos())

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
        if not self._position_locked and event.button() == Qt.MouseButton.LeftButton:
            self.pet.begin_drag(event.globalPosition().toPoint())
            self._set_chrome_visible(True)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self.pet._drag_offset is not None and event.buttons() & Qt.MouseButton.LeftButton:
            self.pet.drag_to(event.globalPosition().toPoint())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton and self.pet._drag_offset is not None:
            self.pet._finish_drag()
            self._track_pointer(event.globalPosition().toPoint())
            event.accept()
            return
        super().mouseReleaseEvent(event)


class TrayController:
    def __init__(self, application: QApplication) -> None:
        logger.debug("Creating tray controller")
        self.application = application
        self.icon = QIcon(str(ICON_PATH))
        self.config = Config()
        localization.set_language(str(self.config["language"]))
        self.stt_manager = SherpaSttProvider()
        self.sovits_manager = SovitsTtsProvider()
        self._local_dictation_thread: _LocalDictationThread | None = None
        self._local_speech_thread: (
            _LocalSpeechThread | _QueuedLocalSpeechThread | None
        ) = None
        self._local_voice_longest_text = ""
        self._local_voice_queued_sentences: list[str] = []
        self._local_voice_interrupted = False
        self._local_voice_thinking_announced = False
        self._local_voice_error: str | None = None
        self._pending_local_announcements: list[tuple[str, bool]] = []
        self._stt_preload_thread: threading.Thread | None = None
        self._local_tts_preload_thread: threading.Thread | None = None
        self._migrate_legacy_hotkeys()
        self.window = OverlayWindow(str(self.config["pet_path"]))
        self.window.pet.set_idle_behavior(str(self.config["pet_idle_mode"]), int(self.config["pet_idle_seconds"]))
        self.window.set_preferred_capture_source(
            str(self.config["capture_source"])
        )
        self.window.set_preferred_chatgpt_window(
            str(self.config["chatgpt_window"])
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
        self._short_voice_text: str | None = None
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
            parent=self.window,
            hold_without_screenshot=bindings["hold_without_screenshot"],
            enabled=self._hotkey_enabled_states(),
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
        self.browser_monitor.response_links_changed.connect(self.window.set_response_links)
        self.browser_monitor.response_html_changed.connect(self.window.set_response_html)
        self.browser_monitor.debug_connection_changed.connect(self.window.set_debug_connection)
        self.browser_monitor.status_changed.connect(
            self.window.set_browser_status
        )
        self.browser_monitor.send_finished.connect(
            self._on_send_finished
        )
        self.browser_monitor.response_changed.connect(
            self._on_response_update
        )
        self.browser_monitor.response_finished.connect(
            self._on_response_finished
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
        self.browser_monitor.local_voice_announcement.connect(
            self._speak_local_announcement
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
                send_on_finish=True,
                include_screenshot=True,
                initial_hold_seconds=self.hotkey_monitor.hold_delay_seconds,
            )
        )
        self.hotkey_monitor.hold_released.connect(self.finish_dictation)
        self.hotkey_monitor.hold_without_screenshot_pressed.connect(
            lambda: self.start_dictation(
                send_on_finish=True,
                include_screenshot=False,
                initial_hold_seconds=self.hotkey_monitor.hold_delay_seconds,
            )
        )
        self.hotkey_monitor.hold_without_screenshot_released.connect(
            self.finish_dictation
        )

        self._position_overlay()
        self._restore_window_geometry()
        self.window.lock_button.setChecked(
            bool(self.config["window_locked"])
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
        self._screen_recovery_timer = QTimer(self.window)
        self._screen_recovery_timer.setSingleShot(True)
        self._screen_recovery_timer.timeout.connect(self.window.ensure_on_screen)
        self.application.screenAdded.connect(self._watch_screen)
        self.application.screenRemoved.connect(self._schedule_screen_recovery)
        self.application.primaryScreenChanged.connect(self._schedule_screen_recovery)
        for screen in self.application.screens():
            self._watch_screen(screen)

        self.browser_monitor.start()
        self.hotkey_monitor.start()

        self.capture_refresh_timer = QTimer(self.window)
        self.capture_refresh_timer.timeout.connect(
            self._refresh_capture_sources
        )
        self.capture_refresh_timer.start(2_000)
        self._refresh_capture_sources()

        self.menu = QMenu()
        self.show_action = QAction(tr("Show Live GPT"), self.menu)
        self.show_action.triggered.connect(self._show_from_tray)
        self.menu.addAction(self.show_action)
        self.exit_action = QAction(tr("Exit"), self.menu)
        self.exit_action.triggered.connect(self._exit_application)
        self.menu.addAction(self.exit_action)
        self._menu_translations = UiTranslations(self.menu)

        self.tray_icon = QSystemTrayIcon(self.icon, self.application)
        self.tray_icon.setToolTip("Live GPT")
        self.tray_icon.setContextMenu(self.menu)
        self.tray_icon.activated.connect(self._handle_activation)
        self.tray_icon.show()
        self.hotkey_monitor.administrator_required.connect(self._warn_hotkey_privileges)
        logger.info("System tray icon is ready")
        self.show_window()
        self._start_stt_preload()
        self._start_local_tts_preload()

    def _warn_hotkey_privileges(self) -> None:
        title = tr("Microphone shortcuts may not work")
        message = (
            tr("The foreground window is running as administrator. Windows may block "
            "Live GPT's microphone shortcuts while that window is in front.\n\n"
            "Exit Live GPT and run it as administrator (or use start-game-mode.ps1) "
            "to allow microphone shortcuts in administrator windows.\n\n"
            "This warning appears only once per app run.")
        )
        logger.warning(title + ": foreground process is elevated")
        self.tray_icon.showMessage(title, message, QSystemTrayIcon.MessageIcon.Warning, 10_000)
        self.window.show_for_auto_hide()
        self.show_window()
        self._hotkey_privilege_notice = QMessageBox(
            QMessageBox.Icon.Warning, title, message, QMessageBox.StandardButton.Ok, self.window
        )
        self._hotkey_privilege_notice.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
        self._hotkey_privilege_notice.open()
        self._hotkey_privilege_notice.finished.connect(
            lambda _result: self.window.schedule_auto_hide(5_000)
        )
        self.window._set_chrome_visible(True)
        self._hotkey_privilege_notice.raise_()
        self._hotkey_privilege_notice.activateWindow()

    def _hotkey_enabled_states(self) -> dict[str, bool]:
        return {
            name: bool(self.config[key])
            for name, key in HOTKEY_CONFIG_KEYS.items()
        }

    @staticmethod
    def _bindings_for_sequences(
        sequences: dict[str, str],
    ) -> dict[str, HotkeyBinding]:
        bindings = {
            name: HotkeyBinding.from_sequence(sequence)
            for name, sequence in sequences.items()
        }
        if len({binding.text.casefold() for binding in bindings.values() if binding.text}) != sum(bool(binding.text) for binding in bindings.values()):
            raise ValueError(tr("Each action must use a different hotkey"))
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

    def _load_hotkey_sequences(self) -> dict[str, str]:
        defaults = {
            "hold": DEFAULT_HOLD_MIC_HOTKEY,
            "hold_without_screenshot": (
                DEFAULT_HOLD_WITHOUT_SCREENSHOT_HOTKEY
            ),
        }
        sequences = {
            name: str(self.config[HOTKEY_CONFIG_KEYS[name]])
            for name in defaults
        }
        try:
            self._bindings_for_sequences(sequences)
        except ValueError as error:
            logger.warning(f"Invalid saved hotkey configuration: {error}")
            sequences = {
                name: default
                for name, default in defaults.items()
            }
            self.config.update(
                {
                    HOTKEY_CONFIG_KEYS[name]: shortcut_text(sequence)
                    for name, sequence in sequences.items()
                }
            )
        return sequences

    def _open_configuration(self) -> None:
        self.hotkey_monitor.stop()
        try:
            dialog = HotkeyConfigDialog(
                self._hotkey_sequences["hold"],
                parent=self.window,
                hold_without_screenshot=self._hotkey_sequences[
                    "hold_without_screenshot"
                ],
                language=str(self.config["language"]),
                recording_backend=str(self.config["recording_backend"]),
                playing_backend=str(self.config["playing_backend"]),
                stt_language=str(self.config["stt_language"]),
                stt_model=str(self.config["stt_model"]),
                pypi_mirror=str(self.config["pypi_mirror"]),
                sovits_installation=str(self.config["sovits_installation"]),
                sovits_ckpt_path=str(self.config["sovits_ckpt_path"]),
                sovits_pth_path=str(self.config["sovits_pth_path"]),
                sovits_text_lang=str(self.config["sovits_text_lang"]),
                sovits_ref_audio_path=str(self.config["sovits_ref_audio_path"]),
                sovits_prompt_text=str(self.config["sovits_prompt_text"]),
                sovits_prompt_lang=str(self.config["sovits_prompt_lang"]),
                config=self.config,
                stt_manager=self.stt_manager,
                sovits_manager=self.sovits_manager,
            )
            dialog.pet_settings_changed.connect(self._apply_pet_settings)
            dialog.exec()

            sequences = self._load_hotkey_sequences()
            bindings = self._bindings_for_sequences(sequences)
            self._hotkey_sequences = sequences
            self.browser_monitor.set_use_browser_voice(
                self.config["playing_backend"] == "web"
            )
            self._start_stt_preload()
            self._start_local_tts_preload()
            self.hotkey_monitor.update_bindings(
                bindings["hold"],
                hold_without_screenshot=bindings["hold_without_screenshot"],
                enabled=self._hotkey_enabled_states(),
            )
            self.window.set_status(tr("Settings updated"))
            logger.info(
                "Updated global hotkeys "
                + ", ".join(
                    f"{name}={binding.text!r}"
                    for name, binding in bindings.items()
                )
            )
        finally:
            self.hotkey_monitor.start()

    def _apply_pet_settings(self, path: str, mode: str, seconds: int) -> None:
        self.window.pet.load_pet(path)
        self.window.pet.set_idle_behavior(mode, seconds)

    def _handle_activation(
        self, reason: QSystemTrayIcon.ActivationReason
    ) -> None:
        logger.debug(f"Tray icon activated: {reason.name}")
        if reason in (
            QSystemTrayIcon.ActivationReason.Trigger,
            QSystemTrayIcon.ActivationReason.DoubleClick,
        ):
            self._show_from_tray()

    def _show_from_tray(self, checked: bool = False) -> None:
        del checked
        self.window.disable_auto_hide()
        self.show_window()
        self.window.reveal_controls()

    def show_window(self) -> None:
        logger.info("Showing the overlay window")
        self.window.remember_foreground_app()
        self.window.showNormal()
        self.window.ensure_on_screen()
        self.window.raise_()
        self.window.activateWindow()

    def _watch_screen(self, screen) -> None:
        screen.geometryChanged.connect(self._schedule_screen_recovery)
        screen.availableGeometryChanged.connect(self._schedule_screen_recovery)
        screen.logicalDotsPerInchChanged.connect(self._schedule_screen_recovery)
        self._schedule_screen_recovery()

    def _schedule_screen_recovery(self, *_args) -> None:
        # Run after Qt finishes applying the display change to its coordinates.
        self._screen_recovery_timer.start(0)

    def _local_tts_configuration(
        self,
    ) -> tuple[TextToSpeechProvider, str, str, str] | None:
        if self.config["playing_backend"] != "sovits":
            return None
        manager = self.sovits_manager
        try:
            manager.configure(
                str(self.config["sovits_prompt_text"]),
                str(self.config["sovits_prompt_lang"]),
                str(self.config["sovits_ckpt_path"]),
                str(self.config["sovits_pth_path"]),
            )
        except ValueError as error:
            logger.warning(f"GPT-SoVITS playback configuration is invalid: {error}")
            return None
        return (manager, str(self.config["sovits_installation"]),
                str(self.config["sovits_ref_audio_path"]), str(self.config["sovits_text_lang"]))

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
            if (
                match.end() < len(text)
                and not text[match.end()].isspace()
                and not any(character in "。！？" for character in match.group())
            ):
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

    def _ensure_local_speech_worker(self, *, display_playback: bool = True) -> _QueuedLocalSpeechThread | None:
        if self._local_speech_thread is None:
            configuration = self._local_tts_configuration()
            if configuration is None:
                return None
            manager, model_key, speaker, language = configuration
            worker = _QueuedLocalSpeechThread(manager, model_key, speaker, language)
            self._local_speech_thread = worker
            if display_playback:
                worker.started.connect(self.window.begin_reading)
                worker.progress.connect(self.window.set_reading_subtitle)
            worker.completed.connect(self._on_local_speech_completed)
            worker.finished.connect(self._local_speech_finished)
            worker.start()
        worker = self._local_speech_thread
        return worker if isinstance(worker, _QueuedLocalSpeechThread) else None

    def _speak_local_announcement(self, message: str, *, final: bool = False) -> None:
        if (
            getattr(self, "config", {}).get("playing_backend") != "sovits"
            or getattr(self, "_local_voice_interrupted", False)
        ):
            return
        worker = self._local_speech_thread
        if isinstance(worker, _QueuedLocalSpeechThread) and worker._queue_finished:
            pending = getattr(self, "_pending_local_announcements", [])
            pending.append((message, final))
            self._pending_local_announcements = pending
            return
        worker = self._ensure_local_speech_worker(display_playback=not final)
        if worker is not None:
            worker.enqueue_announcement(translate_message(message))
            if final:
                worker.finish_queue()

    def _on_send_finished(self, success: bool, text: str, message: str) -> None:
        self.window.set_send_result(success, text, message)
        if not success:
            self._local_voice_error = message
            worker = self._local_speech_thread
            self._speak_local_announcement(
                message, final=worker is None or getattr(worker, "_queue_finished", False)
            )

    def _on_response_update(self, status: str, text: str) -> None:
        self.window.set_response_update(status, text)
        if (
            self.config["playing_backend"] != "sovits"
            or getattr(self, "_local_voice_interrupted", False)
            or getattr(self, "_local_voice_thinking_announced", False)
            or not BrowserMonitor._is_thinking_status(status)
        ):
            return
        worker = self._ensure_local_speech_worker()
        if worker is not None:
            worker.enqueue_announcement(tr(status.strip()))
            self._local_voice_thinking_announced = True

    def _on_response_finished(self, success: bool, message: str) -> None:
        self.window.set_response_finished(success, message)
        # An announcement can start playback before any reply text exists.
        # Close that queue even if the browser fails without producing text.
        worker = self._local_speech_thread
        if not success:
            self._local_voice_error = message
            pending = getattr(self, "_pending_local_announcements", [])
            if pending:
                pending[-1] = (pending[-1][0], True)
            if isinstance(worker, _QueuedLocalSpeechThread):
                worker.finish_queue()

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
        queued_sentences = self._local_voice_queued_sentences
        matched_count = 0
        for sentence, queued_sentence in zip(sentences, queued_sentences):
            if sentence != queued_sentence:
                break
            matched_count += 1
        # ChatGPT can replace an intermediate reply with a separate final answer.
        # Only skip sentences whose text still matches the current response.
        response_replaced = matched_count < min(len(sentences), len(queued_sentences))
        new_sentences = sentences[matched_count:]
        if new_sentences and self._local_speech_thread is None:
            if self._ensure_local_speech_worker() is None:
                return

        worker = self._local_speech_thread
        if isinstance(worker, _QueuedLocalSpeechThread):
            if response_replaced:
                logger.debug(
                    "Local voice response replaced; queuing changed sentences "
                    f"matching_prefix={matched_count} sentences={len(sentences)}"
                )
                del queued_sentences[matched_count:]
            worker.update_full_text(text)
            for sentence in new_sentences:
                worker.enqueue(sentence, text)
            self._local_voice_queued_sentences.extend(new_sentences)
            if final:
                worker.finish_queue()

    def _play_local_voice(self, text: str) -> None:
        """Compatibility entry point for a complete local-voice response."""
        self._update_local_voice(text, True)

    def _start_local_tts_preload(self) -> None:
        """Warm the selected local model without delaying the settings or UI."""
        configuration = self._local_tts_configuration()
        if configuration is None:
            return
        existing = getattr(self, "_local_tts_preload_thread", None)
        if existing is not None and existing.is_alive():
            return
        manager, model_key, _speaker, _language = configuration
        provider_name = "GPT-SoVITS"

        def preload() -> None:
            try:
                manager.model_status("tts", model_key)
                runtime_ok, runtime_message = manager.dependency_status()
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
        self._local_tts_preload_thread = worker
        worker.start()

    def _start_stt_preload(self) -> None:
        """Warm local transcription before the first microphone press."""
        if str(self.config["recording_backend"]) != "sherpa":
            return
        existing = getattr(self, "_stt_preload_thread", None)
        if existing is not None and existing.is_alive():
            return
        model_key = str(self.config["stt_model"])
        language = self.config.get("stt_language", STT_MODELS[model_key].supported_languages[0])

        def preload() -> None:
            try:
                message = self.stt_manager.preload(model_key, language)
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
        pending = getattr(self, "_pending_local_announcements", [])
        self._pending_local_announcements = []
        for message, final in pending:
            self._speak_local_announcement(message, final=final)

    def _on_local_speech_completed(self, success: bool, message: str) -> None:
        if getattr(self, "_dictation_state", "idle") == "idle":
            error = getattr(self, "_local_voice_error", None)
            if error:
                self.window.finish_reading(False, error)
            else:
                self.window.finish_reading(success, message)

    def _on_browser_reading_finished(self, success: bool, message: str) -> None:
        if getattr(self, "_dictation_state", "idle") == "idle":
            self.window.finish_reading(success, message)

    def start_dictation(
        self,
        *,
        send_on_finish: bool = False,
        include_screenshot: bool = True,
        initial_hold_seconds: float = 0.0,
    ) -> None:
        self._stop_playback_for_recording()
        self.window.prepare_for_dictation()
        self._dictation_send_on_finish = send_on_finish
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
                tr("Select a ChatGPT window first"),
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
            tr("Waiting for the browser to start listening…"),
        )
        self._schedule_dictation_screenshot_upload()
        self.browser_monitor.request_start_dictation(tab_id)

    def _stop_playback_for_recording(self) -> None:
        self.browser_monitor.request_stop_reading()
        self._local_voice_interrupted = True
        worker = getattr(self, "_local_speech_thread", None)
        if worker is not None:
            self._local_voice_interrupted = True
            worker.request_stop()
            self.window.finish_reading(False, tr("Playback stopped for recording"))

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
            screenshot = capture_webp(
                source, include_cursor=bool(self.config["capture_cursor"])
            )
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
            screenshot=screenshot,
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
            tr("Starting the local microphone…"),
        )
        session = LocalDictationSession(self.stt_manager, stt_model, self.config["stt_language"])
        worker = _LocalDictationThread(session)
        self._local_dictation_thread = worker
        worker.listening.connect(
            lambda: self._on_dictation_started(
                True,
                tr("Sherpa-ONNX is listening"),
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
            tr("Transcribing local audio…") if local else tr("Finishing ChatGPT dictation…"),
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
            tr("Dictation was too short; cancelling…"),
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

        recognized_text = text.strip()
        minimum_characters = (
            MIN_ENGLISH_VOICE_SEND_CHARACTERS if recognized_text.isascii() else 1
        )
        voice_text_too_short = bool(
            not was_cancelled
            and recognized_text
            and len(recognized_text) < minimum_characters
        )
        if was_cancelled:
            self._discard_preuploaded_dictation_screenshot()
            self.window.set_microphone_state("idle", message)
        else:
            self._short_voice_text = (
                recognized_text if voice_text_too_short else None
            )
            self.window.set_transcript(recognized_text)
            self.window.set_microphone_state(
                "saved",
                (
                    tr("Voice input must contain at least 2 characters to send")
                    if voice_text_too_short
                    else message
                ),
            )
        should_send = (
            getattr(self, "_dictation_send_on_finish", False)
            and not was_cancelled and not restart
            and len(recognized_text) >= minimum_characters
        )
        if should_send:
            self.window.request_send_from_hotkey(
                getattr(self, "_dictation_include_screenshot", True)
            )
        elif recognized_text:
            self.window.show_for_auto_hide()
        else:
            self._discard_preuploaded_dictation_screenshot()
            self.window.schedule_auto_hide()
        if restart:
            send_on_finish = getattr(self, "_dictation_send_on_finish", False)
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
                        send_on_finish=send_on_finish,
                        include_screenshot=include_screenshot,
                        initial_hold_seconds=restart_held_seconds,
                    ),
                )
            else:
                self.start_dictation(
                    send_on_finish=send_on_finish,
                    include_screenshot=include_screenshot,
                    initial_hold_seconds=restart_held_seconds,
                )

    def _handle_send_requested(
        self,
        text: str,
        capture_source: CaptureSource | None,
    ) -> None:
        text = text.strip()
        if text == getattr(self, "_short_voice_text", None):
            self.window.set_status(
                tr("Voice input must contain at least 2 characters to send"),
                error=True,
            )
            return
        self._short_voice_text = None
        pending_capture = getattr(
            self,
            "_pending_dictation_capture",
            None,
        )
        self._pending_dictation_capture = None
        tab_id = self.selected_chatgpt_tab_id
        if tab_id is None:
            self.window.set_status(
                tr("Select a ChatGPT window first"),
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
            self.window.set_status(tr("Capturing screenshot…"))
            try:
                screenshot = capture_webp(
                    capture_source, include_cursor=bool(self.config["capture_cursor"])
                )
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
        self.window.set_status(tr("Sending to ChatGPT…"))
        self._local_voice_interrupted = False
        self._local_voice_thinking_announced = False
        self._local_voice_error = None
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
        self._short_voice_text = None
        tab_id = self.selected_chatgpt_tab_id
        if tab_id is None:
            self.window.set_status(
                "Text cleared locally; no ChatGPT window selected",
                error=True,
            )
            return

        logger.info(f"Clearing ChatGPT input tab_id={tab_id!r}")
        self.window.set_status(tr("Clearing ChatGPT input…"))
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
                tr("Retrying connection… approve it in the browser")
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
            or self.window._input_collapsed_geometry
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
                tr("No system tray is available on this desktop."),
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
