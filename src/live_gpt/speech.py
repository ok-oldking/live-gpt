from __future__ import annotations

import os
import queue
import tarfile
import threading
import urllib.request
from array import array
from pathlib import Path

import sherpa_onnx
from PySide6.QtCore import QObject, Signal

from .logger import Logger


logger = Logger.get_logger(__name__)

MODEL_NAME = "sherpa-onnx-streaming-zipformer-small-bilingual-zh-en-2023-02-16"
MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
    f"{MODEL_NAME}.tar.bz2"
)


class SpeechTranscriber(QObject):
    transcript_changed = Signal(str)
    ready_changed = Signal(bool)
    status_changed = Signal(str)
    failed = Signal(str)

    def __init__(self, models_directory: Path | str = "models") -> None:
        super().__init__()
        self.models_directory = Path(models_directory).resolve()
        self._commands: queue.Queue[tuple[str, object | None]] = queue.Queue(
            maxsize=256
        )
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._closed = threading.Event()

    @property
    def is_ready(self) -> bool:
        return self._ready.is_set()

    def prepare(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run,
            name="sherpa-onnx",
            daemon=True,
        )
        self._thread.start()

    def start_session(self, sample_rate: int) -> None:
        self._put_command("start", int(sample_rate))

    def feed_audio(self, pcm_data: bytes, sample_rate: int) -> None:
        if not self.is_ready or self._closed.is_set():
            return
        try:
            self._commands.put_nowait(("audio", (pcm_data, int(sample_rate))))
        except queue.Full:
            logger.warning("Speech recognition queue is full; dropping audio")

    def finish_session(self) -> None:
        self._put_command("stop", None)

    def close(self) -> None:
        if self._closed.is_set():
            return
        try:
            self._commands.put(("shutdown", None), timeout=1)
        except queue.Full:
            logger.warning("Unable to enqueue speech shutdown command")
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=5)
        self._closed.set()

    def _put_command(self, command: str, value: object | None) -> None:
        if self._closed.is_set() and command != "shutdown":
            return
        try:
            self._commands.put_nowait((command, value))
        except queue.Full:
            logger.warning(f"Unable to enqueue speech command {command!r}")

    def _run(self) -> None:
        try:
            model_directory = self._ensure_model()
            self.status_changed.emit("Loading speech model...")
            recognizer = self._create_recognizer(model_directory)
        except Exception as error:
            logger.error("Unable to prepare sherpa-onnx", error)
            self.failed.emit(str(error))
            return

        self._ready.set()
        self.ready_changed.emit(True)
        self.status_changed.emit("Speech model ready")
        logger.info(f"Sherpa-onnx model ready path={model_directory}")

        stream = None
        sample_rate = 16_000
        committed_segments: list[str] = []

        while True:
            command, value = self._commands.get()
            if command == "shutdown":
                break

            try:
                if command == "start":
                    sample_rate = int(value)
                    stream = recognizer.create_stream()
                    committed_segments = []
                    self.transcript_changed.emit("")
                    logger.info(
                        f"Sherpa-onnx transcription started sample_rate={sample_rate} Hz"
                    )
                elif command == "audio" and stream is not None:
                    pcm_data, chunk_sample_rate = value
                    sample_rate = int(chunk_sample_rate)
                    integer_samples = array("h")
                    integer_samples.frombytes(pcm_data)
                    samples = array(
                        "f",
                        (sample / 32768.0 for sample in integer_samples),
                    )
                    stream.accept_waveform(sample_rate, samples)
                    self._decode_available(
                        recognizer,
                        stream,
                        committed_segments,
                    )
                elif command == "stop" and stream is not None:
                    final_text = self._finish_stream(
                        recognizer,
                        stream,
                        sample_rate,
                        committed_segments,
                    )
                    self.transcript_changed.emit(final_text)
                    logger.info(f"Sherpa-onnx final transcript={final_text!r}")
                    stream = None
            except Exception as error:
                logger.error("Sherpa-onnx streaming recognition failed", error)
                self.failed.emit(str(error))
                stream = None

        self._ready.clear()
        self.ready_changed.emit(False)

    def _decode_available(
        self,
        recognizer,
        stream,
        committed_segments: list[str],
    ) -> None:
        while recognizer.is_ready(stream):
            recognizer.decode_stream(stream)

        partial_text = recognizer.get_result(stream).strip()
        self.transcript_changed.emit(
            self._join_transcript(committed_segments, partial_text)
        )

        if recognizer.is_endpoint(stream):
            if partial_text:
                committed_segments.append(partial_text)
            recognizer.reset(stream)
            self.transcript_changed.emit(
                self._join_transcript(committed_segments)
            )

    def _finish_stream(
        self,
        recognizer,
        stream,
        sample_rate: int,
        committed_segments: list[str],
    ) -> str:
        tail = array("f", [0.0]) * int(0.5 * sample_rate)
        stream.accept_waveform(sample_rate, tail)
        stream.input_finished()
        while recognizer.is_ready(stream):
            recognizer.decode_stream(stream)

        final_segment = recognizer.get_result(stream).strip()
        if final_segment:
            committed_segments.append(final_segment)
        return self._join_transcript(committed_segments)

    @staticmethod
    def _join_transcript(
        committed_segments: list[str],
        partial_text: str = "",
    ) -> str:
        pieces = [*committed_segments]
        if partial_text:
            pieces.append(partial_text)
        return " ".join(piece.strip() for piece in pieces if piece.strip())

    def _ensure_model(self) -> Path:
        model_directory = self.models_directory / MODEL_NAME
        required_files = self._required_model_files(model_directory)
        if all(path.is_file() for path in required_files.values()):
            return model_directory

        self.models_directory.mkdir(parents=True, exist_ok=True)
        archive_path = self.models_directory / f"{MODEL_NAME}.tar.bz2"
        partial_path = archive_path.with_suffix(archive_path.suffix + ".part")

        if not archive_path.is_file():
            self.status_changed.emit("Downloading speech model...")
            logger.info(f"Downloading sherpa-onnx model from {MODEL_URL}")
            urllib.request.urlretrieve(MODEL_URL, partial_path)
            os.replace(partial_path, archive_path)

        self.status_changed.emit("Extracting speech model...")
        logger.info(f"Extracting sherpa-onnx model archive={archive_path}")
        with tarfile.open(archive_path, "r:bz2") as model_archive:
            model_archive.extractall(self.models_directory, filter="data")

        missing = [
            str(path)
            for path in self._required_model_files(model_directory).values()
            if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "Speech model is incomplete; missing: " + ", ".join(missing)
            )
        return model_directory

    @staticmethod
    def _required_model_files(model_directory: Path) -> dict[str, Path]:
        return {
            "tokens": model_directory / "tokens.txt",
            "encoder": model_directory / "encoder-epoch-99-avg-1.int8.onnx",
            "decoder": model_directory / "decoder-epoch-99-avg-1.onnx",
            "joiner": model_directory / "joiner-epoch-99-avg-1.int8.onnx",
        }

    def _create_recognizer(self, model_directory: Path):
        files = self._required_model_files(model_directory)
        return sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=str(files["tokens"]),
            encoder=str(files["encoder"]),
            decoder=str(files["decoder"]),
            joiner=str(files["joiner"]),
            num_threads=2,
            sample_rate=16_000,
            enable_endpoint_detection=True,
            decoding_method="greedy_search",
            model_type="zipformer",
        )
