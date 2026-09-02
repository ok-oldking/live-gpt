from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
import tempfile
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ..config import default_config_path
from ..logger import Logger
from .base import LogCallback, ProgressCallback
from .dependencies import (
    OperationCancelled,
    PackageRequirement,
    dependency_status,
    install_packages,
)

SHERPA_ONNX_VERSION = "1.13.6"
SOUNDDEVICE_VERSION = "0.5.6"
NUMPY_VERSION = "2.5.2"
logger = Logger.get_logger(__name__)
_STREAM_POLL_SECONDS = 0.02
_RELEASE_TAIL_SECONDS = 0.25


@dataclass(frozen=True)
class ModelAsset:
    filename: str
    url: str
    size: int
    sha256: str
    directory: str


@dataclass(frozen=True)
class SpeechToTextModel:
    key: str
    label: str
    language: str
    mode: str
    accuracy: str
    compute: str
    model_size: str
    ram: str
    best_for: str
    kind: str
    asset: ModelAsset
    required_files: tuple[str, ...]

    @property
    def description(self) -> str:
        return (
            f"{self.mode} · {self.accuracy} · 5-sec compute {self.compute} · "
            f"model {self.model_size} · RAM {self.ram} · {self.best_for}"
        )


_ASR_RELEASE = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models"


def _asset(release: str, filename: str, size: int, sha256: str) -> ModelAsset:
    return ModelAsset(
        filename=filename,
        url=f"{release}/{filename}",
        size=size,
        sha256=sha256,
        directory=filename.removesuffix(".tar.bz2"),
    )


STT_MODELS: dict[str, SpeechToTextModel] = {
    "zh_zipformer_ctc_int8_2025_07_03": SpeechToTextModel(
        "zh_zipformer_ctc_int8_2025_07_03", "🥇 Zipformer CTC zh INT8 2025-07-03",
        "Chinese / Mandarin", "Offline", "1.74% AISHELL", "~310 ms", "350 MB",
        "~450–700 MB", "Best overall", "offline_zipformer_ctc",
        _asset(_ASR_RELEASE, "sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03.tar.bz2",
               301_377_906, "f3ad1814fea34c407eab0cc3df6f6b625419ac9a60d8aebd8efe772a8e85ef67"),
        ("model.int8.onnx", "tokens.txt"),
    ),
    "zh_streaming_zipformer_ctc_int8_2025_06_30": SpeechToTextModel(
        "zh_streaming_zipformer_ctc_int8_2025_06_30", "🥈 Streaming Zipformer zh INT8 2025-06-30",
        "Chinese / Mandarin", "Streaming", "Very good", "~700 ms total, overlapped", "155 MB",
        "~250–400 MB", "Quality streaming", "online_zipformer_ctc",
        _asset(_ASR_RELEASE, "sherpa-onnx-streaming-zipformer-ctc-zh-int8-2025-06-30.tar.bz2",
               127_965_713, "f2ab7a5deb02717801f6a5b26c751b42f8a2db891b07f5b095e6da7442081448"),
        ("model.int8.onnx", "tokens.txt"),
    ),
    "zh_streaming_zipformer_small_ctc_int8_2025_04_01": SpeechToTextModel(
        "zh_streaming_zipformer_small_ctc_int8_2025_04_01", "🥉 Streaming Zipformer Small zh INT8 2025-04-01",
        "Chinese / Mandarin", "Streaming", "Good", "~190 ms total, overlapped", "25 MB",
        "~60–120 MB", "Tiny / fastest", "online_zipformer_ctc",
        _asset(_ASR_RELEASE, "sherpa-onnx-streaming-zipformer-small-ctc-zh-int8-2025-04-01.tar.bz2",
               21_264_113, "b3b309f7ce4a737195fcc6963ea19b0653a7d3401580af5ae0d3e284cbb71f0b"),
        ("model.int8.onnx", "tokens.txt"),
    ),
    "zh_paraformer_int8": SpeechToTextModel(
        "zh_paraformer_int8", "Paraformer Large zh INT8", "Chinese / Mandarin", "Offline",
        "~1.9% AISHELL-family", "~700 ms", "217 MB", "~350–550 MB",
        "Dialects / robust Mandarin", "offline_paraformer",
        _asset(_ASR_RELEASE, "sherpa-onnx-paraformer-zh-int8-2025-10-07.tar.bz2",
               228_262_632, "a071ee5419e14adb34d7f970ab98105a45e6608018b168f023ca2e4810744abe"),
        ("model.int8.onnx", "tokens.txt"),
    ),
    "zh_sense_voice_small_int8": SpeechToTextModel(
        "zh_sense_voice_small_int8", "SenseVoiceSmall INT8", "Chinese / Mandarin", "Offline",
        "~3% class", "~500 ms", "~230 MB", "~450–550 MB", "Multilingual / emotion",
        "offline_sense_voice",
        _asset(_ASR_RELEASE, "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09.tar.bz2",
               165_783_878, "7305f7905bfcf77fa0b39388a313f3da35c68d971661a65475b56fb2162c8e63"),
        ("model.int8.onnx", "tokens.txt"),
    ),
    "en_parakeet_tdt_ctc_110m_int8": SpeechToTextModel(
        "en_parakeet_tdt_ctc_110m_int8", "🥇 NVIDIA Parakeet TDT-CTC 110M INT8", "English",
        "Offline", "2.40% WER LS clean", "~330 ms", "126 MB", "~220–350 MB",
        "Best overall", "offline_nemo_ctc",
        _asset(_ASR_RELEASE, "sherpa-onnx-nemo-parakeet_tdt_ctc_110m-en-36000-int8.tar.bz2",
               104_337_827, "17f945007b52ccd8b7200ffc7c5652e9e8e961dfdf479cefcabd06cf5703630b"),
        ("model.int8.onnx", "tokens.txt"),
    ),
    "en_nemo_conformer_ctc_small": SpeechToTextModel(
        "en_nemo_conformer_ctc_small", "🥈 NeMo Conformer CTC Small", "English", "Offline",
        "Lower quality", "~120 ms", "44 MB", "~80–150 MB", "Fastest / tiny",
        "offline_nemo_ctc",
        _asset(_ASR_RELEASE, "sherpa-onnx-nemo-ctc-en-conformer-small.tar.bz2",
               76_482_338, "83dcb462aece5bef4e8072c267419389f0b8d1f91152d8851765f284ff664caa"),
        ("model.int8.onnx", "tokens.txt"),
    ),
    "en_moonshine_tiny_int8": SpeechToTextModel(
        "en_moonshine_tiny_int8", "🥉 Moonshine Tiny INT8", "English", "Offline",
        "~12% benchmark family", "~160 ms", "~118 MB", "~180–300 MB",
        "Short voice commands", "offline_moonshine",
        _asset(_ASR_RELEASE, "sherpa-onnx-moonshine-tiny-en-int8.tar.bz2",
               107_600_538, "d5fe6ec4334fef36255b2a4010412cad4c007e33103fec62fb5d17cad88086f2"),
        ("preprocess.onnx", "encode.int8.onnx", "uncached_decode.int8.onnx",
         "cached_decode.int8.onnx", "tokens.txt"),
    ),
    "en_moonshine_base_int8": SpeechToTextModel(
        "en_moonshine_base_int8", "Moonshine Base INT8", "English", "Offline",
        "Better than Tiny", "~330 ms", "~272 MB", "~400–600 MB", "Compact seq2seq",
        "offline_moonshine",
        _asset(_ASR_RELEASE, "sherpa-onnx-moonshine-base-en-int8.tar.bz2",
               250_807_309, "21870cecaa2e44e4e2bf63e02d1072bed183ccd10284871353bd9d24dad14e5e"),
        ("preprocess.onnx", "encode.int8.onnx", "uncached_decode.int8.onnx",
         "cached_decode.int8.onnx", "tokens.txt"),
    ),
    "en_paraformer_int8": SpeechToTextModel(
        "en_paraformer_int8", "Paraformer EN INT8", "English", "Offline", "Good", "~975 ms",
        "220 MB", "~350–550 MB", "Alternative", "offline_paraformer",
        _asset(_ASR_RELEASE, "sherpa-onnx-paraformer-en-2024-03-09.tar.bz2",
               1_021_641_665, "8ecca99e86f295b6f84bdab8498361f1658aeddc3e6e1cf2f51dfef26d1048b3"),
        ("model.int8.onnx", "tokens.txt"),
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class SherpaSttProvider:
    """Install, verify, download, and run Sherpa-ONNX recording models."""

    def __init__(self, model_root: str | Path | None = None) -> None:
        self.model_root = Path(model_root) if model_root is not None else (
            default_config_path().parent / "models" / "sherpa-onnx"
        )
        self._recognizer_lock = threading.RLock()
        self._recognizers: dict[str, Any] = {}
        self._verified_models: set[str] = set()

    @staticmethod
    def dependency_status() -> tuple[bool, str]:
        ok, detail = dependency_status(
            (
                PackageRequirement(
                    "sherpa-onnx", SHERPA_ONNX_VERSION,
                    import_name="sherpa_onnx",
                    import_version_attribute="__version__",
                ),
                PackageRequirement("sherpa-onnx-core", SHERPA_ONNX_VERSION),
                PackageRequirement(
                    "sounddevice", SOUNDDEVICE_VERSION,
                    import_name="sounddevice",
                ),
                PackageRequirement(
                    "numpy", NUMPY_VERSION,
                    import_name="numpy",
                    import_version_attribute="__version__",
                ),
            )
        )
        if not ok:
            return False, detail
        return (
            True,
            "Sherpa-ONNX 1.13.6, NumPy 2.5.2, and the audio runtime "
            "passed integrity checks",
        )

    @staticmethod
    def install_dependencies(
        progress: ProgressCallback | None = None,
        log: LogCallback | None = None,
        mirror: str = "default",
        cancel_event: threading.Event | None = None,
    ) -> str:
        notify = progress or (lambda _message, _percent=None: None)
        show_log = log or (lambda _message: None)

        def write_log(line: str) -> None:
            logger.info(f"pip:{line}")
            show_log(line)

        notify("Installing the pinned local voice runtime…", None)
        write_log("Starting pinned Sherpa-ONNX dependency installation")
        install_packages(
            (
                f"sherpa-onnx=={SHERPA_ONNX_VERSION}",
                f"sherpa-onnx-core=={SHERPA_ONNX_VERSION}",
                f"numpy=={NUMPY_VERSION}",
                f"sounddevice=={SOUNDDEVICE_VERSION}",
            ),
            mirror=mirror,
            log=write_log,
            timeout=900,
            options=("--only-binary=:all:",),
            cancel_event=cancel_event,
        )
        if cancel_event is not None and cancel_event.is_set():
            raise OperationCancelled("Dependency installation cancelled")
        notify("Checking installed versions and file integrity…", None)
        write_log("pip completed; validating installed files")
        ok, message = SherpaSttProvider.dependency_status()
        if not ok:
            raise RuntimeError(message)
        notify(message, 100)
        write_log(message)
        return message

    def model_directory(self, model_type: str, model_key: str) -> Path:
        if model_type != "stt" or model_key not in STT_MODELS:
            raise ValueError(f"Unknown STT model {model_key!r}")
        return self.model_root / "stt" / model_key

    def model_status(self, model_type: str, model_key: str) -> tuple[bool, str]:
        if model_type != "stt":
            raise ValueError("Sherpa-ONNX only provides recording/STT")
        model = STT_MODELS[model_key]
        directory = self.model_directory(model_type, model_key)
        try:
            manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
            if manifest.get("model_type") != model_type or manifest.get("model") != model.key:
                raise RuntimeError("model manifest does not match the selection")
            hashes = manifest.get("files")
            if not isinstance(hashes, dict):
                raise RuntimeError("model manifest is incomplete")
            for relative in model.required_files:
                if not (directory / relative).is_file():
                    raise RuntimeError(f"model file is missing: {relative}")
            for relative, expected_hash in hashes.items():
                if not isinstance(relative, str) or not isinstance(expected_hash, str):
                    raise RuntimeError("model manifest contains invalid hashes")
                path = (directory / relative).resolve()
                if directory.resolve() not in path.parents or not path.is_file():
                    raise RuntimeError(f"model file is missing: {relative}")
                if _sha256(path) != expected_hash:
                    raise RuntimeError(f"model file failed integrity check: {relative}")
            return True, f"{model.label} is downloaded and verified"
        except FileNotFoundError:
            return False, f"{model.label} has not been downloaded"
        except Exception as error:
            return False, f"Model verification failed: {error}"

    def download_model(
        self,
        model_type: str,
        model_key: str,
        progress: ProgressCallback | None = None,
        log: LogCallback | None = None,
        cancel_event: threading.Event | None = None,
    ) -> str:
        if model_type != "stt":
            raise ValueError("Sherpa-ONNX only provides recording/STT")
        model = STT_MODELS[model_key]
        asset = model.asset
        notify = progress or (lambda _message, _percent=None: None)
        show_log = log or (lambda _message: None)

        def write_log(line: str) -> None:
            logger.info(f"model-install:{line}")
            show_log(line)

        parent = self.model_root / model_type
        parent.mkdir(parents=True, exist_ok=True)
        staging_parent = Path(tempfile.mkdtemp(prefix=f".{model.key}-", dir=parent))
        staging = staging_parent / model.key
        staging.mkdir()
        try:
            archive = staging_parent / asset.filename
            last_logged_percent = -5

            def download_progress(downloaded: int) -> None:
                nonlocal last_logged_percent
                percent = min(int(downloaded * 100 / asset.size), 99)
                notify(f"Downloading {asset.filename}…", percent)
                if percent >= last_logged_percent + 5:
                    last_logged_percent = percent
                    write_log(
                        f"Downloaded {downloaded / 1024**2:.1f} of "
                        f"{asset.size / 1024**2:.1f} MB ({percent}%)"
                    )

            write_log(f"Downloading {asset.url}")
            self._download_asset(
                asset,
                archive,
                download_progress,
                cancel_event,
            )
            if cancel_event is not None and cancel_event.is_set():
                raise OperationCancelled("Model download cancelled")
            notify(f"Verifying {asset.filename}…", None)
            write_log(f"Archive checksum verified: {asset.sha256}")
            extracted = staging_parent / "extracted"
            extracted.mkdir()
            self._safe_extract(archive, extracted)
            write_log(f"Extracted {asset.filename}")
            source = extracted / asset.directory
            if not source.is_dir():
                raise RuntimeError(f"Archive did not contain {asset.directory}")
            for child in source.iterdir():
                shutil.move(str(child), str(staging / child.name))
            for relative in model.required_files:
                if not (staging / relative).is_file():
                    raise RuntimeError(f"Downloaded model is missing {relative}")
            file_hashes: dict[str, str] = {}
            for path in staging.rglob("*"):
                if cancel_event is not None and cancel_event.is_set():
                    raise OperationCancelled("Model download cancelled")
                if path.is_file():
                    relative = path.relative_to(staging).as_posix()
                    file_hashes[relative] = _sha256(path)
                    write_log(f"Verified model file: {relative}")
            (staging / "manifest.json").write_text(
                json.dumps({"model_type": model_type, "model": model.key,
                            "sherpa_onnx_version": SHERPA_ONNX_VERSION,
                            "asset": asset.sha256, "files": file_hashes}, indent=2) + "\n",
                encoding="utf-8",
            )
            destination = self.model_directory(model_type, model.key)
            backup = parent / f".{model.key}-backup-{time.time_ns()}"
            self._invalidate_model(model.key)
            if destination.exists():
                os.replace(destination, backup)
            try:
                os.replace(staging, destination)
            except Exception:
                if backup.exists() and not destination.exists():
                    os.replace(backup, destination)
                raise
            if backup.exists():
                shutil.rmtree(backup)
            notify("Download complete; model files passed SHA-256 checks", 100)
            write_log(f"Installed and verified {model.label}")
            return f"{model.label} downloaded and verified"
        finally:
            shutil.rmtree(staging_parent, ignore_errors=True)

    @staticmethod
    def _download_asset(asset: ModelAsset, destination: Path,
                        progress: Callable[[int], None],
                        cancel_event: threading.Event | None = None) -> None:
        request = urllib.request.Request(asset.url, headers={"User-Agent": "Live-GPT voice model manager"})
        digest = hashlib.sha256()
        downloaded = 0
        with urllib.request.urlopen(request, timeout=60) as response:
            with destination.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    if cancel_event is not None and cancel_event.is_set():
                        raise OperationCancelled("Model download cancelled")
                    output.write(chunk)
                    digest.update(chunk)
                    downloaded += len(chunk)
                    progress(downloaded)
        if downloaded != asset.size:
            raise RuntimeError(f"{asset.filename} size mismatch: expected {asset.size}, received {downloaded}")
        if digest.hexdigest() != asset.sha256:
            raise RuntimeError(f"{asset.filename} failed its SHA-256 check")

    @staticmethod
    def _safe_extract(archive: Path, destination: Path) -> None:
        root = destination.resolve()
        with tarfile.open(archive, "r:bz2") as bundle:
            for member in bundle.getmembers():
                candidate = (destination / member.name).resolve()
                if candidate != root and root not in candidate.parents:
                    raise RuntimeError("Model archive contains an unsafe path")
                if member.issym() or member.islnk():
                    raise RuntimeError("Model archive contains an unsafe link")
            bundle.extractall(destination, filter="data")

    @staticmethod
    def _resample(samples: Any, sample_rate: int) -> Any:
        import numpy as np
        source = np.asarray(samples, dtype=np.float32)
        if sample_rate == 16_000:
            return source
        target_length = max(round(len(source) * 16_000 / sample_rate), 1)
        return np.interp(np.linspace(0, len(source) - 1, target_length),
                         np.arange(len(source)), source).astype(np.float32)

    def transcribe(self, model_key: str, samples: Any, sample_rate: int) -> str:
        selected = STT_MODELS[model_key]
        samples = self._resample(samples, sample_rate)
        if selected.kind == "online_zipformer_ctc":
            recognizer = self._get_recognizer(model_key, verify=False)
            stream = recognizer.create_stream()
            stream.accept_waveform(16_000, samples)
            stream.input_finished()
            while recognizer.is_ready(stream):
                recognizer.decode_stream(stream)
            return str(recognizer.get_result(stream)).strip()

        recognizer = self._get_recognizer(model_key, verify=False)
        stream = recognizer.create_stream()
        stream.accept_waveform(16_000, samples)
        recognizer.decode_stream(stream)
        return str(stream.result.text).strip()

    def _create_recognizer(self, model_key: str) -> Any:
        """Construct a recognizer. Callers must hold ``_recognizer_lock``."""
        import sherpa_onnx

        selected = STT_MODELS[model_key]
        root = self.model_directory("stt", model_key)
        threads = max(min(os.cpu_count() or 1, 4), 1)
        if selected.kind == "online_zipformer_ctc":
            return sherpa_onnx.OnlineRecognizer.from_zipformer2_ctc(
                tokens=str(root / "tokens.txt"),
                model=str(root / "model.int8.onnx"),
                num_threads=threads,
                enable_endpoint_detection=False,
                decoding_method="greedy_search",
                provider="cpu",
                debug=False,
            )

        common = {
            "tokens": str(root / "tokens.txt"),
            "num_threads": threads,
            "provider": "cpu",
            "debug": False,
        }
        # OfflineRecognizer() is intentionally a zero-argument wrapper in the
        # Python API. Its public factories build the compatible native config.
        if selected.kind == "offline_zipformer_ctc":
            recognizer = sherpa_onnx.OfflineRecognizer.from_zipformer_ctc(
                model=str(root / "model.int8.onnx"), **common
            )
        elif selected.kind == "offline_paraformer":
            recognizer = sherpa_onnx.OfflineRecognizer.from_paraformer(
                paraformer=str(root / "model.int8.onnx"), **common
            )
        elif selected.kind == "offline_nemo_ctc":
            recognizer = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
                model=str(root / "model.int8.onnx"), **common
            )
        elif selected.kind == "offline_sense_voice":
            recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
                model=str(root / "model.int8.onnx"),
                language="auto",
                use_itn=True,
                **common,
            )
        else:
            recognizer = sherpa_onnx.OfflineRecognizer.from_moonshine(
                preprocessor=str(root / "preprocess.onnx"),
                encoder=str(root / "encode.int8.onnx"),
                uncached_decoder=str(root / "uncached_decode.int8.onnx"),
                cached_decoder=str(root / "cached_decode.int8.onnx"),
                **common,
            )
        return recognizer

    def _get_recognizer(self, model_key: str, *, verify: bool) -> Any:
        if model_key not in STT_MODELS:
            raise ValueError(f"Unknown STT model {model_key!r}")
        with self._recognizer_lock:
            if verify and model_key not in self._verified_models:
                ok, message = self.model_status("stt", model_key)
                if not ok:
                    raise RuntimeError(message)
                self._verified_models.add(model_key)
            cached = self._recognizers.get(model_key)
            if cached is not None:
                return cached

            recognizer = self._create_recognizer(model_key)
            self._recognizers[model_key] = recognizer
            return recognizer

    def _invalidate_model(self, model_key: str) -> None:
        with self._recognizer_lock:
            self._recognizers.pop(model_key, None)
            self._verified_models.discard(model_key)

    def prepare(self, model_key: str) -> Any:
        """Verify and load a model once, then reuse its recognizer."""
        # Import PortAudio during background preload, but do not open the input
        # device until recording actually starts.
        import sounddevice  # noqa: F401

        return self._get_recognizer(model_key, verify=True)

    def preload(self, model_key: str) -> str:
        started = time.perf_counter()
        with self._recognizer_lock:
            already_loaded = model_key in self._recognizers
            self.prepare(model_key)
        label = STT_MODELS[model_key].label
        if already_loaded:
            return f"{label} is already preloaded"
        return f"{label} preloaded in {time.perf_counter() - started:.1f} s"

    def create_streaming_recognizer(self, model_key: str) -> Any:
        """Return the reusable Sherpa online recognizer for a streaming model."""
        selected = STT_MODELS[model_key]
        if selected.kind != "online_zipformer_ctc":
            raise ValueError(f"{selected.label} is not a streaming model")
        return self._get_recognizer(model_key, verify=False)


class LocalDictationSession:
    """Record microphone samples until stopped, then transcribe locally."""

    def __init__(self, manager: SherpaSttProvider, stt_model: str) -> None:
        self.manager = manager
        self.stt_model = stt_model
        self._created_at = time.perf_counter()
        self._stop = threading.Event()
        self._cancelled = False

    def stop(self, *, cancel: bool = False) -> None:
        self._cancelled = self._cancelled or cancel
        self._stop.set()

    def run(
        self,
        started: Callable[[], None],
        partial: Callable[[str], None] | None = None,
    ) -> tuple[bool, str, str]:
        try:
            import numpy as np
            import sounddevice as sd

            chunks: list[Any] = []
            chunk_lock = threading.Lock()
            captured_samples = 0

            def receive(indata: Any, _frames: int, _time: Any, status: Any) -> None:
                nonlocal captured_samples
                if status:
                    logger.warning(f"Microphone stream status: {status}")
                chunk = indata[:, 0].copy()
                with chunk_lock:
                    chunks.append(chunk)
                    captured_samples += len(chunk)

            def take_chunks() -> list[Any]:
                with chunk_lock:
                    pending = chunks[:]
                    chunks.clear()
                    return pending

            streaming = STT_MODELS[self.stt_model].mode == "Streaming"
            prepared_recognizer = self.manager.prepare(self.stt_model)
            recognizer = prepared_recognizer if streaming else None
            stream = recognizer.create_stream() if recognizer is not None else None
            with sd.InputStream(
                samplerate=16_000,
                channels=1,
                dtype="float32",
                latency="low",
                callback=receive,
            ):
                started_at = time.perf_counter()
                logger.info(
                    "Local microphone ready "
                    f"startup_ms={(started_at - self._created_at) * 1000:.0f} "
                    f"model={self.stt_model!r}"
                )
                started()
                if recognizer is None or stream is None:
                    self._stop.wait()
                    if not self._cancelled:
                        time.sleep(_RELEASE_TAIL_SECONDS)
                else:
                    last_partial = ""
                    release_deadline: float | None = None
                    while True:
                        if self._stop.is_set():
                            if self._cancelled:
                                break
                            if release_deadline is None:
                                release_deadline = (
                                    time.perf_counter()
                                    + _RELEASE_TAIL_SECONDS
                                )
                            if time.perf_counter() >= release_deadline:
                                break
                            time.sleep(_STREAM_POLL_SECONDS)
                        else:
                            self._stop.wait(_STREAM_POLL_SECONDS)
                        pending = take_chunks()
                        if pending:
                            stream.accept_waveform(
                                16_000, np.concatenate(pending)
                            )
                            while recognizer.is_ready(stream):
                                recognizer.decode_stream(stream)
                            current = str(
                                recognizer.get_result(stream)
                            ).strip()
                            if current != last_partial:
                                last_partial = current
                                if partial is not None:
                                    partial(current)
            if self._cancelled:
                return True, "", "Short local dictation cancelled"
            if captured_samples == 0:
                raise RuntimeError("The microphone captured no audio")
            inference_at = time.perf_counter()
            if recognizer is not None and stream is not None:
                pending = take_chunks()
                if pending:
                    stream.accept_waveform(16_000, np.concatenate(pending))
                stream.input_finished()
                while recognizer.is_ready(stream):
                    recognizer.decode_stream(stream)
                text = str(recognizer.get_result(stream)).strip()
                if partial is not None:
                    partial(text)
            else:
                text = self.manager.transcribe(
                    self.stt_model, np.concatenate(take_chunks()), 16_000
                )
            latency_ms = (time.perf_counter() - inference_at) * 1000
            duration = inference_at - started_at
            timing = "Finalized" if streaming else "Recognized"
            return (
                True,
                text,
                f"{timing} in {latency_ms:.0f} ms · recorded {duration:.1f} s",
            )
        except Exception as error:
            logger.error("Local dictation failed", error)
            return False, "", f"Local dictation failed: {error}"


__all__ = [
    "LocalDictationSession",
    "SHERPA_ONNX_VERSION",
    "STT_MODELS",
    "SherpaSttProvider",
    "SpeechToTextModel",
]
