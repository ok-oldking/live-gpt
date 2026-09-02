from __future__ import annotations

import gc
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..logger import Logger
from .base import LogCallback, ProgressCallback
from .dependencies import (
    OperationCancelled,
    PackageRequirement,
    dependency_status,
    install_packages,
    run_logged_process,
)
from .qwen_tts import (
    ALIYUN_PYTORCH_CUDA_INDEX_URL,
    HUGGINGFACE_HUB_VERSION,
    MODEL_DOWNLOAD_SOURCES,
    MODELSCOPE_VERSION,
    PYTORCH_CUDA_INDEX_URL,
    PYTORCH_VERSION,
    QwenTtsProvider,
    SJTUG_PYTORCH_CUDA_INDEX_URL,
    SOUNDDEVICE_VERSION,
    TORCHAUDIO_VERSION,
    TRANSFORMERS_VERSION,
)


COSYVOICE_REPOSITORY = "https://github.com/FunAudioLLM/CosyVoice.git"
COSYVOICE_ONNXRUNTIME_VERSION = "1.29.0"
COSYVOICE_WHISPER_VERSION = "20250625"
COSYVOICE_PYWORLD_PREBUILT_VERSION = "0.3.5.post2"
COSYVOICE_HYDRA_VERSION = "1.3.2"
COSYVOICE_LIGHTNING_VERSION = "2.2.4"
COSYVOICE_RICH_VERSION = "15.0.0"
COSYVOICE_GDOWN_VERSION = "5.1.0"
# Matcha pins 3.7.5, which forces NumPy below 2. Use its Python 3.12-
# compatible successor because Live GPT shares this environment with Qwen.
COSYVOICE_MATPLOTLIB_VERSION = "3.9.4"
COSYVOICE_PYARROW_VERSION = "18.1.0"
COSYVOICE_SETUPTOOLS_VERSION = "80.9.0"
COSYVOICE_RUNTIME_IMPORTS = (
    "cosyvoice.cli.cosyvoice",
    "cosyvoice.llm.llm",
    "cosyvoice.flow.flow",
    "cosyvoice.flow.flow_matching",
    "cosyvoice.flow.DiT.dit",
    "cosyvoice.transformer.upsample_encoder",
    "cosyvoice.hifigan.generator",
    "cosyvoice.hifigan.f0_predictor",
    "cosyvoice.hifigan.hifigan",
    "cosyvoice.hifigan.discriminator",
    "cosyvoice.dataset.processor",
    "cosyvoice.tokenizer.tokenizer",
    "matcha.utils.audio",
    "matcha.hifigan.models",
)
DEFAULT_PROMPT_TEXT = (
    "You are a helpful assistant.<|endofprompt|>"
    "希望你以后能够做的比我还好呦。"
)
logger = Logger.get_logger(__name__)


@dataclass(frozen=True)
class CosyVoiceTtsModel:
    key: str
    label: str
    repository: str
    revision: str
    download_size: int
    test_text: str

    @property
    def description(self) -> str:
        return (
            "Local native bi-streaming · NVIDIA GPU recommended · "
            "zero-shot voice cloning · 9 languages and Chinese dialects · "
            f"about {self.download_size / 1024**3:.2f} GB download"
        )


COSYVOICE_TTS_MODELS: dict[str, CosyVoiceTtsModel] = {
    "fun_cosyvoice3_0_5b_2512": CosyVoiceTtsModel(
        key="fun_cosyvoice3_0_5b_2512",
        label="Fun-CosyVoice3 0.5B 2512",
        repository="FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
        revision="main",
        download_size=3_200_000_000,
        test_text="你好，Live GPT 的 CosyVoice 3 流式语音已经可以使用。",
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _remove_tree(path: Path) -> None:
    """Remove a Git checkout whose packed objects may be read-only on Windows."""
    if not path.exists():
        return

    def make_writable(function: Any, filename: str, _error: BaseException) -> None:
        os.chmod(filename, stat.S_IWRITE | stat.S_IREAD)
        function(filename)

    shutil.rmtree(path, onexc=make_writable)


def _load_wav_with_soundfile(
    wav: str | Path, target_sr: int, min_sr: int = 16_000
) -> Any:
    """Load prompt WAVs without Torchaudio's TorchCodec/FFmpeg backend."""
    import soundfile
    import torch

    samples, sample_rate = soundfile.read(
        wav, dtype="float32", always_2d=True
    )
    speech = torch.from_numpy(samples.T).mean(dim=0, keepdim=True)
    if sample_rate != target_sr:
        if sample_rate < min_sr:
            raise ValueError(
                f"WAV sample rate {sample_rate} must be at least {min_sr}"
            )
        import torchaudio

        speech = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=target_sr
        )(speech)
    return speech


@contextmanager
def _block_optional_import(module_name: str) -> Iterator[None]:
    """Make an online optional frontend unavailable during model startup."""
    missing = object()
    previous = sys.modules.get(module_name, missing)
    sys.modules[module_name] = None
    try:
        yield
    finally:
        if previous is missing:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous


@contextmanager
def _filter_onnx_providers() -> Iterator[tuple[str, ...]]:
    """Keep CosyVoice from requesting unavailable ONNX providers."""
    try:
        import onnxruntime
    except ImportError:
        yield ()
        return

    available = tuple(onnxruntime.get_available_providers())
    original_session = onnxruntime.InferenceSession

    def create_session(*args: Any, **kwargs: Any) -> Any:
        requested = kwargs.get("providers")
        if requested:
            selected = [provider for provider in requested if provider in available]
            if not selected and "CPUExecutionProvider" in available:
                selected = ["CPUExecutionProvider"]
            kwargs["providers"] = selected or None
        return original_session(*args, **kwargs)

    onnxruntime.InferenceSession = create_session
    try:
        yield available
    finally:
        onnxruntime.InferenceSession = original_session


class CosyVoiceTtsProvider:
    """Install, verify, download, and stream Fun-CosyVoice 3 models."""

    display_name = "CosyVoice 3"
    continuous_audio_stream = True
    stream_prebuffer_seconds = 0.25

    def __init__(self, model_root: str | Path | None = None) -> None:
        self.model_root = (
            Path(model_root) if model_root is not None else Path.cwd() / "models"
        )
        self.runtime_root = self.model_root / "cosyvoice-runtime"
        self._load_lock = threading.RLock()
        self._reference_lock = threading.RLock()
        self._preload_lock = threading.RLock()
        self._loaded_key: str | None = None
        self._loaded_model: Any = None
        self._initial_token_hop_len: int | None = None
        self._cached_references: set[str] = set()
        self._preloaded_keys: set[str] = set()

    @staticmethod
    def _requirements(source: str) -> tuple[PackageRequirement, ...]:
        requirements = [
            PackageRequirement(
                "torch",
                PYTORCH_VERSION,
                verify_integrity=False,
                import_name="torch",
                import_version_attribute="__version__",
            ),
            PackageRequirement(
                "torchaudio",
                TORCHAUDIO_VERSION,
                verify_integrity=False,
                import_name="torchaudio",
                import_version_attribute="__version__",
            ),
            PackageRequirement(
                "HyperPyYAML", "1.2.3", import_name="hyperpyyaml"
            ),
            PackageRequirement(
                "diffusers", "0.29.0", import_name="diffusers"
            ),
            PackageRequirement("librosa", "0.10.2", import_name="librosa"),
            PackageRequirement(
                "omegaconf", "2.3.0", import_name="omegaconf"
            ),
            PackageRequirement("onnx", "1.16.0", import_name="onnx"),
            PackageRequirement(
                "onnxruntime",
                COSYVOICE_ONNXRUNTIME_VERSION,
                import_name="onnxruntime",
            ),
            PackageRequirement(
                "openai-whisper",
                COSYVOICE_WHISPER_VERSION,
                import_name="whisper",
            ),
            PackageRequirement(
                "sounddevice",
                SOUNDDEVICE_VERSION,
                verify_integrity=False,
                import_name="sounddevice",
            ),
            PackageRequirement(
                "soundfile", "0.12.1", import_name="soundfile"
            ),
            PackageRequirement("inflect", "7.3.1", import_name="inflect"),
            PackageRequirement("conformer", "0.3.2", import_name="conformer"),
            PackageRequirement(
                "gdown", COSYVOICE_GDOWN_VERSION, import_name="gdown"
            ),
            PackageRequirement(
                "hydra-core", COSYVOICE_HYDRA_VERSION, import_name="hydra"
            ),
            PackageRequirement(
                "lightning", COSYVOICE_LIGHTNING_VERSION, import_name="lightning"
            ),
            PackageRequirement(
                "pytorch-lightning",
                COSYVOICE_LIGHTNING_VERSION,
                import_name="pytorch_lightning",
            ),
            PackageRequirement(
                "matplotlib",
                COSYVOICE_MATPLOTLIB_VERSION,
                import_name="matplotlib",
            ),
            PackageRequirement(
                "rich", COSYVOICE_RICH_VERSION, import_name="rich"
            ),
            PackageRequirement(
                "pyarrow", COSYVOICE_PYARROW_VERSION, import_name="pyarrow"
            ),
            PackageRequirement(
                "setuptools",
                COSYVOICE_SETUPTOOLS_VERSION,
                import_name="setuptools",
            ),
            # cosyvoice.cli.cosyvoice imports ModelScope at module import time,
            # including when Hugging Face is the selected download source.
            PackageRequirement(
                "modelscope", MODELSCOPE_VERSION, import_name="modelscope"
            ),
            PackageRequirement(
                "transformers", TRANSFORMERS_VERSION, import_name="transformers"
            ),
            PackageRequirement("wget", "3.2", import_name="wget"),
            PackageRequirement(
                "x-transformers", "2.11.24", import_name="x_transformers"
            ),
        ]
        if sys.platform == "win32":
            requirements.append(
                PackageRequirement(
                    "pyworld-prebuilt",
                    COSYVOICE_PYWORLD_PREBUILT_VERSION,
                    import_name="pyworld",
                )
            )
        else:
            requirements.append(
                PackageRequirement("pyworld", "0.3.6", import_name="pyworld")
            )
        if source == "huggingface":
            requirements.append(
                PackageRequirement(
                    "huggingface-hub",
                    HUGGINGFACE_HUB_VERSION,
                    import_name="huggingface_hub",
                )
            )
        elif source != "modelscope":
            raise ValueError(f"Unknown CosyVoice model source {source!r}")
        return tuple(requirements)

    def dependency_status(
        self, source: str = "huggingface"
    ) -> tuple[bool, str]:
        if source not in MODEL_DOWNLOAD_SOURCES:
            return False, f"Unknown CosyVoice model source {source!r}"
        gpu_ok, gpu_message = QwenTtsProvider.nvidia_gpu_status()
        if not gpu_ok:
            return False, gpu_message
        runtime_ok, runtime_message = self.runtime_source_status()
        if not runtime_ok:
            return False, runtime_message
        ok, detail = dependency_status(
            self._requirements(source),
            extra_imports=("numpy", "soundfile", "librosa", "omegaconf"),
            require_nvidia_cuda=True,
        )
        if not ok:
            return False, detail
        runtime_import_ok, runtime_import_message = self.runtime_import_status()
        if not runtime_import_ok:
            return False, runtime_import_message
        return (
            True,
            f"CosyVoice 3 source, {MODEL_DOWNLOAD_SOURCES[source]}, NVIDIA "
            f"CUDA PyTorch, ONNX, and audio dependencies passed checks on "
            f"{gpu_message}",
        )

    def runtime_import_status(self) -> tuple[bool, str]:
        """Import the official runtime in a child process before model preload."""
        marker = "__LIVE_GPT_COSYVOICE_IMPORT_RESULT__:"
        matcha_root = self.runtime_root / "third_party" / "Matcha-TTS"
        script = (
            "import importlib, json, sys\n"
            "sys.path.insert(0, sys.argv[1])\n"
            "sys.path.insert(0, sys.argv[2])\n"
            "try:\n"
            "    for module_name in json.loads(sys.argv[3]):\n"
            "        importlib.import_module(module_name)\n"
            "    result = {'ok': True, 'message': ''}\n"
            "except Exception as error:\n"
            "    result = {'ok': False, 'message': "
            "f'{type(error).__name__}: {error}'}\n"
            f"print({marker!r} + json.dumps(result), flush=True)\n"
        )
        creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        try:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    script,
                    str(self.runtime_root.resolve()),
                    str(matcha_root.resolve()),
                    json.dumps(COSYVOICE_RUNTIME_IMPORTS),
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=180,
                check=False,
                creationflags=creationflags,
            )
            result_line = next(
                (
                    line[len(marker) :]
                    for line in reversed(completed.stdout.splitlines())
                    if line.startswith(marker)
                ),
                "",
            )
            if not result_line:
                detail = completed.stderr.strip() or completed.stdout.strip()
                return (
                    False,
                    "CosyVoice runtime startup check produced no result: "
                    f"{detail or 'unknown child-process error'}",
                )
            result = json.loads(result_line)
            if not result["ok"]:
                return False, f"CosyVoice runtime import failed: {result['message']}"
            return True, "CosyVoice runtime imports passed checks"
        except Exception as error:
            return False, f"CosyVoice runtime startup check failed: {error}"

    def runtime_source_status(self) -> tuple[bool, str]:
        required_runtime_files = (
            self.runtime_root / "cosyvoice" / "cli" / "cosyvoice.py",
            self.runtime_root
            / "third_party"
            / "Matcha-TTS"
            / "matcha"
            / "models"
            / "components"
            / "flow_matching.py",
            self.runtime_root / "asset" / "zero_shot_prompt.wav",
        )
        try:
            missing = [
                path for path in required_runtime_files if not path.is_file()
            ]
        except OSError as error:
            return False, f"Unable to inspect the CosyVoice runtime source: {error}"
        if missing:
            return False, "CosyVoice runtime source has not been installed"
        git = shutil.which("git")
        if git is None:
            return False, "Git is required to verify the CosyVoice runtime source"
        creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        try:
            completed = subprocess.run(
                [
                    git,
                    "-C",
                    str(self.runtime_root),
                    "status",
                    "--porcelain",
                    "--untracked-files=no",
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
                check=False,
                creationflags=creationflags,
            )
            if completed.returncode != 0:
                return False, "Unable to verify the CosyVoice runtime Git checkout"
            if completed.stdout.strip():
                return False, "CosyVoice runtime source files failed integrity check"
        except Exception as error:
            return False, f"Unable to verify the CosyVoice runtime: {error}"
        return True, "Official CosyVoice runtime source passed integrity checks"

    def install_dependencies(
        self,
        progress: ProgressCallback | None = None,
        log: LogCallback | None = None,
        mirror: str = "default",
        cancel_event: threading.Event | None = None,
        source: str = "huggingface",
    ) -> str:
        if source not in MODEL_DOWNLOAD_SOURCES:
            raise ValueError(f"Unknown CosyVoice model source {source!r}")
        gpu_ok, gpu_message = QwenTtsProvider.nvidia_gpu_status()
        if not gpu_ok:
            raise RuntimeError(gpu_message)
        notify = progress or (lambda _message, _percent=None: None)
        show_log = log or (lambda _message: None)

        def write_log(line: str) -> None:
            logger.info(f"pip:{line}")
            show_log(line)

        self.model_root.mkdir(parents=True, exist_ok=True)
        resolved_model_root = self.model_root.resolve()
        for stale_backup in self.model_root.glob(".cosyvoice-runtime-backup-*"):
            resolved_backup = stale_backup.resolve()
            if resolved_model_root not in resolved_backup.parents:
                continue
            try:
                _remove_tree(stale_backup)
                write_log(f"Removed stale CosyVoice runtime backup: {stale_backup.name}")
            except OSError as error:
                write_log(
                    "WARNING: Unable to remove stale CosyVoice runtime backup "
                    f"{stale_backup.name}: {error}"
                )
        runtime_ok, runtime_message = self.runtime_source_status()
        if runtime_ok:
            write_log(f"{runtime_message}; preserving the existing checkout")
        else:
            notify("Installing the official CosyVoice source runtime…", None)
            staging_parent = Path(
                tempfile.mkdtemp(prefix=".cosyvoice-runtime-", dir=self.model_root)
            )
            staging = staging_parent / "runtime"
            try:
                git = shutil.which("git")
                if git is None:
                    raise RuntimeError(
                        "Git is required to install the CosyVoice runtime"
                    )
                command = [
                    git,
                    "-c",
                    "core.longpaths=true",
                    "clone",
                    "--depth",
                    "1",
                    "--recurse-submodules",
                    "--shallow-submodules",
                    COSYVOICE_REPOSITORY,
                    str(staging),
                ]
                write_log(f"Command: {subprocess.list2cmdline(command)}")
                return_code, detail = run_logged_process(
                    command,
                    write_log,
                    timeout=3600,
                    cancel_event=cancel_event,
                )
                if return_code != 0:
                    raise RuntimeError(
                        f"CosyVoice source install failed with exit code "
                        f"{return_code}: {detail}"
                    )
                if cancel_event is not None and cancel_event.is_set():
                    raise OperationCancelled(
                        "CosyVoice runtime installation cancelled"
                    )
                backup = (
                    self.model_root
                    / f".cosyvoice-runtime-backup-{time.time_ns()}"
                )
                if self.runtime_root.exists():
                    os.replace(self.runtime_root, backup)
                try:
                    os.replace(staging, self.runtime_root)
                except Exception:
                    if backup.exists() and not self.runtime_root.exists():
                        os.replace(backup, self.runtime_root)
                    raise
                if backup.exists():
                    try:
                        _remove_tree(backup)
                        write_log("Removed the previous CosyVoice runtime backup")
                    except OSError as error:
                        write_log(
                            "WARNING: CosyVoice runtime was installed, but its "
                            f"old backup could not be removed: {error}"
                        )
            finally:
                try:
                    _remove_tree(staging_parent)
                except OSError as error:
                    write_log(
                        f"WARNING: Unable to remove install staging files: {error}"
                    )

        notify("Installing NVIDIA CUDA PyTorch for CosyVoice…", None)
        write_log(f"Detected NVIDIA GPU: {gpu_message}")
        pytorch_indexes = {
            "ali": (
                ALIYUN_PYTORCH_CUDA_INDEX_URL,
                "Aliyun PyTorch NVIDIA CUDA 12.6 wheels",
            ),
            "sjtug": (
                SJTUG_PYTORCH_CUDA_INDEX_URL,
                "SJTUG PyTorch NVIDIA CUDA 12.6 wheels",
            ),
        }
        index_url, index_label = pytorch_indexes.get(
            mirror,
            (PYTORCH_CUDA_INDEX_URL, "PyTorch NVIDIA CUDA 12.6 wheels"),
        )
        public_version = mirror == "ali"
        torch_version = (
            PYTORCH_VERSION.partition("+")[0]
            if public_version
            else PYTORCH_VERSION
        )
        audio_version = (
            TORCHAUDIO_VERSION.partition("+")[0]
            if public_version
            else TORCHAUDIO_VERSION
        )
        install_packages(
            (f"torch=={torch_version}", f"torchaudio=={audio_version}"),
            mirror=mirror,
            log=write_log,
            timeout=7200,
            cancel_event=cancel_event,
            index_url=index_url,
            index_label=index_label,
        )

        notify("Installing CosyVoice inference and audio dependencies…", None)
        packages = [
            "conformer==0.3.2",
            "diffusers==0.29.0",
            f"gdown=={COSYVOICE_GDOWN_VERSION}",
            "HyperPyYAML==1.2.3",
            f"hydra-core=={COSYVOICE_HYDRA_VERSION}",
            "inflect==7.3.1",
            "librosa==0.10.2",
            f"lightning=={COSYVOICE_LIGHTNING_VERSION}",
            f"pytorch-lightning=={COSYVOICE_LIGHTNING_VERSION}",
            f"matplotlib=={COSYVOICE_MATPLOTLIB_VERSION}",
            f"modelscope=={MODELSCOPE_VERSION}",
            "omegaconf==2.3.0",
            "onnx==1.16.0",
            f"onnxruntime=={COSYVOICE_ONNXRUNTIME_VERSION}",
            f"openai-whisper=={COSYVOICE_WHISPER_VERSION}",
            "protobuf==4.25.8",
            f"pyarrow=={COSYVOICE_PYARROW_VERSION}",
            f"rich=={COSYVOICE_RICH_VERSION}",
            f"setuptools=={COSYVOICE_SETUPTOOLS_VERSION}",
            (
                f"pyworld-prebuilt=={COSYVOICE_PYWORLD_PREBUILT_VERSION}"
                if sys.platform == "win32"
                else "pyworld==0.3.6"
            ),
            "soundfile==0.12.1",
            f"sounddevice=={SOUNDDEVICE_VERSION}",
            f"transformers=={TRANSFORMERS_VERSION}",
            "wget==3.2",
            "x-transformers==2.11.24",
        ]
        if source == "huggingface":
            packages.append(f"huggingface-hub=={HUGGINGFACE_HUB_VERSION}")
        install_packages(
            packages,
            mirror=mirror,
            log=write_log,
            timeout=7200,
            cancel_event=cancel_event,
        )
        if cancel_event is not None and cancel_event.is_set():
            raise OperationCancelled("CosyVoice dependency installation cancelled")
        notify("Checking CosyVoice versions and runtime files…", None)
        ok, message = self.dependency_status(source)
        if not ok:
            raise RuntimeError(message)
        write_log(message)
        notify(message, 100)
        return message

    @staticmethod
    def ensure_download_client(
        progress: ProgressCallback | None = None,
        log: LogCallback | None = None,
        mirror: str = "default",
        cancel_event: threading.Event | None = None,
        source: str = "huggingface",
    ) -> None:
        QwenTtsProvider.ensure_download_client(
            progress, log, mirror, cancel_event, source=source
        )

    def model_directory(self, model_type: str, model_key: str) -> Path:
        if model_type != "tts" or model_key not in COSYVOICE_TTS_MODELS:
            raise ValueError(f"Unknown CosyVoice TTS model {model_key!r}")
        return self.model_root / model_key

    def model_status(self, model_type: str, model_key: str) -> tuple[bool, str]:
        model = COSYVOICE_TTS_MODELS[model_key]
        directory = self.model_directory(model_type, model_key)
        try:
            manifest = json.loads(
                (directory / "manifest.json").read_text(encoding="utf-8")
            )
            if (
                manifest.get("model") != model.key
                or manifest.get("repository") != model.repository
                or manifest.get("source") not in MODEL_DOWNLOAD_SOURCES
            ):
                raise RuntimeError("model manifest does not match the selection")
            files = manifest.get("files")
            if not isinstance(files, dict) or not files:
                raise RuntimeError("model manifest is incomplete")
            for required in (
                "cosyvoice3.yaml",
                "llm.pt",
                "flow.pt",
                "hift.pt",
                "campplus.onnx",
                "speech_tokenizer_v3.onnx",
            ):
                if not (directory / required).is_file():
                    raise RuntimeError(f"model file is missing: {required}")
            for relative, metadata in files.items():
                path = (directory / relative).resolve()
                if directory.resolve() not in path.parents or not path.is_file():
                    raise RuntimeError(f"model file is missing: {relative}")
                if path.stat().st_size != metadata.get("size"):
                    raise RuntimeError(f"model file size changed: {relative}")
                if _sha256(path) != metadata.get("sha256"):
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
        source: str = "huggingface",
    ) -> str:
        if model_type != "tts":
            raise ValueError("CosyVoice 3 only provides playback/TTS")
        if source not in MODEL_DOWNLOAD_SOURCES:
            raise ValueError(f"Unknown CosyVoice model source {source!r}")
        model = COSYVOICE_TTS_MODELS[model_key]
        notify = progress or (lambda _message, _percent=None: None)
        show_log = log or (lambda _message: None)

        def write_log(line: str) -> None:
            logger.info(f"model-install:{line}")
            show_log(line)

        self.model_root.mkdir(parents=True, exist_ok=True)
        staging_parent = Path(
            tempfile.mkdtemp(prefix=f".{model.key}-", dir=self.model_root)
        )
        staging = staging_parent / model.key
        try:
            notify(f"Downloading {model.label}…", None)
            if source == "huggingface":
                source_revision = model.revision
                script = (
                    "import sys\nfrom huggingface_hub import snapshot_download\n"
                    "print(f'Resolving Hugging Face snapshot {sys.argv[1]}', flush=True)\n"
                    "snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2], "
                    "local_dir=sys.argv[3])\n"
                )
            else:
                source_revision = "master"
                script = (
                    "import sys\nfrom modelscope import snapshot_download\n"
                    "print(f'Resolving ModelScope snapshot {sys.argv[1]}', flush=True)\n"
                    "snapshot_download(sys.argv[1], revision=sys.argv[2], "
                    "local_dir=sys.argv[3])\n"
                )
            command = [
                sys.executable,
                "-c",
                script,
                model.repository,
                source_revision,
                str(staging),
            ]
            write_log(
                f"Downloading {model.repository} from "
                f"{MODEL_DOWNLOAD_SOURCES[source]}"
            )
            write_log(
                "Command: " + subprocess.list2cmdline(command).replace("\n", "\\n")
            )
            return_code, detail = run_logged_process(
                command,
                write_log,
                timeout=21_600,
                cancel_event=cancel_event,
            )
            if return_code != 0:
                raise RuntimeError(
                    f"{MODEL_DOWNLOAD_SOURCES[source]} model download failed "
                    f"with exit code {return_code}: {detail}"
                )
            if cancel_event is not None and cancel_event.is_set():
                raise OperationCancelled("CosyVoice model download cancelled")
            for required in (
                "cosyvoice3.yaml",
                "llm.pt",
                "flow.pt",
                "hift.pt",
                "campplus.onnx",
                "speech_tokenizer_v3.onnx",
            ):
                if not (staging / required).is_file():
                    raise RuntimeError(f"Downloaded model is missing {required}")

            notify("Building the local SHA-256 integrity manifest…", None)
            files: dict[str, dict[str, Any]] = {}
            model_files = [
                path
                for path in staging.rglob("*")
                if path.is_file() and ".cache" not in path.relative_to(staging).parts
            ]
            for index, path in enumerate(model_files, start=1):
                if cancel_event is not None and cancel_event.is_set():
                    raise OperationCancelled("CosyVoice model download cancelled")
                relative = path.relative_to(staging).as_posix()
                files[relative] = {
                    "size": path.stat().st_size,
                    "sha256": _sha256(path),
                }
                notify(
                    f"Verifying {relative}…",
                    min(99, int(index * 100 / len(model_files))),
                )
            shutil.rmtree(staging / ".cache", ignore_errors=True)
            (staging / "manifest.json").write_text(
                json.dumps(
                    {
                        "model": model.key,
                        "repository": model.repository,
                        "source": source,
                        "revision": source_revision,
                        "files": files,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            destination = self.model_directory("tts", model.key)
            backup = self.model_root / f".{model.key}-backup-{time.time_ns()}"
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
            self.unload()
            notify("Download complete; CosyVoice model passed SHA-256 checks", 100)
            write_log(f"Installed and verified {model.label}")
            return f"{model.label} downloaded and verified"
        finally:
            shutil.rmtree(staging_parent, ignore_errors=True)

    def _load(self, model_key: str) -> Any:
        with self._load_lock:
            if self._loaded_key == model_key and self._loaded_model is not None:
                return self._loaded_model
            ok, message = self.model_status("tts", model_key)
            if not ok:
                raise RuntimeError(message)
            runtime_paths = (
                self.runtime_root,
                self.runtime_root / "third_party" / "Matcha-TTS",
            )
            for path in reversed(runtime_paths):
                path_text = str(path.resolve())
                if path_text not in sys.path:
                    sys.path.insert(0, path_text)
            import torch

            if not torch.cuda.is_available() or not torch.version.cuda:
                raise RuntimeError(
                    "CosyVoice 3 requires CUDA-enabled PyTorch in Live GPT"
                )
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            from cosyvoice.cli.cosyvoice import AutoModel
            import cosyvoice.cli.frontend as frontend_module
            import cosyvoice.utils.file_utils as file_utils

            # Torchaudio 2.11 ignores backend='soundfile' and requires
            # TorchCodec plus an external FFmpeg runtime. CosyVoice only needs
            # PCM prompt WAVs here, so preserve its intended SoundFile path.
            file_utils.load_wav = _load_wav_with_soundfile
            frontend_module.load_wav = _load_wav_with_soundfile

            self.unload()
            # WeText downloads its normalizer model during construction. Keep
            # local TTS offline and let CosyVoice use its built-in normalizer.
            with (
                _block_optional_import("wetext"),
                _filter_onnx_providers() as providers,
            ):
                self._loaded_model = AutoModel(
                    model_dir=str(self.model_directory("tts", model_key)),
                    fp16=True,
                    load_trt=False,
                    load_vllm=False,
                )
            self._loaded_key = model_key
            self._cached_references.clear()
            self._preloaded_keys.clear()
            runtime_model = getattr(self._loaded_model, "model", None)
            token_hop_len = getattr(runtime_model, "token_hop_len", None)
            self._initial_token_hop_len = (
                int(token_hop_len)
                if isinstance(token_hop_len, int) and token_hop_len > 0
                else None
            )
            onnx_provider = (
                "CUDAExecutionProvider"
                if "CUDAExecutionProvider" in providers
                else "CPUExecutionProvider"
            )
            logger.info(
                f"Loaded {COSYVOICE_TTS_MODELS[model_key].label} on "
                f"{torch.cuda.get_device_name(0)} (FP16, native audio streaming; "
                f"ONNX {onnx_provider})"
            )
            return self._loaded_model

    def unload(self) -> None:
        with self._load_lock:
            self._loaded_model = None
            self._loaded_key = None
            self._initial_token_hop_len = None
            self._cached_references.clear()
            self._preloaded_keys.clear()
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

    def preload(self, model_key: str) -> str:
        with self._preload_lock:
            label = COSYVOICE_TTS_MODELS[model_key].label
            if (
                self._loaded_key == model_key
                and self._loaded_model is not None
                and model_key in self._preloaded_keys
            ):
                return f"{label} is already preloaded"
            started = time.perf_counter()
            model = self._load(model_key)
            default_audio = self.default_prompt_audio()
            if default_audio.is_file():
                self._ensure_reference_cached(
                    model,
                    default_audio,
                    DEFAULT_PROMPT_TEXT,
                )
            self._preloaded_keys.add(model_key)
            elapsed = time.perf_counter() - started
            message = (
                f"{label} preloaded in "
                f"{elapsed:.1f} s"
            )
            logger.info(message)
            return message

    def default_prompt_audio(self) -> Path:
        return self.runtime_root / "asset" / "zero_shot_prompt.wav"

    @staticmethod
    def _text_script(text: str) -> str:
        counts = {"zh": 0, "ja": 0, "ko": 0, "latin": 0}
        for character in text:
            codepoint = ord(character)
            if 0x3040 <= codepoint <= 0x30FF:
                counts["ja"] += 2
            elif 0xAC00 <= codepoint <= 0xD7AF:
                counts["ko"] += 2
            elif 0x3400 <= codepoint <= 0x9FFF:
                counts["zh"] += 1
            elif character.isascii() and character.isalpha():
                counts["latin"] += 1
        if counts["ja"]:
            return "ja"
        if counts["ko"]:
            return "ko"
        # Product names and acronyms can contain more Latin characters than
        # an otherwise Chinese sentence. Two CJK characters and a modest CJK
        # share are enough to select Chinese synthesis.
        if counts["zh"] >= 2 and counts["zh"] * 4 >= counts["latin"]:
            return "zh"
        if counts["latin"]:
            return "latin"
        if counts["zh"]:
            return "zh"
        return "unknown"

    @staticmethod
    def _prompt_transcript(prompt: str) -> str:
        if "<|endofprompt|>" in prompt:
            return prompt.rsplit("<|endofprompt|>", 1)[1].strip()
        return prompt.strip()

    @staticmethod
    def bistream_text_chunks(text: str, target_size: int = 48) -> list[str]:
        """Split completed text into natural chunks for native text streaming."""
        normalized = " ".join(text.split())
        if not normalized:
            return []
        pieces = [
            piece
            for piece in re.split(
                r"(?<=[。！？!?；;，,：:])|(?<=[.])(?=\s|$)",
                normalized,
            )
            if piece
        ]
        chunks: list[str] = []
        pending = ""
        for piece in pieces:
            pending += piece
            if len(pending) >= target_size:
                chunks.append(pending)
                pending = ""
        if pending:
            if chunks and len(pending) < max(12, target_size // 3):
                chunks[-1] += pending
            else:
                chunks.append(pending)
        return chunks

    @classmethod
    def _bistream_text_generator(
        cls,
        text: str,
    ) -> Iterator[str]:
        chunks = cls.bistream_text_chunks(text)
        yield from chunks

    def _ensure_reference_cached(
        self,
        model: Any,
        audio_path: Path,
        prompt_text: str,
    ) -> str:
        stat = audio_path.stat()
        identity = (
            f"{audio_path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}|"
            f"{prompt_text}"
        )
        cache_id = "live_gpt_" + hashlib.sha256(
            identity.encode("utf-8")
        ).hexdigest()[:20]
        with self._reference_lock:
            if cache_id not in self._cached_references:
                started = time.perf_counter()
                model.add_zero_shot_spk(
                    prompt_text,
                    str(audio_path.resolve()),
                    cache_id,
                )
                self._cached_references.add(cache_id)
                logger.info(
                    f"Cached CosyVoice reference features id={cache_id!r} in "
                    f"{time.perf_counter() - started:.2f} s"
                )
        return cache_id

    def synthesize(
        self,
        model_key: str,
        text: str,
        prompt_audio: str = "",
        prompt_text: str = "",
    ) -> tuple[Any, int]:
        chunks = list(
            self.synthesize_stream(model_key, text, prompt_audio, prompt_text)
        )
        if not chunks:
            raise RuntimeError("CosyVoice 3 generated no audio")
        import numpy as np

        return np.concatenate([chunk[0] for chunk in chunks]), chunks[0][1]

    def synthesize_stream(
        self,
        model_key: str,
        text: str,
        prompt_audio: str = "",
        prompt_text: str = "",
    ) -> Iterator[tuple[Any, int, str]]:
        normalized = " ".join(text.split())
        if not normalized:
            raise ValueError("Text is required for CosyVoice 3 playback")
        audio_path = (
            Path(prompt_audio)
            if prompt_audio.strip()
            else self.default_prompt_audio()
        )
        if not audio_path.is_file():
            raise RuntimeError(
                "Choose a reference voice WAV file, or install the CosyVoice "
                "runtime to use its official sample"
            )
        normalized_prompt = prompt_text.strip() or DEFAULT_PROMPT_TEXT
        if "<|endofprompt|>" not in normalized_prompt:
            normalized_prompt = (
                "You are a helpful assistant.<|endofprompt|>" + normalized_prompt
            )
        model = self._load(model_key)
        cache_id = self._ensure_reference_cached(
            model,
            audio_path,
            normalized_prompt,
        )
        target_script = self._text_script(normalized)
        prompt_script = self._text_script(
            self._prompt_transcript(normalized_prompt)
        )
        cross_lingual = (
            target_script != "unknown"
            and prompt_script != "unknown"
            and target_script != prompt_script
        )
        runtime_llm = getattr(getattr(model, "model", None), "llm", None)
        native_bistream = callable(
            getattr(runtime_llm, "inference_bistream", None)
        )
        runtime_model = getattr(model, "model", None)

        def reset_token_hop_len() -> None:
            if (
                runtime_model is not None
                and self._initial_token_hop_len is not None
                and hasattr(runtime_model, "token_hop_len")
            ):
                runtime_model.token_hop_len = self._initial_token_hop_len

        reset_token_hop_len()
        inference_mode = (
            "multilingual-zero-shot"
            if native_bistream and cross_lingual
            else "cross-lingual"
            if cross_lingual
            else "zero-shot"
        )
        logger.info(
            "Starting CosyVoice synthesis "
            f"mode={inference_mode} "
            f"input={'native-bistream' if native_bistream else 'complete-text'} "
            f"audio=native-stream target_script={target_script!r} "
            f"prompt_script={prompt_script!r} "
            "sessions=1 "
            f"token_hop_len={getattr(runtime_model, 'token_hop_len', 'default')}"
        )
        # CosyVoice3 inference_bistream requires the end-of-prompt token in
        # prompt_text. inference_cross_lingual removes prompt_text entirely,
        # so native bi-streaming must use the model's multilingual zero-shot
        # path. The complete-text compatibility path can use cross-lingual.
        def output_streams() -> Iterator[Iterator[dict[str, Any]]]:
            if cross_lingual and not native_bistream:
                prefix = "You are a helpful assistant.<|endofprompt|>"
                yield iter(
                    model.inference_cross_lingual(
                        prefix + normalized,
                        str(audio_path.resolve()),
                        cache_id,
                        stream=True,
                    )
                )
                return
            zero_shot_text: str | Iterator[str] = (
                self._bistream_text_generator(normalized)
                if native_bistream
                else normalized
            )
            yield iter(
                model.inference_zero_shot(
                    zero_shot_text,
                    normalized_prompt,
                    str(audio_path.resolve()),
                    cache_id,
                    stream=True,
                )
            )
        generation_started = time.perf_counter()
        generated_samples = 0
        last_waveform: Any = None
        for output_stream in output_streams():
            for current in output_stream:
                waveform = (
                    current["tts_speech"]
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                    .reshape(-1)
                )
                if len(waveform):
                    generated_samples += len(waveform)
                    last_waveform = waveform
                    yield waveform, int(model.sample_rate), ""
        if last_waveform is None:
            raise RuntimeError("CosyVoice 3 generated no audio")
        # Do not retain an audio chunk just to identify the final subtitle.
        # A zero-audio event updates playback progress after the last chunk.
        yield last_waveform[:0], int(model.sample_rate), normalized
        elapsed = time.perf_counter() - generation_started
        audio_seconds = generated_samples / max(1, int(model.sample_rate))
        logger.info(
            f"CosyVoice generation completed in {elapsed:.2f} s for "
            f"{audio_seconds:.2f} s audio (RTF "
            f"{elapsed / max(audio_seconds, 0.001):.2f})"
        )


__all__ = [
    "COSYVOICE_REPOSITORY",
    "COSYVOICE_TTS_MODELS",
    "DEFAULT_PROMPT_TEXT",
    "CosyVoiceTtsModel",
    "CosyVoiceTtsProvider",
]
