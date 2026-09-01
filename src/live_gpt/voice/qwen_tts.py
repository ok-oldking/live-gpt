from __future__ import annotations

import contextlib
import gc
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
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


QWEN_TTS_VERSION = "0.1.1"
TRANSFORMERS_VERSION = "4.57.3"
ACCELERATE_VERSION = "1.12.0"
SOUNDDEVICE_VERSION = "0.5.6"
HUGGINGFACE_HUB_VERSION = "0.36.2"
MODELSCOPE_VERSION = "1.39.1"
PYTORCH_VERSION = "2.11.0+cu126"
TORCHAUDIO_VERSION = "2.11.0+cu126"
PYTORCH_CUDA_INDEX_URL = "https://download.pytorch.org/whl/cu126"
ALIYUN_PYTORCH_CUDA_INDEX_URL = (
    "https://mirrors.aliyun.com/pytorch-wheels/cu126/"
)
SJTUG_PYTORCH_CUDA_INDEX_URL = (
    "https://mirror.sjtu.edu.cn/pytorch-wheels/cu126"
)
MODEL_DOWNLOAD_SOURCES = {
    "huggingface": "Hugging Face",
    "modelscope": "ModelScope",
}
logger = Logger.get_logger(__name__)


@dataclass(frozen=True)
class QwenSpeaker:
    key: str
    label: str
    description: str


QWEN_SPEAKERS = (
    QwenSpeaker("Vivian", "Vivian", "Chinese female · bright and expressive"),
    QwenSpeaker("Serena", "Serena", "Chinese female · warm and gentle"),
    QwenSpeaker("Uncle_Fu", "Uncle Fu", "Chinese male · seasoned and mellow"),
    QwenSpeaker("Dylan", "Dylan", "Beijing Chinese male"),
    QwenSpeaker("Eric", "Eric", "Sichuan Chinese male"),
    QwenSpeaker("Ryan", "Ryan", "English male · dynamic"),
    QwenSpeaker("Aiden", "Aiden", "English male · clear and composed"),
    QwenSpeaker("Ono_Anna", "Ono Anna", "Japanese female"),
    QwenSpeaker("Sohee", "Sohee", "Korean female"),
)


@dataclass(frozen=True)
class QwenTtsModel:
    key: str
    label: str
    repository: str
    revision: str
    download_size: int
    parameters: str
    speakers: tuple[QwenSpeaker, ...]
    test_text: str

    @property
    def description(self) -> str:
        return (
            f"Local · NVIDIA GPU required · Qwen3-TTS CustomVoice · {self.parameters} · "
            f"10 languages · {len(self.speakers)} selectable speakers · "
            f"{self.download_size / 1024**3:.2f} GB download"
        )


TTS_MODELS: dict[str, QwenTtsModel] = {
    "qwen3_tts_0_6b_custom_voice": QwenTtsModel(
        key="qwen3_tts_0_6b_custom_voice",
        label="Qwen3-TTS 0.6B CustomVoice",
        repository="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        revision="85e237c12c027371202489a0ec509ded67b5e4b5",
        download_size=2_498_388_392,
        parameters="0.6B",
        speakers=QWEN_SPEAKERS,
        test_text="你好，Live GPT 本地语音已经可以使用。",
    ),
    "qwen3_tts_1_7b_custom_voice": QwenTtsModel(
        key="qwen3_tts_1_7b_custom_voice",
        label="Qwen3-TTS 1.7B CustomVoice",
        repository="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        revision="0c0e3051f131929182e2c023b9537f8b1c68adfe",
        download_size=4_520_219_951,
        parameters="1.7B",
        speakers=QWEN_SPEAKERS,
        test_text="Hello, Live GPT local voice is ready to speak.",
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class QwenTtsProvider:
    """Install, verify, download, and run Qwen3-TTS CustomVoice models."""

    def __init__(self, model_root: str | Path | None = None) -> None:
        self.model_root = (
            Path(model_root) if model_root is not None else Path.cwd() / "models"
        )
        self._load_lock = threading.RLock()
        self._loaded_key: str | None = None
        self._loaded_model: Any = None

    @staticmethod
    def nvidia_gpu_status() -> tuple[bool, str]:
        executable = shutil.which("nvidia-smi")
        if executable is None:
            return False, "NVIDIA GPU/driver not found (nvidia-smi is unavailable)"
        creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        try:
            completed = subprocess.run(
                [
                    executable,
                    "--query-gpu=name,driver_version",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=15,
                check=False,
                creationflags=creationflags,
            )
            output = completed.stdout.strip()
            if completed.returncode != 0 or not output:
                return False, completed.stderr.strip() or "NVIDIA GPU not detected"
            if "NVIDIA" not in output.upper():
                return False, "Qwen3-TTS requires an NVIDIA GPU"
            return True, output.splitlines()[0]
        except Exception as error:
            return False, f"Unable to query the NVIDIA GPU: {error}"

    @staticmethod
    def dependency_status(source: str = "huggingface") -> tuple[bool, str]:
        if source not in MODEL_DOWNLOAD_SOURCES:
            return False, f"Unknown Qwen model source {source!r}"
        gpu_ok, gpu_message = QwenTtsProvider.nvidia_gpu_status()
        if not gpu_ok:
            return False, gpu_message
        requirements = [
            PackageRequirement(
                "qwen-tts", QWEN_TTS_VERSION,
                import_name="qwen_tts",
            ),
            PackageRequirement(
                "transformers", TRANSFORMERS_VERSION,
                verify_integrity=False,
                import_name="transformers",
            ),
            PackageRequirement(
                "accelerate", ACCELERATE_VERSION,
                verify_integrity=False,
                import_name="accelerate",
            ),
            PackageRequirement(
                "sounddevice", SOUNDDEVICE_VERSION,
                verify_integrity=False,
                import_name="sounddevice",
            ),
            PackageRequirement(
                "huggingface-hub",
                HUGGINGFACE_HUB_VERSION,
                import_name="huggingface_hub",
            ),
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
        ]
        if source == "modelscope":
            requirements.append(
                PackageRequirement(
                    "modelscope",
                    MODELSCOPE_VERSION,
                    import_name="modelscope",
                )
            )
        ok, detail = dependency_status(
            requirements,
            extra_imports=("numpy",),
            require_nvidia_cuda=True,
        )
        if not ok:
            return False, detail
        return (
            True,
            f"Qwen3-TTS 0.1.1, {MODEL_DOWNLOAD_SOURCES[source]}, NVIDIA CUDA "
            f"PyTorch, and the audio runtime passed checks on {gpu_message}",
        )

    @staticmethod
    def install_dependencies(
        progress: ProgressCallback | None = None,
        log: LogCallback | None = None,
        mirror: str = "default",
        cancel_event: threading.Event | None = None,
        source: str = "huggingface",
    ) -> str:
        if source not in MODEL_DOWNLOAD_SOURCES:
            raise ValueError(f"Unknown Qwen model source {source!r}")
        gpu_ok, gpu_message = QwenTtsProvider.nvidia_gpu_status()
        if not gpu_ok:
            raise RuntimeError(gpu_message)
        notify = progress or (lambda _message, _percent=None: None)
        show_log = log or (lambda _message: None)

        def write_log(line: str) -> None:
            logger.info(f"pip:{line}")
            show_log(line)

        notify("Installing NVIDIA CUDA PyTorch and Torchaudio…", None)
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
        pytorch_index_url, pytorch_index_label = pytorch_indexes.get(
            mirror,
            (PYTORCH_CUDA_INDEX_URL, "PyTorch NVIDIA CUDA 12.6 wheels"),
        )
        # Aliyun's PyTorch wheel index lists CUDA builds using the public
        # version (for example, 2.11.0) rather than the local +cu126 label.
        # The index URL still restricts resolution to the CUDA 12.6 wheels.
        pytorch_version = (
            PYTORCH_VERSION.partition("+")[0]
            if mirror == "ali"
            else PYTORCH_VERSION
        )
        torchaudio_version = (
            TORCHAUDIO_VERSION.partition("+")[0]
            if mirror == "ali"
            else TORCHAUDIO_VERSION
        )
        install_packages(
            (
                f"torch=={pytorch_version}",
                f"torchaudio=={torchaudio_version}",
            ),
            mirror=mirror,
            log=write_log,
            timeout=7200,
            options=("--force-reinstall",),
            cancel_event=cancel_event,
            index_url=pytorch_index_url,
            index_label=pytorch_index_label,
        )
        notify("Installing Qwen3-TTS and its audio dependencies…", None)
        write_log("Starting Qwen3-TTS dependency installation")
        packages = [
            f"qwen-tts=={QWEN_TTS_VERSION}",
            f"sounddevice=={SOUNDDEVICE_VERSION}",
            f"huggingface-hub=={HUGGINGFACE_HUB_VERSION}",
        ]
        if source == "modelscope":
            packages.append(f"modelscope=={MODELSCOPE_VERSION}")
        install_packages(
            packages,
            mirror=mirror,
            log=write_log,
            timeout=3600,
            cancel_event=cancel_event,
        )
        if cancel_event is not None and cancel_event.is_set():
            raise OperationCancelled("Dependency installation cancelled")
        notify("Checking installed versions and package integrity…", None)
        write_log("pip completed; validating installed files")
        ok, message = QwenTtsProvider.dependency_status(source)
        if not ok:
            raise RuntimeError(message)
        notify(message, 100)
        write_log(message)
        return message

    @staticmethod
    def _download_client_requirement(source: str) -> PackageRequirement:
        if source == "huggingface":
            return PackageRequirement(
                "huggingface-hub",
                HUGGINGFACE_HUB_VERSION,
                import_name="huggingface_hub",
            )
        if source == "modelscope":
            return PackageRequirement(
                "modelscope",
                MODELSCOPE_VERSION,
                import_name="modelscope",
            )
        raise ValueError(f"Unknown Qwen model source {source!r}")

    @staticmethod
    def ensure_download_client(
        progress: ProgressCallback | None = None,
        log: LogCallback | None = None,
        mirror: str = "default",
        cancel_event: threading.Event | None = None,
        source: str = "huggingface",
    ) -> None:
        requirement = QwenTtsProvider._download_client_requirement(source)
        ok, _message = dependency_status((requirement,))
        if ok:
            return
        notify = progress or (lambda _message, _percent=None: None)
        show_log = log or (lambda _message: None)

        def write_log(line: str) -> None:
            logger.info(f"pip:{line}")
            show_log(line)

        label = MODEL_DOWNLOAD_SOURCES[source]
        notify(f"Installing the {label} model download client…", None)
        write_log(f"The {label} download client is missing or invalid")
        install_packages(
            (f"{requirement.distribution}=={requirement.version}",),
            mirror=mirror,
            log=write_log,
            timeout=3600,
            cancel_event=cancel_event,
        )
        if cancel_event is not None and cancel_event.is_set():
            raise OperationCancelled("Model download client installation cancelled")
        ok, message = dependency_status((requirement,))
        if not ok:
            raise RuntimeError(
                f"{label} download client failed validation: {message}"
            )
        write_log(f"{label} download client installed and verified")

    def model_directory(self, model_type: str, model_key: str) -> Path:
        if model_type != "tts" or model_key not in TTS_MODELS:
            raise ValueError(f"Unknown Qwen TTS model {model_key!r}")
        return self.model_root / model_key

    def model_status(self, model_type: str, model_key: str) -> tuple[bool, str]:
        model = TTS_MODELS[model_key]
        directory = self.model_directory(model_type, model_key)
        try:
            manifest = json.loads(
                (directory / "manifest.json").read_text(encoding="utf-8")
            )
            source = str(manifest.get("source", "huggingface"))
            expected_revision = (
                model.revision if source == "huggingface" else "master"
            )
            if (
                manifest.get("model") != model.key
                or manifest.get("repository") != model.repository
                or source not in MODEL_DOWNLOAD_SOURCES
                or manifest.get("revision") != expected_revision
            ):
                raise RuntimeError("model manifest does not match the selection")
            files = manifest.get("files")
            if not isinstance(files, dict) or not files:
                raise RuntimeError("model manifest is incomplete")
            for required in (
                "config.json",
                "model.safetensors",
                "speech_tokenizer/model.safetensors",
            ):
                if not (directory / required).is_file():
                    raise RuntimeError(f"model file is missing: {required}")
            for relative, metadata in files.items():
                if not isinstance(relative, str) or not isinstance(metadata, dict):
                    raise RuntimeError("model manifest contains invalid entries")
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
            raise ValueError("Qwen3-TTS only provides playback/TTS")
        if source not in MODEL_DOWNLOAD_SOURCES:
            raise ValueError(f"Unknown Qwen model source {source!r}")
        model = TTS_MODELS[model_key]
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
            notify(
                f"Downloading {model.label} ({model.download_size / 1024**3:.2f} GB)…",
                None,
            )
            write_log(
                f"Downloading {model.repository} from "
                f"{MODEL_DOWNLOAD_SOURCES[source]}"
            )
            if source == "huggingface":
                source_revision = model.revision
                download_script = (
                    "import sys\n"
                    "from huggingface_hub import snapshot_download\n"
                    "print(f'Resolving Hugging Face snapshot {sys.argv[1]}', flush=True)\n"
                    "snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2], "
                    "local_dir=sys.argv[3])\n"
                    "print('Hugging Face snapshot download completed', flush=True)\n"
                )
                download_arguments = [
                    model.repository,
                    source_revision,
                    str(staging),
                ]
            else:
                source_revision = "master"
                download_script = (
                    "import sys\n"
                    "from modelscope import snapshot_download\n"
                    "print(f'Resolving ModelScope snapshot {sys.argv[1]}', flush=True)\n"
                    "snapshot_download(sys.argv[1], revision=sys.argv[2], "
                    "local_dir=sys.argv[3])\n"
                    "print('ModelScope snapshot download completed', flush=True)\n"
                )
                download_arguments = [
                    model.repository,
                    source_revision,
                    str(staging),
                ]
            command = [
                sys.executable,
                "-c",
                download_script,
                *download_arguments,
            ]
            printable_command = subprocess.list2cmdline(command).replace(
                "\n", "\\n"
            )
            write_log(f"Command: {printable_command}")
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
                raise OperationCancelled("Model download cancelled")
            write_log(
                f"{MODEL_DOWNLOAD_SOURCES[source]} snapshot download completed"
            )
            for required in (
                "config.json",
                "model.safetensors",
                "speech_tokenizer/model.safetensors",
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
                    raise OperationCancelled("Model download cancelled")
                relative = path.relative_to(staging).as_posix()
                files[relative] = {
                    "size": path.stat().st_size,
                    "sha256": _sha256(path),
                }
                notify(
                    f"Verifying {relative}…",
                    min(99, int(index * 100 / len(model_files))),
                )
                write_log(
                    f"Verified {relative} ({path.stat().st_size / 1024**2:.1f} MB)"
                )
            shutil.rmtree(staging / ".cache", ignore_errors=True)
            (staging / "manifest.json").write_text(
                json.dumps(
                    {
                        "model": model.key,
                        "repository": model.repository,
                        "source": source,
                        "revision": source_revision,
                        "qwen_tts_version": QWEN_TTS_VERSION,
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
            notify("Download complete; Qwen model passed SHA-256 checks", 100)
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
            import torch

            if (
                not torch.version.cuda
                or not torch.cuda.is_available()
                or torch.cuda.device_count() < 1
            ):
                raise RuntimeError(
                    "Qwen3-TTS requires an NVIDIA GPU and CUDA-enabled PyTorch. "
                    "Use Install / Repair runtime, then restart Live GPT."
                )
            device_name = torch.cuda.get_device_name(0)
            if "NVIDIA" not in device_name.upper():
                raise RuntimeError("Qwen3-TTS requires an NVIDIA CUDA GPU")
            import_output = io.StringIO()
            with contextlib.redirect_stdout(import_output), contextlib.redirect_stderr(
                import_output
            ):
                from qwen_tts import Qwen3TTSModel
            for line in import_output.getvalue().splitlines():
                if line.strip():
                    logger.debug(f"Qwen3-TTS import: {line.strip()}")

            self.unload()
            supports_bfloat16 = bool(
                getattr(torch.cuda, "is_bf16_supported", lambda: False)()
            )
            dtype = torch.bfloat16 if supports_bfloat16 else torch.float16
            device_map = "cuda:0"
            self._loaded_model = Qwen3TTSModel.from_pretrained(
                str(self.model_directory("tts", model_key)),
                device_map=device_map,
                dtype=dtype,
                attn_implementation="sdpa",
            )
            self._loaded_key = model_key
            logger.info(
                f"Loaded {TTS_MODELS[model_key].label} on {device_name} ({device_map})"
            )
            return self._loaded_model

    def unload(self) -> None:
        with self._load_lock:
            self._loaded_model = None
            self._loaded_key = None
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

    def synthesize(
        self, model_key: str, text: str, speaker: str = "Vivian"
    ) -> tuple[Any, int]:
        selected = TTS_MODELS[model_key]
        valid_speakers = {item.key for item in selected.speakers}
        if speaker not in valid_speakers:
            raise ValueError(f"{speaker!r} is not supported by {selected.label}")
        model = self._load(model_key)
        wavs, sample_rate = model.generate_custom_voice(
            text=text,
            language="Auto",
            speaker=speaker,
        )
        if not wavs or len(wavs[0]) == 0:
            raise RuntimeError("Qwen3-TTS generated no audio")
        return wavs[0], int(sample_rate)


__all__ = [
    "ALIYUN_PYTORCH_CUDA_INDEX_URL",
    "MODEL_DOWNLOAD_SOURCES",
    "QWEN_SPEAKERS",
    "QWEN_TTS_VERSION",
    "PYTORCH_CUDA_INDEX_URL",
    "PYTORCH_VERSION",
    "SJTUG_PYTORCH_CUDA_INDEX_URL",
    "QwenSpeaker",
    "QwenTtsModel",
    "QwenTtsProvider",
    "TTS_MODELS",
]
