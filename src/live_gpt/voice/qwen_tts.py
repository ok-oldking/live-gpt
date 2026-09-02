from __future__ import annotations

import contextlib
import gc
import hashlib
import importlib.metadata
import io
import json
import os
import re
import shutil
import subprocess
import sys
import sysconfig
import tarfile
import tempfile
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterator
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
FLASH_ATTN_VERSION = "2.8.3"
TRANSFORMERS_VERSION = "4.57.3"
ACCELERATE_VERSION = "1.12.0"
SOUNDDEVICE_VERSION = "0.5.6"
HUGGINGFACE_HUB_VERSION = "0.36.2"
MODELSCOPE_VERSION = "1.39.1"
PYTORCH_VERSION = "2.11.0+cu126"
TORCHAUDIO_VERSION = "2.11.0+cu126"
NVIDIA_CUDA_NVCC_VERSION = "12.6.85"
NVIDIA_CUDA_RUNTIME_VERSION = "12.6.77"
NVIDIA_CUDA_CCCL_VERSION = "12.6.77"
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
TTS_LANGUAGES = (
    "Auto",
    "Chinese",
    "English",
    "Japanese",
    "Korean",
    "German",
    "French",
    "Russian",
    "Portuguese",
    "Spanish",
    "Italian",
)
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
            f"Local streaming playback · NVIDIA GPU required · "
            f"Qwen3-TTS CustomVoice · {self.parameters} · "
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

    display_name = "Qwen3-TTS"

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
    def flash_attention_status() -> tuple[bool, str]:
        return dependency_status(
            (
                PackageRequirement(
                    "flash-attn",
                    FLASH_ATTN_VERSION,
                    import_name="flash_attn",
                ),
            )
        )

    @staticmethod
    @contextlib.contextmanager
    def _short_build_temporary_directory() -> Iterator[str]:
        """Create a pip build directory short enough for FlashAttention sources."""
        drive_root = Path(Path.cwd().anchor)
        if sys.platform == "win32":
            probe = drive_root / f".lgt-{os.getpid()}-{time.time_ns()}"
            root_writable = False
            try:
                probe.touch(exist_ok=False)
                probe.unlink()
                root_writable = True
            except OSError:
                try:
                    probe.unlink(missing_ok=True)
                except OSError:
                    pass
            if root_writable:
                yield str(drive_root)
                return

            backing = Path.cwd() / ".flash-build-temp"
            backing.mkdir(exist_ok=True)
            mapped_drive = ""
            creationflags = subprocess.CREATE_NO_WINDOW
            for letter in "ZYXWVUT":
                drive = f"{letter}:"
                if Path(f"{drive}\\").exists():
                    continue
                completed = subprocess.run(
                    ["subst", drive, str(backing)],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    check=False,
                    creationflags=creationflags,
                )
                if completed.returncode == 0:
                    mapped_drive = drive
                    break
            if mapped_drive:
                try:
                    yield f"{mapped_drive}\\"
                    return
                finally:
                    subprocess.run(
                        ["subst", mapped_drive, "/D"],
                        capture_output=True,
                        check=False,
                        creationflags=creationflags,
                    )
                    try:
                        backing.rmdir()
                    except OSError:
                        pass

        fallback = Path.cwd() / ".t"
        fallback.mkdir(exist_ok=True)
        try:
            yield str(fallback)
        finally:
            try:
                fallback.rmdir()
            except OSError:
                pass

    @staticmethod
    def _cleanup_invalid_pip_remnants(log: LogCallback) -> None:
        """Remove pip backup names left by the interrupted forced reinstall."""
        package_root = Path(sysconfig.get_path("purelib"))
        prefixes = (
            "~arkupsafe",
            "~ilelock",
            "~inja2",
            "~etworkx",
            "~orch",
            "~orchaudio",
            "~pmath",
            "~sspec",
            "~unctorch",
            "~ympy",
            "~yping_extensions",
            "~etuptools",
        )
        for path in package_root.iterdir():
            if not path.name.casefold().startswith(prefixes):
                continue
            try:
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
                log(f"Removed stale pip backup: {path.name}")
            except OSError as error:
                log(f"WARNING: Unable to remove stale pip backup {path.name}: {error}")

    @staticmethod
    def _nvidia_pip_cuda_environment() -> dict[str, str]:
        """Locate NVIDIA's pip-installed CUDA compiler and development files."""
        nvcc_distribution = importlib.metadata.distribution(
            "nvidia-cuda-nvcc-cu12"
        )
        runtime_distribution = importlib.metadata.distribution(
            "nvidia-cuda-runtime-cu12"
        )
        cccl_distribution = importlib.metadata.distribution("nvidia-cuda-cccl-cu12")

        def located_file(distribution: object, suffix: str) -> Path:
            normalized = suffix.replace("\\", "/").casefold()
            for item in distribution.files or ():
                if str(item).replace("\\", "/").casefold().endswith(normalized):
                    path = Path(distribution.locate_file(item))
                    if path.is_file():
                        return path
            raise RuntimeError(f"NVIDIA CUDA package is missing {suffix}")

        nvcc = located_file(nvcc_distribution, "bin/nvcc.exe")
        runtime_header = located_file(
            runtime_distribution,
            "include/cuda_runtime_api.h",
        )
        runtime_library = located_file(
            runtime_distribution,
            "lib/x64/cudart.lib",
        )
        cccl_header = located_file(cccl_distribution, "include/cub/cub.cuh")
        cuda_home = nvcc.parent.parent
        include_paths = (
            cuda_home / "include",
            runtime_header.parent,
            cccl_header.parents[1],
        )
        return {
            "CUDA_HOME": str(cuda_home),
            "CUDA_PATH": str(cuda_home),
            "PATH": f"{nvcc.parent}{os.pathsep}{os.environ.get('PATH', '')}",
            "INCLUDE": os.pathsep.join(
                str(path) for path in include_paths if path.is_dir()
            )
            + os.pathsep
            + os.environ.get("INCLUDE", ""),
            "LIB": f"{runtime_library.parent}{os.pathsep}{os.environ.get('LIB', '')}",
        }

    @staticmethod
    @contextlib.contextmanager
    def _verified_nvidia_flash_source(
        build_root: str,
        log: LogCallback,
        cancel_event: threading.Event | None,
    ) -> Iterator[Path]:
        """Download the verified sdist while omitting its unused AMD CK tree."""
        metadata_url = (
            f"https://pypi.org/pypi/flash-attn/{FLASH_ATTN_VERSION}/json"
        )
        log(f"Resolving verified FlashAttention source: {metadata_url}")
        with urllib.request.urlopen(metadata_url, timeout=60) as response:
            metadata = json.load(response)
        candidates = [
            item
            for item in metadata.get("urls", ())
            if item.get("packagetype") == "sdist"
            and str(item.get("filename", "")).endswith(".tar.gz")
        ]
        if len(candidates) != 1:
            raise RuntimeError("PyPI did not return one FlashAttention source archive")
        source = candidates[0]
        expected_hash = str(source.get("digests", {}).get("sha256", ""))
        if len(expected_hash) != 64:
            raise RuntimeError("FlashAttention source archive has no SHA-256 digest")
        nonce = f"{os.getpid()}-{time.time_ns()}"
        archive_path = Path(build_root) / f"fa-{nonce}.tar.gz"
        source_root = Path(build_root) / f"fa-{nonce}"
        try:
            source_url = str(source["url"])
            log(f"Downloading verified FlashAttention source: {source_url}")
            digest = hashlib.sha256()
            downloaded = 0
            with (
                urllib.request.urlopen(source_url, timeout=120) as response,
                archive_path.open("wb") as output,
            ):
                total = int(response.headers.get("Content-Length", "0") or 0)
                while True:
                    if cancel_event is not None and cancel_event.is_set():
                        raise OperationCancelled("Dependency installation cancelled")
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
                    digest.update(chunk)
                    downloaded += len(chunk)
                    if total:
                        log(
                            "Downloading FlashAttention source "
                            f"{downloaded / 1024**2:.1f}/{total / 1024**2:.1f} MB"
                        )
            if digest.hexdigest() != expected_hash:
                raise RuntimeError("FlashAttention source failed its SHA-256 check")
            source_root.mkdir()
            skipped = 0
            with tarfile.open(archive_path, "r:gz") as archive:
                selected: list[tarfile.TarInfo] = []
                for member in archive.getmembers():
                    parts = Path(member.name.replace("\\", "/")).parts
                    if len(parts) < 2:
                        continue
                    relative_parts = parts[1:]
                    if relative_parts[:2] == ("csrc", "composable_kernel"):
                        skipped += 1
                        continue
                    member.name = "/".join(relative_parts)
                    selected.append(member)
                archive.extractall(source_root, members=selected, filter="data")
            if not (source_root / "setup.py").is_file():
                raise RuntimeError("FlashAttention source archive is incomplete")
            log(
                "Verified FlashAttention source and omitted "
                f"{skipped} AMD Composable Kernel files not used by NVIDIA CUDA"
            )
            yield source_root
        finally:
            archive_path.unlink(missing_ok=True)
            shutil.rmtree(source_root, ignore_errors=True)

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
        flash_ok, _flash_detail = QwenTtsProvider.flash_attention_status()
        attention = (
            f"FlashAttention {FLASH_ATTN_VERSION}"
            if flash_ok
            else "optimized PyTorch SDPA fallback"
        )
        return (
            True,
            f"Qwen3-TTS 0.1.1, {MODEL_DOWNLOAD_SOURCES[source]}, NVIDIA CUDA "
            f"PyTorch, {attention}, and the audio runtime passed checks on "
            f"{gpu_message}",
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
        write_log(
            "Preserving matching PyTorch files; only missing or mismatched "
            "CUDA packages will be changed"
        )
        install_packages(
            (
                f"torch=={pytorch_version}",
                f"torchaudio=={torchaudio_version}",
            ),
            mirror=mirror,
            log=write_log,
            timeout=7200,
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
            "packaging>=24.0",
            "psutil>=5.9",
            "ninja>=1.11",
            "einops>=0.8",
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
        write_log("Validating the core Qwen CUDA runtime before optional tools")
        core_ok, core_message = QwenTtsProvider.dependency_status(source)
        if not core_ok:
            raise RuntimeError(core_message)
        QwenTtsProvider._cleanup_invalid_pip_remnants(write_log)
        notify("Installing and verifying FlashAttention 2…", None)
        write_log(
            "Attempting the official FlashAttention 2 installation; "
            "Windows source builds are experimental"
        )
        try:
            with QwenTtsProvider._short_build_temporary_directory() as build_temp:
                write_log(f"Using short FlashAttention build path: {build_temp}")
                build_environment = {
                    "MAX_JOBS": "4",
                    "TEMP": build_temp,
                    "TMP": build_temp,
                    "TMPDIR": build_temp,
                }
                if sys.platform == "win32":
                    notify("Installing NVIDIA CUDA 12.6 build tools…", None)
                    cuda_build_packages = (
                        f"nvidia-cuda-nvcc-cu12=={NVIDIA_CUDA_NVCC_VERSION}",
                        "nvidia-cuda-runtime-cu12=="
                        f"{NVIDIA_CUDA_RUNTIME_VERSION}",
                        f"nvidia-cuda-cccl-cu12=={NVIDIA_CUDA_CCCL_VERSION}",
                    )
                    try:
                        install_packages(
                            cuda_build_packages,
                            mirror=mirror,
                            log=write_log,
                            timeout=1800,
                            cancel_event=cancel_event,
                        )
                    except OperationCancelled:
                        raise
                    except Exception:
                        if mirror == "default":
                            raise
                        write_log(
                            "Selected mirror could not provide NVIDIA CUDA "
                            "build tools; retrying from official PyPI"
                        )
                        install_packages(
                            cuda_build_packages,
                            mirror="default",
                            log=write_log,
                            timeout=1800,
                            cancel_event=cancel_event,
                        )
                    build_environment.update(
                        QwenTtsProvider._nvidia_pip_cuda_environment()
                    )
                    write_log(
                        "Using pip-installed NVIDIA CUDA compiler: "
                        f"{Path(build_environment['CUDA_HOME']) / 'bin' / 'nvcc.exe'}"
                    )
                    with QwenTtsProvider._verified_nvidia_flash_source(
                        build_temp,
                        write_log,
                        cancel_event,
                    ) as source_directory:
                        install_packages(
                            (str(source_directory),),
                            mirror=mirror,
                            log=write_log,
                            timeout=7200,
                            options=("--no-build-isolation",),
                            cancel_event=cancel_event,
                            environment=build_environment,
                        )
                else:
                    try:
                        install_packages(
                            (f"flash-attn=={FLASH_ATTN_VERSION}",),
                            mirror=mirror,
                            log=write_log,
                            timeout=7200,
                            options=("--no-build-isolation",),
                            cancel_event=cancel_event,
                            environment=build_environment,
                        )
                    except OperationCancelled:
                        raise
                    except Exception:
                        if mirror == "default":
                            raise
                        write_log(
                            "Selected mirror could not provide FlashAttention; "
                            "retrying from official PyPI"
                        )
                        install_packages(
                            (f"flash-attn=={FLASH_ATTN_VERSION}",),
                            mirror="default",
                            log=write_log,
                            timeout=7200,
                            options=("--no-build-isolation",),
                            cancel_event=cancel_event,
                            environment=build_environment,
                        )
            flash_ok, flash_message = QwenTtsProvider.flash_attention_status()
            if not flash_ok:
                raise RuntimeError(flash_message)
            write_log(f"FlashAttention {FLASH_ATTN_VERSION} passed integrity checks")
        except OperationCancelled:
            raise
        except Exception as error:
            logger.error(
                "FlashAttention installation or verification failed; "
                "Qwen will use optimized PyTorch SDPA",
                error,
            )
            write_log(
                "WARNING: FlashAttention 2 is unavailable; continuing with "
                f"optimized PyTorch SDPA: {error}"
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
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            for function_name in (
                "enable_flash_sdp",
                "enable_mem_efficient_sdp",
                "enable_math_sdp",
            ):
                function = getattr(torch.backends.cuda, function_name, None)
                if callable(function):
                    function(True)
            attention = "sdpa"
            try:
                import flash_attn  # noqa: F401

                attention = "flash_attention_2"
            except Exception as error:
                logger.info(f"FlashAttention unavailable; using SDPA: {error}")
            self._loaded_model = Qwen3TTSModel.from_pretrained(
                str(self.model_directory("tts", model_key)),
                device_map=device_map,
                dtype=dtype,
                attn_implementation=attention,
            )
            runtime_model = getattr(self._loaded_model, "model", None)
            if runtime_model is not None:
                runtime_model.eval()
            self._loaded_key = model_key
            logger.info(
                f"Loaded {TTS_MODELS[model_key].label} on {device_name} "
                f"({device_map}, {dtype}, {attention})"
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

    def preload(self, model_key: str) -> str:
        """Load and retain a verified model before playback is requested."""
        started = time.perf_counter()
        self._load(model_key)
        elapsed = time.perf_counter() - started
        message = f"{TTS_MODELS[model_key].label} preloaded in {elapsed:.1f} s"
        logger.info(message)
        return message

    def synthesize(
        self,
        model_key: str,
        text: str,
        speaker: str = "Vivian",
        language: str = "Auto",
    ) -> tuple[Any, int]:
        selected = TTS_MODELS[model_key]
        valid_speakers = {item.key for item in selected.speakers}
        if speaker not in valid_speakers:
            raise ValueError(f"{speaker!r} is not supported by {selected.label}")
        if language not in TTS_LANGUAGES:
            raise ValueError(f"Unsupported Qwen3-TTS language {language!r}")
        model = self._load(model_key)
        wavs, sample_rate = model.generate_custom_voice(
            text=text,
            language=language,
            speaker=speaker,
            non_streaming_mode=False,
            do_sample=False,
            subtalker_dosample=False,
        )
        if not wavs or len(wavs[0]) == 0:
            raise RuntimeError("Qwen3-TTS generated no audio")
        return wavs[0], int(sample_rate)

    @staticmethod
    def stream_text_chunks(
        text: str,
        max_characters: int = 220,
    ) -> tuple[str, ...]:
        """Split prose at natural pauses for low-latency sequential synthesis."""
        normalized = " ".join(text.split())
        if not normalized:
            return ()
        sentences = re.split(r"(?<=[.!?。！？;；:：])\s+", normalized)
        pieces: list[str] = []
        for sentence in sentences:
            remaining = sentence.strip()
            while len(remaining) > max_characters:
                search_start = max(80, max_characters // 2)
                split_at = max(
                    remaining.rfind(mark, search_start, max_characters + 1)
                    for mark in (", ", "; ", ": ", " ")
                )
                if split_at < search_start:
                    split_at = max_characters
                elif remaining[split_at] in ",;:":
                    split_at += 1
                pieces.append(remaining[:split_at].strip())
                remaining = remaining[split_at:].strip()
            if remaining:
                pieces.append(remaining)
        chunks: list[str] = []
        for piece in pieces:
            if chunks and len(chunks[-1]) + 1 + len(piece) <= max_characters:
                chunks[-1] = f"{chunks[-1]} {piece}"
            else:
                chunks.append(piece)
        return tuple(chunks)

    def synthesize_stream(
        self,
        model_key: str,
        text: str,
        speaker: str = "Vivian",
        language: str = "Auto",
    ) -> Iterator[tuple[Any, int, str]]:
        selected = TTS_MODELS[model_key]
        valid_speakers = {item.key for item in selected.speakers}
        if speaker not in valid_speakers:
            raise ValueError(f"{speaker!r} is not supported by {selected.label}")
        if language not in TTS_LANGUAGES:
            raise ValueError(f"Unsupported Qwen3-TTS language {language!r}")
        chunks = self.stream_text_chunks(text)
        if not chunks:
            raise ValueError("Text is required for Qwen3-TTS playback")
        model = self._load(model_key)
        offset = 0
        while offset < len(chunks):
            batch_size = 1 if offset == 0 else 2
            batch = chunks[offset : offset + batch_size]
            # The local Qwen wrapper simulates streaming text but returns a
            # complete waveform. Generate the first chunk alone for earlier
            # playback, then batch two at a time to keep playback buffered.
            wavs, sample_rate = model.generate_custom_voice(
                text=list(batch),
                language=[language] * len(batch),
                speaker=[speaker] * len(batch),
                non_streaming_mode=False,
                do_sample=False,
                subtalker_dosample=False,
            )
            if len(wavs) != len(batch):
                raise RuntimeError("Qwen3-TTS generated no audio")
            for waveform, chunk in zip(wavs, batch):
                if len(waveform) == 0:
                    raise RuntimeError("Qwen3-TTS generated no audio")
                yield waveform, int(sample_rate), chunk
            offset += len(batch)


__all__ = [
    "ALIYUN_PYTORCH_CUDA_INDEX_URL",
    "FLASH_ATTN_VERSION",
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
    "TTS_LANGUAGES",
]
