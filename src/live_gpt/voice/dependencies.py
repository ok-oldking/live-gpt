from __future__ import annotations

import base64
import hashlib
import importlib
import importlib.metadata
import json
import re
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from typing import Sequence

from .base import LogCallback


class OperationCancelled(RuntimeError):
    """Raised after the user stops an install or model download."""


@dataclass(frozen=True)
class PypiMirror:
    key: str
    label: str
    index_url: str


PYPI_MIRRORS: dict[str, PypiMirror] = {
    "default": PypiMirror(
        "default", "Default (PyPI)", "https://pypi.org/simple/"
    ),
    "ali": PypiMirror(
        "ali", "Aliyun", "https://mirrors.aliyun.com/pypi/simple/"
    ),
    "sjtug": PypiMirror(
        "sjtug",
        "SJTUG (Shanghai Jiao Tong University)",
        "https://mirror.sjtu.edu.cn/pypi/web/simple/",
    ),
}


@dataclass(frozen=True)
class PackageRequirement:
    distribution: str
    version: str
    verify_integrity: bool = True
    import_name: str = ""
    import_version_attribute: str = ""


class PipProgressFormatter:
    """Turn pip's machine-readable byte counters into a live status line."""

    _progress_pattern = re.compile(r"^\s*Progress\s+(\d+)\s+of\s+(\d+)\s*$")
    _bar_width = 40

    def __init__(self) -> None:
        self._samples: deque[tuple[float, int]] = deque()

    @staticmethod
    def _amounts(current: int, total: int) -> str:
        units = (
            (1_000_000_000, "GB"),
            (1_000_000, "MB"),
            (1_000, "kB"),
        )
        for scale, suffix in units:
            if total >= scale:
                return f"{current / scale:.1f}/{total / scale:.1f} {suffix}"
        return f"{current}/{total} bytes"

    @staticmethod
    def _speed(value: float) -> str:
        units = (
            (1_000_000_000, "GB/s"),
            (1_000_000, "MB/s"),
            (1_000, "kB/s"),
        )
        for scale, suffix in units:
            if value >= scale:
                return f"{value / scale:.1f} {suffix}"
        return f"{value:.0f} B/s"

    @staticmethod
    def _eta(seconds: float) -> str:
        rounded = max(0, int(seconds + 0.5))
        hours, remainder = divmod(rounded, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours}:{minutes:02d}:{seconds:02d}"

    def format(self, line: str) -> str:
        if line.lstrip().startswith("Downloading "):
            self._samples.clear()
            return line
        match = self._progress_pattern.fullmatch(line)
        if match is None:
            return line
        current, total = (int(value) for value in match.groups())
        now = time.monotonic()
        self._samples.append((now, current))
        while len(self._samples) > 2 and now - self._samples[0][0] > 5:
            self._samples.popleft()
        elapsed = now - self._samples[0][0]
        transferred = current - self._samples[0][1]
        speed = transferred / elapsed if elapsed > 0 else 0.0
        ratio = min(1.0, current / total) if total > 0 else 0.0
        filled = min(self._bar_width, int(ratio * self._bar_width))
        bar = "━" * filled + "─" * (self._bar_width - filled)
        details = [bar, self._amounts(current, total)]
        if speed > 0:
            details.append(self._speed(speed))
            if total > current:
                details.extend(("eta", self._eta((total - current) / speed)))
        return "   " + " ".join(details)


def run_logged_process(
    command: Sequence[str],
    log: LogCallback,
    *,
    timeout: float,
    cancel_event: threading.Event | None = None,
    pip_progress: bool = False,
) -> tuple[int, str]:
    """Run a child process while forwarding its merged output line by line."""
    creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    process = subprocess.Popen(
        list(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        creationflags=creationflags,
    )
    timed_out = threading.Event()
    cancelled = threading.Event()
    watcher_finished = threading.Event()

    def kill_process() -> None:
        try:
            process.kill()
        except OSError:
            pass

    def terminate() -> None:
        if process.poll() is not None:
            return
        timed_out.set()
        kill_process()

    def watch_cancellation() -> None:
        if cancel_event is None:
            return
        while process.poll() is None and not watcher_finished.is_set():
            if cancel_event.wait(0.1):
                cancelled.set()
                log("Cancellation requested; stopping child process")
                kill_process()
                return

    timer = threading.Timer(timeout, terminate)
    timer.daemon = True
    timer.start()
    watcher = threading.Thread(
        target=watch_cancellation,
        name="dependency-cancel-watcher",
        daemon=True,
    )
    watcher.start()
    tail: deque[str] = deque(maxlen=40)
    progress_formatter = PipProgressFormatter() if pip_progress else None
    try:
        if process.stdout is not None:
            for raw_line in process.stdout:
                line = raw_line.rstrip("\r\n")
                if not line:
                    continue
                if progress_formatter is not None:
                    line = progress_formatter.format(line)
                tail.append(line)
                log(line)
        return_code = process.wait()
    finally:
        watcher_finished.set()
        timer.cancel()
        if process.stdout is not None:
            process.stdout.close()
    if cancelled.is_set():
        raise OperationCancelled("Installation or download cancelled")
    if timed_out.is_set():
        raise TimeoutError(
            f"Process exceeded its {timeout:.0f}-second time limit"
        )
    return return_code, "\n".join(tail)


def install_packages(
    packages: Sequence[str],
    *,
    mirror: str,
    log: LogCallback,
    timeout: float,
    options: Sequence[str] = (),
    cancel_event: threading.Event | None = None,
    index_url: str = "",
    index_label: str = "",
) -> None:
    """Install packages with a selected index without changing pip config."""
    selected = PYPI_MIRRORS.get(mirror)
    if selected is None:
        raise ValueError(f"Unknown PyPI mirror {mirror!r}")
    resolved_index_url = index_url or selected.index_url
    resolved_index_label = index_label or selected.label
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--progress-bar",
        "raw",
        "--index-url",
        resolved_index_url,
        "--upgrade",
        *options,
        *packages,
    ]
    log(f"Using {resolved_index_label}: {resolved_index_url}")
    if index_url and selected.index_url != resolved_index_url:
        extra_index_position = command.index("--upgrade")
        command[extra_index_position:extra_index_position] = [
            "--extra-index-url",
            selected.index_url,
        ]
        log(f"Using {selected.label} for dependencies: {selected.index_url}")
    log(f"Command: {subprocess.list2cmdline(command)}")
    return_code, detail = run_logged_process(
        command,
        log,
        timeout=timeout,
        cancel_event=cancel_event,
        pip_progress=True,
    )
    if return_code != 0:
        raise RuntimeError(f"pip install failed with exit code {return_code}: {detail}")


def _sha256_file(path: object) -> bytes:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.digest()


def _verify_requirements(
    requirements: Sequence[PackageRequirement],
    extra_imports: Sequence[str],
) -> None:
    for requirement in requirements:
        distribution = importlib.metadata.distribution(requirement.distribution)
        if distribution.version != requirement.version:
            raise RuntimeError(
                f"{requirement.distribution} {distribution.version} is installed; "
                f"{requirement.version} is required"
            )
        if requirement.verify_integrity:
            checked = 0
            for item in distribution.files or ():
                if item.hash is None or item.hash.mode != "sha256":
                    continue
                path = distribution.locate_file(item)
                if not path.is_file():
                    raise RuntimeError(
                        f"{requirement.distribution} file is missing: {item}"
                    )
                actual = base64.urlsafe_b64encode(
                    _sha256_file(path)
                ).decode("ascii").rstrip("=")
                if actual != item.hash.value:
                    raise RuntimeError(
                        f"{requirement.distribution} file failed integrity "
                        f"check: {item}"
                    )
                checked += 1
            if checked == 0:
                raise RuntimeError(
                    f"{requirement.distribution} has no verifiable wheel records"
                )
        if requirement.import_name:
            module = importlib.import_module(requirement.import_name)
            if requirement.import_version_attribute:
                imported_version = getattr(
                    module, requirement.import_version_attribute, None
                )
                if imported_version != requirement.version:
                    raise RuntimeError(
                        f"{requirement.import_name} import version does not match "
                        "its package"
                    )
    for module_name in extra_imports:
        importlib.import_module(module_name)


def _isolated_check(payload: str) -> None:
    parsed = json.loads(payload)
    requirements = [
        PackageRequirement(**requirement)
        for requirement in parsed["requirements"]
    ]
    if parsed.get("require_nvidia_cuda"):
        import torch

        if not torch.version.cuda:
            raise RuntimeError(
                "CPU-only PyTorch is installed; NVIDIA CUDA PyTorch is required"
            )
        if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
            raise RuntimeError(
                "NVIDIA CUDA is not available; check the GPU and NVIDIA driver"
            )
        device_names = [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ]
        if not any("NVIDIA" in name.upper() for name in device_names):
            raise RuntimeError("Qwen3-TTS requires an NVIDIA CUDA GPU")
    _verify_requirements(requirements, parsed["extra_imports"])


def dependency_status(
    requirements: Sequence[PackageRequirement],
    *,
    extra_imports: Sequence[str] = (),
    timeout: float = 180,
    require_nvidia_cuda: bool = False,
) -> tuple[bool, str]:
    """Check pinned versions, imports, and wheel RECORD hashes in a clean process."""
    payload = json.dumps(
        {
            "requirements": [asdict(requirement) for requirement in requirements],
            "extra_imports": list(extra_imports),
            "require_nvidia_cuda": require_nvidia_cuda,
        }
    )
    result_marker = "__LIVE_GPT_DEPENDENCY_RESULT__:"
    script = (
        "import json, sys\n"
        "from live_gpt.voice.dependencies import _isolated_check\n"
        "try:\n"
        "    _isolated_check(sys.argv[1])\n"
        "    result = {'ok': True, 'message': ''}\n"
        "except Exception as error:\n"
        "    result = {'ok': False, 'message': str(error)}\n"
        f"print({result_marker!r} + json.dumps(result), flush=True)\n"
    )
    try:
        completed = subprocess.run(
            [sys.executable, "-c", script, payload],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
        output = completed.stdout.splitlines()
        result_line = next(
            (
                line[len(result_marker) :]
                for line in reversed(output)
                if line.startswith(result_marker)
            ),
            "",
        )
        if not result_line:
            detail = completed.stderr.strip() or completed.stdout.strip()
            return False, detail or "Dependency check produced no result"
        result = json.loads(result_line)
        return bool(result["ok"]), str(result["message"])
    except Exception as error:
        return False, f"Dependency check failed: {error}"


__all__ = [
    "PYPI_MIRRORS",
    "OperationCancelled",
    "PackageRequirement",
    "PipProgressFormatter",
    "PypiMirror",
    "dependency_status",
    "install_packages",
    "run_logged_process",
]
