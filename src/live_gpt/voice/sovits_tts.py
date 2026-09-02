from __future__ import annotations

import atexit
import json
import shutil
import socket
import struct
import subprocess
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterable
from pathlib import Path
from typing import Any


SOVITS_LANGUAGES = (
    "auto", "auto_yue", "zh", "en", "ja", "yue", "ko",
    "all_zh", "all_ja", "all_yue", "all_ko",
)


class SovitsTtsProvider:
    """Run an existing GPT-SoVITS installation through its embedded Python."""

    display_name = "GPT-SoVITS"
    continuous_audio_stream = True
    stream_prebuffer_seconds = 0.25

    def __init__(self) -> None:
        self.prompt_text = ""
        self.prompt_lang = "auto"
        self._process: subprocess.Popen[bytes] | None = None
        self._installation: Path | None = None
        self._server_installation: Path | None = None
        self._port: int | None = None
        self._lock = threading.RLock()
        atexit.register(self.close)

    def configure(self, prompt_text: str = "", prompt_lang: str = "auto") -> None:
        self.prompt_text = prompt_text.strip()
        self.prompt_lang = prompt_lang.lower()

    @staticmethod
    def _paths(installation: str | Path) -> tuple[Path, Path, Path]:
        root = Path(installation).expanduser().resolve()
        python = root / "runtime" / "python.exe"
        config = root / "GPT_SoVITS" / "configs" / "tts_infer.yaml"
        return root, python, config

    def dependency_status(self) -> tuple[bool, str]:
        if self._installation is None:
            return False, "Choose an existing GPT-SoVITS installation folder"
        root, python, config = self._paths(self._installation)
        missing = [str(path) for path in (python, config) if not path.is_file()]
        if missing:
            return False, "Missing GPT-SoVITS file: " + missing[0]
        return True, f"Embedded runtime found in {root}"

    def install_dependencies(self, *args: Any, **kwargs: Any) -> str:
        raise RuntimeError("GPT-SoVITS must already be installed; choose its folder")

    def download_model(self, *args: Any, **kwargs: Any) -> str:
        raise RuntimeError("GPT-SoVITS models are managed by its installation")

    def model_status(self, model_type: str, model_key: str) -> tuple[bool, str]:
        if model_type != "tts":
            return False, "GPT-SoVITS only provides text to speech"
        self._installation = Path(model_key).expanduser() if model_key else None
        ok, message = self.dependency_status()
        return ok, message if not ok else "GPT-SoVITS model configuration is available"

    @staticmethod
    def _free_port() -> int:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])

    def _health(self, timeout: float = 1.0) -> dict[str, Any]:
        assert self._port is not None
        with urllib.request.urlopen(
            f"http://127.0.0.1:{self._port}/health", timeout=timeout
        ) as response:
            return json.loads(response.read())

    def preload(self, model_key: str) -> str:
        started = time.perf_counter()
        self._ensure_server(model_key)
        return f"GPT-SoVITS server warmed in {time.perf_counter() - started:.1f} s"

    def _ensure_server(self, installation: str) -> None:
        root, python, config = self._paths(installation)
        with self._lock:
            if (
                self._process is not None
                and self._process.poll() is None
                and self._server_installation == root
            ):
                self._health()
                return
            self.close()
            self._installation = root
            ok, message = self.dependency_status()
            if not ok:
                raise RuntimeError(message)
            source = Path(__file__).with_name("sovits_server.py")
            destination = root / "sovits_server.py"
            shutil.copy2(source, destination)
            self._port = self._free_port()
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            self._process = subprocess.Popen(
                [
                    str(python), "-u", str(destination), "--port", str(self._port),
                    "--config", str(config),
                ],
                cwd=root,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=creationflags,
            )
            self._server_installation = root
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                if self._process.poll() is not None:
                    raise RuntimeError(
                        f"GPT-SoVITS server exited with code {self._process.returncode}"
                    )
                try:
                    if self._health().get("ready"):
                        return
                except (OSError, ValueError, urllib.error.URLError):
                    time.sleep(0.25)
            self.close()
            raise TimeoutError("GPT-SoVITS model warmup timed out after 180 seconds")

    def close(self) -> None:
        process = self._process
        self._process = None
        self._port = None
        self._server_installation = None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()

    @staticmethod
    def _read_exact(response: Any, size: int) -> bytes:
        chunks = bytearray()
        while len(chunks) < size:
            part = response.read(size - len(chunks))
            if not part:
                break
            chunks.extend(part)
        return bytes(chunks)

    def synthesize_stream(
        self, model_key: str, text: str, speaker: str, language: str = "auto"
    ) -> Iterable[tuple[Any, int, str]]:
        import numpy as np

        if not text.strip():
            raise ValueError("Text is required for GPT-SoVITS playback")
        if language not in SOVITS_LANGUAGES:
            raise ValueError(f"Unsupported GPT-SoVITS language {language!r}")
        self._ensure_server(model_key)
        assert self._port is not None
        payload = json.dumps(
            {
                "text": text,
                "text_lang": language,
                "ref_audio_path": speaker,
                "prompt_text": self.prompt_text,
                "prompt_lang": self.prompt_lang,
                "text_split_method": "cut5",
                "fragment_interval": 0.15,
            }
        ).encode()
        request = urllib.request.Request(
            f"http://127.0.0.1:{self._port}/tts",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                while True:
                    header = self._read_exact(response, 8)
                    if not header:
                        break
                    if len(header) != 8:
                        raise RuntimeError("Truncated GPT-SoVITS audio frame")
                    sample_rate, byte_count = struct.unpack("!II", header)
                    pcm = self._read_exact(response, byte_count)
                    if len(pcm) != byte_count:
                        raise RuntimeError("Truncated GPT-SoVITS audio data")
                    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32)
                    samples /= 32768.0
                    yield samples, sample_rate, ""
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            try:
                detail = json.loads(detail).get("message", detail)
            except ValueError:
                pass
            raise RuntimeError(f"GPT-SoVITS request failed: {detail}") from error

    def synthesize(
        self, model_key: str, text: str, speaker: str, language: str = "auto"
    ) -> tuple[Any, int]:
        import numpy as np

        chunks = list(self.synthesize_stream(model_key, text, speaker, language))
        if not chunks:
            raise RuntimeError("GPT-SoVITS generated no audio")
        rate = chunks[0][1]
        if any(chunk_rate != rate for _, chunk_rate, _ in chunks):
            raise RuntimeError("GPT-SoVITS changed sample rate during synthesis")
        return np.concatenate([samples for samples, _, _ in chunks]), rate


__all__ = ["SOVITS_LANGUAGES", "SovitsTtsProvider"]
