from __future__ import annotations

import threading
from collections.abc import Iterable
from typing import Any, Callable, Protocol


ProgressCallback = Callable[[str, int | None], None]
LogCallback = Callable[[str], None]


class ModelProvider(Protocol):
    """Common setup surface used by the settings window."""

    def dependency_status(self) -> tuple[bool, str]: ...

    def install_dependencies(
        self,
        progress: ProgressCallback | None = None,
        log: LogCallback | None = None,
        mirror: str = "default",
        cancel_event: threading.Event | None = None,
    ) -> str: ...

    def model_status(self, model_type: str, model_key: str) -> tuple[bool, str]: ...

    def download_model(
        self,
        model_type: str,
        model_key: str,
        progress: ProgressCallback | None = None,
        log: LogCallback | None = None,
        cancel_event: threading.Event | None = None,
    ) -> str: ...


class TextToSpeechProvider(ModelProvider, Protocol):
    def synthesize(
        self, model_key: str, text: str, speaker: str, language: str = "Auto"
    ) -> tuple[Any, int]: ...

    def synthesize_stream(
        self, model_key: str, text: str, speaker: str, language: str = "Auto"
    ) -> Iterable[tuple[Any, int, str]]: ...


__all__ = [
    "LogCallback",
    "ModelProvider",
    "ProgressCallback",
    "TextToSpeechProvider",
]
