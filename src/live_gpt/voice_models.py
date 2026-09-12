"""Compatibility imports for callers using the former combined voice module.

Provider implementations now live in separate modules under ``live_gpt.voice``.
"""

from .voice.sherpa_stt import (
    LocalDictationSession,
    SHERPA_ONNX_VERSION,
    STT_MODELS,
    SherpaSttProvider,
    SpeechToTextModel,
)

# Temporary compatibility alias for external imports. It is deliberately STT-only.
SherpaVoiceManager = SherpaSttProvider

__all__ = [
    "LocalDictationSession",
    "SHERPA_ONNX_VERSION",
    "STT_MODELS",
    "SherpaSttProvider",
    "SherpaVoiceManager",
    "SpeechToTextModel",
]
