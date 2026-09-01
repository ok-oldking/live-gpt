"""Compatibility imports for callers using the former combined voice module.

Provider implementations now live in separate modules under ``live_gpt.voice``.
Sherpa-ONNX supports recording/STT only; Qwen3-TTS supports playback/TTS.
"""

from .voice.qwen_tts import (
    QWEN_SPEAKERS,
    QWEN_TTS_VERSION,
    QwenSpeaker,
    QwenTtsModel,
    QwenTtsProvider,
    TTS_MODELS,
)
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
    "QWEN_SPEAKERS",
    "QWEN_TTS_VERSION",
    "QwenSpeaker",
    "QwenTtsModel",
    "QwenTtsProvider",
    "SHERPA_ONNX_VERSION",
    "STT_MODELS",
    "SherpaSttProvider",
    "SherpaVoiceManager",
    "SpeechToTextModel",
    "TTS_MODELS",
]
