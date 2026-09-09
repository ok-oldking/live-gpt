from __future__ import annotations

import copy
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .logger import Logger


logger = Logger.get_logger(__name__)

DEFAULT_CONFIG: dict[str, Any] = {
    "version": 1,
    "language": "en",
    "hotkey_hold": "Right Alt",
    "hotkey_hold_without_screenshot": "Right Ctrl",
    "auto_hide": False,
    "capture_source": "",
    "chatgpt_window": "",
    "window_geometry": [],
    "window_locked": False,
    "pet_path": "",
    "pet_idle_mode": "always",
    "pet_idle_seconds": 10,
    "recording_backend": "web",
    "playing_backend": "web",
    "pypi_mirror": "default",
    "qwen_model_source": "huggingface",
    "cosyvoice_model_source": "huggingface",
    "stt_model": "zh_zipformer_ctc_int8_2025_07_03",
    "tts_model": "qwen3_tts_0_6b_custom_voice",
    "tts_speaker": "Vivian",
    "tts_language": "Auto",
    "cosyvoice_model": "fun_cosyvoice3_0_5b_2512",
    "cosyvoice_prompt_audio": "",
    "cosyvoice_prompt_text": "",
    "sovits_installation": "",
    "sovits_text_lang": "auto",
    "sovits_ref_audio_path": "",
    "sovits_prompt_text": "",
    "sovits_prompt_lang": "auto",
}


def default_config_path() -> Path:
    """Return the per-user JSON configuration path."""
    if sys.platform == "win32":
        root = Path(
            os.environ.get(
                "APPDATA",
                Path.home() / "AppData" / "Roaming",
            )
        )
        return root / "Live GPT" / "config.json"
    if sys.platform == "darwin":
        return (
            Path.home()
            / "Library"
            / "Application Support"
            / "Live GPT"
            / "config.json"
        )
    root = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return root / "live-gpt" / "config.json"


def _valid_value(key: str, value: Any, default: Any) -> bool:
    if type(value) is not type(default):
        return False
    if key == "pet_idle_mode":
        return value in ("always", "never", "timed")
    if key == "pet_idle_seconds":
        return 1 <= value <= 3600
    if key == "version":
        return value == DEFAULT_CONFIG["version"]
    if key == "language":
        return value in ("en", "zh")
    if key == "recording_backend":
        return value in ("web", "sherpa")
    if key == "playing_backend":
        return value in ("web", "qwen", "cosyvoice", "sovits")
    if key == "pypi_mirror":
        return value in ("default", "ali", "sjtug")
    if key == "qwen_model_source":
        return value in ("huggingface", "modelscope")
    if key == "cosyvoice_model_source":
        return value in ("huggingface", "modelscope")
    if key == "stt_model":
        return value in (
            "zh_zipformer_ctc_int8_2025_07_03",
            "zh_streaming_zipformer_ctc_int8_2025_06_30",
            "zh_streaming_zipformer_small_ctc_int8_2025_04_01",
            "zh_paraformer_int8",
            "zh_sense_voice_small_int8",
            "en_parakeet_tdt_ctc_110m_int8",
            "en_nemo_conformer_ctc_small",
            "en_moonshine_tiny_int8",
            "en_moonshine_base_int8",
            "en_paraformer_int8",
        )
    if key == "tts_model":
        return value in (
            "qwen3_tts_0_6b_custom_voice",
            "qwen3_tts_1_7b_custom_voice",
        )
    if key == "tts_speaker":
        return value in (
            "Vivian",
            "Serena",
            "Uncle_Fu",
            "Dylan",
            "Eric",
            "Ryan",
            "Aiden",
            "Ono_Anna",
            "Sohee",
        )
    if key == "tts_language":
        return value in (
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
    if key == "cosyvoice_model":
        return value == "fun_cosyvoice3_0_5b_2512"
    if key in ("sovits_text_lang", "sovits_prompt_lang"):
        return value in (
            "auto", "auto_yue", "zh", "en", "ja", "yue", "ko",
            "all_zh", "all_ja", "all_yue", "all_ko",
        )
    if key.startswith("hotkey_") and not key.endswith("_enabled"):
        return True
    if key == "window_geometry":
        return value == [] or (
            len(value) == 4
            and all(type(part) is int for part in value)
            and value[2] >= 760
            and value[3] >= 180
        )
    return True


class Config(dict[str, Any]):
    """Validated JSON settings that save automatically when changed."""

    def __init__(
        self,
        path: str | Path | None = None,
        default: Mapping[str, Any] | None = None,
    ) -> None:
        self.default = copy.deepcopy(dict(default or DEFAULT_CONFIG))
        self.path = Path(path) if path is not None else default_config_path()
        self.file_existed = self.path.is_file()
        loaded = self._read_file()
        migrated = False
        if isinstance(loaded, dict) and "hotkey_send" in loaded:
            # Update the old defaults while preserving customized recording keys.
            for key, old_default in (
                ("hotkey_hold", "CapsLock"),
                ("hotkey_hold_without_screenshot", "Shift"),
            ):
                if loaded.get(key) == old_default:
                    loaded[key] = DEFAULT_CONFIG[key]
                    migrated = True
        if isinstance(loaded, dict):
            for key in ("hotkey_hold", "hotkey_hold_without_screenshot"):
                if loaded.get(key + "_enabled") is False:
                    loaded[key] = ""
                    migrated = True
        if isinstance(loaded, dict) and "voice_backend" in loaded:
            loaded = dict(loaded)
            legacy_backend = loaded.pop("voice_backend")
            loaded.setdefault("recording_backend", legacy_backend)
            loaded.setdefault(
                "playing_backend",
                "qwen" if legacy_backend == "sherpa" else "web",
            )
            migrated = True
        if isinstance(loaded, dict) and loaded.get("playing_backend") == "sherpa":
            loaded = dict(loaded)
            loaded["playing_backend"] = "qwen"
            migrated = True
        verified, modified = self._verify(loaded)
        dict.__init__(self, verified)
        if modified or migrated:
            self.save_file()

    def _read_file(self) -> Any:
        if not self.file_existed:
            return None
        try:
            with self.path.open("r", encoding="utf-8") as stream:
                return json.load(stream)
        except Exception as error:
            logger.error(f"Unable to load config {self.path}", error)
            return None

    def _verify(self, loaded: Any) -> tuple[dict[str, Any], bool]:
        if not isinstance(loaded, dict):
            return copy.deepcopy(self.default), True

        modified = set(loaded) != set(self.default)
        verified: dict[str, Any] = {}
        for key, default in self.default.items():
            value = loaded.get(key, default)
            if not _valid_value(key, value, default):
                value = copy.deepcopy(default)
                modified = True
            verified[key] = value
        return verified, modified

    def save_file(self) -> None:
        """Atomically write the current settings to disk."""
        temporary_path = self.path.with_suffix(self.path.suffix + ".tmp")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with temporary_path.open("w", encoding="utf-8") as stream:
                json.dump(self, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
            os.replace(temporary_path, self.path)
        except Exception as error:
            logger.error(f"Unable to save config {self.path}", error)
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass

    def __setitem__(self, key: str, value: Any) -> None:
        if key not in self.default:
            logger.warning(f"Ignoring unknown config key {key!r}")
            return
        if not _valid_value(key, value, self.default[key]):
            logger.warning(f"Ignoring invalid config value for {key!r}")
            return
        if self.get(key) == value:
            return
        dict.__setitem__(self, key, copy.deepcopy(value))
        self.save_file()

    def update(self, *args: Any, **kwargs: Any) -> None:
        updates = dict(*args, **kwargs)
        changed = False
        for key, value in updates.items():
            if key not in self.default:
                logger.warning(f"Ignoring unknown config key {key!r}")
                continue
            if not _valid_value(key, value, self.default[key]):
                logger.warning(f"Ignoring invalid config value for {key!r}")
                continue
            if self.get(key) != value:
                dict.__setitem__(self, key, copy.deepcopy(value))
                changed = True
        if changed:
            self.save_file()

    def pop(self, key: str, default: Any = None) -> Any:
        result = dict.pop(self, key, default)
        self.save_file()
        return result

    def popitem(self) -> tuple[str, Any]:
        result = dict.popitem(self)
        self.save_file()
        return result

    def clear(self) -> None:
        if self:
            dict.clear(self)
            self.save_file()

    def reset_to_default(self) -> None:
        dict.clear(self)
        dict.update(self, copy.deepcopy(self.default))
        self.save_file()

    def get_default(self, key: str) -> Any:
        return copy.deepcopy(self.default.get(key))


__all__ = ["Config", "DEFAULT_CONFIG", "default_config_path"]
