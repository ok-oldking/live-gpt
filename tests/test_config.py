import json
import tempfile
import unittest
from pathlib import Path

from live_gpt.config import Config, DEFAULT_CONFIG


class ConfigTests(unittest.TestCase):
    def test_recording_language_migration_and_model_compatibility(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            for model, expected in (("en_moonshine_tiny_int8", "en"), ("zh_sense_voice_small_int8", "auto")):
                path.write_text(json.dumps({"stt_model": model}), encoding="utf-8")
                config = Config(path)
                self.assertEqual(config["stt_language"], expected)
                self.assertEqual(config["stt_model"], model)
            path.write_text(json.dumps({"stt_model": "en_moonshine_tiny_int8", "stt_language": "auto"}), encoding="utf-8")
            config = Config(path)
            self.assertEqual(config["stt_model"], "zh_sense_voice_small_int8")
            self.assertEqual(Config(path)["stt_language"], "auto")

    def test_disabled_legacy_shortcut_becomes_empty(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({
                "hotkey_hold": "Right Alt",
                "hotkey_hold_enabled": False,
                "hotkey_hold_without_screenshot": "Right Ctrl",
                "hotkey_hold_without_screenshot_enabled": True,
            }), encoding="utf-8")
            config = Config(path)
            self.assertEqual(config["hotkey_hold"], "")
            self.assertEqual(config["hotkey_hold_without_screenshot"], "Right Ctrl")
            self.assertNotIn("hotkey_hold_enabled", config)
            self.assertEqual(Config(path)["hotkey_hold"], "")

    def test_legacy_shortcuts_migrate_and_custom_bindings_survive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({
                "hotkey_hold": "CapsLock",
                "hotkey_hold_without_screenshot": "Shift",
                "hotkey_send": "Ctrl+S",
                "hotkey_send_enabled": True,
            }), encoding="utf-8")
            config = Config(path)
            self.assertEqual(config["hotkey_hold"], "Right Alt")
            self.assertEqual(config["hotkey_hold_without_screenshot"], "Right Ctrl")
            self.assertNotIn("hotkey_send", config)
            self.assertNotIn("hotkey_send_enabled", config)
            config["hotkey_hold"] = "Left Shift+A"
            self.assertEqual(Config(path)["hotkey_hold"], "Left Shift+A")

    def test_missing_file_uses_defaults_and_creates_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"

            config = Config(path)

            self.assertEqual(dict(config), DEFAULT_CONFIG)
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")),
                DEFAULT_CONFIG,
            )

    def test_assignment_saves_automatically(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            config = Config(path)

            config["language"] = "zh"
            config["auto_hide"] = True

            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(saved["language"], "zh")
            self.assertTrue(saved["auto_hide"])

    def test_invalid_json_falls_back_to_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text("not valid json", encoding="utf-8")

            config = Config(path)

            self.assertEqual(dict(config), DEFAULT_CONFIG)
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")),
                DEFAULT_CONFIG,
            )

    def test_invalid_values_and_unknown_keys_are_repaired_on_load(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            loaded = dict(DEFAULT_CONFIG)
            loaded["language"] = "invalid"
            loaded["window_geometry"] = [1, 2, 3]
            loaded["unknown"] = True
            path.write_text(json.dumps(loaded), encoding="utf-8")

            config = Config(path)

            self.assertEqual(config["language"], "en")
            self.assertEqual(config["window_geometry"], [])
            self.assertNotIn("unknown", config)

    def test_invalid_assignment_keeps_last_valid_value(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            config = Config(path)

            config["language"] = "zh"
            config["language"] = "unsupported"

            self.assertEqual(config["language"], "zh")
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8"))["language"],
                "zh",
            )

    def test_voice_backends_models_and_speaker_are_validated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            config = Config(path)

            config["recording_backend"] = "sherpa"
            config["playing_backend"] = "qwen"
            config["pypi_mirror"] = "sjtug"
            config["qwen_model_source"] = "modelscope"
            config["stt_model"] = "en_moonshine_tiny_int8"
            config["tts_model"] = "qwen3_tts_1_7b_custom_voice"
            config["tts_speaker"] = "Ryan"
            config["tts_language"] = "English"
            config["playing_backend"] = "cosyvoice"
            config["cosyvoice_model_source"] = "modelscope"
            config["cosyvoice_model"] = "fun_cosyvoice3_0_5b_2512"
            config["cosyvoice_prompt_audio"] = "E:/voices/reference.wav"
            config["cosyvoice_prompt_text"] = "Reference speech"
            config["playing_backend"] = "sovits"
            config["sovits_installation"] = "E:/tts/GPT-SoVITS"
            config["sovits_text_lang"] = "zh"
            config["sovits_ref_audio_path"] = "E:/voices/sovits.wav"
            config["sovits_prompt_text"] = "参考文本"
            config["sovits_prompt_lang"] = "all_zh"
            config["recording_backend"] = "unknown"
            config["playing_backend"] = "unknown"
            config["pypi_mirror"] = "unknown"
            config["qwen_model_source"] = "unknown"
            config["stt_model"] = "unknown"
            config["tts_model"] = "unknown"
            config["tts_speaker"] = "unknown"
            config["tts_language"] = "unknown"

            self.assertEqual(config["recording_backend"], "sherpa")
            self.assertEqual(config["playing_backend"], "sovits")
            self.assertEqual(config["pypi_mirror"], "sjtug")
            self.assertEqual(config["qwen_model_source"], "modelscope")
            self.assertEqual(config["stt_model"], "en_moonshine_tiny_int8")
            self.assertEqual(config["tts_model"], "qwen3_tts_1_7b_custom_voice")
            self.assertEqual(config["tts_speaker"], "Ryan")
            self.assertEqual(config["tts_language"], "English")
            self.assertEqual(config["cosyvoice_model_source"], "modelscope")
            self.assertEqual(
                config["cosyvoice_model"], "fun_cosyvoice3_0_5b_2512"
            )
            self.assertEqual(
                config["cosyvoice_prompt_audio"], "E:/voices/reference.wav"
            )
            self.assertEqual(
                config["cosyvoice_prompt_text"], "Reference speech"
            )
            self.assertEqual(config["sovits_installation"], "E:/tts/GPT-SoVITS")
            self.assertEqual(config["sovits_text_lang"], "zh")
            self.assertEqual(config["sovits_ref_audio_path"], "E:/voices/sovits.wav")
            self.assertEqual(config["sovits_prompt_text"], "参考文本")
            self.assertEqual(config["sovits_prompt_lang"], "all_zh")

    def test_legacy_voice_backend_migrates_to_both_independent_backends(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            legacy = dict(DEFAULT_CONFIG)
            legacy.pop("recording_backend")
            legacy.pop("playing_backend")
            legacy["voice_backend"] = "sherpa"
            path.write_text(json.dumps(legacy), encoding="utf-8")

            config = Config(path)

            self.assertEqual(config["recording_backend"], "sherpa")
            self.assertEqual(config["playing_backend"], "qwen")
            self.assertNotIn("voice_backend", config)
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertNotIn("voice_backend", saved)
            self.assertEqual(saved["recording_backend"], "sherpa")

    def test_removed_sherpa_playback_migrates_to_qwen(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            old = dict(DEFAULT_CONFIG)
            old["playing_backend"] = "sherpa"
            path.write_text(json.dumps(old), encoding="utf-8")

            config = Config(path)

            self.assertEqual(config["playing_backend"], "qwen")
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8"))["playing_backend"],
                "qwen",
            )

    def test_removed_pypi_mirror_migrates_to_default(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            old = dict(DEFAULT_CONFIG)
            old["pypi_mirror"] = "tsinghua"
            path.write_text(json.dumps(old), encoding="utf-8")

            config = Config(path)

            self.assertEqual(config["pypi_mirror"], "default")
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8"))["pypi_mirror"],
                "default",
            )


if __name__ == "__main__":
    unittest.main()
