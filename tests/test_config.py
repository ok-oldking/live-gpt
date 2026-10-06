import json
import tempfile
import unittest
from pathlib import Path

from live_gpt.config import Config, DEFAULT_CONFIG
from live_gpt.localization import resolve_language


class ConfigTests(unittest.TestCase):
    def test_existing_config_defaults_cursor_capture_on(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text('{"version": 1}', encoding="utf-8")
            config = Config(path)
            self.assertTrue(config["capture_cursor"])
            config["capture_cursor"] = False
            self.assertFalse(Config(path)["capture_cursor"])

    def test_weight_paths_are_saved_atomically_and_incomplete_pair_is_repaired(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            config = Config(path)
            config["sovits_ckpt_path"] = "voice.ckpt"
            self.assertEqual(config["sovits_ckpt_path"], "")
            config.update(sovits_ckpt_path="voice.ckpt", sovits_pth_path="voice.pth")
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual((saved["sovits_ckpt_path"], saved["sovits_pth_path"]),
                             ("voice.ckpt", "voice.pth"))
            config["sovits_pth_path"] = ""
            self.assertEqual(config["sovits_pth_path"], "voice.pth")
            config.update(sovits_ckpt_path="", sovits_pth_path="")
            self.assertEqual(Config(path)["sovits_ckpt_path"], "")
            saved["sovits_pth_path"] = ""
            path.write_text(json.dumps(saved), encoding="utf-8")
            repaired = Config(path)
            self.assertEqual((repaired["sovits_ckpt_path"], repaired["sovits_pth_path"]), ("", ""))

    def test_removed_playback_engines_migrate_to_browser(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            for engine in ("qwen", "cosyvoice"):
                path.write_text(json.dumps({"playing_backend": engine, "tts_model": "old"}), encoding="utf-8")
                config = Config(path)
                self.assertEqual(config["playing_backend"], "web")
                self.assertNotIn("tts_model", config)

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

            expected = {**DEFAULT_CONFIG, "language": resolve_language()}
            self.assertEqual(dict(config), expected)
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")),
                expected,
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

            expected = {**DEFAULT_CONFIG, "language": resolve_language()}
            self.assertEqual(dict(config), expected)
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")),
                expected,
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

            self.assertEqual(config["language"], resolve_language())
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
            config["stt_model"] = "en_moonshine_tiny_int8"
            config["playing_backend"] = "cosyvoice"
            config["playing_backend"] = "sovits"
            config["sovits_installation"] = "E:/tts/GPT-SoVITS"
            config["sovits_text_lang"] = "zh"
            config["sovits_ref_audio_path"] = "E:/voices/sovits.wav"
            config["sovits_prompt_text"] = "参考文本"
            config["sovits_prompt_lang"] = "all_zh"
            config["recording_backend"] = "unknown"
            config["playing_backend"] = "unknown"
            config["pypi_mirror"] = "unknown"
            config["stt_model"] = "unknown"

            self.assertEqual(config["recording_backend"], "sherpa")
            self.assertEqual(config["playing_backend"], "sovits")
            self.assertEqual(config["pypi_mirror"], "sjtug")
            self.assertEqual(config["stt_model"], "en_moonshine_tiny_int8")
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
            self.assertEqual(config["playing_backend"], "web")
            self.assertNotIn("voice_backend", config)
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertNotIn("voice_backend", saved)
            self.assertEqual(saved["recording_backend"], "sherpa")

    def test_removed_sherpa_playback_migrates_to_browser(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            old = dict(DEFAULT_CONFIG)
            old["playing_backend"] = "sherpa"
            path.write_text(json.dumps(old), encoding="utf-8")

            config = Config(path)

            self.assertEqual(config["playing_backend"], "web")
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8"))["playing_backend"],
                "web",
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
