import json
import tempfile
import unittest
from pathlib import Path

from live_gpt.config import Config, DEFAULT_CONFIG


class ConfigTests(unittest.TestCase):
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
            config["auto_send"] = True

            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(saved["language"], "zh")
            self.assertTrue(saved["auto_send"])

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
            config["recording_backend"] = "unknown"
            config["playing_backend"] = "unknown"
            config["pypi_mirror"] = "unknown"
            config["qwen_model_source"] = "unknown"
            config["stt_model"] = "unknown"
            config["tts_model"] = "unknown"
            config["tts_speaker"] = "unknown"

            self.assertEqual(config["recording_backend"], "sherpa")
            self.assertEqual(config["playing_backend"], "qwen")
            self.assertEqual(config["pypi_mirror"], "sjtug")
            self.assertEqual(config["qwen_model_source"], "modelscope")
            self.assertEqual(config["stt_model"], "en_moonshine_tiny_int8")
            self.assertEqual(config["tts_model"], "qwen3_tts_1_7b_custom_voice")
            self.assertEqual(config["tts_speaker"], "Ryan")

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
