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


if __name__ == "__main__":
    unittest.main()
