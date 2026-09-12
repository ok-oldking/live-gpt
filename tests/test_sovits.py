from __future__ import annotations

import io
import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from live_gpt.voice.sovits_tts import SovitsTtsProvider


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None


class SovitsTtsProviderTests(unittest.TestCase):
    def test_model_pair_validation(self):
        self.assertEqual(SovitsTtsProvider.validate_weights("  ", ""), ("", ""))
        with tempfile.TemporaryDirectory() as directory:
            ckpt = Path(directory) / "中文 voice.ckpt"
            pth = Path(directory) / "中文 voice.pth"
            ckpt.touch()
            pth.touch()
            self.assertEqual(SovitsTtsProvider.validate_weights(str(ckpt), str(pth)),
                             (str(ckpt.resolve()), str(pth.resolve())))
            for pair in ((str(ckpt), ""), ("", str(pth)), (str(pth), str(ckpt)),
                         (str(ckpt), str(pth.with_name("missing.pth")))):
                with self.subTest(pair=pair), self.assertRaises(ValueError):
                    SovitsTtsProvider.validate_weights(*pair)

    def test_server_reloads_for_changed_pair_and_restores_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "runtime").mkdir()
            (root / "runtime/python.exe").touch()
            (root / "GPT_SoVITS/configs").mkdir(parents=True)
            (root / "GPT_SoVITS/configs/tts_infer.yaml").touch()
            ckpt, pth = root / "voice.ckpt", root / "voice.pth"
            ckpt.touch()
            pth.touch()
            provider = SovitsTtsProvider()
            self.addCleanup(provider.close)
            processes = [Mock(poll=Mock(return_value=None)) for _ in range(3)]
            with (patch("live_gpt.voice.sovits_tts.subprocess.Popen", side_effect=processes) as launch,
                  patch.object(provider, "_health", return_value={"ready": True})):
                provider.preload(str(root))
                self.assertNotIn("--ckpt", launch.call_args.args[0])
                provider.configure(ckpt_path=str(ckpt), pth_path=str(pth))
                provider.preload(str(root))
                processes[0].terminate.assert_called_once()
                command = launch.call_args.args[0]
                self.assertEqual(command[-4:], ["--ckpt", str(ckpt), "--pth", str(pth)])
                provider.preload(str(root))
                self.assertEqual(launch.call_count, 2)
                provider.configure()
                provider.preload(str(root))
                processes[1].terminate.assert_called_once()
                self.assertEqual(launch.call_count, 3)
                self.assertNotIn("--ckpt", launch.call_args.args[0])

    def test_existing_embedded_runtime_is_validated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "runtime").mkdir()
            (root / "runtime" / "python.exe").touch()
            (root / "GPT_SoVITS" / "configs").mkdir(parents=True)
            (root / "GPT_SoVITS" / "configs" / "tts_infer.yaml").touch()
            provider = SovitsTtsProvider()

            ok, _message = provider.model_status("tts", str(root))

            self.assertTrue(ok)
            self.assertTrue(provider.dependency_status()[0])

    def test_stream_frames_are_decoded_and_request_uses_config(self) -> None:
        first = np.asarray([0, 16384, -16384], dtype="<i2").tobytes()
        second = np.asarray([32767], dtype="<i2").tobytes()
        body = (
            struct.pack("!II", 32000, len(first)) + first
            + struct.pack("!II", 32000, len(second)) + second
        )
        captured = {}
        provider = SovitsTtsProvider()
        provider.configure("reference words", "en")
        provider._port = 12345

        def urlopen(request, timeout=0):
            captured["payload"] = json.loads(request.data)
            return _Response(body)

        with (
            patch.object(provider, "_ensure_server"),
            patch("urllib.request.urlopen", side_effect=urlopen),
        ):
            chunks = list(
                provider.synthesize_stream(
                    "C:/GPT-SoVITS", "hello", "C:/voice.wav", "en"
                )
            )

        self.assertEqual([chunk[1] for chunk in chunks], [32000, 32000])
        np.testing.assert_allclose(chunks[0][0], [0.0, 0.5, -0.5])
        self.assertEqual(captured["payload"]["prompt_text"], "reference words")
        self.assertEqual(captured["payload"]["prompt_lang"], "en")
        self.assertEqual(captured["payload"]["text_lang"], "en")


if __name__ == "__main__":
    unittest.main()
