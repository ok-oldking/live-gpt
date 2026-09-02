from __future__ import annotations

import io
import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from live_gpt.voice.sovits_tts import SovitsTtsProvider


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None


class SovitsTtsProviderTests(unittest.TestCase):
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
