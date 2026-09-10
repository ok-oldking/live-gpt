from __future__ import annotations

import io
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from live_gpt.app import HotkeyConfigDialog
from live_gpt.config import Config
from live_gpt.pet import available_pets, default_pet_path
from live_gpt.pet_download import download_pet, pet_source


URL = "https://github.com/legeling/awesome-codex-pet/tree/main/pets/citlali--zaytsevzy"


class PetDownloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def responses(self):
        manifest = json.loads((default_pet_path() / "pet.json").read_text(encoding="utf-8-sig"))
        sheet = (default_pet_path() / manifest.get("spritesheetPath", "spritesheet.webp")).read_bytes()
        manifest["displayName"] = "Downloaded pet"
        manifest["spritesheetPath"] = "spritesheet.webp"
        return [io.BytesIO(json.dumps(manifest).encode()), io.BytesIO(sheet)]

    def test_download_validates_discovers_and_reuses_pet(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "pets"
            with patch("live_gpt.pet_download.urllib.request.urlopen", side_effect=self.responses()) as fetch:
                path = download_pet(URL, root)
                self.assertEqual(path, root / "citlali--zaytsevzy")
                self.assertEqual({p.name for p in path.iterdir()}, {"pet.json", "spritesheet.webp"})
                self.assertEqual(download_pet(URL, root), path)
                self.assertEqual(fetch.call_count, 2)
                self.assertTrue(fetch.call_args_list[0].args[0].full_url.endswith("/pet.json"))
            with patch("live_gpt.pet.downloaded_pets_path", return_value=root):
                self.assertIn(("Downloaded pet", str(path)), available_pets())

    def test_invalid_urls_do_not_download(self):
        for url in ("", "https://example.com/o/r/tree/main/pet", URL + "/%2e%2e", URL + "/%2fescape", URL + "/CON", URL.replace("/tree/", "/raw/")):
            with self.subTest(url=url), self.assertRaises(ValueError):
                pet_source(url)

    def test_blob_folder_download_uses_same_raw_files_as_tree(self):
        url = "https://github.com/legeling/awesome-codex-pet/blob/main/pets/rem--l1"
        self.assertEqual(pet_source(url), pet_source(url.replace("/blob/", "/tree/")))
        with tempfile.TemporaryDirectory() as temporary:
            with patch("live_gpt.pet_download.urllib.request.urlopen", side_effect=self.responses()) as fetch:
                path = download_pet(url, Path(temporary) / "pets")
                self.assertEqual(path.name, "rem--l1")
                self.assertEqual(
                    [call.args[0].full_url for call in fetch.call_args_list],
                    [f"https://raw.githubusercontent.com/legeling/awesome-codex-pet/main/pets/rem--l1/{name}"
                     for name in ("pet.json", "spritesheet.webp")],
                )

    def test_failed_or_invalid_download_never_publishes_pet(self):
        for responses in (
            [io.BytesIO(b"{}"), OSError("network failure")],
            [io.BytesIO(b"{}"), io.BytesIO(b"invalid image")],
            [io.BytesIO(b'{"spritesheetPath":"../outside.webp"}'), io.BytesIO(b"invalid image")],
        ):
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary) / "pets"
                with patch("live_gpt.pet_download.urllib.request.urlopen", side_effect=responses):
                    with self.assertRaises((OSError, ValueError)):
                        download_pet(URL, root)
                self.assertEqual(list(root.iterdir()), [])
                self.assertEqual(list(root.parent.glob(".pet-*")), [])

    def test_background_download_selects_and_saves_pet(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "pets"
            config = Config(Path(temporary) / "config.json")
            dialog = HotkeyConfigDialog("Right Alt", config=config)
            changes = []
            dialog.pet_settings_changed.connect(lambda *args: changes.append(args))
            try:
                with (
                    patch("live_gpt.pet_download.downloaded_pets_path", return_value=root),
                    patch("live_gpt.pet.downloaded_pets_path", return_value=root),
                    patch("live_gpt.pet_download.urllib.request.urlopen", side_effect=self.responses()),
                ):
                    dialog.pet_url_edit.setText(URL)
                    dialog.pet_download_button.click()
                    self.assertFalse(dialog.pet_download_button.isEnabled())
                    deadline = time.monotonic() + 5
                    while dialog._pet_download_thread is not None and time.monotonic() < deadline:
                        QApplication.processEvents()
                        time.sleep(0.01)
                    self.assertIsNone(dialog._pet_download_thread)
                    selected = str(root / "citlali--zaytsevzy")
                    self.assertEqual(config["pet_path"], selected)
                    self.assertEqual(changes[-1][0], selected)
                    self.assertEqual(dialog.pet_list.currentItem().text(), "Downloaded pet")
                    self.assertEqual(Config(config.path)["pet_path"], selected)
                    self.assertTrue(dialog.pet_download_button.isEnabled())
                    dialog._pet_download_completed(False, "network failure")
                    self.assertEqual(config["pet_path"], selected)
            finally:
                if dialog._pet_download_thread is not None:
                    dialog._pet_download_thread.wait(5000)
                    QApplication.processEvents()
                dialog.close()


if __name__ == "__main__":
    unittest.main()
