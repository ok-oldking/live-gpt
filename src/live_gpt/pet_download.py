"""Download and validate a pet from a public GitHub folder."""
from __future__ import annotations

import json
import re
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

from PySide6.QtGui import QImage


def downloaded_pets_path() -> Path:
    return Path(__file__).resolve().parents[2] / "download" / "pets"


def pet_source(url: str) -> tuple[str, str]:
    parsed = urllib.parse.urlsplit(url.strip())
    parts = [urllib.parse.unquote(part) for part in parsed.path.strip("/").split("/")]
    if (
        parsed.scheme != "https" or parsed.netloc != "github.com"
        or len(parts) < 5 or parts[2] not in ("tree", "blob")
        or any(not re.fullmatch(r"[\w.-]+", part) or part in (".", "..") for part in parts)
        or any(part.endswith(".") for part in parts)
    ):
        raise ValueError("Enter a GitHub pet folder URL containing /tree/ or /blob/ before the branch name")
    name = parts[-1]
    if name.split(".")[0].upper() in {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}:
        raise ValueError("The pet folder name is not supported on Windows")
    raw_parts = parts[:2] + parts[3:]
    base = "https://raw.githubusercontent.com/" + "/".join(urllib.parse.quote(part) for part in raw_parts)
    return name, base


def validate_downloaded_pet(directory: Path) -> None:
    data = json.loads((directory / "pet.json").read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError("Pet manifest must be a JSON object")
    version = data.get("spriteVersionNumber", 1)
    if type(version) is not int or version not in (1, 2):
        raise ValueError("Pet spriteVersionNumber must be 1 or 2")
    if data.get("spritesheetPath", "spritesheet.webp") not in ("spritesheet.webp", "./spritesheet.webp"):
        raise ValueError("The pet manifest must reference spritesheet.webp in the same folder")
    sheet = QImage(str(directory / "spritesheet.webp"))
    height = (11 if version == 2 else 9) * 208
    if sheet.isNull() or sheet.width() != 1536 or sheet.height() != height:
        raise ValueError(f"Pet v{version} requires a 1536×{height} sprite sheet")


def download_pet(url: str, root: Path | None = None) -> Path:
    name, base = pet_source(url)
    root = (root or downloaded_pets_path()).resolve()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / name
    if destination.is_symlink() or destination.resolve().parent != root:
        raise ValueError("Pet destination must be inside download/pets")
    if destination.exists():
        validate_downloaded_pet(destination)
        return destination
    # Staging is outside the discovery root; incomplete pets never enter the list.
    with tempfile.TemporaryDirectory(prefix=".pet-", dir=root.parent) as temporary:
        staging = Path(temporary)
        for filename, limit in (("pet.json", 1024 * 1024), ("spritesheet.webp", 64 * 1024 * 1024)):
            request = urllib.request.Request(f"{base}/{filename}", headers={"User-Agent": "Live-GPT pet downloader"})
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = response.read(limit + 1)
            if len(payload) > limit:
                raise ValueError(f"{filename} exceeds the download size limit")
            (staging / filename).write_bytes(payload)
        validate_downloaded_pet(staging)
        staging.rename(destination)
    return destination
