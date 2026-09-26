"""macOS built-in ``say`` command (offline)."""

from __future__ import annotations

import re
import shutil
import subprocess
from functools import cache
from pathlib import Path

from . import TTSError, Voice

_LINE = re.compile(r"^(?P<name>.+?)\s+(?P<locale>[a-z]{2,3}_[A-Z0-9]{2,3})\s+#")
BASE_WPM = 180


@cache
def _voices() -> tuple[Voice, ...]:
    if not shutil.which("say"):
        return ()
    out = subprocess.run(["say", "-v", "?"], capture_output=True, text=True).stdout
    voices = []
    for line in out.splitlines():
        if m := _LINE.match(line):
            locale = m["locale"].replace("_", "-")
            voices.append(Voice(m["name"].strip(), m["name"].strip(), locale.split("-")[0], locale))
    return tuple(sorted(voices, key=lambda v: (v.locale, v.name)))


class MacSay:
    name = "macos"
    label = "macOS voices"
    suffix = ".aiff"

    def voices(self) -> list[Voice]:
        return list(_voices())

    def default_voice(self, language: str) -> str:
        matches = [v for v in _voices() if v.language == language]
        if not matches:
            matches = [v for v in _voices() if v.language == "en"]
        preferred = [v for v in matches if "(Enhanced)" in v.name or "(Premium)" in v.name]
        return (preferred or matches)[0].id if matches else "Samantha"

    def synthesize(self, text: str, voice: str, rate: float, out: Path) -> None:
        if not shutil.which("say"):
            raise TTSError("The macOS 'say' command is not available on this system")
        subprocess.run(
            ["say", "-v", voice, "-r", str(round(BASE_WPM * rate)), "-o", str(out), text],
            check=True,
            capture_output=True,
        )
