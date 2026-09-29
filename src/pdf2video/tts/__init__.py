"""Text-to-speech engines."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class TTSError(Exception):
    pass


@dataclass(frozen=True)
class Voice:
    id: str
    name: str
    language: str  # ISO 639-1, e.g. "en"
    locale: str  # e.g. "en-US"
    gender: str = ""

    @property
    def label(self) -> str:
        extra = f", {self.gender}" if self.gender else ""
        return f"{self.name} ({self.locale}{extra})"


class TTSEngine(Protocol):
    name: str
    label: str
    suffix: str  # extension of the files ``synthesize`` writes, e.g. ".mp3"

    def voices(self) -> list[Voice]: ...

    def default_voice(self, language: str) -> str: ...

    def synthesize(self, text: str, voice: str, rate: float, out: Path) -> None:
        """Write speech for ``text`` to ``out`` (whose extension is ``suffix``)."""
        ...

    # Optional: engines with downloadable voices also implement
    # ``prepare(voice, on_progress)`` to fetch the voice before synthesis starts,
    # and engines for only some languages implement ``supports(language) -> bool``
    # (the narrator then uses FALLBACK_ENGINE for the others).


ENGINES = {
    "kokoro": "Kokoro neural voices (offline, most natural; Piper for other languages)",
    "piper": "Piper neural voices (offline)",
    "edge": "Microsoft Edge neural voices (online)",
    "macos": "macOS voices (offline)",
}


FALLBACK_ENGINE = "piper"


def get_engine(name: str) -> TTSEngine:
    match name:
        case "edge":
            from .edge import EdgeTTS

            return EdgeTTS()
        case "macos":
            from .macos import MacSay

            return MacSay()
        case "piper":
            from .piper import PiperTTS

            return PiperTTS()
        case "kokoro":
            from .kokoro import KokoroTTS

            return KokoroTTS()
    raise TTSError(f"Unknown TTS engine: {name}")


def synthesize_with_retry(
    engine: TTSEngine, text: str, voice: str, rate: float, out: Path, attempts: int = 3
) -> None:
    for attempt in range(1, attempts + 1):
        try:
            engine.synthesize(text, voice, rate, out)
            if out.exists() and out.stat().st_size > 0:
                return
            raise TTSError("TTS engine produced no audio")
        except Exception as exc:
            if attempt == attempts:
                raise TTSError(f"{engine.label}: speech synthesis failed: {exc}") from exc
            time.sleep(1.5 * attempt)
