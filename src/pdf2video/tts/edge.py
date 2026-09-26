"""Microsoft Edge online neural voices via the ``edge-tts`` package."""

from __future__ import annotations

import asyncio
from functools import cache
from pathlib import Path

import edge_tts

from . import Voice

PREFERRED = {
    "en": "en-US-AndrewMultilingualNeural",
    "cs": "cs-CZ-AntoninNeural",
    "de": "de-DE-FlorianMultilingualNeural",
    "fr": "fr-FR-RemyMultilingualNeural",
    "es": "es-ES-AlvaroNeural",
    "it": "it-IT-GiuseppeMultilingualNeural",
    "pt": "pt-BR-AntonioNeural",
    "nl": "nl-NL-MaartenNeural",
    "pl": "pl-PL-MarekNeural",
}


@cache
def _voices() -> tuple[Voice, ...]:
    raw = asyncio.run(edge_tts.list_voices())
    voices = [
        Voice(
            id=v["ShortName"],
            name=v["ShortName"].split("-", 2)[-1].removesuffix("Neural"),
            language=v["Locale"].split("-")[0],
            locale=v["Locale"],
            gender=v.get("Gender", ""),
        )
        for v in raw
    ]
    return tuple(sorted(voices, key=lambda v: (v.locale, v.name)))


class EdgeTTS:
    name = "edge"
    label = "Edge neural voices"
    suffix = ".mp3"

    def voices(self) -> list[Voice]:
        return list(_voices())

    def default_voice(self, language: str) -> str:
        return PREFERRED.get(language, PREFERRED["en"])

    def synthesize(self, text: str, voice: str, rate: float, out: Path) -> None:
        percent = round((rate - 1.0) * 100)
        communicate = edge_tts.Communicate(text, voice, rate=f"{percent:+d}%")
        communicate.save_sync(str(out))
