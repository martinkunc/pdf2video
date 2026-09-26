from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from pdf2video.tts import Voice


class SilentTTS:
    """Offline fake engine: about 150 words per minute of silence."""

    name = "fake"
    label = "Fake"
    suffix = ".wav"

    def voices(self) -> list[Voice]:
        return [Voice("fake", "Fake", "en", "en-US")]

    def default_voice(self, language: str) -> str:
        return "fake"

    def synthesize(self, text: str, voice: str, rate: float, out: Path) -> None:
        seconds = max(0.5, len(text.split()) / 2.5)
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "anullsrc=r=22050:cl=mono",
                "-t",
                f"{seconds:.2f}",
                str(out),
            ],
            check=True,
        )


@pytest.fixture
def silent_tts() -> SilentTTS:
    return SilentTTS()


LOREM = (
    "The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs. "
    "How vexingly quick daft zebras jump. "
)


@pytest.fixture
def lorem() -> str:
    return LOREM
