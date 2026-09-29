"""Offline neural voices with Kokoro-82M (https://github.com/thewh1teagle/kokoro-onnx).

Much more natural intonation than Piper, but only for some languages: English
(US and UK), Spanish, French, Italian, Portuguese (Brazil) and Hindi. For other
languages the narrator falls back to Piper (see :meth:`KokoroTTS.supports`).

The model (``kokoro-v1.0.onnx``, 326 MB) and the voice styles (``voices-v1.0.bin``,
28 MB) are downloaded on first use into ``~/.cache/pdf2video/kokoro`` (resumable,
sha256-verified). After that, synthesis runs fully offline.
"""

from __future__ import annotations

import os
import threading
import wave
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from . import TTSError, Voice

RELEASE_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"


@dataclass(frozen=True)
class _File:
    name: str
    size: int
    sha256: str

    @property
    def path(self) -> Path:
        return cache_dir() / self.name


MODEL = _File(
    "kokoro-v1.0.onnx",
    325_532_387,
    "7d5df8ecf7d4b1878015a32686053fd0eebe2bc377234608764cc0ef3636a6c5",
)
VOICES = _File(
    "voices-v1.0.bin",
    28_214_398,
    "bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d",
)

# The first letter of a voice id is its language: prefix → (language, locale, espeak code).
# Japanese and Chinese voices need extra phonemizers and are left out.
LANGUAGES = {
    "a": ("en", "en-US", "en-us"),
    "b": ("en", "en-GB", "en-gb"),
    "e": ("es", "es-ES", "es"),
    "f": ("fr", "fr-FR", "fr-fr"),
    "h": ("hi", "hi-IN", "hi"),
    "i": ("it", "it-IT", "it"),
    "p": ("pt", "pt-BR", "pt-br"),
}
PREFERRED = {
    "en": "af_heart",
    "es": "ef_dora",
    "fr": "ff_siwis",
    "hi": "hf_alpha",
    "it": "if_sara",
    "pt": "pf_dora",
}
PARAGRAPH_PAUSE = 0.45  # seconds of silence between paragraphs (sentences get ~0.25 s)

_lock = threading.Lock()  # one synthesis at a time: ONNX Runtime already uses all cores
_model = None  # kokoro_onnx.Kokoro, kept between jobs


def cache_dir() -> Path:
    base = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return base / "pdf2video" / "kokoro"


def is_downloaded() -> bool:
    return all(f.path.is_file() and f.path.stat().st_size == f.size for f in (MODEL, VOICES))


# The ids stored in voices-v1.0.bin (checked by test_kokoro_voice_ids).
VOICE_IDS = (
    "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica", "af_kore", "af_nicole",
    "af_nova", "af_river", "af_sarah", "af_sky", "am_adam", "am_echo", "am_eric", "am_fenrir",
    "am_liam", "am_michael", "am_onyx", "am_puck", "am_santa", "bf_alice", "bf_emma",
    "bf_isabella", "bf_lily", "bm_daniel", "bm_fable", "bm_george", "bm_lewis", "ef_dora",
    "em_alex", "em_santa", "ff_siwis", "hf_alpha", "hf_beta", "hm_omega", "hm_psi", "if_sara",
    "im_nicola", "pf_dora", "pm_alex", "pm_santa",
)  # fmt: skip


class KokoroTTS:
    name = "kokoro"
    label = "Kokoro neural voices (offline, most natural)"
    suffix = ".wav"

    def supports(self, language: str) -> bool:
        return language in PREFERRED

    def voices(self) -> list[Voice]:
        note = "" if is_downloaded() else "350 MB download"
        voices = []
        for voice_id in VOICE_IDS:
            language, locale, _ = LANGUAGES[voice_id[0]]
            gender = "female" if voice_id[1] == "f" else "male"
            name = voice_id[3:].replace("_", " ").title()
            extra = f"{gender}, {note}" if note else gender
            voices.append(Voice(voice_id, name, language, locale, extra))
        return sorted(voices, key=lambda v: (v.locale, v.name))

    def default_voice(self, language: str) -> str:
        return PREFERRED.get(language, PREFERRED["en"])

    def prepare(self, voice: str, on_progress: Callable[[float, str], None] = lambda f, m: None):
        """Download the model and voices if they are not cached yet."""
        if is_downloaded():
            return
        from ..llm.store import fetch_file, format_size

        missing = [f for f in (MODEL, VOICES) if not f.path.is_file()]
        total = sum(f.size for f in missing) or 1
        offset = 0
        for file in missing:

            def progress(done: int, _total: int, offset: int = offset) -> None:
                on_progress(
                    (offset + done) / total,
                    f"Downloading Kokoro voices: {format_size(offset + done)} / "
                    f"{format_size(total)}",
                )

            try:
                fetch_file(
                    f"{RELEASE_URL}/{file.name}", file.path, f"sha256:{file.sha256}", file.size,
                    progress,
                )  # fmt: skip
            except Exception as exc:
                raise TTSError(f"Could not download {file.name}: {exc}") from exc
            offset += file.size

    def _load(self):
        global _model
        if _model is None:
            self.prepare("")
            from kokoro_onnx import Kokoro

            _model = Kokoro(str(MODEL.path), str(VOICES.path))
        return _model

    def synthesize(self, text: str, voice: str, rate: float, out: Path) -> None:
        import numpy as np

        prefix = LANGUAGES.get(voice[:1])
        if prefix is None or voice not in VOICE_IDS:
            raise TTSError(f"Unknown Kokoro voice {voice!r}")
        paragraphs = [" ".join(p.split()) for p in text.split("\n\n") if p.strip()]
        speed = min(2.0, max(0.5, rate))
        parts = []
        with _lock:
            model = self._load()
            for paragraph in paragraphs:
                audio, sample_rate = model.create(paragraph, voice, speed, prefix[2])
                parts.extend([audio, np.zeros(int(PARAGRAPH_PAUSE * sample_rate), audio.dtype)])
        if not parts:
            raise TTSError("nothing to say")
        samples = np.clip(np.concatenate(parts[:-1]), -1.0, 1.0)
        with wave.open(str(out), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes((samples * 32767).astype(np.int16).tobytes())
