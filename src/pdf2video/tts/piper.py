"""Offline neural voices with Piper (https://github.com/OHF-Voice/piper1-gpl).

The voice catalogue (``voices.json``) and the voice models (``.onnx`` + ``.onnx.json``,
60-120 MB each) come from the ``rhasspy/piper-voices`` repository on Hugging Face.
They are downloaded on first use into ``~/.cache/pdf2video/voices`` and checked
against the catalogue's MD5 checksums. After that, synthesis runs fully offline.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import wave
from collections.abc import Callable
from pathlib import Path

import httpx

from . import TTSError, Voice

REPO_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main"
CATALOG_MAX_AGE = 7 * 24 * 3600

# Good-sounding default voice per language; other languages use the best-quality
# voice available in the catalogue.
PREFERRED = {
    "en": "en_US-ryan-high",
    "cs": "cs_CZ-jirka-medium",
    "de": "de_DE-thorsten-high",
    "fr": "fr_FR-siwis-medium",
    "es": "es_ES-davefx-medium",
    "it": "it_IT-paola-medium",
    "pt": "pt_BR-faber-medium",
    "nl": "nl_NL-mls-medium",
    "pl": "pl_PL-darkman-medium",
    "sk": "sk_SK-lili-medium",
}
_QUALITY_RANK = {"x_low": 0, "low": 1, "medium": 2, "high": 3}

_catalog_lock = threading.Lock()
_catalog: dict | None = None
_voice_lock = threading.Lock()
_loaded: dict[str, object] = {}  # voice id → PiperVoice
_synth_locks: dict[str, threading.Lock] = {}


def voices_dir() -> Path:
    base = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return base / "pdf2video" / "voices"


def _catalog_data() -> dict:
    """voices.json: from cache if fresh, else downloaded (falling back to a stale cache)."""
    global _catalog
    with _catalog_lock:
        if _catalog is not None:
            return _catalog
        path = voices_dir() / "voices.json"
        fresh = path.exists() and time.time() - path.stat().st_mtime < CATALOG_MAX_AGE
        if not fresh:
            try:
                response = httpx.get(f"{REPO_URL}/voices.json", follow_redirects=True, timeout=15)
                response.raise_for_status()
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(response.content)
            except httpx.HTTPError:
                if not path.exists():
                    return {}  # offline and never downloaded: only local voices
        _catalog = json.loads(path.read_text(encoding="utf-8"))
        return _catalog


def _model_files(voice_id: str) -> tuple[Path, Path]:
    return voices_dir() / f"{voice_id}.onnx", voices_dir() / f"{voice_id}.onnx.json"


def is_downloaded(voice_id: str) -> bool:
    return all(p.is_file() for p in _model_files(voice_id))


def _download_file(
    url: str,
    target: Path,
    md5: str | None,
    on_bytes: Callable[[int], None],
) -> None:
    partial = target.with_name(target.name + ".partial")
    hasher = hashlib.md5()
    try:
        with httpx.stream(
            "GET", url, follow_redirects=True, timeout=httpx.Timeout(30, read=120)
        ) as r:
            r.raise_for_status()
            with partial.open("wb") as f:
                for chunk in r.iter_bytes(1 << 20):
                    f.write(chunk)
                    hasher.update(chunk)
                    on_bytes(len(chunk))
    except httpx.HTTPError as exc:
        partial.unlink(missing_ok=True)
        raise TTSError(f"Could not download voice file {target.name}: {exc}") from exc
    if md5 and hasher.hexdigest() != md5:
        partial.unlink(missing_ok=True)
        raise TTSError(f"Voice file {target.name} is corrupted (checksum mismatch)")
    partial.rename(target)


class PiperTTS:
    name = "piper"
    label = "Piper neural voices (offline)"
    suffix = ".wav"

    def voices(self) -> list[Voice]:
        catalog = _catalog_data()
        voices = []
        ids = set(catalog)
        # Voices downloaded earlier stay usable even without the catalogue.
        ids |= {p.name.removesuffix(".onnx") for p in voices_dir().glob("*.onnx")}
        for voice_id in ids:
            info = catalog.get(voice_id, {})
            lang = info.get("language", {})
            code = lang.get("code") or voice_id.split("-")[0]
            name = info.get("name") or voice_id.split("-")[1] if "-" in voice_id else voice_id
            quality = info.get("quality", "")
            if is_downloaded(voice_id):
                note = "downloaded"
            else:
                size = sum(
                    f.get("size_bytes", 0)
                    for path, f in info.get("files", {}).items()
                    if path.endswith(".onnx")
                )
                note = f"{size / 1e6:.0f} MB download" if size else ""
            voices.append(
                Voice(
                    id=voice_id,
                    name=f"{name} · {quality}" if quality else name,
                    language=lang.get("family") or code.split("_")[0],
                    locale=code.replace("_", "-"),
                    gender=note,
                )
            )
        return sorted(voices, key=lambda v: (v.locale, v.name))

    def default_voice(self, language: str) -> str:
        preferred = PREFERRED.get(language)
        catalog = _catalog_data()
        if preferred and (preferred in catalog or is_downloaded(preferred)):
            return preferred
        candidates = [
            (_QUALITY_RANK.get(v.get("quality", ""), 0), v.get("num_speakers", 1) == 1, k)
            for k, v in catalog.items()
            if v.get("language", {}).get("family") == language
        ]
        if candidates:
            return max(candidates)[2]
        return PREFERRED["en"]

    def prepare(
        self, voice_id: str, on_progress: Callable[[float, str], None] = lambda f, m: None
    ) -> None:
        """Download the voice model if it is not cached yet."""
        if is_downloaded(voice_id):
            return
        info = _catalog_data().get(voice_id)
        if not info:
            raise TTSError(
                f"Unknown Piper voice {voice_id!r} (or no internet connection to download it)"
            )
        voices_dir().mkdir(parents=True, exist_ok=True)
        files = {
            path: meta
            for path, meta in info["files"].items()
            if path.endswith((".onnx", ".onnx.json"))
        }
        total = sum(meta["size_bytes"] for meta in files.values()) or 1
        done = 0

        def on_bytes(n: int) -> None:
            nonlocal done
            done += n
            on_progress(
                done / total,
                f"Downloading voice {voice_id}: {done / 1e6:.0f} / {total / 1e6:.0f} MB",
            )

        onnx, config = _model_files(voice_id)
        for path, meta in sorted(files.items(), key=lambda kv: kv[0].endswith(".onnx")):
            target = onnx if path.endswith(".onnx") else config
            _download_file(f"{REPO_URL}/{path}", target, meta.get("md5_digest"), on_bytes)

    def _voice(self, voice_id: str):
        with _voice_lock:
            if voice_id not in _loaded:
                self.prepare(voice_id)
                from piper import PiperVoice

                onnx, config = _model_files(voice_id)
                _loaded[voice_id] = PiperVoice.load(onnx, config_path=config)
                _synth_locks[voice_id] = threading.Lock()
            return _loaded[voice_id], _synth_locks[voice_id]

    def synthesize(self, text: str, voice: str, rate: float, out: Path) -> None:
        from piper import SynthesisConfig

        piper_voice, lock = self._voice(voice)
        config = SynthesisConfig(length_scale=1.0 / max(0.25, rate))
        # ONNX Runtime already uses all cores; one synthesis per voice at a time.
        with lock, wave.open(str(out), "wb") as wav:
            piper_voice.synthesize_wav(text, wav, syn_config=config)
