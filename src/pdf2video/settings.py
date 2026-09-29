"""User settings persisted as TOML in the XDG config directory."""

from __future__ import annotations

import os
import re
import tomllib
from dataclasses import asdict, dataclass, fields
from pathlib import Path


def config_path() -> Path:
    base = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return base / "pdf2video" / "settings.toml"


@dataclass
class Settings:
    max_chapter_minutes: float = 10.0
    tts_engine: str = "kokoro"  # "kokoro" (default) | "piper" | "edge" | "macos"
    voice: str = ""  # empty = pick automatically from the document language
    rate: float = 1.0  # speaking speed multiplier
    video_script: str = "auto"  # "auto" | "ai" | "extractive"
    llm_model: str = "qwen3:8b"  # Ollama-style name or path to a .gguf file
    llm_auto_download: bool = True  # download the models on first use if missing
    illustrations: bool = False  # AI adds diagrams / generated pictures to some scenes
    image_model: str = "sdxl-turbo"  # see pdf2video.imagegen.IMAGE_MODELS, or a file path
    book_images: bool = True  # with illustrations: may show the book's own images instead
    vision_model: str = "qwen3-vl-4b"  # describes the book's images (llm.vision.VISION_MODELS)
    ocr: str = "auto"  # "auto" (scanned pages only) | "always" | "off"
    ocr_languages: str = "auto"  # Tesseract codes, e.g. "eng+ces", or "auto"
    bitrate: str = "64k"

    @property
    def max_chapter_seconds(self) -> float:
        return self.max_chapter_minutes * 60

    def parse_options(self, on_progress=None):
        from .parsers import ParseOptions

        return ParseOptions(self.ocr, self.ocr_languages, on_progress)

    def llm(self):
        from .llm import LocalLLM

        return LocalLLM(self.llm_model, self.llm_auto_download)

    @classmethod
    def load(cls) -> Settings:
        path = config_path()
        settings = cls()
        if not path.exists():
            return settings
        try:
            text = path.read_text(encoding="utf-8")
            # Older versions wrote Python booleans (True/False), which TOML rejects.
            text = re.sub(r"= (True|False)\s*$", lambda m: f"= {m[1].lower()}", text, flags=re.M)
            data = tomllib.loads(text)
        except (OSError, tomllib.TOMLDecodeError):
            return settings
        for f in fields(cls):
            if f.name in data and isinstance(data[f.name], type(getattr(settings, f.name))):
                setattr(settings, f.name, data[f.name])
            elif f.name in data and isinstance(data[f.name], int) and f.type == "float":
                setattr(settings, f.name, float(data[f.name]))
        return settings

    def save(self) -> None:
        path = config_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        for key, value in asdict(self).items():
            if isinstance(value, str):
                escaped = value.replace("\\", "\\\\").replace('"', '\\"')
                lines.append(f'{key} = "{escaped}"')
            elif isinstance(value, bool):  # TOML booleans are lowercase
                lines.append(f"{key} = {str(value).lower()}")
            else:
                lines.append(f"{key} = {value}")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
