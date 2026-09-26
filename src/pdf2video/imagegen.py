"""Local text-to-image generation (stable-diffusion.cpp via ``stable-diffusion-cpp-python``).

Used for the *picture* illustrations of the video. Like the LLM, it runs in-process
and needs no server: the weights (GGUF files from Hugging Face) are downloaded on
first use into ``~/.cache/pdf2video/image-models`` (resumable, sha256-verified).

A path to a local ``.gguf`` / ``.safetensors`` checkpoint (SD 1.x/2.x/SDXL) works as
the model name too.
"""

from __future__ import annotations

import gc
import hashlib
import logging
import math
import os
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .llm.store import ModelError, fetch_file, format_size

if TYPE_CHECKING:
    from PIL import Image

log = logging.getLogger(__name__)

# Appended to every prompt: a consistent look that suits explanatory slides.
STYLE = (
    "clean modern editorial illustration, flat vector art style, soft lighting, "
    "harmonious muted colors, simple uncluttered composition, high quality, "
    "no text, no letters, no labels"
)
NEGATIVE = (
    "text, letters, words, caption, watermark, signature, logo, blurry, low quality, deformed"
)


@dataclass(frozen=True)
class ModelFile:
    role: str  # StableDiffusion() argument: model_path, diffusion_model_path, llm_path, vae_path
    repo: str
    name: str
    size: int
    sha256: str

    @property
    def url(self) -> str:
        return f"https://huggingface.co/{self.repo}/resolve/main/{self.name}"

    @property
    def path(self) -> Path:
        return cache_dir() / self.repo.replace("/", "--") / self.name


@dataclass(frozen=True)
class ImageModel:
    name: str
    note: str
    files: tuple[ModelFile, ...]
    resolution: int  # native side length; other aspect ratios keep the pixel count
    steps: int
    cfg_scale: float
    sampler: str = "euler"

    @property
    def size(self) -> int:
        return sum(f.size for f in self.files)


IMAGE_MODELS = {
    m.name: m
    for m in (
        ImageModel(
            "sdxl-turbo",
            "fast (default), 4.1 GB, a few seconds per picture",
            (
                ModelFile(
                    "model_path",
                    "OlegSkutte/sdxl-turbo-GGUF",
                    "sd_xl_turbo_1.0.q8_0.gguf",
                    4_098_988_672,
                    "f5659e6cdad699b1432870cc9fe811359e45c8a331687be505d0ba05c8364720",
                ),
            ),
            resolution=512,
            steps=4,
            cfg_scale=1.0,
            sampler="euler_a",
        ),
        ImageModel(
            "z-image-turbo",
            "best quality, 7.9 GB, ~1-2 min per picture on Apple Silicon",
            (
                ModelFile(
                    "diffusion_model_path",
                    "wbruna/Z-Image-Turbo-sdcpp-GGUF",
                    "z_image_turbo-Q4_0.gguf",
                    3_470_775_264,
                    "545c2a06510949e5c62a7154a44729f92dbf1c2006836508aedd72c537c35ba6",
                ),
                ModelFile(
                    "llm_path",
                    "wbruna/Z-Image-Turbo-sdcpp-GGUF",
                    "qwen_3_4b-Q8_0.gguf",
                    4_273_904_096,
                    "3a7216f3680bb1cfc3fbf01643d6104543ac0ec2456c3e928dae85f5da9941b6",
                ),
                ModelFile(
                    "vae_path",
                    "wbruna/Z-Image-Turbo-sdcpp-GGUF",
                    "ae-f16.gguf",
                    167_656_704,
                    "1bed7b05318709e46a8cb9accc211168fc7f0b61ab594661860bbfe4d785cc46",
                ),
            ),
            resolution=1024,
            steps=8,
            cfg_scale=1.0,
        ),
    )
}
DEFAULT_IMAGE_MODEL = "sdxl-turbo"

_lock = threading.Lock()
_loaded: tuple[str, object] | None = None  # (model name, StableDiffusion) kept between jobs


def cache_dir() -> Path:
    base = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return base / "pdf2video" / "image-models"


def runtime_available() -> bool:
    try:
        import stable_diffusion_cpp  # noqa: F401
    except ImportError:
        return False
    return True


def _custom(path: str) -> ImageModel:
    file = Path(path).expanduser()
    size = file.stat().st_size if file.is_file() else 0
    return ImageModel(
        path,
        "local file",
        (ModelFile("model_path", "", str(file), size, ""),),
        resolution=768 if "xl" in file.name.lower() else 512,
        steps=20,
        cfg_scale=7.0,
    )


def resolve(name: str) -> ImageModel:
    if name.lower().endswith((".gguf", ".safetensors", ".ckpt")):
        return _custom(name)
    try:
        return IMAGE_MODELS[name]
    except KeyError:
        raise ModelError(
            f"Unknown image model {name!r} (choose {', '.join(IMAGE_MODELS)} "
            "or a path to a .gguf/.safetensors file)"
        ) from None


def _local_path(file: ModelFile) -> Path:
    return Path(file.name) if not file.repo else file.path


def _is_present(file: ModelFile) -> bool:
    path = _local_path(file)
    return path.is_file() and (not file.repo or path.stat().st_size == file.size)


def dimensions(model: ImageModel, aspect: float) -> tuple[int, int]:
    """Width and height (multiples of 64) with the model's pixel count and ``aspect``."""
    aspect = min(max(aspect, 0.5), 2.0)
    area = model.resolution**2
    width = math.sqrt(area * aspect)
    return max(256, round(width / 64) * 64), max(256, round(area / width / 64) * 64)


def unload() -> None:
    global _loaded
    with _lock:
        if _loaded is None:
            return
        sd = _loaded[1]
        _loaded = None
        try:
            sd.close()
        except Exception:  # pragma: no cover - best effort
            log.debug("closing the image model failed", exc_info=True)
    gc.collect()


@dataclass
class ImageStatus:
    ok: bool
    message: str


class ImageGenerator:
    def __init__(self, model: str = DEFAULT_IMAGE_MODEL, auto_download: bool = True):
        self.model = model
        self.auto_download = auto_download

    def status(self) -> ImageStatus:
        if not runtime_available():
            return ImageStatus(False, "stable-diffusion-cpp-python is not installed")
        try:
            spec = resolve(self.model)
        except ModelError as exc:
            return ImageStatus(False, str(exc))
        missing = [f for f in spec.files if not _is_present(f)]
        if not missing:
            return ImageStatus(True, f"{spec.name} · {format_size(spec.size)} · downloaded")
        if any(not f.repo for f in missing):
            return ImageStatus(False, f"File not found: {self.model}")
        if not self.auto_download:
            return ImageStatus(False, f"{spec.name} is not downloaded")
        size = sum(f.size for f in missing)
        return ImageStatus(
            True, f"{spec.name} will be downloaded on first use ({format_size(size)})"
        )

    def prepare(
        self,
        on_progress: Callable[[float, str], None] = lambda fraction, message: None,
        check_cancel: Callable[[], None] = lambda: None,
    ) -> None:
        """Download (if needed) and load the model."""
        global _loaded
        if not runtime_available():
            raise ModelError("stable-diffusion-cpp-python is not installed")
        spec = resolve(self.model)
        missing = [f for f in spec.files if not _is_present(f)]
        if missing:
            if any(not f.repo for f in missing):
                raise ModelError(f"File not found: {self.model}")
            if not self.auto_download:
                raise ModelError(f"Image model {spec.name} is not downloaded")
            total = sum(f.size for f in missing)
            offset = 0
            for file in missing:

                def progress(done: int, _total: int, offset: int = offset) -> None:
                    on_progress(
                        (offset + done) / total,
                        f"Downloading {spec.name}: "
                        f"{format_size(offset + done)} / {format_size(total)}",
                    )

                fetch_file(
                    file.url, file.path, f"sha256:{file.sha256}", file.size, progress, check_cancel
                )
                offset += file.size
        on_progress(1.0, f"Loading {spec.name}…")
        with _lock:
            if _loaded is not None and _loaded[0] == spec.name:
                return
            if _loaded is not None:
                _loaded[1].close()
                _loaded = None
            from stable_diffusion_cpp import StableDiffusion

            kwargs = {f.role: str(_local_path(f)) for f in spec.files}
            sd = StableDiffusion(**kwargs, vae_decode_only=True, verbose=False)
            _loaded = (spec.name, sd)

    def generate(self, prompt: str, aspect: float = 1.0) -> Image.Image:
        """One picture for ``prompt`` (deterministic: the seed comes from the prompt)."""
        spec = resolve(self.model)
        width, height = dimensions(spec, aspect)
        seed = int.from_bytes(hashlib.sha256(prompt.encode()).digest()[:4], "big") & 0x7FFFFFFF
        with _lock:
            if _loaded is None or _loaded[0] != spec.name:
                raise ModelError("the image model is not loaded")
            images = _loaded[1].generate_image(
                prompt=f"{prompt.rstrip('. ')}. {STYLE}",
                negative_prompt=NEGATIVE if spec.cfg_scale > 1 else "",
                width=width,
                height=height,
                cfg_scale=spec.cfg_scale,
                sample_steps=spec.steps,
                sample_method=spec.sampler,
                seed=seed,
            )
        if not images:
            raise ModelError("the image model returned no picture")
        return images[0]

    def cache_key(self, prompt: str, aspect: float) -> str:
        spec = resolve(self.model)
        width, height = dimensions(spec, aspect)
        text = f"{spec.name}|{width}x{height}|{STYLE}|{prompt}"
        return hashlib.sha256(text.encode()).hexdigest()[:16]
