"""Local vision-language model (llama.cpp + libmtmd) that describes the images of a book.

Used to summarise a document's own images, so that the script writer can decide
whether one of them fits a scene better than a generated picture. Like the other
models it runs in-process: the weights (a GGUF model plus its vision projector,
from Hugging Face) are downloaded on first use into
``~/.cache/pdf2video/vision-models`` (resumable, sha256-verified).
"""

from __future__ import annotations

import atexit
import base64
import gc
import io
import logging
import os
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from ..media import Cancelled
from . import stream_chat
from .store import ModelError, fetch_file, format_size

log = logging.getLogger(__name__)

N_CTX = 8192
MAX_SIDE = 896  # images are downscaled to this before they are shown to the model

ImageKind = Literal[
    "photo", "painting", "drawing", "diagram", "chart", "map", "table", "text", "decorative"
]


class ImageDescription(BaseModel):
    kind: ImageKind
    description: str


SYSTEM_PROMPT = (
    "You describe images from books for an assistant who cannot see them and has to "
    "decide where each image fits in an explanatory video about the book."
)
USER_PROMPT = """\
Describe this image from a book{caption}.
Answer with JSON only:
- "kind": photo, painting, drawing, diagram, chart, map, table, text (mostly a page of \
text), or decorative (a logo, ornament, border or blank area).
- "description": 1-3 English sentences: what the image shows (subject, setting, \
notable details) and any readable text or labels in it."""


@dataclass(frozen=True)
class HFFile:
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
class VisionModel:
    name: str
    note: str
    model: HFFile
    mmproj: HFFile  # the vision projector

    @property
    def files(self) -> tuple[HFFile, HFFile]:
        return (self.model, self.mmproj)

    @property
    def size(self) -> int:
        return self.model.size + self.mmproj.size


VISION_MODELS = {
    m.name: m
    for m in (
        VisionModel(
            "qwen3-vl-4b",
            "default, 3.0 GB",
            HFFile(
                "Qwen/Qwen3-VL-4B-Instruct-GGUF",
                "Qwen3VL-4B-Instruct-Q4_K_M.gguf",
                2_497_281_664,
                "66358cb18bb6b3b1b6675aa412c7a88ef01d228f481184d13668e5201c730a0a",
            ),
            HFFile(
                "Qwen/Qwen3-VL-4B-Instruct-GGUF",
                "mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf",
                453_974_304,
                "30ba2c7dd3127a4561b6cba9d13d0f711c91bdb38742e2f56d73c8cb596bd06d",
            ),
        ),
        VisionModel(
            "gemma3-4b",
            "3.3 GB",
            HFFile(
                "ggml-org/gemma-3-4b-it-GGUF",
                "gemma-3-4b-it-Q4_K_M.gguf",
                2_489_757_856,
                "882e8d2db44dc554fb0ea5077cb7e4bc49e7342a1f0da57901c0802ea21a0863",
            ),
            HFFile(
                "ggml-org/gemma-3-4b-it-GGUF",
                "mmproj-model-f16.gguf",
                851_251_104,
                "8c0fb064b019a6972856aaae2c7e4792858af3ca4561be2dbf649123ba6c40cb",
            ),
        ),
    )
}
DEFAULT_VISION_MODEL = "qwen3-vl-4b"

_lock = threading.Lock()
_loaded: tuple[str, object] | None = None  # (model name, llama_cpp.Llama) kept between jobs
_abort_epoch = 0  # bumped by abort(); a running description stops when it changes


def cache_dir() -> Path:
    base = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return base / "pdf2video" / "vision-models"


def resolve(name: str) -> VisionModel:
    try:
        return VISION_MODELS[name]
    except KeyError:
        raise ModelError(
            f"Unknown vision model {name!r} (choose {', '.join(VISION_MODELS)})"
        ) from None


def _is_present(file: HFFile) -> bool:
    return file.path.is_file() and file.path.stat().st_size == file.size


def runtime_available() -> bool:
    try:
        from llama_cpp.llama_chat_format import MTMDChatHandler  # noqa: F401
    except ImportError:
        return False
    return True


def abort() -> None:
    """Stop a running description after its current token (it raises Cancelled)."""
    global _abort_epoch
    _abort_epoch += 1


def unload() -> None:
    """Free the model (see :func:`pdf2video.llm.unload` for why this must run at exit)."""
    global _loaded
    abort()
    if not _lock.acquire(timeout=10):
        log.warning("vision model still busy; not freeing it")
        return
    try:
        if _loaded is None:
            return
        llm = _loaded[1]
        _loaded = None
        try:
            llm.close()
        except Exception:  # pragma: no cover - best effort
            log.debug("closing the vision model failed", exc_info=True)
    finally:
        _lock.release()
    gc.collect()


atexit.register(unload)


def _image_url(data: bytes) -> str:
    """The image as a downscaled JPEG data URL (large scans would use many tokens)."""
    from PIL import Image

    with Image.open(io.BytesIO(data)) as img:
        img = img.convert("RGB")
        img.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        img.save(buffer, "JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


@dataclass
class VisionStatus:
    ok: bool
    message: str


class VisionLLM:
    def __init__(self, model: str = DEFAULT_VISION_MODEL, auto_download: bool = True):
        self.model = model
        self.auto_download = auto_download

    def status(self) -> VisionStatus:
        if not runtime_available():
            return VisionStatus(False, "llama-cpp-python has no vision support")
        try:
            spec = resolve(self.model)
        except ModelError as exc:
            return VisionStatus(False, str(exc))
        missing = [f for f in spec.files if not _is_present(f)]
        if not missing:
            return VisionStatus(True, f"{spec.name} · {format_size(spec.size)} · downloaded")
        if not self.auto_download:
            return VisionStatus(False, f"{spec.name} is not downloaded")
        size = sum(f.size for f in missing)
        return VisionStatus(
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
            raise ModelError("llama-cpp-python has no vision support")
        spec = resolve(self.model)
        missing = [f for f in spec.files if not _is_present(f)]
        if missing:
            if not self.auto_download:
                raise ModelError(f"Vision model {spec.name} is not downloaded")
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
            from llama_cpp import Llama, mtmd_cpp
            from llama_cpp._logger import llama_log_callback
            from llama_cpp.llama_chat_format import MTMDChatHandler

            # libmtmd prints every prompt to stderr unless its log goes through
            # llama-cpp-python's logger, which honours verbose=False.
            mtmd_cpp.mtmd_log_set(llama_log_callback, None)
            mtmd_cpp.mtmd_helper_log_set(llama_log_callback, None)
            handler = MTMDChatHandler(clip_model_path=str(spec.mmproj.path), verbose=False)
            llm = Llama(
                model_path=str(spec.model.path),
                chat_handler=handler,
                n_ctx=N_CTX,
                n_gpu_layers=-1,
                verbose=False,
            )
            _loaded = (spec.name, llm)

    def describe(self, image: bytes, caption: str = "") -> ImageDescription:
        """What ``image`` shows; ``caption`` is the book's caption or alt text, if any."""
        spec = resolve(self.model)
        hint = f' (its caption in the book: "{caption}")' if caption else ""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": _image_url(image)}},
                    {"type": "text", "text": USER_PROMPT.format(caption=hint)},
                ],
            },
        ]
        epoch = _abort_epoch
        with _lock:
            if _abort_epoch != epoch:
                raise Cancelled()
            if _loaded is None or _loaded[0] != spec.name:
                raise ModelError("the vision model is not loaded")
            content, finish = stream_chat(
                _loaded[1],
                lambda: _abort_epoch != epoch,
                messages=messages,
                response_format={
                    "type": "json_object",
                    "schema": ImageDescription.model_json_schema(),
                },
                max_tokens=400,
                temperature=0.2,
            )
        if finish == "length":
            raise ValueError("the model's answer was cut off")
        described = ImageDescription.model_validate_json(content)
        described.description = " ".join(described.description.split())
        return described
