"""Built-in local LLM runner (llama.cpp via ``llama-cpp-python``).

No external server is needed: GGUF weights are found in Ollama's model folder or
the app cache, or downloaded on first use (see :mod:`.store`).
"""

from __future__ import annotations

import atexit
import gc
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field

from ..media import Cancelled
from .store import (
    RECOMMENDED,
    LocalModel,
    ModelError,
    download,
    find_local,
    format_size,
    list_local,
    remote_size,
)

__all__ = [
    "RECOMMENDED",
    "LLMStatus",
    "LocalLLM",
    "ModelError",
    "list_local",
]

log = logging.getLogger(__name__)

N_CTX = 16_384
MAX_OUTPUT_TOKENS = 4096

_load_lock = threading.Lock()
_infer_lock = threading.Lock()  # one generation at a time; unload() waits for it
_loaded: tuple[str, object] | None = None  # (model path, llama_cpp.Llama) kept between jobs
_abort_epoch = 0  # bumped by abort(); a running generation stops when it changes


@dataclass
class LLMStatus:
    ok: bool  # usable now, or will be after an automatic download
    message: str
    models: list[str] = field(default_factory=list)  # names available offline
    local: LocalModel | None = None
    download_size: int | None = None


def abort() -> None:
    """Stop a running generation after its current token (it raises Cancelled)."""
    global _abort_epoch
    _abort_epoch += 1


def stream_chat(llm, aborted: Callable[[], bool], **kwargs) -> tuple[str, str | None]:
    """``create_chat_completion`` streamed token by token, so it can be stopped midway.

    Returns the answer and its finish reason; raises Cancelled once ``aborted()``.
    """
    parts: list[str] = []
    finish = None
    for chunk in llm.create_chat_completion(stream=True, **kwargs):
        if aborted():
            raise Cancelled()
        choice = chunk["choices"][0]
        parts.append(choice["delta"].get("content") or "")
        finish = choice.get("finish_reason") or finish
    return "".join(parts), finish


def unload() -> None:
    """Free the cached model (and its Metal/GPU buffers), stopping a running generation.

    Must run before the process exits: llama.cpp's Metal backend asserts in its
    static destructor if a model is still alive (GGML_ASSERT rsets count == 0).
    """
    global _loaded
    abort()
    with _load_lock:
        if _loaded is None:
            return
        llm = _loaded[1]
        _loaded = None
    # Never free the model under a running generation; abort() stops it within a token.
    if not _infer_lock.acquire(timeout=10):
        log.warning("model still busy at shutdown; not freeing it")
        return
    try:
        llm.close()
    except Exception:  # pragma: no cover - best effort during shutdown
        log.debug("closing the model failed", exc_info=True)
    finally:
        _infer_lock.release()
    gc.collect()


atexit.register(unload)


def runtime_available() -> bool:
    try:
        import llama_cpp  # noqa: F401
    except ImportError:
        return False
    return True


class LocalLLM:
    def __init__(self, model: str, auto_download: bool = True, n_ctx: int = N_CTX):
        self.model = model
        self.auto_download = auto_download
        self.n_ctx = n_ctx
        self._llm = None
        self._no_think = False

    # ------------------------------------------------------------- status

    def status(self, check_remote: bool = True) -> LLMStatus:
        models = [m.name for m in list_local()]
        if not runtime_available():
            return LLMStatus(False, "llama-cpp-python is not installed", models)
        try:
            local = find_local(self.model)
        except (ModelError, OSError) as exc:
            return LLMStatus(False, str(exc), models)
        if local is not None:
            where = {"ollama": "from Ollama's folder", "cache": "downloaded", "file": "file"}
            return LLMStatus(
                True,
                f"{local.name} · {format_size(local.size)} · {where[local.source]}",
                models,
                local,
            )
        if self.model.lower().endswith(".gguf"):
            return LLMStatus(False, f"File not found: {self.model}", models)
        if not self.auto_download:
            return LLMStatus(False, f"{self.model} is not downloaded", models)
        size = None
        if check_remote:
            try:
                size = remote_size(self.model)
            except ModelError as exc:
                return LLMStatus(False, str(exc), models)
        size_text = f" ({format_size(size)})" if size else ""
        return LLMStatus(
            True,
            f"{self.model} will be downloaded on first use{size_text}",
            models,
            download_size=size,
        )

    # ------------------------------------------------------------ loading

    def prepare(
        self,
        on_progress: Callable[[float, str], None] = lambda fraction, message: None,
        check_cancel: Callable[[], None] = lambda: None,
    ) -> None:
        """Make sure the model is on disk (downloading it if needed) and loaded."""
        global _loaded
        if self._llm is not None:
            return
        if not runtime_available():
            raise ModelError("llama-cpp-python is not installed")
        local = find_local(self.model)
        if local is None:
            if self.model.lower().endswith(".gguf") or not self.auto_download:
                raise ModelError(f"Model {self.model} is not available locally")

            def progress(done: int, total: int) -> None:
                on_progress(
                    done / total,
                    f"Downloading {self.model}: {format_size(done)} / {format_size(total)}",
                )

            local = download(self.model, progress, check_cancel)
        on_progress(1.0, f"Loading {local.name}…")
        with _load_lock:
            path = str(local.path)
            if _loaded is not None and _loaded[0] == path:
                self._llm = _loaded[1]
            else:
                from llama_cpp import Llama

                if _loaded is not None:  # release the previous model first
                    _loaded[1].close()
                    _loaded = None
                self._llm = Llama(model_path=path, n_ctx=self.n_ctx, n_gpu_layers=-1, verbose=False)
                _loaded = (path, self._llm)
        arch = str(self._llm.metadata.get("general.architecture", ""))
        self._no_think = arch.startswith("qwen3")

    # ---------------------------------------------------------- inference

    def chat_json(self, system: str, user: str, schema: dict) -> str:
        """Chat completion constrained to JSON matching ``schema``."""
        if self._llm is None:
            self.prepare()
        if self._no_think:
            user += "\n/no_think"  # Qwen3: skip the reasoning phase, answer directly
        epoch = _abort_epoch
        with _infer_lock:  # a llama.cpp context is not thread-safe
            if _abort_epoch != epoch:
                raise Cancelled()
            if _loaded is None or _loaded[1] is not self._llm:
                raise ModelError("the model was unloaded")
            content, finish = stream_chat(
                self._llm,
                lambda: _abort_epoch != epoch,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object", "schema": schema},
                max_tokens=MAX_OUTPUT_TOKENS,
                temperature=0.4,
            )
        if finish == "length":
            raise ValueError("the model's answer was cut off")
        return content
