"""Find GGUF model files locally, or download them.

Models are named like in Ollama (``qwen3:8b``, ``user/model:tag``,
``hf.co/org/repo:tag``) or given as a path to a ``.gguf`` file.

Lookup order:

1. Ollama's model folder (``$OLLAMA_MODELS`` or ``~/.ollama/models``). Ollama stores
   the weights as plain GGUF blobs, so they can be used directly, without Ollama.
2. This app's own cache (``~/.cache/pdf2video/models``), in the same layout.
3. Download from the registry (the same OCI-style registry ``ollama pull`` uses)
   into the app cache, with resume support and sha256 verification. Ollama's
   folder is never written to.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import httpx

DEFAULT_HOST = "registry.ollama.ai"
MODEL_MEDIA_TYPE = "application/vnd.ollama.image.model"
MANIFEST_ACCEPT = "application/vnd.docker.distribution.manifest.v2+json"

# Suggested models that are known to work well for the video scripts.
RECOMMENDED = {
    "qwen3:4b": "fast, needs ~4 GB RAM",
    "qwen3:8b": "good balance (default), needs ~7 GB RAM",
    "qwen3:14b": "better quality, needs ~12 GB RAM",
    "qwen3:30b": "best quality, mixture-of-experts, needs ~20 GB RAM",
}


class ModelError(Exception):
    pass


@dataclass(frozen=True)
class ModelRef:
    host: str
    namespace: str
    name: str
    tag: str

    @classmethod
    def parse(cls, text: str) -> ModelRef:
        text = text.strip()
        if not text:
            raise ModelError("No model name given")
        name, _, tag = text.rpartition(":") if ":" in text.split("/")[-1] else (text, "", "")
        tag = tag or "latest"
        parts = name.split("/")
        if len(parts) == 1:
            return cls(DEFAULT_HOST, "library", parts[0], tag)
        if len(parts) == 2:
            return cls(DEFAULT_HOST, parts[0], parts[1], tag)
        return cls(parts[0], "/".join(parts[1:-1]), parts[-1], tag)

    @property
    def display(self) -> str:
        if self.host == DEFAULT_HOST and self.namespace == "library":
            base = self.name
        elif self.host == DEFAULT_HOST:
            base = f"{self.namespace}/{self.name}"
        else:
            base = f"{self.host}/{self.namespace}/{self.name}"
        return f"{base}:{self.tag}"

    def manifest_path(self, root: Path) -> Path:
        return root / "manifests" / self.host / self.namespace / self.name / self.tag

    def registry_url(self, kind: str, ref: str) -> str:
        return f"https://{self.host}/v2/{self.namespace}/{self.name}/{kind}/{ref}"


@dataclass
class LocalModel:
    name: str
    path: Path
    size: int
    source: str  # "ollama" | "cache" | "file"


def ollama_models_dir() -> Path | None:
    candidates = [
        os.environ.get("OLLAMA_MODELS"),
        Path.home() / ".ollama" / "models",
        "/usr/share/ollama/.ollama/models",  # Linux system service
    ]
    for candidate in candidates:
        if candidate and (Path(candidate) / "manifests").is_dir():
            return Path(candidate)
    return None


def cache_dir() -> Path:
    base = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return base / "pdf2video" / "models"


def _blob_path(root: Path, digest: str) -> Path:
    return root / "blobs" / digest.replace(":", "-")


def _model_layer(manifest: dict) -> dict:
    for layer in manifest.get("layers", []):
        if layer.get("mediaType") == MODEL_MEDIA_TYPE:
            return layer
    raise ModelError("The manifest has no model weights layer")


def _from_store(ref: ModelRef, root: Path, source: str) -> LocalModel | None:
    manifest_file = ref.manifest_path(root)
    if not manifest_file.is_file():
        return None
    try:
        layer = _model_layer(json.loads(manifest_file.read_text()))
    except (ValueError, ModelError):
        return None
    blob = _blob_path(root, layer["digest"])
    if not blob.is_file() or blob.stat().st_size != layer.get("size", blob.stat().st_size):
        return None
    return LocalModel(ref.display, blob, blob.stat().st_size, source)


def find_local(name: str) -> LocalModel | None:
    """Locate a model without touching the network."""
    if name.lower().endswith(".gguf"):
        path = Path(name).expanduser()
        return LocalModel(path.name, path, path.stat().st_size, "file") if path.is_file() else None
    ref = ModelRef.parse(name)
    stores = [(ollama_models_dir(), "ollama"), (cache_dir(), "cache")]
    for root, source in stores:
        if root is not None and (found := _from_store(ref, root, source)):
            return found
    return None


def list_local() -> list[LocalModel]:
    """All models available offline (Ollama's folder + app cache)."""
    found: dict[str, LocalModel] = {}
    for root, source in ((ollama_models_dir(), "ollama"), (cache_dir(), "cache")):
        if root is None or not (root / "manifests").is_dir():
            continue
        for manifest in (root / "manifests").rglob("*"):
            if not manifest.is_file():
                continue
            parts = manifest.relative_to(root / "manifests").parts
            if len(parts) < 4:
                continue
            ref = ModelRef(parts[0], "/".join(parts[1:-2]), parts[-2], parts[-1])
            if ref.display not in found and (model := _from_store(ref, root, source)):
                found[ref.display] = model
    return sorted(found.values(), key=lambda m: m.name)


def remote_size(name: str, timeout: float = 5) -> int:
    """Size in bytes of the model weights in the registry."""
    ref = ModelRef.parse(name)
    return int(_model_layer(_fetch_manifest(ref, timeout))["size"])


def _fetch_manifest(ref: ModelRef, timeout: float = 30) -> dict:
    try:
        response = httpx.get(
            ref.registry_url("manifests", ref.tag),
            headers={"Accept": MANIFEST_ACCEPT},
            timeout=timeout,
            follow_redirects=True,
        )
    except httpx.HTTPError as exc:
        raise ModelError(f"Cannot reach the model registry ({exc})") from exc
    if response.status_code == 404:
        raise ModelError(f"Model {ref.display} does not exist in the registry")
    if response.status_code != 200:
        raise ModelError(f"Registry error {response.status_code} for {ref.display}")
    return response.json()


ProgressFn = Callable[[int, int], None]  # (downloaded bytes, total bytes)


def fetch_file(
    url: str,
    dest: Path,
    digest: str,
    total: int,
    on_progress: ProgressFn = lambda done, total: None,
    check_cancel: Callable[[], None] = lambda: None,
) -> None:
    """Download ``url`` to ``dest`` via ``dest.partial`` (resumable, sha256-verified).

    ``digest`` is ``"sha256:<hex>"``.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".partial")
    hasher = hashlib.sha256()
    done = 0
    if partial.exists():  # resume: hash what we already have
        with partial.open("rb") as f:
            while chunk := f.read(1 << 22):
                hasher.update(chunk)
                done += len(chunk)
        if done > total:
            partial.unlink()
            hasher, done = hashlib.sha256(), 0
    headers = {"Range": f"bytes={done}-"} if done else {}
    try:
        with httpx.stream(
            "GET",
            url,
            headers=headers,
            follow_redirects=True,
            timeout=httpx.Timeout(30, read=120),
        ) as response:
            if response.status_code not in (200, 206):
                raise ModelError(f"Download failed: HTTP {response.status_code}")
            if response.status_code == 200 and done:  # server ignored Range
                hasher, done = hashlib.sha256(), 0
            mode = "ab" if done else "wb"
            with partial.open(mode) as f:
                for chunk in response.iter_bytes(1 << 20):
                    check_cancel()
                    f.write(chunk)
                    hasher.update(chunk)
                    done += len(chunk)
                    on_progress(done, total)
    except httpx.HTTPError as exc:
        raise ModelError(f"Download interrupted ({exc}); it will resume next time") from exc
    if done != total:
        raise ModelError(f"Download incomplete ({done} of {total} bytes)")
    if f"sha256:{hasher.hexdigest()}" != digest:
        partial.unlink(missing_ok=True)
        raise ModelError("Downloaded file is corrupted (checksum mismatch); please retry")
    partial.rename(dest)


def download(
    name: str,
    on_progress: ProgressFn = lambda done, total: None,
    check_cancel: Callable[[], None] = lambda: None,
) -> LocalModel:
    """Download ``name`` into the app cache (resumable, sha256-verified)."""
    ref = ModelRef.parse(name)
    root = cache_dir()
    manifest = _fetch_manifest(ref)
    layer = _model_layer(manifest)
    digest, total = layer["digest"], int(layer["size"])
    blob = _blob_path(root, digest)
    blob.parent.mkdir(parents=True, exist_ok=True)

    if not (blob.is_file() and blob.stat().st_size == total):
        fetch_file(
            ref.registry_url("blobs", digest), blob, digest, total, on_progress, check_cancel
        )

    manifest_file = ref.manifest_path(root)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    manifest_file.write_text(json.dumps(manifest))
    return LocalModel(ref.display, blob, total, "cache")


def format_size(size: int) -> str:
    return f"{size / 1e9:.1f} GB" if size >= 1e9 else f"{size / 1e6:.0f} MB"
