import hashlib
import json

import httpx
import pytest

from pdf2video.llm import store
from pdf2video.llm.store import ModelError, ModelRef, download, find_local, list_local


def test_model_ref_parsing():
    assert ModelRef.parse("qwen3").display == "qwen3:latest"
    ref = ModelRef.parse("qwen3:8b")
    assert (ref.host, ref.namespace, ref.name, ref.tag) == (
        "registry.ollama.ai",
        "library",
        "qwen3",
        "8b",
    )
    assert ModelRef.parse("someone/model:q4").namespace == "someone"
    hf = ModelRef.parse("hf.co/org/repo-GGUF:Q4_K_M")
    assert (hf.host, hf.namespace, hf.name, hf.tag) == ("hf.co", "org", "repo-GGUF", "Q4_K_M")
    assert hf.display == "hf.co/org/repo-GGUF:Q4_K_M"


def _write_store(root, name, blob: bytes):
    ref = ModelRef.parse(name)
    digest = "sha256:" + hashlib.sha256(blob).hexdigest()
    (root / "blobs").mkdir(parents=True, exist_ok=True)
    (root / "blobs" / digest.replace(":", "-")).write_bytes(blob)
    manifest = ref.manifest_path(root)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            {"layers": [{"mediaType": store.MODEL_MEDIA_TYPE, "digest": digest, "size": len(blob)}]}
        )
    )


@pytest.fixture
def stores(tmp_path, monkeypatch):
    ollama = tmp_path / "ollama"
    (ollama / "manifests").mkdir(parents=True)
    monkeypatch.setenv("OLLAMA_MODELS", str(ollama))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    return ollama


def test_finds_models_in_ollama_folder(stores):
    _write_store(stores, "qwen3:8b", b"GGUF-weights")
    found = find_local("qwen3:8b")
    assert found.source == "ollama" and found.path.read_bytes() == b"GGUF-weights"
    assert find_local("qwen3:4b") is None
    assert [m.name for m in list_local()] == ["qwen3:8b"]


def test_incomplete_blob_is_ignored(stores):
    _write_store(stores, "qwen3:8b", b"GGUF-weights")
    blob = next((stores / "blobs").iterdir())
    blob.write_bytes(b"GG")  # truncated
    assert find_local("qwen3:8b") is None


def _mock_registry(monkeypatch, blob: bytes, fail_after: int | None = None):
    digest = "sha256:" + hashlib.sha256(blob).hexdigest()
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if "/manifests/" in request.url.path:
            return httpx.Response(
                200,
                json={
                    "layers": [
                        {"mediaType": store.MODEL_MEDIA_TYPE, "digest": digest, "size": len(blob)}
                    ]
                },
            )
        start = 0
        if rng := request.headers.get("Range"):
            start = int(rng.removeprefix("bytes=").rstrip("-"))
        body = blob[start:]
        if fail_after is not None and not rng:
            body = blob[:fail_after]  # simulate a connection drop
        return httpx.Response(206 if start else 200, content=body)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(store.httpx, "get", client.get)
    monkeypatch.setattr(store.httpx, "stream", client.stream)
    return requests


def test_download_verifies_and_resumes(stores, monkeypatch):
    blob = bytes(range(256)) * 4000
    _mock_registry(monkeypatch, blob, fail_after=100_000)
    with pytest.raises(ModelError, match="incomplete"):
        download("qwen3:4b")
    assert find_local("qwen3:4b") is None
    requests = _mock_registry(monkeypatch, blob)
    model = download("qwen3:4b")
    assert requests[-1].headers["Range"] == "bytes=100000-"
    assert model.path.read_bytes() == blob and model.source == "cache"
    assert find_local("qwen3:4b").source == "cache"


def test_download_rejects_corrupted_blob(stores, monkeypatch):
    blob = b"x" * 1000

    def handler(request):
        if "/manifests/" in request.url.path:
            return httpx.Response(
                200,
                json={
                    "layers": [
                        {
                            "mediaType": store.MODEL_MEDIA_TYPE,
                            "digest": "sha256:" + "0" * 64,
                            "size": len(blob),
                        }
                    ]
                },
            )
        return httpx.Response(200, content=blob)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(store.httpx, "get", client.get)
    monkeypatch.setattr(store.httpx, "stream", client.stream)
    with pytest.raises(ModelError, match="checksum"):
        download("qwen3:4b")
