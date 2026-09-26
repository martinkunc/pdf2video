import hashlib

import httpx
import pytest

from pdf2video.tts import TTSError
from pdf2video.tts import piper as piper_mod
from pdf2video.tts.piper import PiperTTS, is_downloaded


def _entry(key, family, quality, files):
    return {
        "key": key,
        "name": key.split("-")[1],
        "quality": quality,
        "num_speakers": 1,
        "language": {"code": key.split("-")[0], "family": family},
        "files": files,
    }


@pytest.fixture
def catalog(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    onnx, config = b"fake-onnx-model", b'{"audio": {}}'
    files = {
        "cs/cs_CZ/jirka/medium/cs_CZ-jirka-medium.onnx": {
            "size_bytes": len(onnx),
            "md5_digest": hashlib.md5(onnx).hexdigest(),
        },
        "cs/cs_CZ/jirka/medium/cs_CZ-jirka-medium.onnx.json": {
            "size_bytes": len(config),
            "md5_digest": hashlib.md5(config).hexdigest(),
        },
    }
    data = {
        "cs_CZ-jirka-medium": _entry("cs_CZ-jirka-medium", "cs", "medium", files),
        "hu_HU-anna-low": _entry("hu_HU-anna-low", "hu", "low", {}),
        "hu_HU-imre-medium": _entry("hu_HU-imre-medium", "hu", "medium", {}),
    }
    monkeypatch.setattr(piper_mod, "_catalog_data", lambda: data)

    def handler(request):
        return httpx.Response(200, content=onnx if request.url.path.endswith(".onnx") else config)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(piper_mod.httpx, "stream", client.stream)
    return data


def test_default_voice(catalog):
    engine = PiperTTS()
    assert engine.default_voice("cs") == "cs_CZ-jirka-medium"
    assert engine.default_voice("hu") == "hu_HU-imre-medium"  # best quality wins


def test_voice_list_and_download(catalog):
    engine = PiperTTS()
    voice = next(v for v in engine.voices() if v.id == "cs_CZ-jirka-medium")
    assert voice.language == "cs" and voice.locale == "cs-CZ"
    assert "download" in voice.gender
    progress = []
    engine.prepare("cs_CZ-jirka-medium", lambda f, m: progress.append(f))
    assert is_downloaded("cs_CZ-jirka-medium")
    assert progress[-1] == pytest.approx(1.0)
    assert next(v for v in engine.voices() if v.id == "cs_CZ-jirka-medium").gender == ("downloaded")


def test_checksum_mismatch(catalog):
    for meta in catalog["cs_CZ-jirka-medium"]["files"].values():
        meta["md5_digest"] = "0" * 32
    with pytest.raises(TTSError, match="checksum"):
        PiperTTS().prepare("cs_CZ-jirka-medium")
    assert not is_downloaded("cs_CZ-jirka-medium")


def test_unknown_voice(catalog):
    with pytest.raises(TTSError, match="Unknown"):
        PiperTTS().prepare("xx_XX-nobody-low")
