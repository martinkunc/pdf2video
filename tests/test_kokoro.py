import wave

import pytest

from pdf2video import narration
from pdf2video.settings import Settings
from pdf2video.tts import Voice, get_engine, kokoro

needs_model = pytest.mark.skipif(not kokoro.is_downloaded(), reason="Kokoro not downloaded")

CZECH = "Toto je krátký český text o požadavcích na software a o tom, proč jsou důležité."
ENGLISH = "This is a short English text about software requirements and why they matter."


def test_voices_and_defaults():
    engine = get_engine("kokoro")
    voices = {v.id: v for v in engine.voices()}
    assert voices["af_heart"].locale == "en-US" and "female" in voices["af_heart"].gender
    assert voices["bm_george"].language == "en" and voices["bm_george"].locale == "en-GB"
    assert engine.default_voice("fr") == "ff_siwis"
    assert engine.supports("en") and not engine.supports("cs")
    assert Settings().tts_engine == "kokoro"


class FakeEngine:
    def __init__(self, name):
        self.name = name

    def voices(self):
        return [Voice(f"{self.name}-voice", "V", "en", "en-US")]

    def default_voice(self, language):
        return f"{self.name}-{language}"


class FakeKokoro(FakeEngine):
    def supports(self, language):
        return language == "en"


@pytest.fixture
def engines(monkeypatch):
    fakes = {"kokoro": FakeKokoro("kokoro"), "piper": FakeEngine("piper")}
    monkeypatch.setattr(narration, "get_engine", lambda name: fakes[name])
    return fakes


def test_narrator_falls_back_to_piper_for_other_languages(engines):
    czech = narration.Narrator(Settings(tts_engine="kokoro"), CZECH)
    assert czech.engine is engines["piper"] and czech.voice == "piper-cs"
    english = narration.Narrator(Settings(tts_engine="kokoro"), ENGLISH)
    assert english.engine is engines["kokoro"] and english.voice == "kokoro-en"
    # A voice picked explicitly is respected.
    chosen = narration.Narrator(Settings(tts_engine="kokoro", voice="kokoro-voice"), CZECH)
    assert chosen.engine is engines["kokoro"] and chosen.voice == "kokoro-voice"


@needs_model
def test_kokoro_voice_ids():
    from kokoro_onnx import Kokoro

    model = Kokoro(str(kokoro.MODEL.path), str(kokoro.VOICES.path))
    usable = {v for v in model.get_voices() if v[0] in kokoro.LANGUAGES}
    assert usable == set(kokoro.VOICE_IDS)


@needs_model
def test_kokoro_synthesis_with_paragraph_pause(tmp_path):
    engine = get_engine("kokoro")
    one, two = tmp_path / "one.wav", tmp_path / "two.wav"
    engine.synthesize("Hello there. How are you?", "af_heart", 1.0, one)
    engine.synthesize("Hello there.\n\nHow are you?", "af_heart", 1.0, two)

    def seconds(path):
        with wave.open(str(path)) as w:
            assert w.getframerate() == 24000 and w.getsampwidth() == 2
            return w.getnframes() / w.getframerate()

    assert 1.0 < seconds(one) < 4.0
    assert seconds(two) > seconds(one) + 0.2  # the paragraph break adds a pause
    with pytest.raises(Exception, match="Unknown Kokoro voice"):
        engine.synthesize("Hi.", "xx_nobody", 1.0, tmp_path / "x.wav")
