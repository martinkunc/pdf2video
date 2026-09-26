"""Parallel speech synthesis shared by the audiobook and video pipelines."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path

from . import media
from .jobs import JobContext
from .settings import Settings
from .textutil import detect_language
from .tts import TTSEngine, get_engine, synthesize_with_retry

MAX_WORKERS = 4


@dataclass
class Clip:
    path: Path
    seconds: float


class Narrator:
    def __init__(self, settings: Settings, sample_text: str, engine: TTSEngine | None = None):
        self.engine = engine or get_engine(settings.tts_engine)
        self.rate = settings.rate
        self.language = detect_language(sample_text)
        voice = settings.voice
        if not voice or voice not in {v.id for v in self.engine.voices()}:
            voice = self.engine.default_voice(self.language)
        self.voice = voice

    def prepare(self, ctx: JobContext, share: float = 0.0) -> None:
        """Download the voice if the engine needs to (reported as progress up to ``share``)."""
        prepare = getattr(self.engine, "prepare", None)
        if prepare is not None:
            ctx.check()
            prepare(self.voice, lambda f, message: ctx.progress(share * f, message))

    def synthesize_many(
        self,
        texts: Sequence[str],
        workdir: Path,
        prefix: str,
        ctx: JobContext,
        on_done: Callable[[int], None] = lambda n: None,
    ) -> list[Clip]:
        """Synthesize ``texts`` concurrently; ``on_done`` gets the number finished so far."""
        paths = [workdir / f"{prefix}_{i:04d}{self.engine.suffix}" for i in range(len(texts))]
        finished = 0

        def job(i: int) -> None:
            ctx.check()
            synthesize_with_retry(self.engine, texts[i], self.voice, self.rate, paths[i])

        pool = ThreadPoolExecutor(MAX_WORKERS)
        try:
            pending = {pool.submit(job, i) for i in range(len(texts))}
            while pending:
                done, pending = wait(pending, timeout=0.3, return_when=FIRST_EXCEPTION)
                for fut in done:
                    if (exc := fut.exception()) is not None:
                        raise exc
                    finished += 1
                if done:
                    on_done(finished)
                ctx.check()
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
        return [Clip(p, media.duration(p)) for p in paths]
