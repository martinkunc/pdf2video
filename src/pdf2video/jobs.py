"""Progress reporting and cancellation shared by the pipelines."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field

from .media import Cancelled

ProgressFn = Callable[[float, str], None]


@dataclass
class JobContext:
    """Passed to pipelines: report progress, check for cancellation."""

    on_progress: ProgressFn = lambda fraction, message: None
    cancel_event: threading.Event = field(default_factory=threading.Event)

    def progress(self, fraction: float, message: str) -> None:
        self.on_progress(max(0.0, min(1.0, fraction)), message)

    def check(self) -> None:
        if self.cancel_event.is_set():
            raise Cancelled()

    def cancel(self) -> None:
        self.cancel_event.set()

    @property
    def cancelled(self) -> bool:
        return self.cancel_event.is_set()
