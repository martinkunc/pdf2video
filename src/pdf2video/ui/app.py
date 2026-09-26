"""Adw.Application wiring."""

from __future__ import annotations

import logging
import sys

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, Gio  # noqa: E402

from .window import MainWindow  # noqa: E402

APP_ID = "io.github.pdf2video"


class Application(Adw.Application):
    def __init__(self):
        super().__init__(
            application_id=APP_ID,
            flags=Gio.ApplicationFlags.HANDLES_OPEN | Gio.ApplicationFlags.NON_UNIQUE,
        )
        self.set_accels_for_action("app.quit", ["<Primary>q"])
        quit_action = Gio.SimpleAction.new("quit", None)
        quit_action.connect("activate", lambda *a: self.quit())
        self.add_action(quit_action)

    def _window(self) -> MainWindow:
        window = self.get_active_window()
        if window is None:
            window = MainWindow(self)
        return window

    def do_shutdown(self) -> None:
        from ..llm import unload

        for window in self.get_windows():
            if getattr(window, "job", None) is not None:
                window.job.cancel()
        unload()  # free GPU buffers before exit (llama.cpp Metal asserts otherwise)
        Adw.Application.do_shutdown(self)

    def do_activate(self) -> None:
        self._window().present()

    def do_open(self, files, n_files, hint) -> None:
        window = self._window()
        window.present()
        if files and files[0].get_path():
            window.open_path(files[0].get_path())


def run(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    return Application().run(argv if argv is not None else sys.argv)
