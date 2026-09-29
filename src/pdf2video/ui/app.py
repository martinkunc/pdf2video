"""Adw.Application wiring."""

from __future__ import annotations

import logging
import os
import sys

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, Gio  # noqa: E402

from .window import MainWindow  # noqa: E402

APP_ID = "io.github.pdf2video"
JOB_STOP_TIMEOUT = 10  # seconds to wait for a cancelled job when quitting

log = logging.getLogger(__name__)


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
        self.force_exit = False
        self.main_windows: list[MainWindow] = []  # also closed ones, whose job may still run

    def _window(self) -> MainWindow:
        window = self.get_active_window()
        if window is None:
            window = MainWindow(self)
            self.main_windows.append(window)
        return window

    def do_shutdown(self) -> None:
        from .. import imagegen, llm
        from ..llm import vision

        # Stop a running job; its thread also terminates ffmpeg.
        stopped = all([window.stop_job(JOB_STOP_TIMEOUT) for window in self.main_windows])
        if stopped:
            # Free GPU buffers before exit (llama.cpp Metal asserts otherwise).
            llm.unload()
            vision.unload()
            imagegen.unload()
        else:
            # A model is stuck mid-computation (e.g. an image being drawn): freeing it
            # would block, and the exit handlers would crash, so skip them in run().
            log.warning("the job did not stop in time; exiting without cleanup")
            self.force_exit = True
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
    app = Application()
    status = app.run(argv if argv is not None else sys.argv)
    if app.force_exit:
        logging.shutdown()
        os._exit(status)
    return status
