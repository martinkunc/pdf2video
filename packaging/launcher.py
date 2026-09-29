"""PyInstaller entry point for the packaged app.

Puts the bundled ffmpeg/ffprobe (in `<bundle>/bin`) on PATH before starting, so
the app works without a system ffmpeg. A system ffmpeg is still used as a fallback.
"""

import os
import sys


def _setup_frozen() -> None:
    base = getattr(sys, "_MEIPASS", None)
    if not base:
        return
    bin_dir = os.path.join(base, "bin")
    if os.path.isdir(bin_dir):
        os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
    # Apps started from Finder get a minimal PATH; add Homebrew for tesseract etc.
    if sys.platform == "darwin":
        extra = [p for p in ("/opt/homebrew/bin", "/usr/local/bin") if os.path.isdir(p)]
        os.environ["PATH"] = os.pathsep.join([os.environ["PATH"], *extra])


_setup_frozen()

from pdf2video.__main__ import cli, main  # noqa: E402

if __name__ == "__main__":
    # On Windows the GUI build has no console, so a separate pdf2video-cli.exe
    # (a console build of this same script) runs the command-line interface.
    name = os.path.basename(sys.executable).lower()
    sys.exit(cli() if name.startswith("pdf2video-cli") else main())
