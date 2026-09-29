"""Build a standalone pdf2video package for the current platform.

    uv sync --group build
    uv run python packaging/build.py            # → dist/pdf2video-<version>-<os>-<arch>.<ext>

Steps: download static ffmpeg/ffprobe into packaging/ffmpeg/ (skip with --no-ffmpeg),
run PyInstaller with packaging/pdf2video.spec, then archive the result:
a .zip on Windows, a .dmg on macOS and a .tar.gz on Linux.
"""

from __future__ import annotations

import argparse
import io
import os
import platform
import shutil
import stat
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PACKAGING = ROOT / "packaging"
FFMPEG_DIR = PACKAGING / "ffmpeg"
DIST = ROOT / "dist"
APP = "pdf2video"

BTBN = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest"
RIEDL = "https://ffmpeg.martin-riedl.de/redirect/latest/macos"


def arch() -> str:
    machine = platform.machine().lower()
    return {"amd64": "x86_64", "x64": "x86_64", "aarch64": "arm64"}.get(machine, machine)


def os_name() -> str:
    return {"win32": "windows", "darwin": "macos"}.get(sys.platform, "linux")


def version() -> str:
    """The version for the archive name: $PDF2VIDEO_VERSION (set by CI from the tag) or the
    package version."""
    if os.environ.get("PDF2VIDEO_VERSION"):
        return os.environ["PDF2VIDEO_VERSION"]
    sys.path.insert(0, str(ROOT / "src"))
    from pdf2video import __version__

    return __version__


def download(url: str) -> bytes:
    print(f"Downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "pdf2video-build"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        return resp.read()


def fetch_ffmpeg() -> None:
    """Put static ffmpeg and ffprobe binaries into packaging/ffmpeg/."""
    exe = ".exe" if sys.platform == "win32" else ""
    if all((FFMPEG_DIR / f"{t}{exe}").exists() for t in ("ffmpeg", "ffprobe")):
        return
    FFMPEG_DIR.mkdir(parents=True, exist_ok=True)
    wanted = {f"ffmpeg{exe}", f"ffprobe{exe}"}

    def unzip(url: str) -> zipfile.ZipFile:
        return zipfile.ZipFile(io.BytesIO(download(url)))

    def save(name: str, data: bytes) -> None:
        target = FFMPEG_DIR / name
        target.write_bytes(data)
        target.chmod(target.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    if sys.platform == "win32":
        with unzip(f"{BTBN}/ffmpeg-master-latest-win64-gpl.zip") as z:
            for info in z.infolist():
                name = info.filename.rsplit("/", 1)[-1]
                if name in wanted:
                    save(name, z.read(info))
    elif sys.platform == "darwin":
        mac_arch = "arm64" if arch() == "arm64" else "amd64"
        for tool in ("ffmpeg", "ffprobe"):
            with unzip(f"{RIEDL}/{mac_arch}/release/{tool}.zip") as z:
                save(tool, z.read(tool))
    else:
        linux_arch = "linuxarm64" if arch() == "arm64" else "linux64"
        data = download(f"{BTBN}/ffmpeg-master-latest-{linux_arch}-gpl.tar.xz")
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:xz") as t:
            for member in t.getmembers():
                name = member.name.rsplit("/", 1)[-1]
                if name in wanted and member.isfile():
                    save(name, t.extractfile(member).read())
    missing = wanted - {p.name for p in FFMPEG_DIR.iterdir()}
    if missing:
        sys.exit(f"ffmpeg download incomplete, missing: {', '.join(sorted(missing))}")


def pyinstaller() -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "PyInstaller",
            "--noconfirm",
            "--clean",
            "--distpath",
            str(DIST),
            "--workpath",
            str(ROOT / "build"),
            str(PACKAGING / "pdf2video.spec"),
        ],
        check=True,
    )


def archive() -> Path:
    base = DIST / f"{APP}-{version()}-{os_name()}-{arch()}"
    if sys.platform == "darwin":
        target = Path(f"{base}.dmg")
        target.unlink(missing_ok=True)
        subprocess.run(
            [
                "hdiutil",
                "create",
                "-volname",
                APP,
                "-srcfolder",
                str(DIST / f"{APP}.app"),
                "-ov",
                "-format",
                "UDZO",
                str(target),
            ],
            check=True,
        )
        return target
    fmt = "zip" if sys.platform == "win32" else "gztar"
    return Path(shutil.make_archive(str(base), fmt, root_dir=DIST, base_dir=APP))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--no-ffmpeg",
        action="store_true",
        help="don't bundle ffmpeg (the app then needs ffmpeg on PATH)",
    )
    parser.add_argument("--no-archive", action="store_true", help="skip the archive step")
    args = parser.parse_args()

    if args.no_ffmpeg:
        shutil.rmtree(FFMPEG_DIR, ignore_errors=True)
    else:
        fetch_ffmpeg()
    pyinstaller()
    if not args.no_archive:
        print(f"Built {archive()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
