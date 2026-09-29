# PyInstaller spec for pdf2video. Build with `uv run python packaging/build.py`,
# which also fetches ffmpeg into packaging/ffmpeg/ and archives the result.
# -*- mode: python -*-

import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_dynamic_libs

ROOT = Path(SPECPATH).parent
sys.path.insert(0, str(ROOT / "src"))
from pdf2video import __version__  # noqa: E402

APP_NAME = "pdf2video"
WINDOWS = sys.platform == "win32"
MACOS = sys.platform == "darwin"

datas, binaries, hiddenimports = [], [], []

# Packages that load native libraries or data files at runtime.
PACKAGES = ("llama_cpp", "stable_diffusion_cpp", "piper", "espeakng_loader", "kokoro_onnx",
            "onnxruntime", "phonemizer", "pymupdf", "edge_tts")
# Training/conversion tools that need torch/onnx and aren't used at runtime.
SKIP = ("piper.train", "onnxruntime.quantization", "onnxruntime.tools", "onnxruntime.transformers")

for package in PACKAGES:
    try:
        d, b, h = collect_all(package, filter_submodules=lambda name: not name.startswith(SKIP))
    except Exception:
        continue
    datas += d
    binaries += b
    hiddenimports += h
binaries += collect_dynamic_libs("llama_cpp") + collect_dynamic_libs("stable_diffusion_cpp")

# Bundled ffmpeg/ffprobe (put on PATH by launcher.py).
ffmpeg_dir = ROOT / "packaging" / "ffmpeg"
if ffmpeg_dir.is_dir():
    for tool in ffmpeg_dir.iterdir():
        if tool.stem in ("ffmpeg", "ffprobe"):
            binaries.append((str(tool), "bin"))

a = Analysis(
    [str(ROOT / "packaging" / "launcher.py")],
    pathex=[str(ROOT / "src")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports + ["pdf2video.ui.app", "pdf2video.cli"],
    hooksconfig={
        "gi": {
            "module-versions": {"Gtk": "4.0", "Gdk": "4.0", "Adw": "1"},
            "icons": ["Adwaita", "hicolor"],
            "themes": ["Adwaita"],
            "languages": ["en_US", "en_GB", "cs_CZ", "de_DE", "es_ES", "fr_FR", "it_IT",
                          "pt_BR", "pt_PT"],
        },
    },
    excludes=["pytest", "ruff", "tkinter", "matplotlib"],
    noarchive=False,
)
pyz = PYZ(a.pure)

gui = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP_NAME,
    console=not WINDOWS,
    argv_emulation=False,
    upx=False,
)
executables = [gui]
if WINDOWS:
    # A console build of the same launcher for the command-line interface.
    executables.append(
        EXE(pyz, a.scripts, [], exclude_binaries=True, name=f"{APP_NAME}-cli",
            console=True, upx=False)
    )

coll = COLLECT(*executables, a.binaries, a.datas, name=APP_NAME, upx=False)

if MACOS:
    app = BUNDLE(
        coll,
        name=f"{APP_NAME}.app",
        bundle_identifier="io.github.pdf2video",
        version=__version__,
        info_plist={
            "CFBundleName": APP_NAME,
            "CFBundleDisplayName": APP_NAME,
            "CFBundleShortVersionString": __version__,
            "NSHighResolutionCapable": True,
            "LSMinimumSystemVersion": os.environ.get("MACOSX_DEPLOYMENT_TARGET", "13.0"),
            "CFBundleDocumentTypes": [
                {
                    "CFBundleTypeName": "Document",
                    "CFBundleTypeRole": "Viewer",
                    "LSHandlerRank": "Alternate",
                    "CFBundleTypeExtensions": ["pdf", "docx", "epub", "txt", "md"],
                }
            ],
        },
    )
