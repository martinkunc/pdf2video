"""Thin ffmpeg/ffprobe wrappers (run as subprocesses, cancellable)."""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
from collections.abc import Sequence
from pathlib import Path

SAMPLE_RATE = 44100


class MediaError(Exception):
    pass


class Cancelled(Exception):
    """Raised when the user cancels a running job."""


def check_ffmpeg() -> str | None:
    """Return an error message if ffmpeg/ffprobe are missing, else None."""
    missing = [tool for tool in ("ffmpeg", "ffprobe") if not shutil.which(tool)]
    if missing:
        return (
            f"{' and '.join(missing)} not found. Install it, e.g. `brew install ffmpeg` (macOS) "
            "or `sudo apt install ffmpeg` (Debian/Ubuntu)."
        )
    return None


def run(cmd: Sequence[str], cancel: threading.Event | None = None) -> None:
    proc = subprocess.Popen(
        [str(c) for c in cmd], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True
    )
    try:
        while True:
            try:
                _, stderr = proc.communicate(timeout=0.25)
                break
            except subprocess.TimeoutExpired:
                if cancel is not None and cancel.is_set():
                    proc.terminate()
                    proc.wait(5)
                    raise Cancelled() from None
    except BaseException:
        if proc.poll() is None:
            proc.kill()
        raise
    if proc.returncode != 0:
        tail = "\n".join((stderr or "").strip().splitlines()[-6:])
        raise MediaError(f"{Path(cmd[0]).name} failed ({proc.returncode}):\n{tail}")


def duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    try:
        return float(json.loads(out)["format"]["duration"])
    except (KeyError, ValueError) as exc:
        raise MediaError(f"Cannot determine duration of {path.name}") from exc


def concat_audio_to_mp3(
    inputs: Sequence[Path],
    out: Path,
    *,
    lead_silence: float = 0.5,
    gap: float = 0.25,
    tags: dict[str, str] | None = None,
    bitrate: str = "64k",
    cancel: threading.Event | None = None,
) -> None:
    """Concatenate audio files (any format) into one mono MP3 with ID3 tags."""
    if not inputs:
        raise MediaError("Nothing to concatenate")
    cmd: list[str] = ["ffmpeg", "-y", "-hide_banner", "-nostdin"]
    for path in inputs:
        cmd += ["-i", str(path)]
    fmt = f"aresample={SAMPLE_RATE},aformat=sample_fmts=fltp:channel_layouts=mono"
    parts = [f"anullsrc=r={SAMPLE_RATE}:cl=mono,atrim=duration={lead_silence}[lead]"]
    labels = ["[lead]"]
    for i in range(len(inputs)):
        parts.append(f"[{i}:a]{fmt},apad=pad_dur={gap}[a{i}]")
        labels.append(f"[a{i}]")
    parts.append(f"{''.join(labels)}concat=n={len(labels)}:v=0:a=1[out]")
    cmd += ["-filter_complex", ";".join(parts), "-map", "[out]"]
    cmd += ["-c:a", "libmp3lame", "-b:a", bitrate, "-ar", str(SAMPLE_RATE), "-ac", "1"]
    cmd += ["-id3v2_version", "3", "-write_id3v1", "1"]
    for key, value in (tags or {}).items():
        cmd += ["-metadata", f"{key}={value}"]
    cmd.append(str(out))
    run(cmd, cancel)


def _audio_filter(index: int) -> str:
    return (
        f"[{index}:a]aresample={SAMPLE_RATE},aformat=channel_layouts=stereo,"
        "adelay=200:all=1,apad[a]"
    )


def _encode_args(length: float, fps: int, still: bool, out: Path) -> list[str]:
    # Identical stream parameters for every clip type, so that concat_videos can
    # join them without re-encoding.
    return [
        "-map", "[v]", "-map", "[a]", "-t", f"{length + 0.2:.3f}",
        "-c:v", "libx264", "-preset", "medium", *(["-tune", "stillimage"] if still else []),
        "-crf", "20", "-profile:v", "high", "-pix_fmt", "yuv420p",
        "-r", str(fps), "-g", str(fps * 10),
        "-c:a", "aac", "-b:a", "160k", "-ar", str(SAMPLE_RATE),
        "-movflags", "+faststart", str(out),
    ]  # fmt: skip


def still_clip(
    image: Path,
    audio: Path,
    out: Path,
    *,
    pad: float = 0.6,
    fps: int = 30,
    cancel: threading.Event | None = None,
) -> None:
    """Encode a still image + narration into an H.264/AAC clip."""
    length = duration(audio) + pad
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-nostdin",
        "-loop", "1", "-framerate", str(fps), "-i", str(image),
        "-i", str(audio),
        "-filter_complex", f"{_audio_filter(1)};[0:v]format=yuv420p[v]",
        *_encode_args(length, fps, True, out),
    ]  # fmt: skip
    run(cmd, cancel)


def frames_clip(
    frames: Sequence[tuple[Path, float]],
    audio: Path,
    out: Path,
    *,
    pad: float = 0.6,
    fps: int = 30,
    fade: float = 0.4,
    cancel: threading.Event | None = None,
) -> None:
    """Like :func:`still_clip`, but the slide changes: ``frames`` are (image, start
    second) pairs, the first starting at 0; consecutive images cross-fade."""
    length = duration(audio) + pad
    total = length + 0.2
    starts = [max(0.0, min(start, total - 0.1)) for _, start in frames]
    if len(frames) == 1:
        still_clip(frames[0][0], audio, out, pad=pad, fps=fps, cancel=cancel)
        return
    gaps = [b - a for a, b in zip(starts, starts[1:], strict=False)]
    fade = max(0.04, min(fade, min(gaps) * 0.8))
    cmd = ["ffmpeg", "-y", "-hide_banner", "-nostdin"]
    for k, (image, _) in enumerate(frames):
        # Each input lasts until the next transition has finished.
        end = starts[k + 1] + fade if k + 1 < len(frames) else total + fade
        cmd += [
            "-loop",
            "1",
            "-framerate",
            str(fps),
            "-t",
            f"{end - starts[k]:.3f}",
            "-i",
            str(image),
        ]
    cmd += ["-i", str(audio)]
    parts = [
        f"[{k}:v]format=yuv420p,settb=AVTB,setpts=PTS-STARTPTS[f{k}]" for k in range(len(frames))
    ]
    previous = "[f0]"
    for k in range(1, len(frames)):
        label = "[v]" if k == len(frames) - 1 else f"[x{k}]"
        parts.append(
            f"{previous}[f{k}]xfade=transition=fade:duration={fade:.3f}:offset={starts[k]:.3f}{label}"
        )
        previous = label
    parts.append(_audio_filter(len(frames)))
    cmd += ["-filter_complex", ";".join(parts), *_encode_args(length, fps, False, out)]
    run(cmd, cancel)


MOTIONS = ("zoom_in", "zoom_out", "pan", "fade_in")


def motion_clip(
    base: Path,
    picture: Path,
    position: tuple[int, int],
    audio: Path,
    out: Path,
    *,
    motion: str,
    pad: float = 0.6,
    fps: int = 30,
    cancel: threading.Event | None = None,
) -> None:
    """``picture`` over the ``base`` slide at ``position``, with a slow camera
    ``motion`` (see :data:`MOTIONS`) inside the picture's frame."""
    from PIL import Image

    with Image.open(picture) as img:
        width, height = img.size
    length = duration(audio) + pad
    n = round((length + 0.2) * fps) + 1
    zoom = 0.12
    if motion == "zoom_in":
        z, x, y = f"1+{zoom}*on/{n}", "iw/2-iw/zoom/2", "ih/2-ih/zoom/2"
    elif motion == "zoom_out":
        z, x, y = f"{1 + zoom}-{zoom}*on/{n}", "iw/2-iw/zoom/2", "ih/2-ih/zoom/2"
    elif motion == "pan":
        z, x, y = f"{1 + zoom}", f"(iw-iw/zoom)*on/{n}", "ih/2-ih/zoom/2"
    else:
        z, x, y = "1", "0", "0"
    # zoompan rounds the crop position to whole pixels; working on a 4x larger
    # image keeps the motion smooth.
    picture_filter = (
        f"[1:v]scale={width * 4}:{height * 4}:flags=lanczos,"
        f"zoompan=z='{z}':x='{x}':y='{y}':d={n}:s={width}x{height}:fps={fps},format=rgba"
    )
    if motion == "fade_in":
        picture_filter += ",fade=t=in:st=0.5:d=1.0:alpha=1"
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-nostdin",
        "-loop", "1", "-framerate", str(fps), "-i", str(base),
        "-i", str(picture),
        "-i", str(audio),
        "-filter_complex",
        f"{picture_filter}[p];[0:v][p]overlay={position[0]}:{position[1]}:format=auto,"
        f"format=yuv420p[v];{_audio_filter(2)}",
        *_encode_args(length, fps, False, out),
    ]  # fmt: skip
    run(cmd, cancel)


def concat_videos(
    clips: Sequence[Path],
    out: Path,
    *,
    tags: dict[str, str] | None = None,
    cancel: threading.Event | None = None,
) -> None:
    """Join clips with identical encoding parameters without re-encoding."""
    listing = out.with_suffix(".concat.txt")
    listing.write_text(
        "".join("file '{}'\n".format(str(c).replace("'", r"'\''")) for c in clips),
        encoding="utf-8",
    )
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-nostdin",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(listing),
        "-c",
        "copy",
        "-movflags",
        "+faststart",
    ]
    for key, value in (tags or {}).items():
        cmd += ["-metadata", f"{key}={value}"]
    cmd.append(str(out))
    try:
        run(cmd, cancel)
    finally:
        listing.unlink(missing_ok=True)
