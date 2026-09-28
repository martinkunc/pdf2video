# pdf2video

A GTK 4 / libadwaita desktop app that turns **PDF, DOCX, EPUB, TXT and Markdown**
documents into

- **audiobooks**: one MP3 per chapter in `<document folder>/<name>_audio/`. Chapters
  longer than the maximum length (10 min by default) are split into
  `NN - Title_1.mp3`, `NN - Title_2.mp3`, …
- **explanatory videos**: narrated slides as MP4 (+ `.srt` subtitles) in
  `<document folder>/<name>_video/`. A **local LLM** (built in, `qwen3:8b` by default)
  writes a teaching-style script for each chapter. Without it, the slides show the
  key sentences and the narration reads the original text. Optionally the AI also
  adds **illustrations**: diagrams, or pictures drawn by a local image model.

Scanned PDFs are read with **OCR** (Tesseract), and the OCR language is detected
automatically.

See [docs/DESIGN.md](docs/DESIGN.md) for the design.

## Install

### Prebuilt packages

Each [GitHub release](../../releases) has standalone packages that include Python,
GTK and ffmpeg:

- **Windows**: `pdf2video-<version>-windows-x86_64.zip`. Unzip it and run
  `pdf2video.exe` (GUI) or `pdf2video-cli.exe` (command line).
- **macOS**: `pdf2video-<version>-macos-arm64.dmg` (Apple Silicon) or `…-x86_64.dmg`
  (Intel). Drag `pdf2video.app` to Applications. The app isn't signed, so on first
  launch right-click it → **Open**, or run `xattr -dr com.apple.quarantine /Applications/pdf2video.app`.
- **Linux**: `pdf2video-<version>-linux-x86_64.tar.gz`. Unpack it and run `pdf2video/pdf2video`.

OCR still needs a system Tesseract (see [OCR](#ocr)). Voices and AI models are
downloaded on first use, as usual.

### From source

System dependencies (macOS / Homebrew):

```sh
brew install uv gtk4 libadwaita gobject-introspection pkg-config cairo ffmpeg \
  tesseract tesseract-lang cmake
```

Debian/Ubuntu:

```sh
sudo apt install libgtk-4-dev libadwaita-1-dev libgirepository-2.0-dev libcairo2-dev \
  pkg-config ffmpeg tesseract-ocr tesseract-ocr-all cmake build-essential
```

Then:

```sh
CMAKE_ARGS="-DSD_METAL=ON" uv sync   # on Linux: plain `uv sync` (CUDA: -DSD_CUDA=ON)
uv run pdf2video            # GUI
uv run pdf2video book.epub  # GUI with a document preloaded
```

## Usage

1. Drop a document onto the window, type or paste its path, or click the folder
   button.
2. Check the detected chapters. All of them are selected; untick the ones you
   don't want, or click **None** and tick just the ones you need (**All**
   selects everything again). Both the audio book and the video process only
   the selected chapters, and the files keep the book's chapter numbers
   (`03 - …`, `07 - …`).
3. Adjust the settings (max chapter length, voice, speed, video script).
4. Click **Make audio book** or **Make video**.

### Voices

- **Kokoro neural voices** (default): **fully offline** and the most natural
  sounding, with good intonation and pauses. English (US and UK, about 30 voices;
  the default is `af_heart`), Spanish, French, Italian, Portuguese and Hindi. The
  model (350 MB) is downloaded on first use into `~/.cache/pdf2video/kokoro`.
  About 6–8× faster than real time on Apple Silicon. For other languages (Czech,
  German, …) a Piper voice is used automatically.
- **Microsoft Edge neural voices**: high quality and many languages (also a good
  Czech voice). Needs an internet connection.
- **Piper neural voices**: **fully offline**, 50+ languages including Czech
  (e.g. `cs_CZ-jirka-medium`). The chosen voice (60–120 MB) is downloaded
  automatically the first time it's used, into `~/.cache/pdf2video/voices`.
  After that no internet is needed, and it's fast: about 1 minute of speech in
  under 2 seconds on Apple Silicon.
- **macOS voices**: offline, using the system `say` command. The *Premium* and
  *Enhanced* voices sound best. Download them in System Settings →
  Accessibility → Spoken Content → System voice → Manage Voices.

For a completely offline setup, use Kokoro (or Piper) voices together with the built-in AI
model.

The voice is chosen automatically from the document's language unless you pick
one.

### AI video scripts (local, built in)

The scripts are written by a language model that runs **inside the app**
(llama.cpp). There's no API key and no server to install, and nothing leaves
your machine.

- If you use [Ollama](https://ollama.com), the models it has already downloaded
  are found in `~/.ollama/models` and used directly. Ollama doesn't have to be
  running.
- Otherwise the selected model is **downloaded automatically on first use** into
  `~/.cache/pdf2video/models`. The download resumes if interrupted and is
  checksum-verified. It uses the same names as Ollama, e.g. `qwen3:8b`
  (5.2 GB, default), `qwen3:4b` (2.5 GB, faster) or `qwen3:30b` (best).
- A path to any `.gguf` file works as the model name too.

The script is saved as `script.json` next to the videos.

### Illustrations

Turn on **Illustrations** (or pass `--illustrations`) and the AI decides, scene by
scene, whether a visual would help and what kind:

- **Diagrams** for processes, cycles, hierarchies, a concept and its aspects,
  comparisons, timelines and numbers. The AI supplies the structure and labels, and
  the app draws them in the slide style, so the text is always sharp and correct.
- **Pictures** for concrete things: objects, places, organisms, historical
  scenes. The AI writes a description, and a local image model paints it:
  `sdxl-turbo` (default, 4.1 GB, a few seconds per picture) or `z-image-turbo`
  (7.9 GB, better, about 1–2 minutes per picture). The model is downloaded on
  first use into `~/.cache/pdf2video/image-models`.

The AI also picks the **placement** (beside the bullets or large, under the title)
and an **animation**: diagrams can build up item by item while the narrator
explains them or fade in, and pictures can slowly zoom, pan or fade in. At most
60 % of the scenes get an illustration.

**Book images.** If the chapter has images of its own (figures, photos, maps), they
are saved in `<name>_video/book_images/` and a local vision model (`qwen3-vl-4b`,
3.0 GB, downloaded on first use) writes a short summary of each into
`book_images.json`. Before any picture is generated, the AI compares the picture it
planned with the book's images and uses the original where it fits the scene
better. Its choice and the reason are in `script.json` (`book_image`,
`book_image_reason`). Turn this off with the **Book images** switch or
`--no-book-images`.

Pictures are saved in `<name>_video/illustrations/` and reused when you re-render.
You can edit a prompt, placement or animation in `script.json` and run
`video --reuse-script --illustrations`, or replace a PNG with your own image.

### OCR

With OCR set to *Automatic*, PDF pages without a text layer are OCR'd. *Always*
re-reads every page, which helps with PDFs whose text layer is broken. The
language is detected automatically, or you can set Tesseract codes such as
`eng+ces`.

### Command line (no GUI)

Every GUI feature is also available from the terminal, which is handy for
testing:

```sh
uv run pdf2video-cli info   book.pdf                       # chapters, language
uv run pdf2video-cli text   scan.pdf --ocr always --chapters 1   # extracted/OCR'd text
uv run pdf2video-cli audio  book.epub --max-minutes 15 --chapters 2-3
uv run pdf2video-cli script book.pdf --model qwen3:8b      # only the LLM script
uv run pdf2video-cli video  book.pdf --reuse-script        # re-render, no LLM
uv run pdf2video-cli video  book.pdf --illustrations       # AI diagrams + pictures
uv run pdf2video-cli script book.pdf --illustrations       # see what the AI picks
uv run pdf2video-cli voices --engine piper --language cs
uv run pdf2video-cli models                                # local/downloadable LLMs
uv run pdf2video-cli models --pull qwen3:4b                # download a model
uv run pdf2video-cli models --pull-image sdxl-turbo        # download an image model
uv run pdf2video-cli models --pull-vision qwen3-vl-4b      # download the vision model
```

`uv run pdf2video audio book.pdf` works too. Run `pdf2video-cli COMMAND --help`
for all options (`--engine`, `--voice`, `--rate`, `--script-mode`, `--image-model`,
`--vision-model`, `--book-images`,
`--ocr-lang`, …).

Settings are stored in `~/.config/pdf2video/settings.toml`.

## Development

```sh
uv run pytest
uv run ruff check src tests && uv run ruff format src tests
```

### Building packages

`packaging/build.py` builds a standalone package for the current OS with
[PyInstaller](https://pyinstaller.org). It downloads static ffmpeg/ffprobe into
`packaging/ffmpeg/`, bundles them, and writes a `.zip` (Windows), `.dmg` (macOS) or
`.tar.gz` (Linux) to `dist/`:

```sh
uv sync --group build --no-binary-package pillow --reinstall-package pillow  # macOS
uv sync --group build                                                        # Linux
uv run --no-sync python packaging/build.py
```

On macOS, Pillow must be built from source so that it uses the same Homebrew
harfbuzz/freetype as GTK. Pillow's wheel ships its own copies, and they stop GTK
from loading in the bundle. On Windows, GTK comes from
[gvsbuild](https://github.com/wingtk/gvsbuild); see the workflow for the setup.

When a GitHub release is published, `.github/workflows/release.yml` builds the
packages for Windows, macOS (arm64 and x86_64) and Linux and attaches them to the
release. It can also be started by hand from the Actions tab, and then it only uploads
them as workflow artifacts.
# pdf2video
