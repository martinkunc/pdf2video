# pdf2video — Design Document

Status: v4 · 2026-09-26 (illustrations: LLM-designed diagrams and locally generated pictures)

## 1. Goal

A desktop app (GTK 4 + libadwaita, Python) that turns a document — **PDF, DOCX,
EPUB, TXT/Markdown** — into:

1. **An audiobook**: a set of MP3 files, one per chapter (or chapter part),
   written to `<doc dir>/<doc stem>_audio/`.
2. **An explanatory video**: MP4 files (narrated slides) written to
   `<doc dir>/<doc stem>_video/`.

### Non-goals (v1)

- Editing the text inside the app.
- Avatars, stock footage or generative video; videos are narrated slides with
  optional illustrations and simple motion (see §3.5.1).
- OCR of images embedded in DOCX/EPUB files (only PDF pages are OCR'd).
- Cloud AI services: everything runs locally except the default Edge TTS voices.

## 2. User experience

```
┌──────────────────────────────────────────────────────────────┐
│  pdf2video                                          [≡]      │
├──────────────────────────────────────────────────────────────┤
│  [ /path/to/book.pdf                         ] [Browse…]     │
│  ┌────────────────────────────────────────────────────────┐  │
│  │        Drop a PDF, DOCX, EPUB or TXT file here          │  │
│  └────────────────────────────────────────────────────────┘  │
│  Title: The Book · 12 chapters · ~84 000 words · ~9 h 20 min │
│  ┌ Chapters ──────────────────────────────────────────────┐  │
│  │ 01 Introduction                 1 200 words  ~8 min    │  │
│  │ 02 The Beginning               3 900 words  ~26 min → 3│  │
│  │ …                                                      │  │
│  └────────────────────────────────────────────────────────┘  │
│  Settings: Max chapter length [10] min · Voice [▼] · Speed   │
│  OCR: [Automatic ▼]  languages [auto]                        │
│  Video script: [Local AI ▼ / Extractive]  Model [qwen3:8b ▼] │
│  [ Make audio book ]   [ Make video ]            [Cancel]    │
│  ▓▓▓▓▓▓▓▓░░░░░░░ 42 %  Synthesizing chapter 5/12…            │
└──────────────────────────────────────────────────────────────┘
```

- The file is chosen by typing or pasting a path into the entry (then Enter),
  with **Browse…** (`Gtk.FileDialog`), or by **drag and drop** anywhere on the
  window (`Gtk.DropTarget` accepting `Gdk.FileList` / `Gio.File`).
- As soon as a file is set, it is **parsed in a background thread**. The chapter
  list then shows each chapter's word count, estimated duration and how many
  parts it will be split into.
- **Make audio book** / **Make video** start a background job. A progress bar,
  a status line and a Cancel button are shown. When the job finishes, a toast
  offers **Open folder**.
- **Chapter selection:** every chapter row has a checkbox, and clicking the row
  toggles it. The chapter list header has **All** / **None** buttons and a
  summary ("3 of 12 chapters selected · ~41 min"). Everything is selected after
  a document loads. Both jobs run on `Document.select(numbers)`, a copy with
  only the ticked chapters. The make buttons are disabled when nothing is
  selected. Chapters keep their original `number`, so file names, slide
  labels and tags match the full book. `VideoSection.number` records the
  chapter number in `script.json`, and a reused script is filtered to the
  current selection. The CLI's `--chapters 1,3-5` uses the same mechanism.
- Settings persist in `~/.config/pdf2video/settings.toml`.

## 3. Architecture

```
src/pdf2video/
  __main__.py        entry point (GUI; `--cli` for headless use)
  narration.py       concurrent TTS of many texts (shared by both pipelines)
  cli.py             headless CLI (`pdf2video-cli info|text|audio|video|script|voices|models`)
  settings.py        dataclass + TOML load/save
  model.py           Document / Chapter data model
  textutil.py        cleanup, sentence split, chunking, duration estimate
  parsers/
    __init__.py      load_document(path) → Document (dispatch by extension)
    pdf.py           PyMuPDF
    docx.py          python-docx
    epub.py          ebooklib + BeautifulSoup
    text.py          .txt / .md
  tts/
    __init__.py      TTSEngine protocol + registry
    edge.py          edge-tts (Microsoft neural voices, online, default)
    macos.py         macOS `say` (offline fallback)
  media.py           ffmpeg/ffprobe helpers (encode, concat, duration, still→video)
  audiobook.py       audiobook pipeline
  video/
    script.py        Document → VideoScript (local LLM, or extractive); illustration schema
    diagrams.py      Pillow diagram renderer (flow, cycle, hierarchy, hub, comparison, …)
  llm/
    __init__.py      LocalLLM: built-in llama.cpp runner (load, JSON-constrained chat)
    store.py         model lookup (Ollama folder, app cache) + registry download
    slides.py        Pillow slide renderer (1920×1080)
    builder.py       video pipeline
  imagegen.py        local text-to-image (stable-diffusion.cpp) + model registry/download
  jobs.py            cancellable background job + progress reporting
  ui/
    app.py           Adw.Application
    window.py        main window
```

The core packages (`parsers`, `tts`, `audiobook`, `video`) do not depend on
GTK. The GUI and the CLI are thin layers over them, which makes the core
testable without a display.

### 3.1 Data model

```python
@dataclass
class Chapter:
    title: str
    paragraphs: list[str]
    images: list[bytes] = []   # optional, used by video slides (PDF/EPUB/DOCX)

@dataclass
class Document:
    path: Path
    title: str
    author: str | None
    chapters: list[Chapter]
```

### 3.2 Parsing and chapter detection

Each parser tries **structural** chapter information first and falls back to
heuristics.

| Format | Primary source | Fallback |
|---|---|---|
| PDF | Outline/TOC (`doc.get_toc()`), top 1–2 levels, split by page (and by heading text within the page when possible) | Font-size heuristic: lines whose font is clearly larger than the body text and short → headings |
| DOCX | Paragraph styles `Heading 1` / `Heading 2` / `Title` | Same “short, bold line” heuristic |
| EPUB | Navigation (TOC) mapped to spine documents; `<h1>`/`<h2>` inside | One chapter per spine document |
| TXT/MD | Markdown `#`/`##` headings; lines like `Chapter 12`, `CHAPTER XII`, `Part One` | Whole text as one chapter |

Cleanup (`textutil`, `parsers.postprocess`): strip Project Gutenberg header and
licence, de-hyphenate line breaks (`exam-\nple`), join
wrapped lines into paragraphs, drop repeating page headers/footers and
bare page numbers (PDF), normalise whitespace and quotes, and drop empty
chapters. Table-of-contents chapters are dropped, and runs of tiny leading
chapters (title page, dedication) are merged into one.

#### OCR (scanned PDFs)

PDF pages are OCR'd with **Tesseract through PyMuPDF**
(`page.get_textpage_ocr(dpi=300, full=True)`). The result is an ordinary PyMuPDF
text page, so OCR'd lines go through exactly the same pipeline as native text:
header/footer removal, font-size heading detection (OCR span sizes come from the
glyph box heights) and paragraph joining.

- **Mode** (`ocr` setting): `auto` (default) OCRs only pages with fewer than 25
  extractable characters that show images or drawings. `always` OCRs every page,
  which helps with PDFs that have a broken text layer. `off` never OCRs.
- **Language** (`ocr_languages`): Tesseract codes such as `eng+ces`, or `auto`.
  In auto mode the first scanned page is OCR'd with `eng`, the language is
  detected with the stop-word heuristic, and the page is re-OCR'd with
  `<lang>+eng` if that language pack is installed. The same set is used for the
  rest of the document.
- Page images on OCR'd pages are not used as slide illustrations, because the
  "image" is the scan of the page itself.
- If Tesseract is missing and the PDF has scanned pages, the user gets a clear
  error with install instructions.
- OCR takes about 0.5 s per page. Progress is reported to the GUI and CLI
  while parsing.

If no chapter structure is found, the whole document is one chapter titled
with the document title.

### 3.3 Text-to-speech

`TTSEngine` protocol:

```python
class TTSEngine(Protocol):
    name: str
    def voices(self) -> list[Voice]: ...
    def synthesize(self, text: str, voice: str, rate: float, out: Path) -> None: ...
```

- **edge-tts (default)**: high-quality neural voices, many languages, no API
  key, fast. Needs an internet connection. The output is MP3.
- **macOS `say` (offline fallback)**: always available on macOS. It writes AIFF,
  which is converted to MP3.
- **Piper (offline neural)**: VITS voices run with ONNX Runtime through the
  `piper-tts` package, which bundles espeak-ng for phonemisation. There are 177
  voices in 53 languages, including Czech. The catalogue (`voices.json`) and the
  voice files (`.onnx` + `.onnx.json`) come from Hugging Face
  `rhasspy/piper-voices` and are downloaded on first use into
  `~/.cache/pdf2video/voices`, MD5-verified against the catalogue. The
  catalogue is cached for 7 days, and downloaded voices keep working offline.
  Engines with downloadable voices implement an optional `prepare(voice,
  on_progress)`, which the pipelines call before synthesis so the download shows
  in the progress bar. Speed maps to `length_scale = 1 / rate`. Output is WAV.
  Licence: GPL-3.0 (fine for this app; relevant only if it is redistributed
  under another licence).

The engine is chosen in settings. If edge-tts fails (for example, when offline),
the job reports a clear error that suggests the offline engine. The voice list
is filtered by the language detected in the document (simple stop-word
heuristic, with English as the default).

Text is sent to the engine in **chunks of at most ~900 characters (smaller for short
max lengths), cut at paragraph or sentence boundaries**. Smaller chunks make it possible to
balance the chapter parts precisely, and a failed chunk can be retried on its own (3 attempts
with backoff). Up to 4 chunks are synthesized concurrently.

### 3.4 Audiobook pipeline and chapter splitting

Requirements:
- Use the document's natural chapters.
- A chapter longer than the **max length (default 10 min)** is split into parts
  whose file names carry an index suffix.
- The output is MP3.

Algorithm (per chapter):

1. Chunk the chapter text into sentence-aligned chunks (≤ ~900 chars).
2. Synthesize every chunk to a temporary MP3 and measure its **real** duration
   with ffprobe. Real durations are used instead of word-count estimates, so
   the split honours the limit whatever the voice speed is.
3. Pack the chunks greedily into parts of **≤ max length**. A new part starts
   only at a chunk (sentence) boundary, preferably at a paragraph boundary.
   The parts are balanced: with remaining duration R and n = ceil(R / max) parts
   left, the target is R / n, recomputed after each cut. This gives, say,
   3 × 8.7 min instead of 10 + 10 + 6.
4. Concatenate each part's chunks and re-encode once to a uniform MP3
   (`libmp3lame`, mono, 44.1 kHz, 64 kbps VBR, which is a common audiobook
   profile). Each part starts with a short silence, and the first part of a
   chapter also narrates the chapter title.
5. Tag the files with ID3 (written by ffmpeg): title, album = document title,
   artist = author, track number, genre “Audiobook”.

File naming (in `<stem>_audio/`):

```
01 - Introduction.mp3
02 - The Beginning_1.mp3
02 - The Beginning_2.mp3
02 - The Beginning_3.mp3
03 - Epilogue.mp3
```

Chapter titles are sanitised for the file system (no `/:*?"<>|`, max 80
chars). An existing output folder is reused, and files in it with the same
names are overwritten.

Temporary files go to a `tempfile.TemporaryDirectory`, which is removed on
completion or cancel.

### 3.5 Explanatory video pipeline

An *explanatory* video should **explain** the content rather than read it
verbatim. The pipeline:

1. **Script generation** (`video/script.py`) produces a `VideoScript`:
   ```python
   class Scene(BaseModel):
       title: str
       bullets: list[str]      # 2–5 short points shown on the slide
       narration: str          # what the narrator says (60–150 words)
   class VideoSection(BaseModel):
       chapter_title: str
       scenes: list[Scene]
   ```
   - **AI mode (default):** a local LLM run by the **built-in llama.cpp runner**
     (`llama-cpp-python`, Metal/CUDA/CPU). No server has to be installed or running
     (see §3.6). Output is constrained to the Pydantic JSON schema with a llama.cpp
     grammar (`response_format={"type": "json_object", "schema": …}`). For Qwen3
     models `/no_think` is appended, so the model answers directly without a
     reasoning phase. The prompt asks for a clear teaching-style explanation in
     2–8 scenes per request, in the document's language.
     - Local models have small context windows. Long chapters are therefore
       split into **segments of ≤ 20 000 characters** (paragraph/sentence
       aligned, never truncated) with `n_ctx = 16384`. Each segment gets its own
       request ("part k of n: introduce / continue / recap"), and the scenes are
       concatenated.
     - Requests run sequentially (one llama.cpp context). Malformed or cut-off
       JSON is retried once.
     - Before the first chapter, `prepare()` makes sure the model is on disk
       (downloading it if needed, as the first 20 % of the script progress) and
       loads it. The loaded model stays in memory for later jobs. In `auto` mode
       an unavailable model or a failing chapter falls back to extractive; in
       `ai` mode it is an error.
     - The script is saved as `script.json` and can be reused (`--reuse-script`)
       to re-render videos without calling the model again.
   - **Extractive mode (no model needed):** each chapter is split into scenes
     of ~120–180 words. Bullets are the leading sentence of each paragraph,
     shortened. The narration is the original text.
2. **Slides** (`video/slides.py`) are rendered with Pillow at 1920×1080 in a
   clean, readable theme: a title bar, bullets with automatic font sizing and
   wrapping, a progress indicator and the chapter name. When a PDF/EPUB/DOCX
   chapter has an embedded image, it can be shown next to the bullets. Each
   chapter also gets a title slide.
3. **Narration** is produced with the same TTS engine, one audio file per scene.
4. **Scene clips**: `ffmpeg -loop 1 -i slide.png -i narration.mp3` is encoded
   with H.264 (`yuv420p`, `-tune stillimage`, 30 fps) and AAC. The clip lasts
   as long as the narration plus 0.6 s of padding.
5. **Chapter videos**: the scene clips are concatenated (concat demuxer,
   stream copy) into `NN - <Chapter>.mp4`. The same max-length rule as for
   audio applies: a chapter longer than the limit is split at scene boundaries
   into `NN - <Chapter>_1.mp4`, `_2`, …
6. **Subtitles**: a `.srt` file per video is generated from the narration
   timings.

Output (in `<stem>_video/`):
```
01 - Introduction.mp4
01 - Introduction.srt
02 - The Beginning.mp4
…
script.json          # the generated script, for review/reuse
```

### 3.5.1 Illustrations (optional)

With `illustrations = true` (GUI switch, CLI `--illustrations`) the AI script also
decides **which scenes need a visual and what kind**. Each `Scene` gets an optional
`illustration`:

```python
class Illustration(BaseModel):
    kind: "diagram" | "picture"
    placement: "right" | "left" | "full"       # full = under the title, no bullets
    animation: "none" | "fade_in" | "reveal" | "zoom_in" | "zoom_out" | "pan"
    prompt: str              # picture: English prompt for the image model
    diagram: Diagram | None  # diagram: type + items (label, detail, value)
    image: str               # picture file, filled in by the builder (informational)
```

- **Choosing the kind** is part of the prompt (`ILLUSTRATION_RULES`): *diagrams*
  for processes, cycles, hierarchies, a concept and its aspects, comparisons,
  timelines and quantities; *pictures* for concrete, visual subjects (objects,
  places, organisms, historical scenes, metaphors). Diagrams are preferred for
  anything with parts and relations, because they are precise and readable.
- **Schema tricks for small local models.** The grammar only enforces what is
  *required*, so the schema sent to the model makes `illustration` required but
  nullable, and every field of it required. Without that, `qwen3:8b` simply
  omitted the field (or kept every default). The answer is still parsed with the
  lenient model, so a missing field never discards a segment. The rules also say
  that illustrations never replace scenes; without that sentence the model
  wrote fewer, longer scenes.
- **Clean-up in code** (`clean_illustration`, `_limit_illustrations`): pictures
  without a real prompt and diagrams with too few items (or bars without
  values) are dropped. Incompatible animations are replaced (reveal→zoom for
  pictures, zoom/pan→reveal for diagrams). At most 60 % of a segment's scenes
  keep an illustration: the opening scene's goes first, then pictures, then the
  latest diagrams.
- **Diagrams are data, not drawings.** Local models are poor at emitting precise
  SVG but good at choosing a structure. `video/diagrams.py` lays out seven types
  (flow, cycle, hierarchy, hub, comparison, timeline, bars) with Pillow in the
  slide theme. Font sizes are shared across a diagram's nodes, details are
  dropped if they would make the text too small, and wide or narrow areas get
  different layouts. `render_diagram(d, size, visible=k)` draws only the first
  k items, which is what the **reveal** animation uses.
- **Pictures** come from `imagegen.py`: stable-diffusion.cpp
  (`stable-diffusion-cpp-python`, Metal) with GGUF weights from Hugging Face.
  The registry pins each file's sha256 and size, and downloads reuse the
  LLM store's resumable, verified `fetch_file`.

  | name | files | speed (M4 Max) | notes |
  |---|---|---|---|
  | `sdxl-turbo` (default) | 4.1 GB | ~3.5 s at 512² | 4 steps, cfg 1 |
  | `z-image-turbo` | 7.9 GB (DiT Q4 + Qwen3-4B encoder + VAE) | ~50 s at 768², ~90 s at 1024² | best quality, Apache-2.0 |

  A path to a `.gguf`/`.safetensors` SD checkpoint also works (20 steps, cfg 7).
  The picture's aspect follows its placement (≈1:1 beside the bullets, 2:1 for
  "full"), at the model's native pixel count. A fixed style suffix keeps the
  pictures consistent ("clean modern editorial illustration, flat vector …, no
  text"). The seed is derived from the prompt, so results are reproducible.
- **Pipeline order** in `make_video`: script → *pictures* (the LLM is unloaded
  first so both models never share memory; the image model is unloaded after)
  → slides/narration/clips. Pictures are cached as
  `<stem>_video/illustrations/<hash of model, size, style, prompt>.png`, so
  `--reuse-script` re-renders don't regenerate them. Editing a prompt in
  `script.json` makes a new picture, and replacing a PNG by hand keeps that
  image. If the image model can't be loaded or a picture fails, those scenes
  just have no illustration.
- **Motion** (`media.py`); all clip types use identical stream parameters, so
  the chapter concat stays a stream copy:
  - `frames_clip`: a sequence of slide images with `xfade` cross-fades at given
    times. Used for diagram *reveal* (item k appears at
    `0.2 s + narration × 0.85 × k/n`, so the diagram is complete before the
    scene ends) and *fade_in* (at 0.8 s).
  - `motion_clip`: the slide with an empty picture frame, plus the picture
    overlaid with `zoompan` (*zoom_in/out* ±12 %, *pan*; the picture is
    upscaled 4× first so the motion is smooth) or an alpha *fade_in*.
  - `still_clip`: unchanged, for everything else.
- Illustrations are ignored when the setting is off, even if `script.json`
  contains them, and they are never produced by the extractive script.

### 3.6 Local model runner and model store

Goals: no separate LLM server to install, reuse models the user already has, and
fetch a model automatically when it is missing.

- **Runtime:** `llama-cpp-python` (llama.cpp) runs GGUF models in-process, with
  Metal on Apple Silicon and all layers offloaded to the GPU. The `Llama`
  instance is cached per model path.
- **Model names** follow Ollama's scheme (`qwen3:8b`, `user/model:tag`,
  `hf.co/org/repo:quant`). A path to a `.gguf` file also works.
- **Lookup order** (`llm/store.py`):
  1. **Ollama's folder** (`$OLLAMA_MODELS`, `~/.ollama/models`, or
     `/usr/share/ollama/.ollama/models`). The manifest at
     `manifests/<host>/<namespace>/<model>/<tag>` names the
     `application/vnd.ollama.image.model` layer, whose blob
     `blobs/sha256-<digest>` is a plain GGUF file with an embedded chat
     template, so llama.cpp loads it directly. Ollama does not need to be
     running, and its folder is only read.
  2. **App cache** `~/.cache/pdf2video/models`, in the same layout.
  3. **Download** from the model's registry (`https://<host>/v2/<ns>/<model>/
     manifests/<tag>`, then `/blobs/<digest>`). This is the same protocol
     `ollama pull` uses, so any model from ollama.com (or `hf.co/...` GGUF
     repos) works. Downloads stream to `<blob>.partial`, resume with HTTP
     `Range`, are verified against the sha256 digest, and only then are
     renamed and the manifest written.
- **Recommended models** in the picker are `qwen3:4b`, `qwen3:8b` (default),
  `qwen3:14b` and `qwen3:30b`. The GUI lists the local models first, then the
  downloadable ones.
- **Settings:** `llm_model`, plus `llm_auto_download` (default on; the CLI
  flag `--no-download` turns it off).

### 3.7 Background jobs and progress

`jobs.Job` runs the pipeline in a `threading.Thread`. The pipeline reports
progress through a callback `(fraction: float, message: str)`, which the UI
wraps with `GLib.idle_add` so that widgets are only touched on the main thread.
Cancellation is a `threading.Event` that the pipeline checks between chunks and
scenes. Any running ffmpeg subprocess is terminated.

Parsing also runs in a thread, so large PDFs never freeze the UI.

### 3.8 Error handling

- **Process exit with a loaded model:** llama.cpp's Metal backend asserts in a
  static destructor (`GGML_ASSERT([rsets->data count] == 0)`) if a model is
  still alive when the process exits. `llm.unload()` therefore runs from
  `atexit` and from `Application.do_shutdown`. It waits for any running
  generation to finish (a module-wide inference lock) and then frees the model
  explicitly.

- Unsupported extension / unreadable file / no extractable text → an error
  banner in the window (`Adw.Banner`/toast) with a human-readable message.
- A missing `ffmpeg` is detected at startup and reported with install
  instructions (`brew install ffmpeg`).
- TTS network failure → the chunk is retried, then the job fails with a
  suggestion to switch to the offline engine.

## 4. Technology choices

| Concern | Choice | Why |
|---|---|---|
| Python | 3.13+ (developed on 3.14) | latest |
| Project/packaging | `pyproject.toml` + **uv** (`uv sync`, `uv run`), hatchling build backend | modern, fast, lockfile |
| GUI | GTK 4 + libadwaita via **PyGObject** | requested; native look on Linux, works on macOS via Homebrew |
| PDF | **PyMuPDF** | fastest and most accurate text + TOC + font info |
| DOCX | **python-docx** | standard |
| EPUB | **ebooklib** + **beautifulsoup4**/lxml | standard |
| TTS | **edge-tts** (default), **Piper** (offline neural), macOS `say` | best online quality; offline neural voices incl. Czech; system fallback |
| Audio/video | **ffmpeg** (system binary) via `subprocess` | robust, no heavy Python wrappers |
| Slides | **Pillow** | simple, no browser dependency |
| AI script | **llama-cpp-python** (built-in), GGUF models, default `qwen3:8b` | runs in-process, no server; grammar-constrained JSON; reuses Ollama's downloaded models |
| Pictures | **stable-diffusion-cpp-python**, GGUF from Hugging Face, default SDXL-Turbo | in-process like the LLM, Metal, no torch; auto-download |
| Diagrams | LLM-chosen structure drawn with **Pillow** | precise, legible, animatable; no SVG/Graphviz dependency |
| Model download | **httpx** against the Ollama/OCI registry | resumable, sha256-verified, same names as Ollama |
| OCR | **Tesseract** via PyMuPDF | mature, 100+ languages, reuses the PyMuPDF text pipeline |
| Lint/test | ruff, pytest | |

### System dependencies

macOS (Homebrew): `brew install gtk4 libadwaita gobject-introspection pkg-config cairo cmake ffmpeg uv tesseract tesseract-lang`
Linux (Debian/Ubuntu): `apt install libgtk-4-dev libadwaita-1-dev libgirepository-2.0-dev libcairo2-dev cmake build-essential ffmpeg tesseract-ocr tesseract-ocr-all`

`llama-cpp-python` is compiled on install (about 30 s). On Apple Silicon it
uses Metal automatically. For CUDA, set `CMAKE_ARGS="-DGGML_CUDA=on"`.
`stable-diffusion-cpp-python` is compiled the same way. For Metal, build it with
`CMAKE_ARGS="-DSD_METAL=ON"` (CUDA: `-DSD_CUDA=ON`).

PyGObject is installed from PyPI into the uv venv. It builds against the system
GObject-Introspection, so these libraries must be installed first.

## 5. Configuration

`settings.toml`:
```toml
max_chapter_minutes = 10
tts_engine = "edge"          # "edge" | "piper" | "macos"
voice = "en-US-AndrewMultilingualNeural"
rate = 1.0                   # speaking speed multiplier
video_script = "auto"        # "auto" | "ai" | "extractive"
llm_model = "qwen3:8b"       # Ollama-style name or path to a .gguf file
llm_auto_download = true     # fetch the model on first use if missing
ocr = "auto"                 # "auto" | "always" | "off"
ocr_languages = "auto"       # e.g. "eng+ces"
illustrations = false        # AI adds diagrams / generated pictures to some scenes
image_model = "sdxl-turbo"   # "sdxl-turbo" | "z-image-turbo" | path to a checkpoint
```

## 6. Testing

- A full **CLI** (`pdf2video-cli`) mirrors every GUI function, so everything can
  be tested headless: `info`, `text` (dump parsed/OCR'd text, `--json`),
  `script` (LLM only), `audio`, `video` (`--reuse-script`), `voices`, `models`.
  It supports `--chapters 1,3-5` for quick runs.
- OCR tests render text (including Czech diacritics) into an image-only PDF.
- Model store tests use a fake Ollama folder and a mocked registry: lookup, an
  interrupted download resuming with `Range`, and checksum rejection.
- LLM tests use a fake runner: request shape, segmenting of long
  chapters, fallback and error behaviour.
- Illustration tests: schema (required/nullable, no `image`), clean-up and the
  60 % limit, every diagram type × placement rendering with a growing reveal,
  and an end-to-end video with a fake image generator (reveal, fade, zoom, pan,
  static pictures; decodes cleanly; pictures cached and reused; ignored when
  the setting is off).
- Unit tests: text cleanup, sentence chunking, balanced part splitting,
  filename sanitising, and the chapter detection for each format (small
  fixtures generated in tests: PDF via PyMuPDF, DOCX via python-docx, EPUB via
  ebooklib).
- Pipeline tests use a fake TTS engine that writes silence of a length
  proportional to the text. This lets the splitting and ffmpeg stages be tested
  offline and quickly.
- Manual: a GUI smoke test on macOS.

## 7. Future work

- Offline neural TTS (Kokoro / Piper) engine.
- Chapter-marked single-file M4B output.
- Page crops from the PDF as illustrations.
- Image-to-video models for real animation once they run fast enough locally.
