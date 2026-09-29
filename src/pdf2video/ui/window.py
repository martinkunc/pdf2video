"""Main application window."""

from __future__ import annotations

import logging
import math
import subprocess
import sys
import threading
import traceback
from collections.abc import Callable
from pathlib import Path

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, Gdk, Gio, GLib, Gtk  # noqa: E402

from ..imagegen import IMAGE_MODELS, ImageGenerator  # noqa: E402
from ..jobs import JobContext  # noqa: E402
from ..llm import RECOMMENDED, LLMStatus, LocalLLM  # noqa: E402
from ..llm import abort as abort_llm  # noqa: E402
from ..llm.store import format_size  # noqa: E402
from ..llm.vision import VisionLLM  # noqa: E402
from ..llm.vision import abort as abort_vision  # noqa: E402
from ..media import Cancelled, check_ffmpeg  # noqa: E402
from ..model import Document  # noqa: E402
from ..parsers import SUPPORTED_EXTENSIONS, load_document  # noqa: E402
from ..settings import Settings  # noqa: E402
from ..textutil import detect_language, estimate_seconds, format_duration  # noqa: E402
from ..tts import ENGINES, Voice, get_engine  # noqa: E402

log = logging.getLogger(__name__)

OCR_MODES = [
    ("auto", "Automatic (scanned pages only)"),
    ("always", "Always (re-read every page)"),
    ("off", "Off"),
]

SCRIPT_MODES = [
    ("auto", "Automatic (local AI model if available)"),
    ("ai", "Local AI model explains the text"),
    ("extractive", "Original text on slides (offline)"),
]

CSS = b"""
.drop-zone {
  border: 2px dashed alpha(@accent_color, 0.5);
  border-radius: 12px;
  padding: 24px;
}
.drop-zone.dragging, .drop-zone:drop(active) {
  background: alpha(@accent_color, 0.12);
  border-color: @accent_color;
}
"""


def _idle(fn: Callable, *args) -> None:
    GLib.idle_add(lambda: (fn(*args), False)[1])


class MainWindow(Adw.ApplicationWindow):
    def __init__(self, app: Adw.Application):
        super().__init__(application=app, title="pdf2video", default_width=760, default_height=860)
        self.settings = Settings.load()
        self.document: Document | None = None
        self.language = "en"
        self.voices: list[Voice] = []
        self.voice_ids: list[str] = []
        self.job: JobContext | None = None
        self.job_thread: threading.Thread | None = None
        self._updating_voices = False
        self.selected: set[int] = set()  # numbers of the chapters to process
        self.chapter_checks: dict[int, Gtk.CheckButton] = {}
        self._bulk_selecting = False
        self._updating_models = False
        self.llm: LLMStatus | None = None

        provider = Gtk.CssProvider()
        provider.load_from_data(CSS)
        Gtk.StyleContext.add_provider_for_display(
            Gdk.Display.get_default(), provider, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
        )

        self.toasts = Adw.ToastOverlay()
        toolbar = Adw.ToolbarView()
        header = Adw.HeaderBar()
        toolbar.add_top_bar(header)
        self.banner = Adw.Banner(revealed=False)
        self.banner.set_button_label("Dismiss")
        self.banner.connect("button-clicked", lambda b: b.set_revealed(False))
        toolbar.add_top_bar(self.banner)

        scroller = Gtk.ScrolledWindow(vexpand=True, hscrollbar_policy=Gtk.PolicyType.NEVER)
        clamp = Adw.Clamp(maximum_size=720)
        self.content = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL,
            spacing=18,
            margin_top=18,
            margin_bottom=18,
            margin_start=12,
            margin_end=12,
        )
        clamp.set_child(self.content)
        scroller.set_child(clamp)

        body = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        body.append(scroller)
        body.append(self._build_action_bar())
        toolbar.set_content(body)
        self.toasts.set_child(toolbar)
        self.set_content(self.toasts)

        self._build_file_section()
        self._build_chapters_section()
        self._build_settings_section()
        self._install_drop_target()
        self._update_buttons()
        self._load_voices()

        if error := check_ffmpeg():
            self._show_error(error)

    # ------------------------------------------------------------------ layout

    def _build_file_section(self) -> None:
        group = Adw.PreferencesGroup(title="Document")
        self.path_row = Adw.EntryRow(
            title="File path (PDF, DOCX, EPUB, TXT, MD)", show_apply_button=True
        )
        self.path_row.connect("apply", lambda row: self.open_path(row.get_text()))
        self.path_row.connect("entry-activated", lambda row: self.open_path(row.get_text()))
        browse = Gtk.Button(
            icon_name="document-open-symbolic", valign=Gtk.Align.CENTER, tooltip_text="Browse…"
        )
        browse.add_css_class("flat")
        browse.connect("clicked", self._on_browse)
        self.path_row.add_suffix(browse)
        group.add(self.path_row)
        self.content.append(group)

        self.drop_zone = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.drop_zone.add_css_class("drop-zone")
        icon = Gtk.Image(icon_name="folder-download-symbolic", pixel_size=48)
        icon.add_css_class("dim-label")
        self.drop_label = Gtk.Label(label="Drop a document here")
        self.drop_label.add_css_class("title-3")
        hint = Gtk.Label(label="or type a path above / click the folder button")
        hint.add_css_class("dim-label")
        self.drop_zone.append(icon)
        self.drop_zone.append(self.drop_label)
        self.drop_zone.append(hint)
        self.content.append(self.drop_zone)

        self.info_label = Gtk.Label(xalign=0, wrap=True, visible=False)
        self.info_label.add_css_class("heading")
        self.content.append(self.info_label)

    def _build_chapters_section(self) -> None:
        self.chapters_group = Adw.PreferencesGroup(title="Chapters", visible=False)
        selection_buttons = Gtk.Box(spacing=6, valign=Gtk.Align.CENTER)
        for label, tooltip, select_all in (
            ("All", "Select all chapters", True),
            ("None", "Deselect all chapters", False),
        ):
            button = Gtk.Button(label=label, tooltip_text=tooltip)
            button.add_css_class("flat")
            button.connect("clicked", lambda _b, value=select_all: self._select_all(value))
            selection_buttons.append(button)
        self.selection_buttons = selection_buttons
        self.chapters_group.set_header_suffix(selection_buttons)
        self.chapters_list = Gtk.ListBox(selection_mode=Gtk.SelectionMode.NONE)
        self.chapters_list.add_css_class("boxed-list")
        self.chapters_group.add(self.chapters_list)
        self.content.append(self.chapters_group)

    def _build_settings_section(self) -> None:
        s = self.settings
        audio = Adw.PreferencesGroup(title="Narration")
        self.max_len_row = Adw.SpinRow.new_with_range(1, 180, 1)
        self.max_len_row.set_title("Maximum chapter length")
        self.max_len_row.set_subtitle("Minutes; longer chapters are split into _1, _2, … files")
        self.max_len_row.set_value(s.max_chapter_minutes)
        self.max_len_row.connect("notify::value", self._on_max_len_changed)
        audio.add(self.max_len_row)

        self.engine_keys = list(ENGINES)
        self.engine_row = Adw.ComboRow(
            title="Voice engine", model=Gtk.StringList.new(list(ENGINES.values()))
        )
        if s.tts_engine in self.engine_keys:
            self.engine_row.set_selected(self.engine_keys.index(s.tts_engine))
        self.engine_row.connect("notify::selected", self._on_engine_changed)
        audio.add(self.engine_row)

        self.voice_row = Adw.ComboRow(
            title="Voice", enable_search=True, model=Gtk.StringList.new(["Automatic (by language)"])
        )
        self.voice_row.connect("notify::selected", self._on_voice_changed)
        audio.add(self.voice_row)

        self.rate_row = Adw.SpinRow.new_with_range(0.5, 2.0, 0.05)
        self.rate_row.set_digits(2)
        self.rate_row.set_title("Speaking speed")
        self.rate_row.set_value(s.rate)
        self.rate_row.connect("notify::value", self._on_rate_changed)
        audio.add(self.rate_row)
        self.content.append(audio)

        ocr = Adw.PreferencesGroup(title="Text recognition (OCR)")
        self.ocr_row = Adw.ComboRow(
            title="OCR for PDFs", model=Gtk.StringList.new([m[1] for m in OCR_MODES])
        )
        ocr_keys = [m[0] for m in OCR_MODES]
        self.ocr_row.set_selected(ocr_keys.index(s.ocr) if s.ocr in ocr_keys else 0)
        self.ocr_row.connect("notify::selected", self._on_ocr_changed)
        ocr.add(self.ocr_row)
        self.ocr_lang_row = Adw.EntryRow(
            title="OCR languages (Tesseract codes like eng+ces, or auto)",
            text=s.ocr_languages,
            show_apply_button=True,
        )
        self.ocr_lang_row.connect("apply", self._on_ocr_languages_changed)
        ocr.add(self.ocr_lang_row)
        self.content.append(ocr)

        video = Adw.PreferencesGroup(title="Video")
        self.script_row = Adw.ComboRow(
            title="Script", model=Gtk.StringList.new([m[1] for m in SCRIPT_MODES])
        )
        keys = [m[0] for m in SCRIPT_MODES]
        self.script_row.set_selected(keys.index(s.video_script) if s.video_script in keys else 0)
        self.script_row.connect("notify::selected", self._on_script_changed)
        video.add(self.script_row)
        self.model_row = Adw.ComboRow(
            title="AI model (runs on this computer)",
            model=Gtk.StringList.new([s.llm_model]),
            enable_search=True,
        )
        self.model_names = [s.llm_model]
        self.model_row.connect("notify::selected", self._on_model_changed)
        video.add(self.model_row)

        self.illustrations_row = Adw.SwitchRow(title="Illustrations", active=s.illustrations)
        self.illustrations_row.connect("notify::active", self._on_illustrations_changed)
        video.add(self.illustrations_row)
        self.image_model_names = list(IMAGE_MODELS)
        if s.image_model not in self.image_model_names:
            self.image_model_names.append(s.image_model)  # a checkpoint path from settings
        self.image_model_row = Adw.ComboRow(
            title="Picture model (runs on this computer)",
            model=Gtk.StringList.new(
                [
                    f"{name} · {IMAGE_MODELS[name].note}" if name in IMAGE_MODELS else name
                    for name in self.image_model_names
                ]
            ),
        )
        self.image_model_row.set_selected(self.image_model_names.index(s.image_model))
        self.image_model_row.connect("notify::selected", self._on_image_model_changed)
        video.add(self.image_model_row)
        self.book_images_row = Adw.SwitchRow(title="Book images", active=s.book_images)
        self.book_images_row.connect("notify::active", self._on_book_images_changed)
        video.add(self.book_images_row)
        self.content.append(video)
        self._update_illustrations_rows()
        self._check_llm()

    def _build_action_bar(self) -> Gtk.Widget:
        bar = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL,
            spacing=8,
            margin_top=10,
            margin_bottom=12,
            margin_start=18,
            margin_end=18,
        )
        self.progress = Gtk.ProgressBar(show_text=False, visible=False)
        self.status = Gtk.Label(xalign=0, ellipsize=3, visible=False)  # 3 = END
        self.status.add_css_class("dim-label")
        buttons = Gtk.Box(spacing=12, halign=Gtk.Align.END)
        self.audio_button = Gtk.Button(label="Make audio book")
        self.audio_button.add_css_class("suggested-action")
        self.audio_button.add_css_class("pill")
        self.audio_button.connect("clicked", lambda b: self._start_job("audio"))
        self.video_button = Gtk.Button(label="Make video")
        self.video_button.add_css_class("pill")
        self.video_button.connect("clicked", lambda b: self._start_job("video"))
        self.cancel_button = Gtk.Button(label="Cancel", visible=False)
        self.cancel_button.add_css_class("destructive-action")
        self.cancel_button.add_css_class("pill")
        self.cancel_button.connect("clicked", self._on_cancel)
        buttons.append(self.cancel_button)
        buttons.append(self.video_button)
        buttons.append(self.audio_button)
        bar.append(self.progress)
        bar.append(self.status)
        bar.append(buttons)
        wrapper = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        wrapper.append(Gtk.Separator())
        wrapper.append(bar)
        return wrapper

    # -------------------------------------------------------------- file input

    def _install_drop_target(self) -> None:
        target = Gtk.DropTarget.new(Gdk.FileList, Gdk.DragAction.COPY)
        target.connect("drop", self._on_drop)
        target.connect(
            "enter", lambda *a: (self.drop_zone.add_css_class("dragging"), Gdk.DragAction.COPY)[1]
        )
        target.connect("leave", lambda *a: self.drop_zone.remove_css_class("dragging"))
        self.add_controller(target)

    def _on_drop(self, _target, value, _x, _y) -> bool:
        self.drop_zone.remove_css_class("dragging")
        files = value.get_files() if hasattr(value, "get_files") else []
        paths = [f.get_path() for f in files if f.get_path()]
        if not paths:
            self._show_error("Only local files can be dropped here.")
            return False
        self.open_path(paths[0])
        return True

    def _on_browse(self, _button) -> None:
        dialog = Gtk.FileDialog(title="Open document")
        filters = Gio.ListStore.new(Gtk.FileFilter)
        docs = Gtk.FileFilter(name="Documents")
        for ext in SUPPORTED_EXTENSIONS:
            docs.add_suffix(ext.lstrip("."))
        filters.append(docs)
        dialog.set_filters(filters)
        dialog.set_default_filter(docs)
        if self.document:
            dialog.set_initial_folder(Gio.File.new_for_path(str(self.document.path.parent)))

        def done(dlg, result):
            try:
                file = dlg.open_finish(result)
            except GLib.Error:
                return  # dismissed
            if file and file.get_path():
                self.open_path(file.get_path())

        dialog.open(self, None, done)

    def open_path(self, raw: str) -> None:
        raw = raw.strip().strip("'\"")
        if raw.startswith("file://"):
            raw = Gio.File.new_for_uri(raw).get_path() or raw
        if not raw:
            return
        path = Path(raw).expanduser()
        self.path_row.set_text(str(path))
        if self.job:
            self._show_error("Please wait until the current job finishes.")
            return
        self.document = None
        self._update_buttons()
        self.drop_label.set_label(f"Reading {path.name}…")
        self.info_label.set_visible(False)
        self.chapters_group.set_visible(False)

        def work() -> None:
            try:
                options = self.settings.parse_options(
                    lambda f, m: _idle(self.drop_label.set_label, f"{path.name}: {m}")
                )
                doc = load_document(path, options)
                lang = detect_language(" ".join(c.text[:3000] for c in doc.chapters[:5]))
                _idle(self._on_parsed, doc, lang, None)
            except Exception as exc:
                log.debug("parse failed", exc_info=True)
                _idle(self._on_parsed, None, "en", exc)

        threading.Thread(target=work, daemon=True).start()

    def _on_parsed(self, doc: Document | None, language: str, error: Exception | None) -> None:
        self.drop_label.set_label("Drop a document here")
        if error is not None or doc is None:
            self._show_error(str(error))
            return
        self.banner.set_revealed(False)
        self.document = doc
        self.selected = {c.number for c in doc.chapters}  # everything selected by default
        self.language = language
        self.drop_label.set_label(f"{doc.path.name} — drop another file to replace it")
        self._refresh_document_view()
        self._refresh_voice_model()
        self._update_buttons()

    def _refresh_document_view(self) -> None:
        doc = self.document
        if doc is None:
            return
        rate = self.settings.rate
        total = estimate_seconds(doc.word_count, rate)
        author = f" by {doc.author}" if doc.author else ""
        self.info_label.set_label(
            f"{doc.title}{author}\n{len(doc.chapters)} chapters · {doc.word_count:,} words · "
            f"about {format_duration(total)} of narration"
        )
        self.info_label.set_visible(True)
        while (row := self.chapters_list.get_first_child()) is not None:
            self.chapters_list.remove(row)
        self.chapter_checks = {}
        max_seconds = self.settings.max_chapter_seconds
        for chapter in doc.chapters:
            secs = estimate_seconds(chapter.word_count, rate)
            parts = max(1, math.ceil(secs / max_seconds))
            subtitle = f"{chapter.word_count:,} words · ~{format_duration(secs)}"
            if parts > 1:
                subtitle += f" · split into {parts} files"
            if chapter.images:
                subtitle += f" · {len(chapter.images)} images"
            row = Adw.ActionRow(title=GLib.markup_escape_text(chapter.title), subtitle=subtitle)
            check = Gtk.CheckButton(active=chapter.number in self.selected, valign=Gtk.Align.CENTER)
            check.connect("toggled", self._on_chapter_toggled, chapter.number)
            row.add_prefix(check)
            number = Gtk.Label(label=f"{chapter.number:0{doc.number_width}d}")
            number.add_css_class("dim-label")
            number.add_css_class("numeric")
            row.add_prefix(number)
            row.set_activatable_widget(check)  # clicking the row toggles the chapter
            self.chapter_checks[chapter.number] = check
            self.chapters_list.append(row)
        self.chapters_group.set_visible(True)
        self._update_selection_summary()

    # --------------------------------------------------------- chapter selection

    def _on_chapter_toggled(self, check: Gtk.CheckButton, number: int) -> None:
        if check.get_active():
            self.selected.add(number)
        else:
            self.selected.discard(number)
        if not self._bulk_selecting:
            self._update_selection_summary()

    def _select_all(self, value: bool) -> None:
        self._bulk_selecting = True
        try:
            for check in self.chapter_checks.values():
                check.set_active(value)
        finally:
            self._bulk_selecting = False
        self._update_selection_summary()

    def _selected_document(self) -> Document | None:
        """The loaded document restricted to the selected chapters."""
        if self.document is None or not self.selected:
            return None
        if len(self.selected) == len(self.document.chapters):
            return self.document
        return self.document.select(self.selected)

    def _update_selection_summary(self) -> None:
        doc = self.document
        if doc is None:
            return
        chosen = [c for c in doc.chapters if c.number in self.selected]
        words = sum(c.word_count for c in chosen)
        seconds = estimate_seconds(words, self.settings.rate)
        if not chosen:
            text = "No chapters selected — select at least one to make an audio book or video"
        elif len(chosen) == len(doc.chapters):
            text = f"All {len(chosen)} chapters selected · ~{format_duration(seconds)}"
        else:
            text = (
                f"{len(chosen)} of {len(doc.chapters)} chapters selected · "
                f"{words:,} words · ~{format_duration(seconds)}"
            )
        self.chapters_group.set_description(text)
        self._update_buttons()

    # ---------------------------------------------------------------- settings

    def _on_max_len_changed(self, row, _pspec) -> None:
        self.settings.max_chapter_minutes = float(row.get_value())
        self.settings.save()
        self._refresh_document_view()

    def _on_rate_changed(self, row, _pspec) -> None:
        self.settings.rate = round(float(row.get_value()), 2)
        self.settings.save()
        self._refresh_document_view()

    def _on_engine_changed(self, row, _pspec) -> None:
        self.settings.tts_engine = self.engine_keys[row.get_selected()]
        self.settings.voice = ""
        self.settings.save()
        self._load_voices()

    def _on_voice_changed(self, row, _pspec) -> None:
        if self._updating_voices:
            return
        index = row.get_selected()
        if 0 <= index < len(self.voice_ids):
            self.settings.voice = self.voice_ids[index]
            self.settings.save()

    def _on_script_changed(self, row, _pspec) -> None:
        self.settings.video_script = SCRIPT_MODES[row.get_selected()][0]
        self.settings.save()
        self._update_script_subtitle()
        self._update_illustrations_rows()

    def _on_illustrations_changed(self, row, _pspec) -> None:
        self.settings.illustrations = row.get_active()
        self.settings.save()
        self._update_illustrations_rows()

    def _on_image_model_changed(self, row, _pspec) -> None:
        index = row.get_selected()
        if 0 <= index < len(self.image_model_names):
            self.settings.image_model = self.image_model_names[index]
            self.settings.save()
            self._update_illustrations_rows()

    def _on_book_images_changed(self, row, _pspec) -> None:
        self.settings.book_images = row.get_active()
        self.settings.save()
        self._update_illustrations_rows()

    def _update_illustrations_rows(self) -> None:
        if self.settings.video_script == "extractive":
            text = "Needs the AI script (the original-text script has no illustrations)"
        else:
            text = "The AI adds diagrams or generated pictures to the scenes that need them"
        self.illustrations_row.set_subtitle(text)
        self.image_model_row.set_sensitive(self.settings.illustrations)
        status = ImageGenerator(self.settings.image_model, self.settings.llm_auto_download).status()
        self.image_model_row.set_subtitle(GLib.markup_escape_text(status.message))
        self.book_images_row.set_sensitive(self.settings.illustrations)
        vision = VisionLLM(self.settings.vision_model, self.settings.llm_auto_download).status()
        text = "Use the book's own image where it fits a scene better than a generated picture"
        if self.settings.book_images:
            text += f" · {vision.message}"
        self.book_images_row.set_subtitle(GLib.markup_escape_text(text))

    def _on_ocr_changed(self, row, _pspec) -> None:
        self.settings.ocr = OCR_MODES[row.get_selected()][0]
        self.settings.save()
        self._reparse()

    def _on_ocr_languages_changed(self, row) -> None:
        self.settings.ocr_languages = row.get_text().strip() or "auto"
        self.settings.save()
        self._reparse()

    def _reparse(self) -> None:
        """Re-read the current PDF so that changed OCR settings take effect."""
        if self.document is not None and self.document.path.suffix.lower() == ".pdf":
            self.open_path(str(self.document.path))

    def _on_model_changed(self, row, _pspec) -> None:
        if self._updating_models:
            return
        index = row.get_selected()
        if 0 <= index < len(self.model_names):
            self.settings.llm_model = self.model_names[index]
            self.settings.save()
            self._check_llm()

    def _check_llm(self) -> None:
        """Find the model locally (Ollama folder / app cache) or its download size."""
        model = self.settings.llm_model
        llm = LocalLLM(model, self.settings.llm_auto_download)
        threading.Thread(
            target=lambda: _idle(self._on_llm_status, model, llm.status()), daemon=True
        ).start()

    def _on_llm_status(self, model: str, status: LLMStatus) -> None:
        if model != self.settings.llm_model:
            return
        self.llm = status
        from ..llm import list_local

        local = {m.name: m for m in list_local()}
        names = list(local)
        names += [n for n in RECOMMENDED if n not in local]
        if model not in names:
            names.insert(0, model)
        labels = []
        for name in names:
            if name in local:
                where = "Ollama" if local[name].source == "ollama" else "downloaded"
                labels.append(f"{name} ({format_size(local[name].size)}, {where})")
            elif name in RECOMMENDED:
                labels.append(f"{name} (download · {RECOMMENDED[name].split(',')[0]})")
            else:
                labels.append(name)
        self.model_names = names
        self._updating_models = True
        try:
            self.model_row.set_model(Gtk.StringList.new(labels))
            self.model_row.set_selected(names.index(model))
        finally:
            self._updating_models = False
        self.model_row.set_subtitle(GLib.markup_escape_text(status.message))
        self._update_script_subtitle()

    def _update_script_subtitle(self) -> None:
        mode = self.settings.video_script
        ok = self.llm is not None and self.llm.ok
        if mode == "extractive":
            text = "Slides show key sentences, narration reads the original text"
        elif ok:
            text = f"{self.settings.llm_model} explains each chapter (runs locally)"
        elif mode == "ai":
            text = "The local model is not available — see below"
        else:
            text = "Local model not available: the original text will be used"
        self.script_row.set_subtitle(text)

    def _load_voices(self) -> None:
        engine_name = self.settings.tts_engine

        def work() -> None:
            try:
                voices = get_engine(engine_name).voices()
            except Exception as exc:
                log.warning("Could not list voices: %s", exc)
                voices = []
            _idle(self._on_voices_loaded, engine_name, voices)

        threading.Thread(target=work, daemon=True).start()

    def _on_voices_loaded(self, engine_name: str, voices: list[Voice]) -> None:
        if engine_name != self.settings.tts_engine:
            return
        self.voices = voices
        self._refresh_voice_model()

    def _refresh_voice_model(self) -> None:
        matching = [v for v in self.voices if v.language == self.language] or self.voices
        current = self.settings.voice
        if current and current not in {v.id for v in matching}:
            extra = [v for v in self.voices if v.id == current]
            matching = extra + matching
        engine = get_engine(self.settings.tts_engine)
        supports = getattr(engine, "supports", None)
        if self.language and supports is not None and not supports(self.language):
            note = f"{engine.label.split(' (')[0]} has no voices for this language; "
            self.voice_row.set_subtitle(note + "Automatic uses a Piper voice")
        else:
            self.voice_row.set_subtitle("")
        labels = ["Automatic (by language)"] + [v.label for v in matching]
        self.voice_ids = [""] + [v.id for v in matching]
        self._updating_voices = True
        try:
            self.voice_row.set_model(Gtk.StringList.new(labels))
            self.voice_row.set_selected(
                self.voice_ids.index(current) if current in self.voice_ids else 0
            )
        finally:
            self._updating_voices = False

    # -------------------------------------------------------------------- jobs

    def _update_buttons(self) -> None:
        busy = self.job is not None
        ready = self.document is not None and bool(self.selected) and not busy
        self.audio_button.set_sensitive(ready)
        self.video_button.set_sensitive(ready)
        self.chapters_list.set_sensitive(not busy)
        self.selection_buttons.set_sensitive(not busy)
        self.cancel_button.set_visible(busy)
        for row in (
            self.max_len_row,
            self.engine_row,
            self.voice_row,
            self.rate_row,
            self.script_row,
            self.model_row,
            self.ocr_row,
            self.ocr_lang_row,
            self.path_row,
        ):
            row.set_sensitive(not busy)

    def _start_job(self, kind: str) -> None:
        doc = self._selected_document()
        if doc is None or self.job is not None:
            return
        if error := check_ffmpeg():
            self._show_error(error)
            return
        settings = Settings(**vars(self.settings))
        ctx = JobContext(on_progress=lambda f, m: _idle(self._on_progress, f, m))
        self.job = ctx
        self.progress.set_fraction(0)
        self.progress.set_visible(True)
        self.status.set_visible(True)
        self.status.set_label("Starting…")
        self.banner.set_revealed(False)
        self._update_buttons()

        def work() -> None:
            try:
                if kind == "audio":
                    from ..audiobook import make_audiobook

                    result = make_audiobook(doc, settings, ctx)
                else:
                    from ..video.builder import make_video

                    result = make_video(doc, settings, ctx)
                _idle(self._on_job_done, kind, result, None)
            except Cancelled:
                _idle(self._on_job_done, kind, None, None)
            except Exception as exc:
                traceback.print_exc()
                _idle(self._on_job_done, kind, None, exc)

        self.job_thread = threading.Thread(target=work, daemon=True)
        self.job_thread.start()

    def _on_progress(self, fraction: float, message: str) -> None:
        if self.job is None:
            return
        self.progress.set_fraction(fraction)
        self.status.set_label(message)

    def cancel_job(self) -> None:
        """Cancel the running job, also stopping a model in the middle of an answer."""
        if self.job:
            self.job.cancel()
            abort_llm()
            abort_vision()

    def stop_job(self, timeout: float) -> bool:
        """Cancel the job and wait for its thread; False if it's still running."""
        self.cancel_job()
        thread = self.job_thread
        if thread is not None:
            thread.join(timeout)
            return not thread.is_alive()
        return True

    def _on_cancel(self, _button) -> None:
        if self.job:
            self.cancel_job()
            self.status.set_label("Cancelling…")
            self.cancel_button.set_sensitive(False)

    def _on_job_done(self, kind: str, result, error: Exception | None) -> None:
        self.job = None
        self.cancel_button.set_sensitive(True)
        self.progress.set_visible(False)
        self.status.set_visible(False)
        self._update_buttons()
        what = "Audio book" if kind == "audio" else "Video"
        if error is not None:
            self._show_error(f"{what} failed: {error}")
            return
        if result is None:
            self.toasts.add_toast(Adw.Toast(title=f"{what} cancelled", timeout=3))
            return
        toast = Adw.Toast(
            title=f"{what} ready: {len(result.files)} files in {result.out_dir.name}",
            button_label="Open folder",
            timeout=0,
        )
        toast.connect("button-clicked", lambda t: self._open_folder(result.out_dir))
        self.toasts.add_toast(toast)

    def _open_folder(self, folder: Path) -> None:
        if sys.platform == "darwin":
            subprocess.Popen(["open", str(folder)])
            return
        Gtk.FileLauncher.new(Gio.File.new_for_path(str(folder))).launch(self, None, None)

    def _show_error(self, message: str) -> None:
        self.banner.set_title(GLib.markup_escape_text(message.splitlines()[0][:300]))
        self.banner.set_revealed(True)
        log.error("%s", message)
