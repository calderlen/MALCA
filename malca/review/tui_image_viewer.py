"""Persistent, geometry-preserving image window for ``malca review-tui``.

The curses process writes a tiny JSON manifest whenever the current rendered
PNG changes.  This helper owns one Tk window, polls that manifest, and replaces
only the canvas pixels.  It deliberately never reapplies window geometry after
startup, so user resizing and repositioning survive candidate and period
changes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import signal
from typing import Any


POLL_INTERVAL_MS = 100
RESIZE_DEBOUNCE_MS = 75
GEOMETRY_SAVE_DEBOUNCE_MS = 150
GEOMETRY_STATE_FILENAME = "viewer_geometry.json"
_GEOMETRY_PATTERN = re.compile(
    r"^(?P<width>\d+)x(?P<height>\d+)(?P<x>[+-]\d+)(?P<y>[+-]\d+)$"
)


def _normalized_window_geometry(value: object) -> str | None:
    """Return a safe Tk geometry string, preserving multi-monitor coordinates."""

    match = _GEOMETRY_PATTERN.fullmatch(str(value or "").strip())
    if match is None:
        return None
    width = max(420, int(match.group("width")))
    height = max(315, int(match.group("height")))
    x = int(match.group("x"))
    y = int(match.group("y"))
    return f"{width}x{height}{x:+d}{y:+d}"


def _load_window_geometry(path: Path) -> str | None:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, ValueError, TypeError):
        return None
    if not isinstance(payload, dict) or payload.get("version") != 1:
        return None
    return _normalized_window_geometry(payload.get("geometry"))


def _save_window_geometry(path: Path, geometry: object) -> bool:
    normalized = _normalized_window_geometry(geometry)
    if normalized is None:
        return False
    destination = Path(path)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(
            json.dumps({"version": 1, "geometry": normalized}, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(destination)
    except OSError:
        try:
            temporary.unlink()
        except OSError:
            pass
        return False
    return True


class PersistentImageWindow:
    def __init__(self, manifest_path: Path) -> None:
        import tkinter as tk

        self._tk = tk
        self.manifest_path = Path(manifest_path).expanduser()
        self.geometry_state_path = self.manifest_path.with_name(
            GEOMETRY_STATE_FILENAME
        )
        self.root = tk.Tk()
        self.root.title("MALCA Review")
        self.root.minsize(420, 315)
        restored_geometry = _load_window_geometry(self.geometry_state_path)
        if restored_geometry is not None:
            self.root.geometry(restored_geometry)
        self.canvas = tk.Canvas(
            self.root,
            width=960,
            height=720,
            background="#111111",
            highlightthickness=0,
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self._source_image: Any = None
        self._display_image: Any = None
        self._manifest_token: str | None = None
        self._resize_job: str | None = None
        self._geometry_save_job: str | None = None
        self.canvas.bind("<Configure>", self._schedule_redraw)
        self.root.bind("<Configure>", self._schedule_geometry_save, add="+")
        self.root.protocol("WM_DELETE_WINDOW", self._shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown_signal)
        signal.signal(signal.SIGINT, self._handle_shutdown_signal)
        self.root.after(0, self._poll_manifest)

    def run(self) -> None:
        try:
            self.root.mainloop()
        finally:
            self._save_geometry()

    def _handle_shutdown_signal(self, _signum: int, _frame: object) -> None:
        self._shutdown()

    def _shutdown(self) -> None:
        if self._geometry_save_job is not None:
            try:
                self.root.after_cancel(self._geometry_save_job)
            except self._tk.TclError:
                pass
            self._geometry_save_job = None
        self._save_geometry()
        try:
            self.root.destroy()
        except self._tk.TclError:
            pass

    def _schedule_geometry_save(self, event: object = None) -> None:
        if event is not None and getattr(event, "widget", self.root) is not self.root:
            return
        if self._geometry_save_job is not None:
            self.root.after_cancel(self._geometry_save_job)
        self._geometry_save_job = self.root.after(
            GEOMETRY_SAVE_DEBOUNCE_MS,
            self._save_geometry,
        )

    def _save_geometry(self) -> None:
        self._geometry_save_job = None
        try:
            geometry = self.root.winfo_geometry()
        except self._tk.TclError:
            return
        _save_window_geometry(self.geometry_state_path, geometry)

    def _poll_manifest(self) -> None:
        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            token = str(manifest.get("token") or manifest.get("path") or "")
            if token and token != self._manifest_token:
                self._load_image(Path(str(manifest["path"])).expanduser())
                self._manifest_token = token
                title = str(manifest.get("title") or "MALCA Review")
                self.root.title(title)
        except (FileNotFoundError, KeyError, OSError, ValueError, TypeError):
            # A manifest replacement or PNG render may briefly be in flight.
            # Keep the prior image visible and retry on the next poll.
            pass
        self.root.after(POLL_INTERVAL_MS, self._poll_manifest)

    def _load_image(self, image_path: Path) -> None:
        from PIL import Image

        with Image.open(image_path) as opened:
            self._source_image = opened.convert("RGB").copy()
        self._redraw()

    def _schedule_redraw(self, _event: object = None) -> None:
        if self._resize_job is not None:
            self.root.after_cancel(self._resize_job)
        self._resize_job = self.root.after(RESIZE_DEBOUNCE_MS, self._redraw)

    def _redraw(self) -> None:
        self._resize_job = None
        if self._source_image is None:
            return
        from PIL import Image, ImageTk

        width = max(1, int(self.canvas.winfo_width()))
        height = max(1, int(self.canvas.winfo_height()))
        source_width, source_height = self._source_image.size
        scale = min(width / source_width, height / source_height)
        display_size = (
            max(1, int(round(source_width * scale))),
            max(1, int(round(source_height * scale))),
        )
        resized = self._source_image.resize(
            display_size,
            Image.Resampling.LANCZOS,
        )
        self._display_image = ImageTk.PhotoImage(resized)
        self.canvas.delete("all")
        self.canvas.create_image(
            width // 2,
            height // 2,
            image=self._display_image,
            anchor=self._tk.CENTER,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--manifest", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    PersistentImageWindow(args.manifest).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
