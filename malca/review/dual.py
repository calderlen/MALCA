from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from malca.review import tui


def _start_streamlit(db_path: Path | None = None) -> subprocess.Popen:
    app_path = Path(__file__).resolve().parent / "app.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path)]
    if db_path is not None:
        # Pass DB path via Streamlit args to keep both frontends aligned.
        cmd.extend(["--", "--db", str(db_path)])
    return subprocess.Popen(cmd)


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch review GUI and TUI together")
    parser.add_argument("--db", type=Path, default=None, help="SQLite DB path used by both frontends")
    parser.add_argument("--input", type=Path, default=None, help="Optional candidates file import for TUI")
    parser.add_argument("--reviewer", type=str, default="", help="Reviewer name for TUI")
    args = parser.parse_args()

    gui_proc = _start_streamlit(db_path=args.db)
    try:
        tui.main_with_args(db=args.db, input_path=args.input, reviewer=args.reviewer)
    finally:
        if gui_proc.poll() is None:
            gui_proc.terminate()
            try:
                gui_proc.wait(timeout=5)
            except Exception:
                gui_proc.kill()


if __name__ == "__main__":
    main()
