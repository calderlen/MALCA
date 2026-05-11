"""Entry point for the PyQt review GUI (test module)."""
from __future__ import annotations

import argparse
import signal
import sys
from pathlib import Path

# Optional Qt: try PySide6 then PyQt6
try:
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import Qt, QTimer
    QT_BINDING = "PySide6"
except ImportError:
    try:
        from PyQt6.QtWidgets import QApplication
        from PyQt6.QtCore import Qt, QTimer
        QT_BINDING = "PyQt6"
    except ImportError:
        QT_BINDING = None
        QTimer = None  # type: ignore[misc, assignment]

from malca.review_qt.window import ReviewMainWindow


def main() -> int:
    parser = argparse.ArgumentParser(
        description="MALCA Review (Qt) — test desktop GUI; use 'malca review' for the Dash app.",
    )
    parser.add_argument(
        "--review-db",
        type=Path,
        default=None,
        help="SQLite review DB path (default: from env or output/review/review.db)",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=None,
        help="Plot directory for light-curve resolution (optional)",
    )
    args = parser.parse_args()

    if QT_BINDING is None:
        print("review-qt requires PySide6 or PyQt6. Install with: pip install PySide6", file=sys.stderr)
        return 1

    app = QApplication(sys.argv)
    app.setApplicationName("MALCA Review (Qt)")

    # Quit cleanly on Ctrl+C instead of dumping KeyboardInterrupt tracebacks
    def _quit_on_sigint() -> None:
        QTimer.singleShot(0, app.quit)

    signal.signal(signal.SIGINT, lambda *a: _quit_on_sigint())

    window = ReviewMainWindow(db_path=args.review_db, plot_dir=args.plot_dir)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
