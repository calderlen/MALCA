from __future__ import annotations

from pathlib import Path
import os
import socket
import subprocess
import sys


def _find_open_port(preferred_port: int, *, host: str = "127.0.0.1") -> int:
    """Return a free TCP port, preferring the requested one when available."""
    preferred = int(preferred_port)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, preferred))
            return preferred
        except OSError:
            pass

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def launch_detached(command: list[str]) -> None:
    """Spawn a detached child process without inheriting stdio."""
    with open(os.devnull, "rb") as devnull_in, open(os.devnull, "ab") as devnull_out:
        subprocess.Popen(
            [str(part) for part in command],
            stdin=devnull_in,
            stdout=devnull_out,
            stderr=devnull_out,
            start_new_session=True,
        )


def build_review_command(
    *,
    db_path: str | Path,
    candidate: str | None = None,
    plot_dir: str | Path | None = None,
    host: str = "127.0.0.1",
    preferred_port: int = 8050,
    open_browser: bool = False,
) -> tuple[list[str], str]:
    """Build a detached `malca review` launch command and target URL."""
    port = _find_open_port(preferred_port, host=host)
    command = [
        sys.executable,
        "-m",
        "malca",
        "review",
        "--db",
        str(Path(db_path).expanduser().resolve()),
        "--host",
        str(host),
        "--port",
        str(port),
    ]
    if not open_browser:
        command.append("--no-browser")
    if plot_dir:
        command.extend(["--plot-dir", str(Path(plot_dir).expanduser().resolve())])
    if candidate:
        command.extend(["--candidate", str(candidate)])
    return command, f"http://{host}:{port}"


def build_explorer_command(
    *,
    sources: list[str | Path],
    candidate: str | None = None,
    host: str = "127.0.0.1",
    preferred_port: int = 8062,
    plot_dir: str | Path | None = None,
) -> tuple[list[str], str]:
    """Build a detached `malca review-explore` launch command and target URL."""
    port = _find_open_port(preferred_port, host=host)
    command = [
        sys.executable,
        "-m",
        "malca",
        "review-explore",
        "--host",
        str(host),
        "--port",
        str(port),
    ]
    for source in sources:
        command.extend(["--source", str(Path(source).expanduser().resolve())])
    if plot_dir:
        command.extend(["--plot-dir", str(Path(plot_dir).expanduser().resolve())])
    if candidate:
        command.extend(["--candidate", str(candidate)])
    return command, f"http://{host}:{port}"
