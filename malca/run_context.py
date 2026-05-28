from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping
import json
import shlex
import shutil
import sys
import time


@dataclass
class PipelineRunContext:
    timescale: str
    run_dir: Path
    results_dir: Path
    review_dir: Path
    run_params_file: Path
    run_summary_file: Path
    run_log_file: Path
    started_at: datetime
    started_perf: float
    command: str


def timestamped_run_dir(root: Path) -> Path:
    """Return a timestamped run directory under ``root``."""
    return Path(root).expanduser() / datetime.now().strftime("%Y%m%d_%H%M%S")


def run_dir_from_bundle(
    bundle_zip: Path,
    root: Path,
    *,
    collision_suffix: str | None = None,
    overwrite: bool = False,
) -> Path:
    """Derive a run directory from a bundle filename."""
    base_name = Path(bundle_zip).expanduser().stem.removesuffix("_bundle")
    run_root = Path(root).expanduser()
    run_root.mkdir(parents=True, exist_ok=True)
    candidate = run_root / base_name
    if overwrite or not candidate.exists() or collision_suffix is None:
        return candidate
    return run_root / f"{base_name}{collision_suffix}"


def init_pipeline_run_context(timescale: str, run_dir: Path) -> PipelineRunContext:
    """Create common run directories and return run metadata paths."""
    run_path = Path(run_dir).expanduser()
    results_dir = run_path / "results"
    review_dir = run_path / "review"
    results_dir.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)
    return PipelineRunContext(
        timescale=str(timescale),
        run_dir=run_path,
        results_dir=results_dir,
        review_dir=review_dir,
        run_params_file=run_path / "run_params.json",
        run_summary_file=run_path / "run_summary.json",
        run_log_file=run_path / "run.log",
        started_at=datetime.now(),
        started_perf=time.perf_counter(),
        command=shlex.join(getattr(sys, "orig_argv", None) or ([sys.executable] + sys.argv)),
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, default=str), encoding="ascii")


def write_run_params(
    ctx: PipelineRunContext,
    payload: Mapping[str, Any],
    *,
    extra_paths: Iterable[Path] = (),
) -> None:
    """Write run parameters to the canonical file and optional extra files."""
    _write_json(ctx.run_params_file, payload)
    for path in extra_paths:
        _write_json(Path(path), payload)


def write_run_log(ctx: PipelineRunContext, lines: Iterable[str]) -> None:
    """Write the canonical run log."""
    ctx.run_log_file.parent.mkdir(parents=True, exist_ok=True)
    ctx.run_log_file.write_text("\n".join(str(line) for line in lines) + "\n", encoding="ascii")


def write_run_summary(ctx: PipelineRunContext, payload: Mapping[str, Any]) -> None:
    """Write the canonical run summary."""
    _write_json(ctx.run_summary_file, payload)


def update_latest_symlink(run_dir: Path, latest_path: Path, *, label: str) -> None:
    """Best-effort update of a latest symlink."""
    run_path = Path(run_dir).expanduser()
    latest = Path(latest_path).expanduser()
    try:
        if run_path.resolve() == latest.parent.resolve():
            return
        latest.parent.mkdir(parents=True, exist_ok=True)
        if latest.is_symlink() or latest.exists():
            if latest.is_dir() and not latest.is_symlink():
                shutil.rmtree(latest)
            else:
                latest.unlink()
        latest.symlink_to(run_path.resolve(), target_is_directory=True)
    except Exception as exc:
        print(f"Warning: could not update {label} latest symlink: {exc}")


def maybe_sync_review_bundle(
    enabled: bool,
    review_db: Path,
    sync_dir: Path,
    *,
    hash_assets: bool,
    verbose: bool,
) -> None:
    """Optionally export a review sync bundle."""
    if not enabled:
        return
    from malca.review.sync import auto_export_review_bundle

    auto_export_review_bundle(
        Path(review_db).expanduser(),
        Path(sync_dir).expanduser(),
        hash_assets=bool(hash_assets),
        logger=print if verbose else (lambda _msg: None),
    )
