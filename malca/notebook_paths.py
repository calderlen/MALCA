from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


LIGHTCURVE_DIR_CANDIDATES = (
    Path("bundle_assets") / "lightcurves",
    Path("lightcurves"),
)

LIGHTCURVE_FILE_PATTERNS = ("*.dat3", "*.dat2", "*.dat", "*.csv")
LIGHTCURVE_SUFFIX_PRIORITY = (".dat3", ".dat2", ".dat", ".csv", ".raw2")


def find_repo_root(start: Path | str | None = None) -> Path:
    base = Path.cwd() if start is None else Path(start).expanduser()
    try:
        base = base.resolve()
    except Exception:
        base = base.expanduser()

    for candidate in (base, *base.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise FileNotFoundError("Could not find repo root (missing pyproject.toml).")


def resolve_repo_path(path_like: Path | str | None, repo_root: Path | str | None = None) -> Path | None:
    if path_like is None:
        return None

    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path

    root = find_repo_root() if repo_root is None else Path(repo_root).expanduser()
    try:
        root = root.resolve()
    except Exception:
        root = root.expanduser()
    return root / path


def infer_run_dir(path_like: Path | str | None) -> Path | None:
    if path_like is None:
        return None

    try:
        path = Path(path_like).expanduser().resolve()
    except Exception:
        path = Path(path_like).expanduser()

    candidates = [path]
    if path.is_file():
        candidates.extend([path.parent, path.parent.parent, path.parent.parent.parent])
    else:
        candidates.extend([path.parent, path.parent.parent])

    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)

        if candidate.name in {"results", "plots", "bundle_assets"}:
            run_dir = candidate.parent
            if any((run_dir / rel).is_dir() for rel in LIGHTCURVE_DIR_CANDIDATES) or (run_dir / "results").is_dir():
                return run_dir

        if any((candidate / rel).is_dir() for rel in LIGHTCURVE_DIR_CANDIDATES) or (candidate / "results").is_dir():
            return candidate
    return None


def iter_lightcurve_dirs(run_dir: Path | str | None) -> list[Path]:
    if run_dir is None:
        return []

    root = Path(run_dir).expanduser()
    dirs: list[Path] = []
    for rel in LIGHTCURVE_DIR_CANDIDATES:
        candidate = root / rel
        if candidate.is_dir():
            dirs.append(candidate)
    return dirs


def candidate_lightcurve_names(path_like: Path | str | None) -> list[str]:
    text = str(path_like or "").strip()
    if not text:
        return []

    base_name = Path(text).name if any(sep in text for sep in ("/", "\\")) else text
    names: list[str] = []
    seen: set[str] = set()

    def add(name: str) -> None:
        clean = str(name).strip()
        if clean and clean not in seen:
            seen.add(clean)
            names.append(clean)

    add(base_name)
    stem = Path(base_name).stem if Path(base_name).suffix else base_name
    if stem and stem != base_name:
        add(stem)
    for ext in LIGHTCURVE_SUFFIX_PRIORITY:
        add(f"{stem}{ext}")
    return names


def resolve_local_lightcurve_path(
    path_like: Path | str | None,
    *,
    run_dir: Path | str | None = None,
    repo_root: Path | str | None = None,
) -> Path | None:
    if path_like is None:
        return None

    repo_path = resolve_repo_path(path_like, repo_root=repo_root)
    if repo_path is not None and repo_path.exists():
        return repo_path

    resolved_run_dir = infer_run_dir(run_dir) if run_dir is not None else None
    if resolved_run_dir is None and repo_path is not None:
        resolved_run_dir = infer_run_dir(repo_path)

    for lightcurve_dir in iter_lightcurve_dirs(resolved_run_dir):
        for name in candidate_lightcurve_names(path_like):
            candidate = lightcurve_dir / name
            if candidate.exists():
                return candidate
    return None


def localize_lightcurve_frame_paths(
    df: pd.DataFrame,
    *,
    run_dir: Path | str | None = None,
    repo_root: Path | str | None = None,
    path_columns: Sequence[str] = ("dat_path", "path", "lc_path"),
) -> tuple[pd.DataFrame, dict[str, int]]:
    out = df.copy()
    localized_counts: dict[str, int] = {}

    for col in path_columns:
        if col not in out.columns:
            continue

        localized: list[object] = []
        changed = 0
        for value in out[col]:
            if value is None:
                localized.append(value)
                continue
            try:
                if pd.isna(value):
                    localized.append(value)
                    continue
            except Exception:
                pass

            resolved = resolve_local_lightcurve_path(value, run_dir=run_dir, repo_root=repo_root)
            if resolved is None:
                localized.append(str(value))
                continue

            resolved_text = str(resolved)
            if str(value) != resolved_text:
                changed += 1
            localized.append(resolved_text)

        out[col] = localized
        localized_counts[col] = changed

    return out, localized_counts


def discover_bundled_lightcurve_paths(
    search_root: Path | str,
    *,
    file_patterns: Iterable[str] = LIGHTCURVE_FILE_PATTERNS,
    limit: int | None = None,
) -> list[Path]:
    root = Path(search_root).expanduser()
    dirs = iter_lightcurve_dirs(root)
    if not dirs:
        dirs = sorted(root.glob("*/bundle_assets/lightcurves"))
        dirs.extend(sorted(root.glob("*/lightcurves")))

    paths: list[Path] = []
    for lightcurve_dir in dirs:
        for pattern in file_patterns:
            paths.extend(sorted(lightcurve_dir.glob(pattern)))

    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
        if limit is not None and len(unique) >= limit:
            return unique
    return unique
