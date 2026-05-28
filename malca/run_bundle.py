from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import zipfile

import pandas as pd
from tqdm.auto import tqdm


@dataclass(frozen=True)
class BundleFileCollection:
    files: list[tuple[Path, str]]
    rows: int
    candidate_paths: int
    added: int
    missing: int
    skipped_suffix: int
    duplicate_arcname: int


def _resolve_for_dedupe(path: Path) -> Path:
    try:
        return path.expanduser().resolve()
    except OSError:
        return path.expanduser()


def _normalize_arc_prefix(prefix: str) -> str:
    return str(prefix).strip("/")


def _suffix_allowed(path: Path, allowed_suffix_prefixes: tuple[str, ...]) -> bool:
    if not allowed_suffix_prefixes:
        return True
    suffix = path.suffix.lower().lstrip(".")
    return any(suffix.startswith(str(prefix).lower().lstrip(".")) for prefix in allowed_suffix_prefixes)


def _unique_arcname(
    name: str,
    *,
    arc_prefix: str,
    used_arcnames: set[str],
    duplicate_count: int,
) -> tuple[str, int, bool]:
    prefix = _normalize_arc_prefix(arc_prefix)
    arcname = f"{prefix}/{name}" if prefix else name
    if arcname not in used_arcnames:
        return arcname, duplicate_count, False

    duplicate_count += 1
    while True:
        candidate_name = f"{duplicate_count:06d}_{name}"
        candidate = f"{prefix}/{candidate_name}" if prefix else candidate_name
        if candidate not in used_arcnames:
            return candidate, duplicate_count, True
        duplicate_count += 1


def collect_candidate_lightcurve_files(
    df: pd.DataFrame,
    *,
    path_cols: tuple[str, ...],
    arc_prefix: str,
    allowed_suffix_prefixes: tuple[str, ...] = (),
    sidecar_suffixes: tuple[str, ...] = (),
    include_missing: bool = False,
) -> BundleFileCollection:
    """Collect source light-curve files with stable archive names."""
    rows = len(df)
    path_col = next((col for col in path_cols if col in df.columns), None)
    if path_col is None:
        return BundleFileCollection([], rows, 0, 0, 0, 0, 0)

    raw_paths = [
        str(value).strip()
        for value in pd.Series(df[path_col]).dropna().tolist()
        if str(value).strip()
    ]
    candidate_paths = len(raw_paths)

    files: list[tuple[Path, str]] = []
    seen_files: set[Path] = set()
    used_arcnames: set[str] = set()
    missing = 0
    skipped_suffix = 0
    duplicate_arcname = 0
    duplicate_counter = 0

    for raw_path in raw_paths:
        source_path = Path(raw_path).expanduser()
        if not _suffix_allowed(source_path, allowed_suffix_prefixes):
            skipped_suffix += 1
            continue

        related_paths = [source_path]
        for suffix in sidecar_suffixes:
            suffix_text = str(suffix)
            if suffix_text and not suffix_text.startswith("."):
                suffix_text = f".{suffix_text}"
            if suffix_text:
                related_paths.append(source_path.with_suffix(suffix_text))

        for path in related_paths:
            resolved = _resolve_for_dedupe(path)
            if resolved in seen_files:
                continue
            seen_files.add(resolved)

            if not path.is_file():
                missing += 1
                if not include_missing:
                    continue

            arcname, duplicate_counter, had_duplicate = _unique_arcname(
                path.name,
                arc_prefix=arc_prefix,
                used_arcnames=used_arcnames,
                duplicate_count=duplicate_counter,
            )
            if had_duplicate:
                duplicate_arcname += 1
            used_arcnames.add(arcname)
            files.append((path, arcname))

    added = sum(1 for path, _arcname in files if path.is_file())
    return BundleFileCollection(
        files=files,
        rows=rows,
        candidate_paths=candidate_paths,
        added=added,
        missing=missing,
        skipped_suffix=skipped_suffix,
        duplicate_arcname=duplicate_arcname,
    )


def _safe_extract_member(zf: zipfile.ZipFile, member: zipfile.ZipInfo, destination: Path) -> None:
    target = (destination / member.filename).resolve()
    root = destination.resolve()
    if root != target and root not in target.parents:
        raise ValueError(f"Unsafe bundle path: {member.filename}")
    zf.extract(member, destination)


def import_bundle_zip(
    bundle_zip: str | Path,
    run_dir: str | Path,
    *,
    overwrite: bool = False,
    show_progress: bool = False,
) -> None:
    """Validate and extract a run transfer bundle into ``run_dir``."""
    bundle_path = Path(bundle_zip).expanduser()
    destination = Path(run_dir).expanduser()
    if not bundle_path.exists():
        raise FileNotFoundError(f"Bundle not found: {bundle_path}")
    if not zipfile.is_zipfile(bundle_path):
        raise ValueError(f"Bundle is not a valid zip file: {bundle_path}")
    if destination.exists() and overwrite:
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(bundle_path, "r") as zf:
        members = zf.infolist()
        files = [member for member in members if not member.is_dir()]
        total_bytes = sum(member.file_size for member in files)
        if show_progress:
            with tqdm(total=total_bytes, desc="Import bundle", unit="B", unit_scale=True) as pbar:
                for member in members:
                    _safe_extract_member(zf, member, destination)
                    if not member.is_dir():
                        pbar.update(member.file_size)
        else:
            for member in members:
                _safe_extract_member(zf, member, destination)


def _collect_run_files(
    run_dir: Path,
    *,
    include_files: tuple[str, ...] | list[str],
    include_globs: tuple[str, ...] | list[str],
    include_dirs: tuple[str, ...] | list[str],
) -> set[Path]:
    files: set[Path] = set()
    for rel in include_files:
        path = run_dir / rel
        if path.is_file():
            files.add(path)
        elif path.is_dir():
            files.update(child for child in path.rglob("*") if child.is_file())

    for pattern in include_globs:
        for path in run_dir.glob(pattern):
            if path.is_file():
                files.add(path)
            elif path.is_dir():
                files.update(child for child in path.rglob("*") if child.is_file())

    for rel_dir in include_dirs:
        path = run_dir / rel_dir
        if path.is_dir():
            files.update(child for child in path.rglob("*") if child.is_file())
    return files


def export_run_bundle(
    bundle_zip: str | Path,
    run_dir: str | Path,
    *,
    include_files: tuple[str, ...] | list[str] = (),
    include_globs: tuple[str, ...] | list[str] = (),
    include_dirs: tuple[str, ...] | list[str] = (),
    external_files: tuple[tuple[Path, str], ...] | list[tuple[Path, str]] = (),
    description: str = "run",
) -> list[str]:
    """Create a ZIP bundle from a run directory and optional external files."""
    run_path = Path(run_dir).expanduser()
    bundle_path = Path(bundle_zip).expanduser()
    bundle_path.parent.mkdir(parents=True, exist_ok=True)

    files = _collect_run_files(
        run_path,
        include_files=include_files,
        include_globs=include_globs,
        include_dirs=include_dirs,
    )
    files.discard(bundle_path)
    external = [(Path(path), arcname) for path, arcname in external_files if Path(path).is_file()]
    if not files and not external:
        raise FileNotFoundError(f"No {description} bundle files found under {run_path}")

    names: list[str] = []
    used_names: set[str] = set()
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        for path in sorted(files, key=lambda p: str(p.relative_to(run_path))):
            arcname = str(path.relative_to(run_path))
            zf.write(path, arcname=arcname)
            names.append(arcname)
            used_names.add(arcname)
        for source_path, arcname in sorted(external, key=lambda item: item[1]):
            if arcname in used_names:
                continue
            zf.write(source_path, arcname=arcname)
            names.append(arcname)
            used_names.add(arcname)
    return names
