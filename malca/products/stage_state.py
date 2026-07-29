"""Versioned, fail-closed stage/checkpoint metadata.

The scientific meaning of a cached table depends on more than its filename.
This module gives pipeline stages one small contract for input/config/code
fingerprints, completion accounting, and atomic state writes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Sequence

from malca.products.run_metadata import code_fingerprint, json_stable, sha256_file


STAGE_STATE_VERSION = "1"
TERMINAL_STATUSES = {"success", "partial", "error", "skipped", "no_data"}


@dataclass(frozen=True)
class StageResult:
    stage: str
    status: str
    expected: int
    succeeded: int = 0
    failed: int = 0
    skipped: int = 0
    errors: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in TERMINAL_STATUSES | {"running"}:
            raise ValueError(f"Unknown stage status: {self.status}")
        counts = (self.expected, self.succeeded, self.failed, self.skipped)
        if any(int(value) < 0 for value in counts):
            raise ValueError("Stage accounting values must be non-negative")
        handled = int(self.succeeded) + int(self.failed) + int(self.skipped)
        if handled > int(self.expected):
            raise ValueError(
                f"Stage accounting exceeds expected rows: {handled} > {self.expected}"
            )
        if self.status in TERMINAL_STATUSES and handled != int(self.expected):
            raise ValueError(
                "Terminal stage accounting must balance exactly: "
                f"expected={self.expected}, succeeded={self.succeeded}, "
                f"failed={self.failed}, skipped={self.skipped}"
            )
        if self.status == "success" and self.failed:
            raise ValueError("A successful stage cannot contain failed rows")

    @property
    def complete(self) -> bool:
        return self.status in TERMINAL_STATUSES


def file_signature(path: str | Path, *, content_hash: bool = False) -> dict[str, Any]:
    target = Path(path).expanduser()
    signature: dict[str, Any] = {"path": str(target)}
    try:
        stat = target.stat()
    except OSError:
        signature.update({"exists": False, "size": None, "mtime_ns": None})
        return signature
    signature.update({
        "exists": True,
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    })
    if content_hash and target.is_file():
        signature["sha256"] = sha256_file(target)
    elif content_hash and target.is_dir():
        digest = hashlib.sha256()
        file_count = 0
        total_size = 0
        for child in sorted(path for path in target.rglob("*") if path.is_file()):
            relative = child.relative_to(target).as_posix()
            child_stat = child.stat()
            child_hash = sha256_file(child)
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(int(child_stat.st_size)).encode("ascii"))
            digest.update(b"\0")
            digest.update(child_hash.encode("ascii"))
            digest.update(b"\n")
            file_count += 1
            total_size += int(child_stat.st_size)
        signature.update({
            "tree_sha256": digest.hexdigest(),
            "file_count": file_count,
            "total_file_size": total_size,
        })
    return signature


def candidate_set_digest(candidate_ids: Iterable[object]) -> str:
    values = [str(value).strip() for value in candidate_ids]
    if any(not value for value in values):
        raise ValueError("Candidate-set fingerprint cannot contain blank IDs")
    if len(values) != len(set(values)):
        raise ValueError("Candidate-set fingerprint cannot contain duplicate IDs")
    payload = json.dumps(sorted(values), separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_stage_fingerprint(
    *,
    stage: str,
    stage_version: str,
    candidate_ids: Iterable[object],
    input_paths: Iterable[str | Path] = (),
    settings: Mapping[str, Any] | None = None,
    code_base: str | Path | None = None,
    code_paths: Sequence[str] = (),
    hash_input_contents: bool = False,
) -> dict[str, Any]:
    paths = [
        file_signature(path, content_hash=hash_input_contents)
        for path in input_paths
    ]
    payload: dict[str, Any] = {
        "state_version": STAGE_STATE_VERSION,
        "stage": str(stage),
        "stage_version": str(stage_version),
        "candidate_set_hash": candidate_set_digest(candidate_ids),
        "inputs": paths,
        "settings": json_stable(dict(settings or {})),
    }
    if code_base is not None and code_paths:
        payload["code"] = code_fingerprint(Path(code_base), list(code_paths))
    payload["digest"] = fingerprint_digest(payload)
    return payload


def fingerprint_digest(payload: Mapping[str, Any]) -> str:
    normalized = dict(json_stable(dict(payload)))
    normalized.pop("digest", None)
    encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def write_stage_state(
    path: str | Path,
    *,
    fingerprint: Mapping[str, Any],
    result: StageResult,
    outputs: Iterable[str | Path] = (),
) -> Path:
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_version": STAGE_STATE_VERSION,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "fingerprint": json_stable(dict(fingerprint)),
        "result": json_stable(asdict(result)),
        "outputs": [file_signature(output, content_hash=True) for output in outputs],
    }
    fd, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, indent=2, default=str)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, target)
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise
    return target


def read_stage_state(path: str | Path) -> dict[str, Any] | None:
    target = Path(path).expanduser()
    if not target.exists():
        return None
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Invalid stage state file: {target}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("state_version") != STAGE_STATE_VERSION:
        raise ValueError(f"Unsupported stage state schema: {target}")
    return payload


def assert_reusable_stage_state(
    state: Mapping[str, Any] | None,
    *,
    fingerprint: Mapping[str, Any],
    require_complete: bool = False,
) -> None:
    if state is None:
        raise ValueError("Checkpoint/output exists without a versioned stage state")
    stored = state.get("fingerprint")
    if not isinstance(stored, Mapping):
        raise ValueError("Stage state is missing its fingerprint")
    if fingerprint_digest(stored) != fingerprint_digest(fingerprint):
        raise ValueError("Stage checkpoint fingerprint does not match current inputs/settings/code")
    result = state.get("result")
    if not isinstance(result, Mapping):
        raise ValueError("Stage state is missing result accounting")
    if require_complete and str(result.get("status")) != "success":
        raise ValueError(f"Stage is not successfully complete: status={result.get('status')}")
    outputs = state.get("outputs", [])
    if not isinstance(outputs, list):
        raise ValueError("Stage state has malformed output signatures")
    for stored_signature in outputs:
        if not isinstance(stored_signature, Mapping) or not stored_signature.get("path"):
            raise ValueError("Stage state contains a malformed output signature")
        validate_hash = "sha256" in stored_signature or "tree_sha256" in stored_signature
        current_signature = file_signature(
            str(stored_signature["path"]),
            content_hash=validate_hash,
        )
        for key in (
            "exists",
            "size",
            "mtime_ns",
            "sha256",
            "tree_sha256",
            "file_count",
            "total_file_size",
        ):
            if key in stored_signature and current_signature.get(key) != stored_signature.get(key):
                raise ValueError(
                    "Stage output no longer matches its recorded signature: "
                    f"{stored_signature['path']} ({key})"
                )
