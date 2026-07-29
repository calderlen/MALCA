"""Resumable client for IPAC's batch ZTF forced-photometry service.

Unlike the ordinary ZTF catalog-light-curve endpoint, ZFPS is asynchronous
and returns forced difference-image fluxes.  This module deliberately keeps
those products separate from ``ztf_lc_*`` catalog products: credentials are
read only at request time, while the durable journal contains request
parameters and result URLs, never the personal ZFPS password.
"""

from __future__ import annotations

from contextlib import contextmanager
from hashlib import sha256
from io import StringIO
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Any, Callable

import numpy as np
import pandas as pd
import requests

try:
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None

from malca.config import PARQUET_CACHE_COMPRESSION
from malca.external_lc_manifest import upsert_external_lc_manifest_entry


ZTF_FORCED_SUMMARY_COLUMNS = (
    "ztf_forced_lc_n_epochs",
    "ztf_forced_lc_n_good",
    "ztf_forced_lc_n_zg",
    "ztf_forced_lc_n_zr",
    "ztf_forced_lc_n_zi",
)
ZTF_FORCED_SUBMIT_URL = "https://ztfweb.ipac.caltech.edu/cgi-bin/batchfp.py/submit"
ZTF_FORCED_STATUS_URL = "https://ztfweb.ipac.caltech.edu/cgi-bin/getBatchForcedPhotometryRequests.cgi"
ZTF_FORCED_HTTP_AUTH = ("ztffps", "dontgocrazy!")
_REQUEST_VERSION = "ztf-forced-phot-v1"
_LEDGER_NAME = "ztf_forced_phot_tasks.parquet"
_MAX_BATCH_SIZE = 1500
_HTTP_TIMEOUT = 90.0
_LEDGER_COLUMNS = (
    "request_version", "batch_key", "candidate_ids_json", "coordinate_keys_json",
    "jd_start", "jd_end", "status", "submitted_unix", "updated_unix",
    "attempts", "error_message",
)


def parse_ztf_forced_result(text: str) -> pd.DataFrame:
    """Parse a ZFPS ASCII product while retaining every raw QA column."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("ZTF forced-photometry result is empty")
    names: list[str] | None = None
    for line in text.splitlines():
        stripped = line.lstrip("# ").strip()
        if "forcediffimflux" in stripped.lower() and "," in stripped:
            names = [part.strip() for part in stripped.split(",")]
            break
    data = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    if not data.strip():
        return pd.DataFrame(columns=names or [])
    try:
        out = pd.read_csv(StringIO(data), sep=r"\s+", names=names, header=None if names else "infer")
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=names or [])
    return out


def summarize_ztf_forced_lc(lc: pd.DataFrame) -> dict[str, int]:
    summary = {name: 0 for name in ZTF_FORCED_SUMMARY_COLUMNS}
    if lc is None or lc.empty:
        return summary
    summary["ztf_forced_lc_n_epochs"] = int(len(lc))
    status = pd.to_numeric(lc.get("procstatus", pd.Series(0, index=lc.index)), errors="coerce")
    summary["ztf_forced_lc_n_good"] = int(status.fillna(255).eq(0).sum())
    filters = lc.get("filter", pd.Series("", index=lc.index)).astype(str).str.strip().str.lower()
    for band in ("zg", "zr", "zi"):
        summary[f"ztf_forced_lc_n_{band}"] = int(filters.isin({band, f"ztf_{band[-1]}"}).sum())
    return summary


def query_ztf_forced_phot(
    df: pd.DataFrame,
    *,
    email: str | None = None,
    userpass: str | None = None,
    output_dir: Path | str | None = None,
    results_root: Path | str | None = None,
    task_checkpoint: Path | str | None = None,
    batch_size: int = _MAX_BATCH_SIZE,
    jd_start: float = 2458194.5,
    jd_end: float | None = None,
    submit_only: bool = False,
    refresh_cache: bool = False,
    session: Any | None = None,
    progress: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """Submit, monitor, and download batch ZTF forced photometry.

    The ZFPS status service only retains 30 days of history.  Re-running this
    function before then downloads any completed products and maps them back
    to candidates using the requested RA/Dec embedded in each product header.
    A prior ``submitting`` batch is never automatically sent again: that
    avoids violating ZFPS's 90-day duplicate-request policy after an
    interrupted HTTP response.
    """
    if output_dir is None:
        raise ValueError("output_dir is required for ZTF forced photometry")
    out = df.copy()
    for column in ZTF_FORCED_SUMMARY_COLUMNS:
        out[column] = pd.NA
    if "candidate_id" not in out.columns or "ra" not in out.columns or "dec" not in out.columns:
        raise ValueError("ZTF forced photometry requires candidate_id, ra, and dec columns")
    out["ra"] = pd.to_numeric(out["ra"], errors="coerce")
    out["dec"] = pd.to_numeric(out["dec"], errors="coerce")
    valid = out["candidate_id"].notna() & out["ra"].between(0, 360) & out["dec"].between(-90, 90)
    if not bool(valid.any()):
        return out
    batch_size = int(batch_size)
    if not 1 <= batch_size <= _MAX_BATCH_SIZE:
        raise ValueError(f"batch_size must be between 1 and {_MAX_BATCH_SIZE}")
    jd_start = float(jd_start)
    jd_end = float(jd_end if jd_end is not None else (time.time() / 86400.0 + 2440587.5))
    if not (math.isfinite(jd_start) and math.isfinite(jd_end) and jd_end > jd_start):
        raise ValueError("jd_start and jd_end must be finite, with jd_end > jd_start")
    email = email or os.environ.get("MALCA_ZTF_FORCED_EMAIL")
    userpass = userpass or os.environ.get("MALCA_ZTF_FORCED_USERPASS")
    if not email or not userpass:
        raise ValueError("set MALCA_ZTF_FORCED_EMAIL and MALCA_ZTF_FORCED_USERPASS (or pass email/userpass)")

    lc_dir = Path(output_dir).expanduser()
    lc_dir.mkdir(parents=True, exist_ok=True)
    root = Path(results_root).expanduser() if results_root is not None else (lc_dir.parent if lc_dir.name == "external_lcs" else lc_dir)
    checkpoint = Path(task_checkpoint).expanduser() if task_checkpoint else lc_dir / _LEDGER_NAME
    owned = session is None
    client = requests.Session() if owned else session
    try:
        with _locked(checkpoint):
            ledger = _read_ledger(checkpoint)
            if not submit_only:
                _download_ready_products(client, email, userpass, out, lc_dir, root, progress)
                _load_cached_products(out, lc_dir)
            if not refresh_cache:
                submitted = set(ledger.loc[ledger["status"].isin(["submitted", "submitting"]), "batch_key"].astype(str))
            else:
                submitted = set()
            specs = _batch_specs(out.loc[valid], batch_size, jd_start, jd_end)
            for spec in specs:
                if spec["batch_key"] in submitted:
                    continue
                now = time.time()
                new_row = pd.DataFrame([{
                    "request_version": _REQUEST_VERSION, **spec, "status": "submitting",
                    "submitted_unix": now, "updated_unix": now, "attempts": 1, "error_message": "",
                }]).reindex(columns=_LEDGER_COLUMNS)
                ledger = new_row if ledger.empty else pd.concat([ledger, new_row], ignore_index=True)
                _write_ledger(checkpoint, ledger)
                ra_values = spec["ra"]
                dec_values = spec["dec"]
                response = client.post(
                    ZTF_FORCED_SUBMIT_URL, auth=ZTF_FORCED_HTTP_AUTH,
                    data={"ra": json.dumps(ra_values), "dec": json.dumps(dec_values),
                          "jdstart": json.dumps(jd_start), "jdend": json.dumps(jd_end),
                          "email": email, "userpass": userpass}, timeout=_HTTP_TIMEOUT,
                )
                response.raise_for_status()
                ledger.loc[ledger["batch_key"].eq(spec["batch_key"]), ["status", "updated_unix"]] = ["submitted", time.time()]
                _write_ledger(checkpoint, ledger)
                _emit(progress, f"ZTF forced photometry: submitted {len(json.loads(spec['candidate_ids_json']))} coordinate(s)")
            if not submit_only:
                _download_ready_products(client, email, userpass, out, lc_dir, root, progress)
                _load_cached_products(out, lc_dir)
            return out
    finally:
        if owned:
            client.close()


def _batch_specs(df: pd.DataFrame, batch_size: int, jd_start: float, jd_end: float) -> list[dict[str, object]]:
    unique = df.drop_duplicates(subset=["ra", "dec"], keep="first")
    specs = []
    for start in range(0, len(unique), batch_size):
        part = unique.iloc[start:start + batch_size]
        coordinates = [(round(float(row.ra), 7), round(float(row.dec), 7)) for row in part.itertuples()]
        key_input = json.dumps({"v": _REQUEST_VERSION, "coordinates": coordinates, "jd": [jd_start, jd_end]}, sort_keys=True)
        specs.append({"batch_key": sha256(key_input.encode()).hexdigest(),
                      "candidate_ids_json": json.dumps(part["candidate_id"].astype(str).tolist()),
                      "coordinate_keys_json": json.dumps([f"{ra:.7f},{dec:.7f}" for ra, dec in coordinates]),
                      "jd_start": jd_start, "jd_end": jd_end,
                      "ra": [item[0] for item in coordinates], "dec": [item[1] for item in coordinates]})
    return specs


def _download_ready_products(client: Any, email: str, userpass: str, out: pd.DataFrame, lc_dir: Path, root: Path, progress: Callable[[str], None] | None) -> None:
    response = client.get(ZTF_FORCED_STATUS_URL, auth=ZTF_FORCED_HTTP_AUTH,
                          params={"email": email, "userpass": userpass, "option": "All recent jobs", "action": "Query Database"}, timeout=_HTTP_TIMEOUT)
    response.raise_for_status()
    paths = sorted(set(re.findall(r"/ztf/ops.+?lc\.txt\b", response.text)))
    if paths:
        _emit(progress, f"ZTF forced photometry: {len(paths)} recent product(s) ready")
    for path in paths:
        text = client.get("https://ztfweb.ipac.caltech.edu" + path, auth=ZTF_FORCED_HTTP_AUTH, timeout=_HTTP_TIMEOUT).text
        requested = re.search(r"Requested input R\.A\.\s*=\s*([\d.+-]+).*?Requested input Dec\.\s*=\s*([\d.+-]+)", text, re.S | re.I)
        if requested is None:
            continue
        ra, dec = map(float, requested.groups())
        matches = out.index[np.isclose(out["ra"], ra, atol=6e-7) & np.isclose(out["dec"], dec, atol=6e-7)]
        if not len(matches):
            continue
        lc = parse_ztf_forced_result(text)
        for idx in matches:
            candidate_id = str(out.at[idx, "candidate_id"])
            target = lc_dir / f"ztf_forced_lc_{candidate_id}.parquet"
            lc.to_parquet(target, index=False, compression=PARQUET_CACHE_COMPRESSION)
            upsert_external_lc_manifest_entry(root, candidate_id=candidate_id, source="ztf_forced", file_prefix="ztf_forced_lc", path=target)


def _load_cached_products(out: pd.DataFrame, lc_dir: Path) -> None:
    for idx, candidate_id in out["candidate_id"].items():
        path = lc_dir / f"ztf_forced_lc_{candidate_id}.parquet"
        if not path.exists():
            continue
        try:
            summary = summarize_ztf_forced_lc(pd.read_parquet(path))
        except Exception:
            continue
        for key, value in summary.items():
            out.at[idx, key] = value


def _read_ledger(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=_LEDGER_COLUMNS)
    try:
        frame = pd.read_parquet(path)
    except Exception:
        return pd.DataFrame(columns=_LEDGER_COLUMNS)
    for column in _LEDGER_COLUMNS:
        if column not in frame:
            frame[column] = pd.NA
    return frame[list(_LEDGER_COLUMNS)]


def _write_ledger(path: Path, ledger: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        ledger.to_parquet(temporary, index=False, compression=PARQUET_CACHE_COMPRESSION)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def _locked(path: Path):
    lock_path = path.with_suffix(path.suffix + ".lock")
    with open(lock_path, "a", encoding="ascii") as handle:
        if fcntl is not None:
            fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(handle, fcntl.LOCK_UN)


def _emit(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is not None:
        progress(message)
