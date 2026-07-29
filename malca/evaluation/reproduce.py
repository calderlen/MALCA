from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping, Sequence
import argparse
import hashlib
import inspect
import io
import json
import re
import sys

from tqdm import tqdm

import matplotlib.pyplot as pl
import numpy as np
import pandas as pd

from malca.plotting.lightcurve_publication import apply_publication_rcparams, save_publication_figure, FIG_TWO_COL_STANDARD

apply_publication_rcparams(pl)

from malca.core.baseline import (
    global_median_baseline,
    per_camera_median_baseline,
    per_camera_gp_baseline,
    per_camera_gp_baseline_masked,
)
from malca.cli_config import add_config_args, namespace_keys, parse_args_with_config
from malca.enrichment.characterize import query_gaia_by_ids, get_dust_extinction
from malca.enrichment.classify import compute_all_classifications
from malca.config import (
    MIN_TIME_SPAN,
    MIN_POINTS_PER_DAY,
    MIN_CAMERAS,
    VSX_MAX_SEP_ARCSEC,
    MIN_BAYES_FACTOR,
    POST_FILTER_MIN_RUN_CAMERAS,
    POST_FILTER_MIN_RUN_POINTS,
)
from malca.config import REPRODUCE_CHUNK_SIZE
from malca.config import VSX_RAW_CATALOG_PATH
from malca.config import (
    WORKERS,
    TRIGGER_MODE,
    LOGBF_THRESHOLD_DIP,
    LOGBF_THRESHOLD_JUMP,
    SIGNIFICANCE_THRESHOLD,
    P_POINTS,
    MAG_POINTS,
    MIN_MAG_OFFSET,
    RUN_MIN_POINTS,
    RUN_MAX_GAP_POINTS,
    BASELINE_FUNC,
    BASELINE_S0,
    BASELINE_W0,
    BASELINE_Q,
    BASELINE_JITTER,
    JD_OFFSET,
    DEFAULT_OUTPUT_DIR,
    LIGHT_CURVE_FILE_EXTENSION,
)
from malca.stv.events import score_lightcurve, signal_amplitude_pass_mask
from malca.stv.filter import apply_filters, filter_signal_amplitude
from malca.stv.plot import plot_passing_candidates
from malca.stv.plot import read_skypatrol_csv
from malca.stv.score import compute_event_score
from concurrent.futures import ProcessPoolExecutor
from malca.core.stats import median_dt, compute_stats, _enrich_row_worker
from malca.stv.tag import apply_tags
from malca.io.table_io import read_feature_table, read_parquet_table, write_parquet_table
from malca.stv.triggering import normalize_trigger_block
from malca.core.utils import read_lc_dat2








CANDIDATE_USECOLS = {
    "candidate_id",
    "asas_sn_id",
    "path",
    "source_id",
    "mag_bin",
    "ra_deg",
    "dec_deg",
}

REPRODUCTION_SCHEMA_VERSION = 2
_MISSING_ID_VALUES = frozenset({"", "nan", "none", "null", "<na>"})


def _canonical_candidate_id(value: object) -> str | None:
    if value is None or value is pd.NA:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (int, np.integer)):
        text = str(int(value))
    elif isinstance(value, (float, np.floating)) and np.isfinite(value) and float(value).is_integer():
        text = str(int(value))
    else:
        text = str(value).strip()
    return None if text.lower() in _MISSING_ID_VALUES else text


def _validate_unique_ids(df: pd.DataFrame, column: str, label: str) -> pd.Series:
    if column not in df.columns:
        raise ValueError(f"{label} must include a {column!r} column")
    ids = df[column].map(_canonical_candidate_id).astype("string")
    if bool(ids.isna().any()):
        raise ValueError(f"{label} contains blank/null {column} values")
    duplicates = ids.duplicated(keep=False)
    if bool(duplicates.any()):
        examples = sorted(ids.loc[duplicates].astype(str).unique())[:5]
        raise ValueError(f"{label} contains duplicate {column} values: {examples}")
    return ids


def _fingerprintable(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _fingerprintable(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_fingerprintable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_fingerprintable(item) for item in value)
    if callable(value):
        return f"{value.__module__}.{getattr(value, '__qualname__', getattr(value, '__name__', type(value).__name__))}"
    if isinstance(value, Path):
        return str(value.expanduser())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _stable_digest(value: object) -> str:
    payload = json.dumps(_fingerprintable(value), sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def _reproduction_code_fingerprint() -> str:
    functions = {
        "score_lightcurve": score_lightcurve,
        "compute_event_score": compute_event_score,
        "apply_tags": apply_tags,
        "filter_signal_amplitude": filter_signal_amplitude,
        "apply_filters": apply_filters,
        "query_gaia_by_ids": query_gaia_by_ids,
        "get_dust_extinction": get_dust_extinction,
        "compute_all_classifications": compute_all_classifications,
        "enrich_row_worker": _enrich_row_worker,
        "global_median_baseline": global_median_baseline,
        "per_camera_median_baseline": per_camera_median_baseline,
        "per_camera_gp_baseline": per_camera_gp_baseline,
        "per_camera_gp_baseline_masked": per_camera_gp_baseline_masked,
    }
    sources: dict[str, str] = {}
    for name, function in functions.items():
        try:
            sources[name] = inspect.getsource(function)
        except (OSError, TypeError):
            sources[name] = repr(function)
    return _stable_digest(sources)


def _path_fingerprint(path_value: object) -> dict[str, object]:
    if _canonical_candidate_id(path_value) is None:
        return {"path": None, "exists": False, "is_file": False}
    path = Path(str(path_value)).expanduser().resolve(strict=False)
    record: dict[str, object] = {"path": str(path), "exists": path.exists(), "is_file": path.is_file()}
    if path.is_dir():
        children = [
            {
                "relative_path": str(child.relative_to(path)),
                "fingerprint": _path_fingerprint(child),
            }
            for child in sorted(candidate for candidate in path.rglob("*") if candidate.is_file())
        ]
        record.update(
            {
                "is_directory": True,
                "n_files": len(children),
                "contents_sha256": _stable_digest(children),
            }
        )
        return record
    if not path.is_file():
        return record
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    stat = path.stat()
    record.update({"size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns), "sha256": digest.hexdigest()})
    return record


def _strict_boolean_series(series: pd.Series) -> pd.Series:
    def parse(value: object) -> object:
        if value is None or value is pd.NA:
            return pd.NA
        try:
            if bool(pd.isna(value)):
                return pd.NA
        except (TypeError, ValueError):
            pass
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
            return bool(value)
        text = str(value).strip().lower()
        if text in {"true", "1", "t", "yes", "y"}:
            return True
        if text in {"false", "0", "f", "no", "n"}:
            return False
        return pd.NA

    return pd.Series((parse(value) for value in series), index=series.index, dtype="boolean")


def _apply_reproduction_signal_amplitude(
    frame: pd.DataFrame,
    *,
    min_mag_offset: float,
) -> pd.DataFrame:
    """Apply the canonical amplitude gate independently to every band/branch.

    Reproduction rows keep g and V results in separate prefixed columns.  They
    must not be collapsed by taking a baseline from one band and an event
    offset from the other.  This helper also gates dip and jump significance
    independently so a strong jump cannot make an under-amplitude dip count as
    a reproduced dip.
    """
    out = frame.copy()
    threshold = float(min_mag_offset)
    if out.empty or threshold <= 0:
        return out

    successful = out.get(
        "trial_status",
        pd.Series("ok", index=out.index, dtype="string"),
    ).astype("string").eq("ok")
    original: dict[tuple[str, str], pd.Series] = {}
    passed: dict[tuple[str, str], pd.Series] = {}

    for band in ("g", "v"):
        band_frame = pd.DataFrame(index=out.index)
        for kind in ("dip", "jump"):
            significant_col = f"{band}_bayes_{kind}_significant"
            significant = _strict_boolean_series(
                out.get(significant_col, pd.Series(False, index=out.index))
            ).fillna(False)
            original[(band, kind)] = significant & successful
            band_frame[f"{kind}_significant"] = significant

            explicit_delta = f"{band}_{kind}_best_delta_mag"
            legacy_delta = f"{band}_{kind}_best_mag_event"
            source_col = explicit_delta if explicit_delta in out.columns else legacy_delta
            band_frame[f"{kind}_best_delta_mag"] = pd.to_numeric(
                out.get(source_col, pd.Series(np.nan, index=out.index)),
                errors="coerce",
            )

        # Use the same canonical event-gate implementation as stv-events, then
        # retain branch-level decisions for scientifically correct reporting.
        band_any_pass = signal_amplitude_pass_mask(band_frame, threshold) & successful
        for kind in ("dip", "jump"):
            delta = pd.to_numeric(band_frame[f"{kind}_best_delta_mag"], errors="coerce")
            branch_pass = original[(band, kind)] & delta.abs().ge(threshold)
            passed[(band, kind)] = branch_pass
            significant_col = f"{band}_bayes_{kind}_significant"
            if significant_col not in out.columns:
                out[significant_col] = False
            out.loc[successful, significant_col] = branch_pass.loc[successful].to_numpy()

            decision = pd.Series(pd.NA, index=out.index, dtype="boolean")
            decision.loc[successful] = branch_pass.loc[successful].to_numpy()
            out[f"{band}_{kind}_signal_amplitude_pass"] = decision

        # Assert that the explicit branch decisions cannot drift from the
        # canonical candidate-level gate.
        explicit_band_pass = passed[(band, "dip")] | passed[(band, "jump")]
        if not explicit_band_pass.loc[successful].equals(band_any_pass.loc[successful]):
            raise AssertionError(f"{band}-band amplitude decisions disagree with the canonical gate")

    original_dip = original[("g", "dip")] | original[("v", "dip")]
    passed_dip = passed[("g", "dip")] | passed[("v", "dip")]
    original_any = original_dip | original[("g", "jump")] | original[("v", "jump")]
    passed_any = passed_dip | passed[("g", "jump")] | passed[("v", "jump")]

    evaluated = pd.Series(pd.NA, index=out.index, dtype="boolean")
    evaluated.loc[successful] = original_any.loc[successful].to_numpy()
    out["signal_amplitude_evaluated"] = evaluated

    amplitude_pass = pd.Series(pd.NA, index=out.index, dtype="boolean")
    amplitude_pass.loc[successful] = passed_any.loc[successful].to_numpy()
    out["signal_amplitude_pass"] = amplitude_pass

    dip_amplitude_pass = pd.Series(pd.NA, index=out.index, dtype="boolean")
    dip_amplitude_pass.loc[successful] = passed_dip.loc[successful].to_numpy()
    out["dip_signal_amplitude_pass"] = dip_amplitude_pass

    failed = pd.Series(pd.NA, index=out.index, dtype="boolean")
    failed.loc[successful] = (
        original_any.loc[successful] & ~passed_any.loc[successful]
    ).to_numpy()
    out["failed_signal_amplitude"] = failed

    # ``detected`` in the reproduction report is specifically dip recovery.
    # Record a dip-amplitude rejection when every otherwise-significant dip is
    # removed, even if an unrelated jump branch survives.
    dip_rejected = successful & original_dip & ~passed_dip
    if "rejection_reason" not in out.columns:
        out["rejection_reason"] = None
    overall_update = dip_rejected & out["rejection_reason"].isna()
    out.loc[overall_update, "rejection_reason"] = "signal_amplitude"

    for band in ("g", "v"):
        reason_col = f"{band}_rejection_reason"
        if reason_col not in out.columns:
            out[reason_col] = None
        band_rejected = successful & original[(band, "dip")] & ~passed[(band, "dip")]
        update = band_rejected & out[reason_col].isna()
        out.loc[update, reason_col] = "signal_amplitude"

    return out


REPRODUCE_CONFIG_DEFAULTS = {
    "trigger_mode": TRIGGER_MODE,
    "significance_threshold": SIGNIFICANCE_THRESHOLD,
    "logbf_threshold_dip": LOGBF_THRESHOLD_DIP,
    "logbf_threshold_jump": LOGBF_THRESHOLD_JUMP,
    "p_points": P_POINTS,
    "p_min_dip": None,
    "p_max_dip": None,
    "p_min_jump": None,
    "p_max_jump": None,
    "mag_points": MAG_POINTS,
    "baseline_func": BASELINE_FUNC,
    "baseline_s0": BASELINE_S0,
    "baseline_w0": BASELINE_W0,
    "baseline_q": BASELINE_Q,
    "baseline_jitter": BASELINE_JITTER,
    "baseline_sigma_floor": None,
    "mag_min_dip": None,
    "mag_max_dip": None,
    "mag_min_jump": None,
    "mag_max_jump": None,
    "run_min_points": RUN_MIN_POINTS,
    "run_max_gap_points": RUN_MAX_GAP_POINTS,
    "run_max_gap_days": None,
    "run_min_duration_days": 0.0,
    "min_mag_offset": MIN_MAG_OFFSET,
}


brayden_candidates: list[dict[str, object]] = [
    {"source": "J042214+152530", "source_id": "377957522430", "category": "Dippers", "mag_bin": "13_13.5", "search_method": "Pipeline", "expected_detected": True},
    {"source": "J202402+383938", "source_id": "42950993887", "category": "Dippers", "mag_bin": "13_13.5", "search_method": "Pipeline", "expected_detected": True},
    {"source": "J174328+343315", "source_id": "223339338105", "category": "Dippers", "mag_bin": "13_13.5", "search_method": "Pipeline", "expected_detected": True},
    {"source": "J080327-261620", "source_id": "601296043597", "category": "Dippers", "mag_bin": "13_13.5", "search_method": "Pipeline", "expected_detected": False},
    {"source": "J184916-473251", "source_id": "472447294641", "category": "Dippers", "mag_bin": "13_13.5", "search_method": "Known", "expected_detected": True},
    {"source": "J183153-284827", "source_id": "455267102087", "category": "Dippers", "mag_bin": "13.5_14", "search_method": "Known", "expected_detected": False},
    {"source": "J070519+061219", "source_id": "266288137752", "category": "Dippers", "mag_bin": "13.5_14", "search_method": "Known", "expected_detected": False},
    {"source": "J081523-385923", "source_id": "532576686103", "category": "Dippers", "mag_bin": "13.5_14", "search_method": "Known", "expected_detected": False},
    {"source": "J085816-430955", "source_id": "352187470767", "category": "Dippers", "mag_bin": "12_12.5", "search_method": "Known", "expected_detected": False},
    {"source": "J114712-621037", "source_id": "609886184506", "category": "Dippers", "mag_bin": "13_13.5", "search_method": "Known", "expected_detected": False},
    {"source": "J005437+644347", "source_id": "68720274411", "category": "Multiple Eclipse Binaries", "mag_bin": "13_13.5", "search_method": "Known", "expected_detected": True},
    {"source": "J062510-075341", "source_id": "377958261591", "category": "Multiple Eclipse Binaries", "mag_bin": "13.5_14", "search_method": "Pipeline", "expected_detected": True},
    {"source": "J124745-622756", "source_id": "515397118400", "category": "Multiple Eclipse Binaries", "mag_bin": "13.5_14", "search_method": "Pipeline", "expected_detected": True},
    {"source": "J175912-120956", "source_id": "326417831663", "category": "Multiple Eclipse Binaries", "mag_bin": "13_13.5", "search_method": "Pipeline", "expected_detected": True},
    {"source": "J181752-580749", "source_id": "644245387906", "category": "Multiple Eclipse Binaries", "mag_bin": "12_12.5", "search_method": "Known", "expected_detected": True},
    {"source": "J160757-574540", "source_id": "661425129485", "category": "Multiple Eclipse Binaries", "mag_bin": "13.5_14", "search_method": "Pipeline", "expected_detected": False},
    {"source": "J073924-272916", "source_id": "438086977939", "category": "Single Eclipse Binaries", "mag_bin": "13.5_14", "search_method": "Pipeline", "expected_detected": True},
    {"source": "J074007-161608", "source_id": "360777377116", "category": "Single Eclipse Binaries", "mag_bin": "13_13.5", "search_method": "Pipeline", "expected_detected": True},
    {"source": "J094848-545959", "source_id": "635655234580", "category": "Single Eclipse Binaries", "mag_bin": "13_13.5", "search_method": "Pipeline", "expected_detected": True},
    {"source": "J162209-444247", "source_id": "412317159120", "category": "Single Eclipse Binaries", "mag_bin": "13.5_14", "search_method": "Pipeline", "expected_detected": True},
    {"source": "J183606-314826", "source_id": "438086901547", "category": "Single Eclipse Binaries", "mag_bin": "13.5_14", "search_method": "Pipeline", "expected_detected": True},
    {"source": "J205245-713514", "source_id": "463856535113", "category": "Single Eclipse Binaries", "mag_bin": "13_13.5", "search_method": "Pipeline", "expected_detected": True},
    {"source": "J212132+480140", "source_id": "120259184943", "category": "Single Eclipse Binaries", "mag_bin": "13_13.5", "search_method": "Pipeline", "expected_detected": True},
    {"source": "J225702+562312", "source_id": "25770019815", "category": "Single Eclipse Binaries", "mag_bin": "13.5_14", "search_method": "Pipeline", "expected_detected": True},
    {"source": "J190316-195739", "source_id": "515396514761", "category": "Single Eclipse Binaries", "mag_bin": "13.5_14", "search_method": "Pipeline", "expected_detected": True},
    {"source": "J175602+013135", "source_id": "231929175915", "category": "Single Eclipse Binaries", "mag_bin": "14_14.5", "search_method": "Known", "expected_detected": True},
    {"source": "J073234-200049", "source_id": "335007754417", "category": "Single Eclipse Binaries", "mag_bin": "14.5_15", "search_method": "Known", "expected_detected": True},
    {"source": "J223332+565552", "source_id": "60130040391", "category": "Single Eclipse Binaries", "mag_bin": "12.5_13", "search_method": "Known", "expected_detected": True},
    {"source": "J183210-173432", "source_id": "317827964025", "category": "Single Eclipse Binaries", "mag_bin": "12.5_13", "search_method": "Pipeline", "expected_detected": False},
]


def _parse_tzanidakis_candidates() -> list[dict[str, object]]:
    """
    Parse Tzanidakis+2025 candidates from the fixed-width .sty file.
    Returns a list of candidate dictionaries with gaia_id (Gaia DR3 source_id).
    """
    input_path = Path(__file__).parent.parent.parent / "input" / "Tzanidakis+2025.sty"
    if not input_path.exists():
        return []
    
    # Column specifications based on byte positions in the file header
    colspecs = [
        (0, 19),    # source_id (Gaia DR3)
        (20, 28),   # RAdeg
        (29, 37),   # DEdeg
        (38, 43),   # GMAG0
        (44, 49),   # BP-RP0
        (50, 54),   # dist50
        (55, 64),   # t0dip (MJD)
        (65, 66),   # Ndips
    ]
    names = ["gaia_id", "ra", "dec", "gmag", "bp_rp", "distance_kpc", "t0_dip_mjd", "n_dips"]
    
    # Read fixed-width format, skipping header lines
    df = pd.read_fwf(input_path, colspecs=colspecs, names=names, skiprows=19)
    
    # Convert to list of dicts with additional metadata
    candidates = []
    for _, row in df.iterrows():
        gaia_id = str(int(row["gaia_id"]))
        candidates.append({
            "source": f"Gaia-{gaia_id}",
            "gaia_id": gaia_id,
            "source_id": gaia_id,  # Use gaia_id as source_id
            "category": "Tzanidakis2025_Dippers",
            "ra": float(row["ra"]),
            "dec": float(row["dec"]),
            "gmag": float(row["gmag"]),
            "bp_rp": float(row["bp_rp"]),
            "distance_kpc": float(row["distance_kpc"]),
            "t0_dip_mjd": float(row["t0_dip_mjd"]),
            "n_dips": int(row["n_dips"]),
            "search_method": "Literature",
            "expected_detected": True,
        })
    
    return candidates


tzanidakis_candidates: list[dict[str, object]] = _parse_tzanidakis_candidates()


def load_manifest_df(manifest_path: Path | str) -> pd.DataFrame:
    path = Path(manifest_path).expanduser()
    df = read_parquet_table(path)
    if "source_id" not in df.columns and "asas_sn_id" in df.columns:
        df["source_id"] = df["asas_sn_id"]
    df["source_id"] = _validate_unique_ids(df, "source_id", "Reproduction manifest")
    if "candidate_id" not in df.columns:
        df["candidate_id"] = df["source_id"]
    return df


def load_candidates_df(cand_path: Path) -> pd.DataFrame:
    path = Path(cand_path).expanduser()
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = read_feature_table(path)
    if "source_id" not in df.columns and "asas_sn_id" in df.columns:
        df["source_id"] = df["asas_sn_id"].astype(str)
    if "candidate_id" not in df.columns:
        if "asas_sn_id" in df.columns:
            df["candidate_id"] = df["asas_sn_id"].astype(str)
        elif "source_id" in df.columns:
            df["candidate_id"] = df["source_id"].astype(str)
        elif "path" in df.columns:
            df["candidate_id"] = df["path"].map(lambda value: Path(str(value)).stem)
    if "source_id" not in df.columns and "candidate_id" in df.columns:
        df["source_id"] = df["candidate_id"]
    if "source_id" not in df.columns:
        raise ValueError("Candidate table must include source_id/asas_sn_id or a path-derived identity")
    df["source_id"] = _validate_unique_ids(df, "source_id", "Candidate table")
    df["candidate_id"] = _validate_unique_ids(df, "candidate_id", "Candidate table")
    usecols = [col for col in df.columns if col in CANDIDATE_USECOLS]
    return df[usecols].copy() if usecols else df


def dataframe_from_candidates(data: Sequence[Mapping[str, object]] | None = None) -> pd.DataFrame:
    df = pd.DataFrame(data or brayden_candidates).copy()
    if "source_id" not in df.columns:
        raise ValueError("Candidates must include a 'source_id' column.")
    df["source_id"] = _validate_unique_ids(df, "source_id", "Candidates")
    if "candidate_id" not in df.columns:
        df["candidate_id"] = df["source_id"]
    df["candidate_id"] = _validate_unique_ids(df, "candidate_id", "Candidates")
    return df


def target_map(df: pd.DataFrame) -> dict[str, set[str]]:
    grouped: dict[str, set[str]] = {}
    for mag_bin, chunk in df.groupby("mag_bin"):
        grouped[str(mag_bin)] = set(chunk["source_id"].astype(str))
    return grouped


def records_from_manifest(df: pd.DataFrame) -> dict[str, list[dict[str, object]]]:
    source_ids = _validate_unique_ids(df, "source_id", "Manifest subset")
    records: dict[str, list[dict[str, object]]] = {}
    for source_id, rec in zip(source_ids.astype(str), df.to_dict("records")):
        mag_bin = str(rec.get("mag_bin"))
        lc_dir = str(rec.get("lc_dir"))
        dat_value = rec.get("dat_path")
        dat_path = (
            str(dat_value)
            if _canonical_candidate_id(dat_value) is not None
            else str(Path(lc_dir) / f"{source_id}.{LIGHT_CURVE_FILE_EXTENSION}")
        )
        exists_value = rec.get("dat_exists", Path(dat_path).exists())
        exists = _strict_boolean_series(pd.Series([exists_value])).iloc[0]
        if pd.isna(exists):
            raise ValueError(f"Manifest has an invalid dat_exists value for source_id {source_id!r}")
        record = {
            "mag_bin": mag_bin,
            "index_num": rec.get("index_num"),
            "index_csv": rec.get("index_csv"),
            "lc_dir": lc_dir,
            "asas_sn_id": source_id,
            "dat_path": dat_path,
            "found": bool(exists),
            "input_source": "manifest",
        }
        records.setdefault(mag_bin, []).append(record)
    return records


def records_from_skypatrol_dir(df_targets: pd.DataFrame, skypatrol_dir: Path) -> dict[str, list[dict[str, object]]]:
    base = Path(skypatrol_dir)

    records: dict[str, list[dict[str, object]]] = {}
    for _, row in df_targets.iterrows():
        source_id = str(row.get("source_id"))
        mag_bin = str(row.get("mag_bin"))
        csv_candidates = [
            base / f"{source_id}.csv",
            base / f"{source_id}-light-curves.csv",
        ]
        csv_path = next((path for path in csv_candidates if path.exists()), None)
        if csv_path is None:
            csv_path = csv_candidates[0]
        rec = {
            "mag_bin": mag_bin,
            "index_num": None,
            "index_csv": None,
            "lc_dir": str(base),
            "asas_sn_id": source_id,
            "dat_path": str(csv_path),
            "found": csv_path.is_file(),
            "input_source": "skypatrol",
        }
        records.setdefault(mag_bin, []).append(rec)
    return records


def records_from_candidates_with_paths(
    df_targets: pd.DataFrame,
    *,
    path_prefix: Path | str | None = None,
    path_root: Path | str | None = None,
) -> dict[str, list[dict[str, object]]]:
    """
    Build records_map from candidates that have a 'path' column.
    Used when events.py output is passed directly to reproduction.py.
    """
    if "path" not in df_targets.columns:
        return {}

    records: dict[str, list[dict[str, object]]] = {}
    prefix = Path(path_prefix).expanduser() if path_prefix else None
    root = Path(path_root).expanduser() if path_root else None

    for _, row in df_targets.iterrows():
        source_id = str(row.get("source_id"))
        mag_bin = str(row.get("mag_bin", ""))
        path_str = str(row.get("path", ""))
        path_missing = _canonical_candidate_id(row.get("path")) is None
        path = Path(path_str) if not path_missing else Path("__missing_light_curve__") / f"{source_id}.dat2"
        if prefix and root:
            try:
                path = root / path.relative_to(prefix)
            except ValueError:
                pass
        lc_dir = str(path.parent)

        rec = {
            "mag_bin": mag_bin,
            "index_num": None,
            "index_csv": None,
            "lc_dir": lc_dir,
            "asas_sn_id": source_id,
            "dat_path": str(path),
            "found": path.is_file(),
            "input_source": "candidate_path",
        }
        records.setdefault(mag_bin, []).append(rec)
    return records


def _ordered_reproduction_columns(frame: pd.DataFrame, extra_cols: Iterable[str] | None = None) -> list[str]:
    if extra_cols:
        base_extra_cols = [c for c in extra_cols if c in frame.columns]
    else:
        base_extra_cols = []

    ordered = [
        col for col in [
            "source",
            "source_id",
            "category",
            "mag_bin",
            "trial_status",
            "evaluation_error",
            "reproduction_record_id",
            "detected",
            "pipeline_passed",
            "rejection_reason",
            "detection_details",
            "input_source",
            "dat_path",
            "input_record_fingerprint",
            "input_file_fingerprint",
            "input_fingerprint",
            "candidate_set_fingerprint",
            "manifest_fingerprint",
            "auxiliary_input_fingerprint",
            "config_fingerprint",
            "code_fingerprint",
            "run_fingerprint",
        ]
        if col in frame.columns
    ]
    for col in [
        "g_rejection_reason",
        "v_rejection_reason",
        "g_n_peaks",
        "v_n_peaks",
        "g_bayes_dip_significant",
        "v_bayes_dip_significant",
        "g_bayes_dip_max_prob",
        "v_bayes_dip_max_prob",
        "g_bayes_dip_max_logbf",
        "v_bayes_dip_max_logbf",
        "g_bayes_jump_significant",
        "v_bayes_jump_significant",
        "g_bayes_dip_bayes_factor",
        "v_bayes_dip_bayes_factor",
        "g_bayes_jump_bayes_factor",
        "v_bayes_jump_bayes_factor",
        "g_bayes_n_dips",
        "v_bayes_n_dips",
        "g_bayes_n_jumps",
        "v_bayes_n_jumps",
        "g_max_depth",
        "v_max_depth",
        "jd_first",
        "jd_last",
        "g_n_points",
        "v_n_points",
        "g_time_span",
        "v_time_span",
        "g_n_runs",
        "g_n_triggered",
        "g_best_morphology",
        "g_best_t0",
        "g_best_amplitude",
        "g_best_duration",
        "g_best_run_n_points",
        "g_best_run_start_jd",
        "g_best_run_end_jd",
        "v_n_runs",
        "v_n_triggered",
        "v_best_morphology",
        "v_best_t0",
        "v_best_amplitude",
        "v_best_duration",
        "v_best_run_n_points",
        "v_best_run_start_jd",
        "v_best_run_end_jd",
    ]:
        if col in frame.columns:
            ordered.append(col)
    ordered.extend([c for c in base_extra_cols if c not in ordered])
    ordered.extend([c for c in frame.columns if c not in ordered])
    return ordered


def coerce_candidate_records(data) -> list[dict[str, object]]:
    if data is None:
        return list(brayden_candidates)

    if isinstance(data, pd.DataFrame):
        records = data.to_dict("records")
    else:
        records = list(data)

    if not records:
        return []

    first = records[0]
    if isinstance(first, Mapping):

        coerced: list[dict[str, object]] = []
        seen_source_ids: set[str] = set()
        seen_candidate_ids: set[str] = set()
        for row_index, rec in enumerate(records):
            if not isinstance(rec, Mapping):
                raise ValueError(f"Candidate entry {row_index} is not a mapping")
            new = dict(rec)
            source_id = _canonical_candidate_id(new.get("source_id", new.get("asas_sn_id")))
            if source_id is None and "path" in new:
                source_id = _canonical_candidate_id(Path(str(new["path"])).stem)
            if source_id is None:
                raise ValueError(f"Candidate entry {row_index} has no usable source identity")
            candidate_id = _canonical_candidate_id(new.get("candidate_id")) or source_id
            if source_id in seen_source_ids:
                raise ValueError(f"Candidates contain duplicate source_id value {source_id!r}")
            if candidate_id in seen_candidate_ids:
                raise ValueError(f"Candidates contain duplicate candidate_id value {candidate_id!r}")
            seen_source_ids.add(source_id)
            seen_candidate_ids.add(candidate_id)

            new["source_id"] = source_id
            new["candidate_id"] = candidate_id
            new.setdefault("source", new.get("source", source_id))

            if "mag_bin" not in new or not new["mag_bin"]:
                if "path" in new:
                    path_str = str(new["path"])
                    mag_bin_match = re.search(r'(\d+(?:\.\d+)?_\d+(?:\.\d+)?)', path_str)
                    if mag_bin_match:
                        new["mag_bin"] = mag_bin_match.group(1)

            coerced.append(new)
        return coerced

    ids = []
    for entry in records:
        entry_str = str(entry)
        entry_path = Path(entry_str)
        stem = entry_path.stem
        source_id = stem.split("-")[0] if stem else entry_str
        ids.append(source_id)

    lookup = {c["source_id"]: c for c in brayden_candidates}
    coerced = []
    seen = set()
    for source_id in ids:
        if source_id in seen:
            raise ValueError(f"Candidates contain duplicate source identity {source_id!r}")
        seen.add(source_id)
        if source_id in lookup:
            record = dict(lookup[source_id])
            record.setdefault("candidate_id", source_id)
            coerced.append(record)
        else:
            coerced.append({"source": source_id, "source_id": source_id, "candidate_id": source_id, "mag_bin": None})
    return coerced


def resolve_candidates(spec: str | None):
    if spec is None:
        return list(brayden_candidates)
    
    spec_lower = spec.lower().strip()
    if spec_lower in {"brayden", "brayden_candidates"}:
        return list(brayden_candidates)
    if spec_lower in {"tzanidakis", "tzanidakis_candidates", "tzanidakis2025"}:
        return list(tzanidakis_candidates)
    
    path = Path(spec).expanduser()
    if not path.exists():
        print(f"WARNING: candidates path does not exist: {path}")

    cand_path = Path(spec)
    if cand_path.exists():
        df = load_candidates_df(cand_path)
        return coerce_candidate_records(df)

    raise SystemExit(f"Unknown candidates spec '{spec}'. Provide a built-in name or a valid file path.")


def clean_for_bayes(df: pd.DataFrame) -> pd.DataFrame:
    """Keep this consistent with the Bayesian module's cleaning so event_indices align with plotting."""
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    mask = np.ones(len(out), dtype=bool)

    if "saturated" in out.columns:
        mask &= (out["saturated"] == 0)

    mask &= out["JD"].notna() & out["mag"].notna()

    if "error" in out.columns:
        mask &= out["error"].notna() & (out["error"] > 0.0) & (out["error"] < 1.0)

    out = out.loc[mask].sort_values("JD").reset_index(drop=True)
    return out


def plot_light_curve_with_dips(
    dfg: pd.DataFrame,
    dfv: pd.DataFrame,
    res_g: dict,
    res_v: dict,
    source_id: str,
    plot_path: Path,
    accepted_morphologies: set[str] | None = None,
    g_significant: bool = False,
    v_significant: bool = False,
):
    """
    Plot light curves with dips in 2x2 layout (raw + residuals for V and g bands).
    Matches the old plot style with JD offset, thinner baselines, and residual panes.
    """
    # Default: accept gaussian and paczynski, reject noise/none
    if accepted_morphologies is None:
        accepted_morphologies = {"gaussian", "paczynski"}
    
    # Use 2x2 layout: V-band and g-band columns, raw + residuals rows
    fig, axes = pl.subplots(2, 2, figsize=FIG_TWO_COL_STANDARD, sharex="col")

    # Baseline parameters (match the SHO-ish defaults; note baseline function takes q not Q)
    baseline_kwargs = {
        "S0": BASELINE_S0,
        "w0": BASELINE_W0,
        "q": BASELINE_Q,
        "jitter": BASELINE_JITTER,
        "sigma_floor": None,
        "add_sigma_eff_col": True,
    }

    camera_colors = pl.cm.tab10(np.linspace(0, 1, 10))
    band_labels = {0: "g band", 1: "V band"}
    band_markers = {0: "o", 1: "s"}

    def plot_band(band_idx, df_band: pd.DataFrame, res: dict, band: int):
        """Plot one band column (raw + residuals)."""
        band_label = band_labels[band]
        
        if df_band is None or df_band.empty or "JD" not in df_band.columns or "mag" not in df_band.columns:
            axes[0, band_idx].text(0.5, 0.5, f"No {band_label} data", 
                                    ha="center", va="center", transform=axes[0, band_idx].transAxes)
            axes[0, band_idx].set_title(f"{source_id} - {band_label} (no data)", fontsize=12)
            return

        # Filter bad errors
        plot = df_band.copy()
        if "error" in plot.columns:
            plot = plot[plot["error"] <= 1.0]
        
        # Apply JD offset
        median_jd = plot["JD"].median()
        if median_jd > 2000000:
            plot["JD_plot"] = plot["JD"] - JD_OFFSET
        else:
            plot["JD_plot"] = plot["JD"] - 8000.0

        # Compute baseline
        df_baseline = None
        try:
            df_baseline = per_camera_gp_baseline(plot, **baseline_kwargs)
            if "baseline" in df_baseline.columns:
                plot["baseline"] = df_baseline["baseline"]
                plot["resid"] = plot["mag"] - plot["baseline"]
        except Exception:
            pass

        # Main (raw) plot
        ax_main = axes[0, band_idx]
        ax_resid = axes[1, band_idx]
        
        ax_main.invert_yaxis()
        ax_main.grid(True, alpha=0.3)
        ax_main.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
        ax_main.set_xlabel(f"JD - {int(JD_OFFSET)} [d]", fontsize=10)
        ax_main.xaxis.set_label_position("top")
        ax_main.set_ylabel(f"{band_label} [mag]", fontsize=12)

        # Get cameras
        cam_col = "camera#" if "camera#" in plot.columns else None
        camera_ids = sorted(plot[cam_col].dropna().unique()) if cam_col else []
        
        legend_handles = {}
        
        # Plot per camera
        if cam_col and camera_ids:
            for i, cam in enumerate(camera_ids):
                cam_data = plot[plot[cam_col] == cam]
                if cam_data.empty:
                    continue

                color = camera_colors[i % len(camera_colors)]
                marker = band_markers.get(band, "o")
                
                # Data points
                ax_main.errorbar(
                    cam_data["JD_plot"], cam_data["mag"], yerr=cam_data.get("error"),
                    fmt=marker, ms=4, color=color, alpha=0.8,
                    ecolor=color, elinewidth=0.8, capsize=2,
                    markeredgecolor="black", markeredgewidth=0.5,
                )
                
                # Baseline
                if "baseline" in plot.columns:
                    cam_base = plot[plot[cam_col] == cam].sort_values("JD_plot")
                    if not cam_base.empty and cam_base["baseline"].notna().any():
                        ax_main.plot(
                            cam_base["JD_plot"], cam_base["baseline"],
                            color=color, linestyle="-", linewidth=1.6, alpha=0.8, zorder=5
                        )
                
                # Residuals
                if "resid" in plot.columns:
                    ax_resid.scatter(
                        cam_data["JD_plot"], cam_data["resid"],
                        s=10, color=color, alpha=0.8,
                        edgecolor="black", linewidth=0.3, marker=marker, zorder=3
                    )
                
                legend_handles[cam] = pl.Line2D([], [], color=color, marker="o", linestyle="",
                                                markeredgecolor="black", markeredgewidth=0.5,
                                                label=f"{cam}")
        else:
            # No camera info - plot all together
            ax_main.errorbar(
                plot["JD_plot"], plot["mag"], yerr=plot.get("error"),
                fmt="o", ms=4, alpha=0.8, elinewidth=0.8, capsize=2,
                markeredgecolor="black", markeredgewidth=0.5,
            )
            
            if "baseline" in plot.columns:
                plot_sorted = plot.sort_values("JD_plot")
                ax_main.plot(
                    plot_sorted["JD_plot"], plot_sorted["baseline"],
                    color="orange", linestyle="-", linewidth=2, alpha=0.8,
                    label="Baseline", zorder=5
                )
            
            if "resid" in plot.columns:
                ax_resid.scatter(
                    plot["JD_plot"], plot["resid"],
                    s=10, alpha=0.8, edgecolor="black", linewidth=0.3, zorder=3
                )
        
        # Legend for main plot
        if legend_handles:
            ax_main.legend(handles=list(legend_handles.values()), title="Cameras", 
                          loc="best", fontsize="small")
        
        # Residual panel styling
        if "resid" in plot.columns:
            jd_min = plot["JD_plot"].min()
            jd_max = plot["JD_plot"].max()
            ax_resid.fill_between([jd_min, jd_max], 0.3, 100, color="lightgrey", alpha=0.5, zorder=0)
            ax_resid.fill_between([jd_min, jd_max], -0.3, -100, color="lightgrey", alpha=0.45, zorder=0)
            
            ax_resid.axhline(0.0, color="black", linestyle="--", alpha=0.4, zorder=1)
            ax_resid.axhline(0.3, color="black", linestyle="-", linewidth=0.8, zorder=1)
            ax_resid.axhline(-0.3, color="black", linestyle="-", linewidth=0.8, zorder=1)
            
            resid_min, resid_max = plot["resid"].min(), plot["resid"].max()
            pad = (resid_max - resid_min) * 0.1 if resid_max != resid_min else 0.1
            ax_resid.set_ylim(max(resid_max + pad, 0.35), min(resid_min - pad, -0.35))
        ax_resid.set_ylabel(f"{band_label} residual [mag]", fontsize=12)
        ax_resid.set_xlabel("JD", fontsize=10)
        ax_resid.grid(True, alpha=0.3)
        
        # Plot event markers - ONLY if this band passed ALL filters
        # band_idx=0 is V-band (left column), band_idx=1 is g-band (right column)
        is_significant = v_significant if band_idx == 0 else g_significant
        if is_significant:
            run_summaries = res.get("dip", {}).get("run_summaries", [])
            confirmed_count = 0
            
            if run_summaries:
                for summary in run_summaries:
                    morph = summary.get("morphology", "none").lower()
                    if morph in accepted_morphologies:
                        confirmed_count += 1
                        
                        t0 = summary.get("params", {}).get("t0")
                        if t0 is None:
                            start_jd = summary.get("start_jd")
                            end_jd = summary.get("end_jd")
                            if start_jd and end_jd:
                                t0 = (start_jd + end_jd) / 2.0
                        
                        if t0 is not None and np.isfinite(t0):
                            t0_plot = t0 - (JD_OFFSET if median_jd > 2000000 else 8000.0)
                            ax_main.axvline(t0_plot, color='red', alpha=0.7, linestyle="--", linewidth=1.5)
                            if "resid" in plot.columns:
                                ax_resid.axvline(t0_plot, color='red', alpha=0.7, linestyle="--", linewidth=1.5)

    # Plot both bands (V=1, g=0)
    plot_band(0, dfv, res_v, 1)  # V-band in left column
    plot_band(1, dfg, res_g, 0)  # g-band in right column

    # Overall title
    n_trig_v = int(res_v.get("dip", {}).get("n_dips", 0))
    n_trig_g = int(res_g.get("dip", {}).get("n_dips", 0))
    
    # Compute JD range
    jd_min = min(dfv["JD"].min() if not dfv.empty else float('inf'),
                 dfg["JD"].min() if not dfg.empty else float('inf'))
    jd_max = max(dfv["JD"].max() if not dfv.empty else float('-inf'),
                 dfg["JD"].max() if not dfg.empty else float('-inf'))
    
    if np.isfinite(jd_min) and np.isfinite(jd_max):
        jd_label = f"JD {jd_min:.0f}-{jd_max:.0f}"
    else:
        jd_label = "JD range unknown"
    
    fig.suptitle(f"{source_id} – SkyPatrol LC – {jd_label}", fontsize=14)

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    save_publication_figure(fig, plot_path, dpi=150)


def build_reproduction_report(
    candidates: Sequence[Mapping[str, object]] | None = None,
    *,
    out_dir: Path | str = "./peak_results_repro",
    out_format: str = "parquet",
    plot_format: str = "png",
    n_workers: int | None = None,
    chunk_size: int = REPRODUCE_CHUNK_SIZE,
    metrics_baseline_func=None,
    metrics_dip_threshold: float = 0.3,
    # Optional posterior threshold passthrough
    significance_threshold: float | None = None,
    p_points: int = P_POINTS,
    # Triggering mode
    trigger_mode: str = TRIGGER_MODE,
    logbf_threshold_dip: float = LOGBF_THRESHOLD_DIP,     # trigger if max log BF >= this
    logbf_threshold_jump: float = LOGBF_THRESHOLD_JUMP,
    # Probability grid bounds (matching events.py)
    p_min_dip: float | None = None,
    p_max_dip: float | None = None,
    p_min_jump: float | None = None,
    p_max_jump: float | None = None,
    # Magnitude grid
    mag_points: int = MAG_POINTS,
    mag_min_dip: float | None = None,
    mag_max_dip: float | None = None,
    mag_min_jump: float | None = None,
    mag_max_jump: float | None = None,
    # Baseline function
    baseline_func: str = BASELINE_FUNC,            # "gp", "global_median", "per_camera_median"
    # Baseline kwargs (GP kernel parameters)
    baseline_s0: float = BASELINE_S0,
    baseline_w0: float = BASELINE_W0,
    baseline_q: float = BASELINE_Q,
    baseline_jitter: float = BASELINE_JITTER,
    baseline_sigma_floor: float | None = None,
    # Run confirmation filters
    run_min_points: int = RUN_MIN_POINTS,
    max_gap_points: int = RUN_MAX_GAP_POINTS,
    run_max_gap_days: float | None = None,
    run_min_duration_days: float | None = None,
    skypatrol_dir: Path | str | None = None,
    path_prefix: Path | str | None = None,
    path_root: Path | str | None = None,
    extra_columns: Iterable[str] | None = None,
    manifest_path: Path | str | None = None,
    method: str = "bayes",
    verbose: bool = False,
    # Filter options
    skip_tags: bool = False,
    min_time_span: float = MIN_TIME_SPAN,
    min_points_per_day: float = MIN_POINTS_PER_DAY,
    min_cameras: int = MIN_CAMERAS,
    skip_vsx: bool = False,
    vsx_catalog: Path | str = VSX_RAW_CATALOG_PATH,
    vsx_max_sep: float = VSX_MAX_SEP_ARCSEC,
    min_mag_offset: float = MIN_MAG_OFFSET,
    run_filter: bool = False,
    filter_min_run_cameras: int = POST_FILTER_MIN_RUN_CAMERAS,
    filter_min_run_points: int = POST_FILTER_MIN_RUN_POINTS,
    filter_min_bayes_factor: float = MIN_BAYES_FACTOR,
    run_postprocess: bool = False,
    max_plots: int | None = None,
    run_enrich: bool = False,
    enrich_compute_ls: bool = False,
    # Morphology filtering
    accepted_morphologies: set[str] | None = None,
    # Post-processing: classification and characterization
    run_classify: bool = False,
    run_characterize: bool = False,
    gaia_cache: Path | str | None = None,
    index_file: Path | str | None = None,
    run_dust: bool = False,
    **baseline_kwargs,
) -> pd.DataFrame:
    # Default morphologies: gaussian and paczynski (reject noise/none)
    if accepted_morphologies is None:
        accepted_morphologies = {"gaussian", "paczynski", "fred"}
    manifest_df = load_manifest_df(manifest_path) if manifest_path is not None else None

    baseline_candidates = candidates if candidates is not None else brayden_candidates
    df_targets = dataframe_from_candidates(baseline_candidates)

    manifest_subset = None
    if manifest_df is not None:
        manifest_subset = manifest_df[manifest_df["source_id"].isin(df_targets["source_id"])].copy()
        if not manifest_subset.empty:
            cols = [
                col
                for col in ["source_id", "mag_bin", "lc_dir", "index_num", "index_csv", "dat_path", "dat_exists"]
                if col in manifest_subset.columns
            ]
            df_targets = df_targets.merge(manifest_subset[cols], on="source_id", how="left")
            if "mag_bin_x" in df_targets.columns and "mag_bin_y" in df_targets.columns:
                df_targets["mag_bin"] = df_targets["mag_bin_y"].fillna(df_targets["mag_bin_x"])
                df_targets = df_targets.drop(columns=["mag_bin_x", "mag_bin_y"])

    if "mag_bin" not in df_targets.columns:
        df_targets["mag_bin"] = ""
    df_targets["mag_bin"] = df_targets["mag_bin"].map(
        lambda value: "" if _canonical_candidate_id(value) is None else str(value)
    )

    target_map_dict = target_map(df_targets)

    # Priority order for light curve sources:
    # 1. SkyPatrol directory (if provided) - preferred for SkyPatrol CSVs
    # 2. Candidates 'path' column (if present) - for events.py output
    # 3. Manifest (if provided) - fallback to .dat2 files
    records_map = None

    if skypatrol_dir is not None:
        records_map = records_from_skypatrol_dir(df_targets, Path(skypatrol_dir))
        if verbose:
            n_found = sum(len(v) for v in records_map.values())
            print(f"[DEBUG] Built records_map from skypatrol_dir: {n_found} light curves found")

    if (records_map is None or not records_map) and "path" in df_targets.columns:
        records_map = records_from_candidates_with_paths(
            df_targets,
            path_prefix=path_prefix,
            path_root=path_root,
        )
        if verbose:
            n_found = sum(len(v) for v in records_map.values())
            print(f"[DEBUG] Built records_map from candidates 'path' column: {n_found} light curves found")

    if (records_map is None or not records_map) and manifest_subset is not None:
        records_map = records_from_manifest(manifest_subset)
        if verbose:
            n_found = sum(len(v) for v in records_map.values())
            print(f"[DEBUG] Built records_map from manifest: {n_found} light curves found")

    code_fingerprint = _reproduction_code_fingerprint()
    config_fingerprint = _stable_digest(
        {
            "schema_version": REPRODUCTION_SCHEMA_VERSION,
            "code_fingerprint": code_fingerprint,
            "method": method,
            "trigger_mode": trigger_mode,
            "significance_threshold": significance_threshold,
            "logbf_threshold_dip": logbf_threshold_dip,
            "logbf_threshold_jump": logbf_threshold_jump,
            "p_points": p_points,
            "p_bounds": [p_min_dip, p_max_dip, p_min_jump, p_max_jump],
            "mag_points": mag_points,
            "mag_bounds": [mag_min_dip, mag_max_dip, mag_min_jump, mag_max_jump],
            "baseline_func": baseline_func,
            "baseline": [baseline_s0, baseline_w0, baseline_q, baseline_jitter, baseline_sigma_floor],
            "run_confirmation": [run_min_points, max_gap_points, run_max_gap_days, run_min_duration_days],
            "tagging": [skip_tags, min_time_span, min_points_per_day, min_cameras, skip_vsx, str(vsx_catalog), vsx_max_sep],
            "signal_filter": min_mag_offset,
            "post_filter": [run_filter, filter_min_run_cameras, filter_min_run_points, filter_min_bayes_factor],
            "accepted_morphologies": accepted_morphologies,
            "post_processing": [run_enrich, enrich_compute_ls, run_classify, run_characterize, run_dust],
            "extra_baseline_kwargs": baseline_kwargs,
        }
    )
    candidate_fingerprint = _stable_digest(
        df_targets[[column for column in ("candidate_id", "source_id", "mag_bin", "path") if column in df_targets.columns]]
        .sort_values("source_id", kind="mergesort")
        .to_dict("records")
    )
    selected_input_records: list[dict[str, object]] = []
    record_provenance_by_source: dict[str, dict[str, object]] = {}
    if records_map:
        for mag_bin in sorted(records_map):
            for record in records_map[mag_bin]:
                file_fingerprint = _path_fingerprint(record.get("dat_path"))
                record_fingerprint = _stable_digest(
                    {
                        "source_id": _canonical_candidate_id(record.get("asas_sn_id")),
                        "mag_bin": str(record.get("mag_bin")),
                        "input_source": record.get("input_source"),
                        "file": file_fingerprint,
                    }
                )
                record["input_file_fingerprint"] = _stable_digest(file_fingerprint)
                record["input_record_fingerprint"] = record_fingerprint
                source_id = str(record.get("asas_sn_id"))
                input_record = {
                    "source_id": source_id,
                    "input_source": record.get("input_source"),
                    "dat_path": str(record.get("dat_path")),
                    "lc_dir": record.get("lc_dir"),
                    "index_num": record.get("index_num"),
                    "index_csv": record.get("index_csv"),
                    "input_record_fingerprint": record_fingerprint,
                    "input_file_fingerprint": record["input_file_fingerprint"],
                    "file": file_fingerprint,
                }
                selected_input_records.append(input_record)
                record_provenance_by_source[source_id] = input_record
    manifest_fingerprint = _stable_digest(_path_fingerprint(manifest_path)) if manifest_path is not None else None
    auxiliary_inputs = {
        "vsx_catalog": (
            _path_fingerprint(vsx_catalog)
            if manifest_subset is not None and not skip_tags and bool(records_map) and not skip_vsx
            else None
        ),
        "index_file": (
            _path_fingerprint(index_file)
            if index_file is not None and (run_filter or run_characterize or run_dust)
            else None
        ),
        "gaia_cache": (
            _path_fingerprint(gaia_cache)
            if gaia_cache is not None and run_characterize
            else None
        ),
    }
    auxiliary_input_fingerprint = _stable_digest(auxiliary_inputs)
    input_fingerprint = _stable_digest(
        {
            "candidates": candidate_fingerprint,
            "manifest": manifest_fingerprint,
            "light_curves": selected_input_records,
            "auxiliary_inputs": auxiliary_inputs,
        }
    )
    run_fingerprint = _stable_digest(
        {
            "schema_version": REPRODUCTION_SCHEMA_VERSION,
            "config_fingerprint": config_fingerprint,
            "input_fingerprint": input_fingerprint,
        }
    )

    tags_applied = manifest_subset is not None and not skip_tags and bool(records_map)
    tag_rejected_ids: set[str] = set()
    # Apply tag filters if manifest is provided and not skipped
    if tags_applied:
        if verbose:
            total_before = sum(len(v) for v in records_map.values())
            print(f"\n[TAG] Applying tag filters to {total_before} candidates...")
        
        # Prepare dataframe for tagging stage (needs 'path' column pointing to lc_dir)
        df_pre = manifest_subset.rename(columns={"lc_dir": "path"}).copy()
        
        try:
            df_filtered = apply_tags(
                df_pre,
                apply_sparse=True,
                min_time_span=min_time_span,
                min_points_per_day=min_points_per_day,
                apply_vsx=not skip_vsx,
                vsx_max_sep_arcsec=vsx_max_sep,
                vsx_crossmatch_csv=vsx_catalog,
                apply_multi_camera=True,
                min_cameras=min_cameras,
                n_workers=n_workers or WORKERS,
                show_tqdm=verbose,
            )
            
            # Update records_map to only include filtered sources
            filtered_ids = set(_validate_unique_ids(df_filtered, "source_id", "Tagged reproduction candidates").astype(str))
            original_ids = {
                str(record.get("asas_sn_id"))
                for records in records_map.values()
                for record in records
            }
            tag_rejected_ids = original_ids - filtered_ids
            for mag_bin in list(records_map.keys()):
                records_map[mag_bin] = [
                    rec for rec in records_map[mag_bin]
                    if str(rec.get("asas_sn_id")) in filtered_ids
                ]
                # Remove empty mag_bins
                if not records_map[mag_bin]:
                    del records_map[mag_bin]
            
            total_after = sum(len(v) for v in records_map.values())
            if verbose:
                print(f"[TAG] Kept {total_after}/{total_before} candidates after tagging")
                print(f"[TAG] Rejected {total_before - total_after} candidates")
        
        except Exception as e:
            raise RuntimeError("Reproduction tag stage failed; refusing to report an untagged run as tagged") from e

    baseline_func_map = {
        "gp": per_camera_gp_baseline,
        "gp_masked": per_camera_gp_baseline_masked,
        "global_median": global_median_baseline,
        "per_camera_median": per_camera_median_baseline,
    }
    if baseline_func not in baseline_func_map:
        raise ValueError(
            f"Unsupported baseline_func {baseline_func!r}; expected one of {sorted(baseline_func_map)}"
        )
    selected_baseline_func = baseline_func_map[baseline_func]
    baseline_kwargs_dict = dict(
        S0=baseline_s0,
        w0=baseline_w0,
        q=baseline_q,
        jitter=baseline_jitter,
        sigma_floor=baseline_sigma_floor,
        add_sigma_eff_col=True,
    )

    if method != "bayes":
        raise ValueError(f"Unsupported method '{method}'. Only 'bayes' is supported.")
    if records_map is None or (not records_map and not tags_applied):
        raise SystemExit("Bayesian method requires light-curve paths. Provide --manifest or --skypatrol-dir.")

    rows = []

    for mag_bin in sorted(records_map):
        for rec in records_map[mag_bin]:
            asn = _canonical_candidate_id(rec.get("asas_sn_id"))
            lc_dir = rec.get("lc_dir")
            dat_path = rec.get("dat_path")
            has_path = _canonical_candidate_id(dat_path) is not None and Path(str(dat_path)).is_file()
            load_error: str | None = None

            if verbose and not has_path:
                print(f"[DEBUG] {asn}: dat_path missing: {dat_path}")

            try:
                if str(dat_path).endswith('.csv') and has_path:

                    df_all = read_skypatrol_csv(str(dat_path))
                    # read_skypatrol_csv standardizes v_g_band to 0=g, 1=V
                    if not df_all.empty and "v_g_band" in df_all.columns:
                        dfg = df_all[df_all["v_g_band"] == 0].reset_index(drop=True)
                        dfv = df_all[df_all["v_g_band"] == 1].reset_index(drop=True)
                    else:
                        dfg, dfv = pd.DataFrame(), pd.DataFrame()
                else:
                    dfg, dfv = (
                        read_lc_dat2(asn, str(dat_path))
                        if asn and has_path
                        else (pd.DataFrame(), pd.DataFrame())
                    )
            except Exception as e:
                if verbose:
                    print(f"[DEBUG] {asn}: data load failed: {e}")
                dfg, dfv = pd.DataFrame(), pd.DataFrame()
                load_error = f"{type(e).__name__}: {e}"

            if verbose:
                print(f"[DEBUG] {asn}: loaded g={len(dfg)} rows, v={len(dfv)} rows from {dat_path}")

            def apply_triggering(result: dict, band_name: str) -> dict:
                """Normalize dip/jump trigger metadata without overriding trigger logic."""
                for kind in ("dip", "jump"):
                    if not isinstance(result, dict):
                        raise ValueError("score_lightcurve must return a mapping")
                    block = result.get(kind)
                    if not isinstance(block, dict):
                        raise ValueError(f"score_lightcurve result is missing a {kind!r} decision block")

                    block = normalize_trigger_block(
                        block,
                        kind=kind,
                        default_trigger_mode=trigger_mode,
                    )
                    idx = np.asarray(block.get("event_indices", np.array([], dtype=int)), dtype=int)
                    significant = _strict_boolean_series(pd.Series([block.get("significant")])).iloc[0]
                    if pd.isna(significant):
                        raise ValueError(f"score_lightcurve {kind}.significant is not an explicit boolean")
                    block["significant"] = bool(significant)

                    if verbose:
                        mode = str(block.get("trigger_mode", trigger_mode))
                        thr = float(block.get("trigger_threshold", np.nan))
                        trig_max = float(block.get("trigger_max", np.nan))
                        mx = block.get("max_log_bf_local", np.nan)
                        bf = block.get("bayes_factor", np.nan)
                        ct = len(idx)
                        sig = bool(block.get("significant", False))
                        thr_txt = f"{thr:.3g}" if np.isfinite(thr) else "nan"
                        trig_txt = f"{trig_max:.3g}" if np.isfinite(trig_max) else "nan"
                        mx_txt = f"{mx:.2f}" if np.isfinite(mx) else "nan"
                        bf_txt = f"{bf:.2f}" if np.isfinite(bf) else "nan"
                        print(
                            f"[DEBUG] {asn} {band_name}: {kind} mode={mode} thr={thr_txt} "
                            f"trigger_max={trig_txt} max_logBF={mx_txt} count={ct} "
                            f"significant={sig} globalBF={bf_txt}"
                        )

                    result[kind] = block

                return result

            mag_grid_dip = None
            mag_grid_jump = None
            if mag_min_dip is not None and mag_max_dip is not None:
                mag_grid_dip = np.linspace(mag_min_dip, mag_max_dip, mag_points)
            if mag_min_jump is not None and mag_max_jump is not None:
                mag_grid_jump = np.linspace(mag_min_jump, mag_max_jump, mag_points)

            def bayes(df: pd.DataFrame, band_name: str):
                dfc = clean_for_bayes(df)
                if dfc is None or dfc.empty:
                    return {
                        "dip": {"significant": False, "bayes_factor": np.nan, "max_event_prob": np.nan, "n_dips": 0, "max_log_bf_local": np.nan},
                        "jump": {"significant": False, "bayes_factor": np.nan, "max_event_prob": np.nan, "n_jumps": 0, "max_log_bf_local": np.nan},
                        "_evaluation_status": "ineligible_empty_band",
                        "_evaluation_error": None,
                    }

                try:
                    result = score_lightcurve(
                        dfc,
                        baseline_func=selected_baseline_func,
                        baseline_kwargs=baseline_kwargs_dict,
                        significance_threshold=float(significance_threshold) if significance_threshold is not None else SIGNIFICANCE_THRESHOLD,
                        p_points=int(p_points),
                        p_min_dip=p_min_dip,
                        p_max_dip=p_max_dip,
                        p_min_jump=p_min_jump,
                        p_max_jump=p_max_jump,
                        mag_points=mag_points,
                        mag_grid_dip=mag_grid_dip,
                        mag_grid_jump=mag_grid_jump,
                        trigger_mode=trigger_mode,
                        logbf_threshold_dip=logbf_threshold_dip,
                        logbf_threshold_jump=logbf_threshold_jump,
                        run_min_points=run_min_points,
                        max_gap_points=max_gap_points,
                        run_max_gap_days=run_max_gap_days,
                        run_min_duration_days=run_min_duration_days,
                    )
                    result = apply_triggering(result, band_name)
                    result["_evaluation_status"] = "ok"
                    result["_evaluation_error"] = None
                    return result
                except Exception as e:
                    if verbose:
                        print(f"[DEBUG] {asn} {band_name}: run_bayesian_significance failed: {e}")
                    return {
                        "dip": {"significant": False, "bayes_factor": np.nan, "max_event_prob": np.nan, "n_dips": 0, "max_log_bf_local": np.nan},
                        "jump": {"significant": False, "bayes_factor": np.nan, "max_event_prob": np.nan, "n_jumps": 0, "max_log_bf_local": np.nan},
                        "_evaluation_status": "error",
                        "_evaluation_error": f"{type(e).__name__}: {e}",
                    }

            res_g = bayes(dfg, "g")
            res_v = bayes(dfv, "V")
            band_errors = [
                f"{band}: {result.get('_evaluation_error')}"
                for band, result in (("g", res_g), ("v", res_v))
                if result.get("_evaluation_status") == "error"
            ]
            if not has_path:
                trial_status = "input_missing"
                evaluation_error = f"Light-curve path is missing or is not a file: {dat_path}"
            elif load_error is not None:
                trial_status = "error"
                evaluation_error = f"load_error: {load_error}"
            elif band_errors:
                trial_status = "error"
                evaluation_error = "; ".join(band_errors)
            elif res_g.get("_evaluation_status") != "ok" and res_v.get("_evaluation_status") != "ok":
                trial_status = "ineligible_empty_light_curve"
                evaluation_error = None
            else:
                trial_status = "ok"
                evaluation_error = None

            # Apply morphology filtering to results
            def count_accepted_runs(res: dict, kind: str) -> int:
                """Count runs that pass morphology filter."""
                run_summaries = res.get(kind, {}).get("run_summaries", [])
                return sum(
                    1 for s in run_summaries
                    if str(s.get("morphology") or "none").lower() in accepted_morphologies
                )

            # Update significant flags based on morphology filter
            res_g["dip"]["n_accepted"] = count_accepted_runs(res_g, "dip")
            res_g["jump"]["n_accepted"] = count_accepted_runs(res_g, "jump")
            res_v["dip"]["n_accepted"] = count_accepted_runs(res_v, "dip")
            res_v["jump"]["n_accepted"] = count_accepted_runs(res_v, "jump")

            def extract_run_info(res: dict, kind: str) -> dict:
                """Extract summary info from the best accepted run."""
                run_summaries = res.get(kind, {}).get("run_summaries", [])
                n_runs = len(run_summaries)
                n_triggered = len(res.get(kind, {}).get("event_indices", []))

                # Find best accepted run (first one that passes morphology filter)
                best_run = None
                for s in run_summaries:
                    if str(s.get("morphology") or "none").lower() in accepted_morphologies:
                        best_run = s
                        break

                if best_run is None and run_summaries:
                    # No accepted runs, use first run for info
                    best_run = run_summaries[0]

                if best_run:
                    params = best_run.get("params", {})
                    morph = str(best_run.get("morphology") or "none").lower()
                    # Extract width param based on morphology type
                    if morph == "gaussian":
                        width_param = params.get("sigma", np.nan)
                    elif morph == "paczynski":
                        width_param = params.get("tE", np.nan)
                    else:
                        width_param = np.nan
                    return {
                        "n_runs": n_runs,
                        "n_triggered": n_triggered,
                        "best_morphology": morph,
                        "best_t0": params.get("t0", np.nan),
                        "best_amplitude": params.get("amplitude", np.nan),
                        "best_duration": params.get("sigma", params.get("tau", np.nan)),
                        "best_run_n_points": best_run.get("n_points", 0),
                        "best_run_start_jd": best_run.get("start_jd", np.nan),
                        "best_run_end_jd": best_run.get("end_jd", np.nan),
                        # New fields from events.py
                        "best_delta_bic": best_run.get("delta_bic_null", np.nan),
                        "best_width_param": width_param,
                        "best_symmetry_score": best_run.get("symmetry_score", np.nan),
                    }
                return {
                    "n_runs": n_runs,
                    "n_triggered": n_triggered,
                    "best_morphology": "none",
                    "best_t0": np.nan,
                    "best_amplitude": np.nan,
                    "best_duration": np.nan,
                    "best_run_n_points": 0,
                    "best_run_start_jd": np.nan,
                    "best_run_end_jd": np.nan,
                    # New fields from events.py
                    "best_delta_bic": np.nan,
                    "best_width_param": np.nan,
                    "best_symmetry_score": np.nan,
                }

            g_run_info = extract_run_info(res_g, "dip")
            v_run_info = extract_run_info(res_v, "dip")

            # Light curve statistics
            g_n_points = len(clean_for_bayes(dfg)) if not dfg.empty else 0
            v_n_points = len(clean_for_bayes(dfv)) if not dfv.empty else 0
            g_time_span = float(dfg["JD"].max() - dfg["JD"].min()) if not dfg.empty and len(dfg) > 1 else 0.0
            v_time_span = float(dfv["JD"].max() - dfv["JD"].min()) if not dfv.empty and len(dfv) > 1 else 0.0

            # Additional timing stats (from events.py)
            dfg_clean = clean_for_bayes(dfg) if not dfg.empty else pd.DataFrame()
            dfv_clean = clean_for_bayes(dfv) if not dfv.empty else pd.DataFrame()

            g_jd_first = float(dfg_clean["JD"].min()) if not dfg_clean.empty else np.nan
            g_jd_last = float(dfg_clean["JD"].max()) if not dfg_clean.empty else np.nan
            g_cadence_median_days = float(median_dt(dfg_clean["JD"].to_numpy())) if not dfg_clean.empty else np.nan

            v_jd_first = float(dfv_clean["JD"].min()) if not dfv_clean.empty else np.nan
            v_jd_last = float(dfv_clean["JD"].max()) if not dfv_clean.empty else np.nan
            v_cadence_median_days = float(median_dt(dfv_clean["JD"].to_numpy())) if not dfv_clean.empty else np.nan

            # Camera statistics (from events.py)
            def get_camera_stats(df_band: pd.DataFrame) -> dict:
                """Extract camera statistics for a band."""
                if df_band.empty or "camera#" not in df_band.columns:
                    return {
                        "n_cameras": 0,
                        "camera_ids": "",
                        "camera_min_points": 0,
                        "camera_max_points": 0,
                    }
                cams = df_band["camera#"].dropna()
                if len(cams) == 0:
                    return {
                        "n_cameras": 0,
                        "camera_ids": "",
                        "camera_min_points": 0,
                        "camera_max_points": 0,
                    }
                unique_cams = np.unique(cams.astype(str))
                cam_counts = cams.value_counts()
                return {
                    "n_cameras": int(unique_cams.size),
                    "camera_ids": ",".join(sorted(unique_cams)),
                    "camera_min_points": int(cam_counts.min()) if len(cam_counts) else 0,
                    "camera_max_points": int(cam_counts.max()) if len(cam_counts) else 0,
                }

            g_cam_stats = get_camera_stats(dfg_clean)
            v_cam_stats = get_camera_stats(dfv_clean)

            # Dipper score (from events.py) - only computed if significant
            def compute_dipper_stats(df_band: pd.DataFrame, res_band: dict, kind: str = "dip") -> dict:
                """Compute dipper score for a band."""
                if df_band.empty or not res_band.get(kind, {}).get("significant", False):
                    return {
                        "dipper_score": 0.0,
                        "dipper_n_dips": 0,
                        "dipper_n_valid_dips": 0,
                        "dipper_score_status": "not_applicable",
                        "dipper_score_error": None,
                    }
                try:
                    df_base = res_band.get("df_base")
                    if df_base is not None and "baseline" in df_base.columns:
                        baseline_mags = df_base["baseline"].to_numpy()
                    else:
                        baseline_mags = None
                    score, events = compute_event_score(df_band, event_type='dip', baseline_mags=baseline_mags)
                    return {
                        "dipper_score": float(score),
                        "dipper_n_dips": int(len(events)),
                        "dipper_n_valid_dips": int(sum(1 for e in events if e.valid)),
                        "dipper_score_status": "ok",
                        "dipper_score_error": None,
                    }
                except Exception as exc:
                    return {
                        "dipper_score": np.nan,
                        "dipper_n_dips": pd.NA,
                        "dipper_n_valid_dips": pd.NA,
                        "dipper_score_status": "error",
                        "dipper_score_error": f"{type(exc).__name__}: {exc}",
                    }

            g_dipper_stats = compute_dipper_stats(dfg_clean, res_g)
            v_dipper_stats = compute_dipper_stats(dfv_clean, res_v)

            # Also extract jump run info
            g_jump_run_info = extract_run_info(res_g, "jump")
            v_jump_run_info = extract_run_info(res_v, "jump")

            # Override significant flag if no runs pass morphology filter
            if res_g["dip"]["n_accepted"] == 0:
                res_g["dip"]["significant"] = False
            if res_g["jump"]["n_accepted"] == 0:
                res_g["jump"]["significant"] = False
            if res_v["dip"]["n_accepted"] == 0:
                res_v["dip"]["significant"] = False
            if res_v["jump"]["n_accepted"] == 0:
                res_v["jump"]["significant"] = False

            # Determine rejection reason (first filter that fails)
            def get_rejection_reason(res: dict, kind: str) -> str | None:
                """Determine why a detection was rejected."""
                block = res.get(kind, {})

                # Check if any triggers
                n_triggers = len(block.get("event_indices", []))

                if n_triggers == 0:
                    return "no_triggers"

                # Check if runs formed
                n_runs = block.get("n_runs", 0)
                if n_runs == 0:
                    return "run_confirmation"

                # Check if morphology passed
                n_accepted = block.get("n_accepted", 0)
                if n_accepted == 0:
                    return "morphology"

                # Passed all filters
                return None

            g_dip_reason = get_rejection_reason(res_g, "dip")
            v_dip_reason = get_rejection_reason(res_v, "dip")

            # Combined rejection reason: None if either band passes, else first failure
            if res_g["dip"].get("significant") or res_v["dip"].get("significant"):
                combined_rejection = None
            else:
                # Report the "furthest" rejection (morphology > run_confirmation > no_triggers)
                priority = {"morphology": 3, "run_confirmation": 2, "no_triggers": 1, None: 0}
                if priority.get(g_dip_reason, 0) >= priority.get(v_dip_reason, 0):
                    combined_rejection = g_dip_reason
                else:
                    combined_rejection = v_dip_reason

            plot_status = "not_requested"
            plot_error: str | None = None
            if out_dir and trial_status == "ok":
                plot_path = Path(out_dir) / f"{asn}_dips.{plot_format}"
                try:
                    plot_light_curve_with_dips(
                        clean_for_bayes(dfg),
                        clean_for_bayes(dfv),
                        res_g,
                        res_v,
                        str(asn),
                        plot_path,
                        accepted_morphologies=accepted_morphologies,
                        g_significant=res_g["dip"].get("significant", False),
                        v_significant=res_v["dip"].get("significant", False),
                    )
                    plot_status = "ok"
                except Exception as exc:
                    plot_status = "error"
                    plot_error = f"{type(exc).__name__}: {exc}"

            rows.append(
                {
                    "candidate_id": next(
                        (
                            str(value)
                            for value in df_targets.loc[df_targets["source_id"].astype(str).eq(str(asn)), "candidate_id"]
                            if _canonical_candidate_id(value) is not None
                        ),
                        str(asn),
                    ),
                    "mag_bin": str(rec.get("mag_bin")),
                    "asas_sn_id": asn,
                    "index_num": rec.get("index_num"),
                    "index_csv": rec.get("index_csv"),
                    "lc_dir": lc_dir,
                    "dat_path": dat_path,
                    "input_source": rec.get("input_source", "unknown"),
                    "input_record_fingerprint": rec.get("input_record_fingerprint"),
                    "input_file_fingerprint": rec.get("input_file_fingerprint"),
                    "input_fingerprint": input_fingerprint,
                    "candidate_set_fingerprint": candidate_fingerprint,
                    "manifest_fingerprint": manifest_fingerprint,
                    "auxiliary_input_fingerprint": auxiliary_input_fingerprint,
                    "config_fingerprint": config_fingerprint,
                    "code_fingerprint": code_fingerprint,
                    "run_fingerprint": run_fingerprint,
                    "reproduction_schema_version": REPRODUCTION_SCHEMA_VERSION,
                    "trial_status": trial_status,
                    "evaluation_error": evaluation_error,
                    "plot_status": plot_status,
                    "plot_error": plot_error,

                    "g_bayes_dip_significant": bool(res_g["dip"].get("significant", False)),
                    "v_bayes_dip_significant": bool(res_v["dip"].get("significant", False)),
                    "g_bayes_jump_significant": bool(res_g["jump"].get("significant", False)),
                    "v_bayes_jump_significant": bool(res_v["jump"].get("significant", False)),

                    "g_bayes_dip_max_prob": float(res_g["dip"].get("max_event_prob", np.nan)),
                    "v_bayes_dip_max_prob": float(res_v["dip"].get("max_event_prob", np.nan)),

                    "g_bayes_dip_bayes_factor": float(res_g["dip"].get("bayes_factor", np.nan)),
                    "v_bayes_dip_bayes_factor": float(res_v["dip"].get("bayes_factor", np.nan)),
                    "g_bayes_jump_bayes_factor": float(res_g["jump"].get("bayes_factor", np.nan)),
                    "v_bayes_jump_bayes_factor": float(res_v["jump"].get("bayes_factor", np.nan)),

                    "g_bayes_dip_max_logbf": float(res_g["dip"].get("max_log_bf_local", np.nan)),
                    "v_bayes_dip_max_logbf": float(res_v["dip"].get("max_log_bf_local", np.nan)),
                    "g_bayes_jump_max_logbf": float(res_g["jump"].get("max_log_bf_local", np.nan)),
                    "v_bayes_jump_max_logbf": float(res_v["jump"].get("max_log_bf_local", np.nan)),

                    "g_bayes_n_dips": int(res_g["dip"].get("n_dips", 0)),
                    "v_bayes_n_dips": int(res_v["dip"].get("n_dips", 0)),
                    "g_bayes_n_jumps": int(res_g["jump"].get("n_jumps", 0)),
                    "v_bayes_n_jumps": int(res_v["jump"].get("n_jumps", 0)),

                    # Rejection tracking
                    "g_rejection_reason": g_dip_reason,
                    "v_rejection_reason": v_dip_reason,
                    "rejection_reason": combined_rejection,

                    # Light curve statistics
                    "g_n_points": g_n_points,
                    "v_n_points": v_n_points,
                    "g_time_span": g_time_span,
                    "v_time_span": v_time_span,

                    # Timing stats (from events.py)
                    "g_jd_first": g_jd_first,
                    "v_jd_first": v_jd_first,
                    "g_jd_last": g_jd_last,
                    "v_jd_last": v_jd_last,
                    "g_cadence_median_days": g_cadence_median_days,
                    "v_cadence_median_days": v_cadence_median_days,

                    # Run details - g band dips
                    "g_n_runs": g_run_info["n_runs"],
                    "g_n_triggered": g_run_info["n_triggered"],
                    "g_best_morphology": g_run_info["best_morphology"],
                    "g_best_t0": g_run_info["best_t0"],
                    "g_best_amplitude": g_run_info["best_amplitude"],
                    "g_best_duration": g_run_info["best_duration"],
                    "g_best_run_n_points": g_run_info["best_run_n_points"],
                    "g_best_run_start_jd": g_run_info["best_run_start_jd"],
                    "g_best_run_end_jd": g_run_info["best_run_end_jd"],
                    # New morphology fields (from events.py)
                    "g_dip_best_delta_bic": g_run_info["best_delta_bic"],
                    "g_dip_best_width_param": g_run_info["best_width_param"],
                    "g_dip_symmetry_score": g_run_info["best_symmetry_score"],

                    # Run details - V band dips
                    "v_n_runs": v_run_info["n_runs"],
                    "v_n_triggered": v_run_info["n_triggered"],
                    "v_best_morphology": v_run_info["best_morphology"],
                    "v_best_t0": v_run_info["best_t0"],
                    "v_best_amplitude": v_run_info["best_amplitude"],
                    "v_best_duration": v_run_info["best_duration"],
                    "v_best_run_n_points": v_run_info["best_run_n_points"],
                    "v_best_run_start_jd": v_run_info["best_run_start_jd"],
                    "v_best_run_end_jd": v_run_info["best_run_end_jd"],
                    # New morphology fields (from events.py)
                    "v_dip_best_delta_bic": v_run_info["best_delta_bic"],
                    "v_dip_best_width_param": v_run_info["best_width_param"],
                    "v_dip_symmetry_score": v_run_info["best_symmetry_score"],

                    # Run details - g band jumps (from events.py)
                    "g_jump_n_runs": g_jump_run_info["n_runs"],
                    "g_jump_n_triggered": g_jump_run_info["n_triggered"],
                    "g_jump_best_morphology": g_jump_run_info["best_morphology"],
                    "g_jump_best_delta_bic": g_jump_run_info["best_delta_bic"],
                    "g_jump_best_width_param": g_jump_run_info["best_width_param"],

                    # Run details - V band jumps (from events.py)
                    "v_jump_n_runs": v_jump_run_info["n_runs"],
                    "v_jump_n_triggered": v_jump_run_info["n_triggered"],
                    "v_jump_best_morphology": v_jump_run_info["best_morphology"],
                    "v_jump_best_delta_bic": v_jump_run_info["best_delta_bic"],
                    "v_jump_best_width_param": v_jump_run_info["best_width_param"],

                    # Run count/max stats (from events.py - using dip stats)
                    "g_dip_run_count": int(res_g["dip"].get("n_runs", 0)),
                    "v_dip_run_count": int(res_v["dip"].get("n_runs", 0)),
                    "g_jump_run_count": int(res_g["jump"].get("n_runs", 0)),
                    "v_jump_run_count": int(res_v["jump"].get("n_runs", 0)),
                    "g_dip_max_run_points": int(res_g["dip"].get("max_run_points", 0)),
                    "v_dip_max_run_points": int(res_v["dip"].get("max_run_points", 0)),
                    "g_dip_max_run_duration": float(res_g["dip"].get("max_run_duration", np.nan)),
                    "v_dip_max_run_duration": float(res_v["dip"].get("max_run_duration", np.nan)),
                    "g_dip_max_run_sum": float(res_g["dip"].get("max_run_sum", np.nan)),
                    "v_dip_max_run_sum": float(res_v["dip"].get("max_run_sum", np.nan)),
                    "g_dip_max_run_max": float(res_g["dip"].get("max_run_max", np.nan)),
                    "v_dip_max_run_max": float(res_v["dip"].get("max_run_max", np.nan)),
                    "g_dip_max_run_cameras": int(res_g["dip"].get("max_run_cameras", 0)),
                    "v_dip_max_run_cameras": int(res_v["dip"].get("max_run_cameras", 0)),
                    "g_jump_max_run_points": int(res_g["jump"].get("max_run_points", 0)),
                    "v_jump_max_run_points": int(res_v["jump"].get("max_run_points", 0)),
                    "g_jump_max_run_cameras": int(res_g["jump"].get("max_run_cameras", 0)),
                    "v_jump_max_run_cameras": int(res_v["jump"].get("max_run_cameras", 0)),

                    # Detection parameters (from events.py)
                    "g_baseline_mag": float(res_g["dip"].get("baseline_mag", np.nan)),
                    "v_baseline_mag": float(res_v["dip"].get("baseline_mag", np.nan)),
                    "g_dip_best_p": float(res_g["dip"].get("best_p", np.nan)),
                    "v_dip_best_p": float(res_v["dip"].get("best_p", np.nan)),
                    "g_jump_best_p": float(res_g["jump"].get("best_p", np.nan)),
                    "v_jump_best_p": float(res_v["jump"].get("best_p", np.nan)),
                    "g_dip_best_mag_event": float(res_g["dip"].get("best_mag_event", np.nan)),
                    "v_dip_best_mag_event": float(res_v["dip"].get("best_mag_event", np.nan)),
                    "g_jump_best_mag_event": float(res_g["jump"].get("best_mag_event", np.nan)),
                    "v_jump_best_mag_event": float(res_v["jump"].get("best_mag_event", np.nan)),
                    "g_dip_trigger_max": float(res_g["dip"].get("trigger_max", np.nan)),
                    "v_dip_trigger_max": float(res_v["dip"].get("trigger_max", np.nan)),
                    "g_jump_trigger_max": float(res_g["jump"].get("trigger_max", np.nan)),
                    "v_jump_trigger_max": float(res_v["jump"].get("trigger_max", np.nan)),

                    # Camera statistics (from events.py)
                    "g_n_cameras": g_cam_stats["n_cameras"],
                    "v_n_cameras": v_cam_stats["n_cameras"],
                    "g_camera_ids": g_cam_stats["camera_ids"],
                    "v_camera_ids": v_cam_stats["camera_ids"],
                    "g_camera_min_points": g_cam_stats["camera_min_points"],
                    "v_camera_min_points": v_cam_stats["camera_min_points"],
                    "g_camera_max_points": g_cam_stats["camera_max_points"],
                    "v_camera_max_points": v_cam_stats["camera_max_points"],

                    # Dipper score (from events.py)
                    "g_dipper_score": g_dipper_stats["dipper_score"],
                    "v_dipper_score": v_dipper_stats["dipper_score"],
                    "g_dipper_n_dips": g_dipper_stats["dipper_n_dips"],
                    "v_dipper_n_dips": v_dipper_stats["dipper_n_dips"],
                    "g_dipper_n_valid_dips": g_dipper_stats["dipper_n_valid_dips"],
                    "v_dipper_n_valid_dips": v_dipper_stats["dipper_n_valid_dips"],
                    "g_dipper_score_status": g_dipper_stats["dipper_score_status"],
                    "v_dipper_score_status": v_dipper_stats["dipper_score_status"],
                    "g_dipper_score_error": g_dipper_stats["dipper_score_error"],
                    "v_dipper_score_error": v_dipper_stats["dipper_score_error"],

                    # System info (from events.py)
                    "g_used_sigma_eff": bool(res_g["dip"].get("used_sigma_eff", False)),
                    "v_used_sigma_eff": bool(res_v["dip"].get("used_sigma_eff", False)),
                    "g_baseline_source": str(res_g["dip"].get("baseline_source", "unknown")),
                    "v_baseline_source": str(res_v["dip"].get("baseline_source", "unknown")),
                    "g_trigger_mode": str(res_g["dip"].get("trigger_mode", trigger_mode)),
                    "v_trigger_mode": str(res_v["dip"].get("trigger_mode", trigger_mode)),
                    "g_dip_trigger_threshold": float(res_g["dip"].get("trigger_threshold", np.nan)),
                    "v_dip_trigger_threshold": float(res_v["dip"].get("trigger_threshold", np.nan)),
                    "g_jump_trigger_threshold": float(res_g["jump"].get("trigger_threshold", np.nan)),
                    "v_jump_trigger_threshold": float(res_v["jump"].get("trigger_threshold", np.nan)),
                }
            )
    if rows is None:
        rows_df = pd.DataFrame(columns=["source_id", "mag_bin"])
    elif isinstance(rows, list):
        rows_df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["source_id", "mag_bin"])
    elif isinstance(rows, pd.DataFrame):
        rows_df = rows.copy() if not rows.empty else pd.DataFrame(columns=["source_id", "mag_bin"])
    else:
        rows_df = pd.DataFrame(columns=["source_id", "mag_bin"])

    if "path" not in rows_df.columns and "dat_path" in rows_df.columns:
        rows_df["path"] = rows_df["dat_path"]

    # Apply signal amplitude filter if enabled
    # Instead of removing rows, mark rejection_reason for rows that fail
    if not rows_df.empty and min_mag_offset > 0:
        if verbose:
            print(f"\n[SIGNAL-FILTER] Applying signal amplitude filter (min_mag_offset={min_mag_offset})...")
            n_before = len(rows_df)

        try:
            rows_df = _apply_reproduction_signal_amplitude(
                rows_df,
                min_mag_offset=min_mag_offset,
            )

            if verbose:
                n_rejected = int(
                    _strict_boolean_series(rows_df["failed_signal_amplitude"])
                    .fillna(False)
                    .sum()
                )
                print(f"[SIGNAL-FILTER] Marked {n_rejected}/{n_before} as rejected by signal amplitude filter")

        except Exception as e:
            raise RuntimeError("Reproduction signal-amplitude stage failed; refusing unfiltered science output") from e

    # ==========================================================================
    # Post-processing: Load index to get Gaia IDs and coordinates
    # ==========================================================================
    if (run_filter or run_characterize or run_dust) and not rows_df.empty and index_file:
        index_path = Path(index_file) if index_file else None
        if index_path and index_path.exists():
            if verbose:
                print(f"\n[INDEX] Loading ASAS-SN index from {index_path}...")

            try:
                # Load only needed columns from index
                index_cols = ["asas_sn_id", "gaia_id", "ra_deg", "dec_deg"]
                index_df = read_parquet_table(index_path)
                index_df = index_df[[col for col in index_cols if col in index_df.columns]].copy()

                # Ensure asas_sn_id is string for matching
                index_df["asas_sn_id"] = _validate_unique_ids(index_df, "asas_sn_id", "Reproduction index")

                # The source_id in rows_df is actually asas_sn_id
                if "asas_sn_id" in rows_df.columns:
                    merge_col = "asas_sn_id"
                    rows_df[merge_col] = rows_df[merge_col].astype(str)
                elif "source_id" in rows_df.columns:
                    merge_col = "source_id"
                    rows_df["_merge_id"] = rows_df[merge_col].astype(str)
                    merge_col = "_merge_id"
                else:
                    merge_col = None

                if merge_col:
                    rows_df = rows_df.merge(
                        index_df,
                        left_on=merge_col,
                        right_on="asas_sn_id",
                        how="left",
                        suffixes=("", "_idx")
                    )
                    # Clean up merge columns
                    rows_df = rows_df.drop(columns=["_merge_id", "asas_sn_id_idx"], errors="ignore")

                    if verbose:
                        n_with_gaia = rows_df["gaia_id"].notna().sum() if "gaia_id" in rows_df.columns else 0
                        print(f"[INDEX] Matched {n_with_gaia}/{len(rows_df)} sources to index (got gaia_id, ra_deg, dec_deg)")

            except Exception as e:
                raise RuntimeError("Reproduction index merge failed") from e
        else:
            raise FileNotFoundError(f"Requested reproduction index file does not exist: {index_path}")

    # ==========================================================================
    # Post-processing: Filters (apply_filters)
    # ==========================================================================
    if run_filter and not rows_df.empty:
        if verbose:
            print("\n[FILTER] Applying filters...")

        try:
            df_for_post = rows_df.copy()
            if "path" not in df_for_post.columns and "dat_path" in df_for_post.columns:
                df_for_post["path"] = df_for_post["dat_path"]
            if "lc_path" not in df_for_post.columns and "dat_path" in df_for_post.columns:
                df_for_post["lc_path"] = df_for_post["dat_path"]
            df_for_post["_reproduction_order"] = np.arange(len(df_for_post), dtype=int)
            ok_mask = df_for_post.get("trial_status", pd.Series("ok", index=df_for_post.index)).astype("string").eq("ok")
            df_not_evaluated = df_for_post.loc[~ok_mask].copy()
            df_for_post = df_for_post.loc[ok_mask].copy()

            def _combine(primary: str, secondary: str):
                if primary in df_for_post.columns:
                    if secondary in df_for_post.columns:
                        return df_for_post[primary].fillna(df_for_post[secondary])
                    return df_for_post[primary]
                if secondary in df_for_post.columns:
                    return df_for_post[secondary]
                return np.nan

            def _max_cols(a: str, b: str):
                if a in df_for_post.columns and b in df_for_post.columns:
                    return df_for_post[[a, b]].max(axis=1)
                if a in df_for_post.columns:
                    return df_for_post[a]
                if b in df_for_post.columns:
                    return df_for_post[b]
                return np.nan

            df_for_post["dip_bayes_factor"] = _max_cols("g_bayes_dip_bayes_factor", "v_bayes_dip_bayes_factor")
            df_for_post["jump_bayes_factor"] = _max_cols("g_bayes_jump_bayes_factor", "v_bayes_jump_bayes_factor")
            df_for_post["dip_run_count"] = _max_cols("g_dip_run_count", "v_dip_run_count")
            df_for_post["jump_run_count"] = _max_cols("g_jump_run_count", "v_jump_run_count")
            df_for_post["dip_max_run_points"] = _max_cols("g_dip_max_run_points", "v_dip_max_run_points")
            df_for_post["jump_max_run_points"] = _max_cols("g_jump_max_run_points", "v_jump_max_run_points")
            df_for_post["dip_max_run_cameras"] = _max_cols("g_dip_max_run_cameras", "v_dip_max_run_cameras")
            df_for_post["jump_max_run_cameras"] = _max_cols("g_jump_max_run_cameras", "v_jump_max_run_cameras")
            df_for_post["dip_best_morph"] = _combine("g_best_morphology", "v_best_morphology")
            df_for_post["jump_best_morph"] = _combine("g_jump_best_morphology", "v_jump_best_morphology")
            df_for_post["dip_best_delta_bic"] = _max_cols("g_dip_best_delta_bic", "v_dip_best_delta_bic")
            df_for_post["jump_best_delta_bic"] = _max_cols("g_jump_best_delta_bic", "v_jump_best_delta_bic")
            df_for_post["dipper_score"] = _max_cols("g_dipper_score", "v_dipper_score")
            # Required by filter_posterior_strength
            df_for_post["dip_max_log_bf_local"] = _max_cols("g_bayes_dip_max_logbf", "v_bayes_dip_max_logbf")
            df_for_post["jump_max_log_bf_local"] = _max_cols("g_bayes_jump_max_logbf", "v_bayes_jump_max_logbf")
            df_for_post["dip_max_run_duration"] = _max_cols("g_dip_max_run_duration", "v_dip_max_run_duration")
            df_for_post["jump_max_run_duration"] = _max_cols("g_jump_max_run_duration", "v_jump_max_run_duration")
            df_for_post["dip_max_run_points"] = _max_cols("g_dip_max_run_points", "v_dip_max_run_points")
            df_for_post["jump_max_run_points"] = _max_cols("g_jump_max_run_points", "v_jump_max_run_points")

            if df_for_post.empty:
                rows_df = df_not_evaluated.drop(columns=["_reproduction_order"], errors="ignore")
            else:
                filtered_ok = apply_filters(
                    df_for_post,
                    min_bayes_factor=filter_min_bayes_factor,
                    apply_run_robustness=True,
                    min_run_points=filter_min_run_points,
                    min_run_cameras=filter_min_run_cameras,
                    show_tqdm=verbose,
                    verbose=verbose,
                )
                rows_df = (
                    pd.concat([filtered_ok, df_not_evaluated], ignore_index=True, sort=False)
                    .sort_values("_reproduction_order", kind="mergesort")
                    .drop(columns=["_reproduction_order"], errors="ignore")
                    .reset_index(drop=True)
                )
        except Exception as e:
            raise RuntimeError("Reproduction post-filter stage failed; refusing unfiltered science output") from e

    # ==========================================================================
    # Post-processing: Postprocess plots (optional)
    # ==========================================================================
    if run_postprocess:
        if not run_filter:
            raise ValueError("run_postprocess=True requires run_filter=True")
        elif not rows_df.empty:
            if verbose:
                print("\n[POSTPROCESS] Generating postprocess plots...")
            try:
                baseline_map = {
                    "gp": "per_camera_gp",
                    "gp_masked": "gp_masked",
                    "global_median": "global_median",
                    "per_camera_median": "per_camera_median",
                }
                baseline_name = baseline_map.get(str(baseline_func), "per_camera_gp")
                postprocess_dir = Path(out_dir) / "postprocess"
                postprocess_dir.mkdir(parents=True, exist_ok=True)
                plot_mask = rows_df.get("trial_status", pd.Series("ok", index=rows_df.index)).astype("string").eq("ok")
                if "failed_any" in rows_df.columns:
                    failed = _strict_boolean_series(rows_df["failed_any"])
                    plot_mask &= failed.eq(False).fillna(False)
                plot_input = rows_df.loc[plot_mask].copy()
                if not plot_input.empty:
                    plot_passing_candidates(
                        plot_input,
                        postprocess_dir,
                        baseline=baseline_name,
                        logbf_threshold_dip=logbf_threshold_dip,
                        logbf_threshold_jump=logbf_threshold_jump,
                        format=plot_format,
                        max_plots=max_plots,
                        show_tqdm=verbose,
                    )
                    rows_df["postprocess_plot_status"] = "ok"
                else:
                    rows_df["postprocess_plot_status"] = "ineligible_no_passing_candidates"
                rows_df["postprocess_plot_error"] = None
            except Exception as e:
                rows_df["postprocess_plot_status"] = "error"
                rows_df["postprocess_plot_error"] = f"{type(e).__name__}: {e}"

    # ==========================================================================
    # Post-processing: Enrich with compute_stats (optional)
    # ==========================================================================
    if run_enrich:
        if not run_filter:
            raise ValueError("run_enrich=True requires run_filter=True")
        elif not rows_df.empty:
            if verbose:
                print("\n[ENRICH] Enriching with light curve stats...")
            try:
                rows_df = rows_df.copy()
                rows_df["_reproduction_order"] = np.arange(len(rows_df), dtype=int)
                eligible = rows_df.get("trial_status", pd.Series("ok", index=rows_df.index)).astype("string").eq("ok")
                if "failed_any" in rows_df.columns:
                    failed = _strict_boolean_series(rows_df["failed_any"])
                    eligible &= failed.eq(False).fillna(False)
                    df_passed = rows_df.loc[eligible].copy()
                else:
                    df_passed = rows_df.loc[eligible].copy()
                df_passthrough = rows_df.loc[~eligible].copy()

                # Separate pass-through rows from rows needing compute_stats,
                # tracking original position so we can reconstruct order afterwards.
                ordered_results: list[dict | None] = [None] * len(df_passed)
                pending_tasks: list[tuple] = []
                pending_indices: list[int] = []
                for pos, (_, row) in enumerate(df_passed.iterrows()):
                    row_dict = row.to_dict()
                    path_val = row.get("path") or row.get("dat_path")
                    if not path_val:
                        ordered_results[pos] = row_dict
                        continue
                    lc_path = Path(str(path_val))
                    asassn_id = str(row.get("asas_sn_id") or row.get("source_id") or lc_path.stem)
                    pending_tasks.append((row_dict, asassn_id, str(lc_path), enrich_compute_ls, lc_path.suffix.lstrip(".") or None))
                    pending_indices.append(pos)

                n_enrich_workers = max(1, n_workers or WORKERS)
                if verbose:
                    print(f"[ENRICH] {len(pending_tasks)} candidates → compute_stats ({n_enrich_workers} workers)...")

                with ProcessPoolExecutor(max_workers=n_enrich_workers) as executor:
                    for pos, result in zip(pending_indices, tqdm(
                        executor.map(_enrich_row_worker, pending_tasks),
                        total=len(pending_tasks),
                        desc="compute_stats",
                        disable=not verbose,
                    )):
                        ordered_results[pos] = result

                enriched = pd.DataFrame(ordered_results)
                if not enriched.empty:
                    enriched["enrichment_status"] = np.where(
                        enriched.get("stats_compute_status", pd.Series(index=enriched.index, dtype=object)).notna(),
                        enriched.get("stats_compute_status", pd.Series("ok", index=enriched.index)),
                        "error",
                    )
                    enriched["enrichment_error"] = np.where(
                        enriched["enrichment_status"].astype(str).eq("error"),
                        enriched.get("stats_compute_error", pd.Series(None, index=enriched.index)).fillna(
                            "compute_stats worker returned no explicit status"
                        ),
                        enriched.get("stats_compute_error", pd.Series(None, index=enriched.index)),
                    )
                rows_df = (
                    pd.concat([enriched, df_passthrough], ignore_index=True, sort=False)
                    .sort_values("_reproduction_order", kind="mergesort")
                    .drop(columns=["_reproduction_order"], errors="ignore")
                    .reset_index(drop=True)
                )
            except Exception as e:
                raise RuntimeError("Requested reproduction enrichment failed") from e

    # Load index logic has been moved before apply_filters.

    # ==========================================================================
    # Post-processing: Characterization (Gaia DR3, stellar params, photometry)
    # ==========================================================================
    if run_characterize and not rows_df.empty:
        char_mask = rows_df.get("trial_status", pd.Series("ok", index=rows_df.index)).astype("string").eq("ok")
        if "failed_any" in rows_df.columns:
            failed = _strict_boolean_series(rows_df["failed_any"])
            char_mask &= failed.eq(False).fillna(False)
        df_char = rows_df.loc[char_mask].copy()
        if df_char.empty:
            if verbose:
                print("\n[CHARACTERIZE] No rows passed failed_any filter. Skipping Gaia DR3 query.")
        elif verbose:
            print(f"\n[CHARACTERIZE] Querying Gaia DR3 for {len(df_char)} sources...")

        try:
            # Get Gaia source IDs - prefer gaia_id from index, fall back to source_id
            gaia_ids = []
            if "gaia_id" in df_char.columns:
                gaia_ids = df_char["gaia_id"].dropna().astype(int).astype(str).tolist()
            elif "gaia_source_id" in df_char.columns:
                gaia_ids = df_char["gaia_source_id"].dropna().astype(str).tolist()

            if gaia_ids:
                if verbose:
                    print(f"[CHARACTERIZE] Found {len(gaia_ids)} Gaia IDs to query")

                gaia_df = query_gaia_by_ids(
                    gaia_ids,
                    cache_file=str(gaia_cache) if gaia_cache else None,
                )

                if not gaia_df.empty:
                    # Merge Gaia data with results by gaia_id
                    gaia_df["source_id"] = _validate_unique_ids(gaia_df, "source_id", "Gaia characterization result")

                    # Select key columns from Gaia
                    gaia_cols = [
                        "source_id", "ra", "dec", "parallax", "parallax_error", "ruwe",
                        "pmra", "pmdec", "phot_g_mean_mag", "bp_rp",
                        "teff_gspphot", "logg_gspphot", "mh_gspphot",
                        "distance_gspphot", "ag_gspphot",
                        "tmass_j", "tmass_h", "tmass_k",
                        "w1", "w2", "w3", "w4",
                    ]
                    gaia_cols_present = [c for c in gaia_cols if c in gaia_df.columns]
                    gaia_subset = gaia_df[gaia_cols_present].copy()

                    # Rename columns to gaia_ prefix
                    rename_map = {c: f"gaia_{c}" for c in gaia_cols_present if c != "source_id"}
                    gaia_subset = gaia_subset.rename(columns=rename_map)

                    # Merge by gaia_id
                    rows_df["_gaia_id_str"] = rows_df["gaia_id"].astype(str) if "gaia_id" in rows_df.columns else ""
                    rows_df = rows_df.merge(
                        gaia_subset,
                        left_on="_gaia_id_str",
                        right_on="source_id",
                        how="left",
                        suffixes=("", "_gaia_query"),
                        validate="many_to_one",
                    )
                    rows_df = rows_df.drop(columns=["_gaia_id_str", "source_id_gaia_query", "source_id"], errors="ignore")

                    if verbose:
                        n_matched = rows_df["gaia_ruwe"].notna().sum() if "gaia_ruwe" in rows_df.columns else 0
                        print(f"[CHARACTERIZE] Matched {n_matched}/{len(rows_df)} sources to Gaia DR3")
                    rows_df["characterization_status"] = np.where(
                        rows_df.get("gaia_ruwe", pd.Series(index=rows_df.index, dtype=float)).notna(),
                        "matched",
                        "no_match",
                    )
                else:
                    rows_df["characterization_status"] = "no_matches_returned"
            else:
                rows_df["characterization_status"] = "ineligible_no_gaia_id"

        except Exception as e:
            raise RuntimeError("Requested reproduction Gaia characterization failed") from e

    # ==========================================================================
    # Post-processing: Dust extinction
    # ==========================================================================
    def eligible_postprocess_mask(frame: pd.DataFrame) -> pd.Series:
        mask = frame.get("trial_status", pd.Series("ok", index=frame.index)).astype("string").eq("ok")
        if "failed_any" in frame.columns:
            failed = _strict_boolean_series(frame["failed_any"])
            mask &= failed.eq(False).fillna(False)
        return mask

    if run_dust and not rows_df.empty:
        if verbose:
            print(f"\n[DUST] Computing 3D dust extinction for {len(rows_df)} sources...")

        try:
            rows_df = rows_df.copy()
            rows_df["_reproduction_order"] = np.arange(len(rows_df), dtype=int)
            dust_mask = eligible_postprocess_mask(rows_df)
            dust_eligible = rows_df.loc[dust_mask].copy()
            dust_passthrough = rows_df.loc[~dust_mask].copy()
            dust_result = get_dust_extinction(dust_eligible) if not dust_eligible.empty else dust_eligible
            if len(dust_result) != len(dust_eligible):
                raise ValueError("Dust-extinction stage changed the candidate row count")
            rows_df = (
                pd.concat([dust_result, dust_passthrough], ignore_index=True, sort=False)
                .sort_values("_reproduction_order", kind="mergesort")
                .drop(columns=["_reproduction_order"], errors="ignore")
                .reset_index(drop=True)
            )
            if verbose:
                n_with_av = (rows_df["A_v_3d"] > 0).sum() if "A_v_3d" in rows_df.columns else 0
                print(f"[DUST] {n_with_av}/{len(rows_df)} sources have A_V > 0")
        except Exception as e:
            raise RuntimeError("Requested reproduction dust-extinction stage failed") from e

    # ==========================================================================
    # Post-processing: Classification (EB/CV/starspot rejection, YSO)
    # ==========================================================================
    if run_classify and not rows_df.empty:
        if verbose:
            print(f"\n[CLASSIFY] Running classification on {len(rows_df)} sources...")

        try:
            rows_df = rows_df.copy()
            rows_df["_reproduction_order"] = np.arange(len(rows_df), dtype=int)
            classify_mask = eligible_postprocess_mask(rows_df)
            classify_eligible = rows_df.loc[classify_mask].copy()
            classify_passthrough = rows_df.loc[~classify_mask].copy()
            classify_result = (
                compute_all_classifications(classify_eligible)
                if not classify_eligible.empty
                else classify_eligible
            )
            if len(classify_result) != len(classify_eligible):
                raise ValueError("Classification stage changed the candidate row count")
            rows_df = (
                pd.concat([classify_result, classify_passthrough], ignore_index=True, sort=False)
                .sort_values("_reproduction_order", kind="mergesort")
                .drop(columns=["_reproduction_order"], errors="ignore")
                .reset_index(drop=True)
            )
            if verbose:
                if "final_class" in rows_df.columns:
                    print("[CLASSIFY] Classification summary:")
                    print(rows_df["final_class"].value_counts().to_string())
        except Exception as e:
            raise RuntimeError("Requested reproduction classification failed") from e

    if "source_id" not in rows_df.columns:
        if "asas_sn_id" in rows_df.columns:
            rows_df["source_id"] = rows_df["asas_sn_id"].map(_canonical_candidate_id).astype("string")
        else:
            rows_df["source_id"] = pd.Series(pd.NA, index=rows_df.index, dtype="string")
    rows_df = rows_df.drop(columns=["asas_sn_id"], errors="ignore")
    if "mag_bin" not in rows_df.columns:
        rows_df["mag_bin"] = ""
    rows_df["mag_bin"] = rows_df["mag_bin"].map(
        lambda value: "" if _canonical_candidate_id(value) is None else str(value)
    )
    if not rows_df.empty:
        rows_df["source_id"] = _validate_unique_ids(rows_df, "source_id", "Reproduction results")
        target_candidate_by_source = df_targets.set_index("source_id")["candidate_id"].astype(str).to_dict()
        if "candidate_id" in rows_df.columns:
            mismatched = rows_df.apply(
                lambda row: str(row["candidate_id"]) != target_candidate_by_source.get(str(row["source_id"]), str(row["candidate_id"])),
                axis=1,
            )
            if bool(mismatched.any()):
                raise ValueError("Reproduction result candidate_id/source_id pairs do not match the requested candidates")
            rows_df = rows_df.drop(columns=["candidate_id"])

    merged = df_targets.merge(
        rows_df,
        on=["source_id", "mag_bin"],
        how="left",
        suffixes=("", "_det"),
        validate="one_to_one",
    )
    for column in (
        "input_source",
        "dat_path",
        "lc_dir",
        "index_num",
        "index_csv",
        "input_record_fingerprint",
        "input_file_fingerprint",
    ):
        mapped = merged["source_id"].astype(str).map(
            lambda source_id: record_provenance_by_source.get(source_id, {}).get(column)
        )
        has_selected_record = merged["source_id"].astype(str).isin(record_provenance_by_source)
        if column not in merged.columns:
            merged[column] = mapped
        else:
            merged.loc[has_selected_record, column] = mapped.loc[has_selected_record]
        merged = merged.drop(columns=[f"{column}_det"], errors="ignore")
    merged["path"] = merged["dat_path"]
    merged = merged.drop(columns=["path_det"], errors="ignore")

    missing_result = merged.get("trial_status", pd.Series(pd.NA, index=merged.index, dtype="string")).isna()
    if "trial_status" not in merged.columns:
        merged["trial_status"] = pd.Series(pd.NA, index=merged.index, dtype="string")
    merged.loc[missing_result & merged["source_id"].astype(str).isin(tag_rejected_ids), "trial_status"] = "ineligible_tag_filter"
    merged.loc[missing_result & ~merged["source_id"].astype(str).isin(tag_rejected_ids), "trial_status"] = "not_evaluated"
    if "evaluation_error" not in merged.columns:
        merged["evaluation_error"] = None
    merged.loc[merged["trial_status"].eq("not_evaluated") & merged["evaluation_error"].isna(), "evaluation_error"] = (
        "No coherent light-curve record was evaluated for this candidate"
    )
    for column, value in (
        ("input_fingerprint", input_fingerprint),
        ("candidate_set_fingerprint", candidate_fingerprint),
        ("manifest_fingerprint", manifest_fingerprint),
        ("auxiliary_input_fingerprint", auxiliary_input_fingerprint),
        ("config_fingerprint", config_fingerprint),
        ("code_fingerprint", code_fingerprint),
        ("run_fingerprint", run_fingerprint),
        ("reproduction_schema_version", REPRODUCTION_SCHEMA_VERSION),
    ):
        if column not in merged.columns:
            merged[column] = value
        elif value is not None:
            merged[column] = merged[column].fillna(value)
    merged["reproduction_record_id"] = merged.apply(
        lambda row: _stable_digest(
            {
                "candidate_id": row.get("candidate_id"),
                "source_id": row.get("source_id"),
                "input_record_fingerprint": row.get("input_record_fingerprint"),
                "run_fingerprint": row.get("run_fingerprint"),
            }
        ),
        axis=1,
    )
    if bool(merged["reproduction_record_id"].duplicated().any()):
        raise ValueError("Reproduction output contains duplicate per-candidate record identities")

    g_peaks = merged["g_n_peaks"] if "g_n_peaks" in merged.columns else pd.Series(np.nan, index=merged.index)
    v_peaks = merged["v_n_peaks"] if "v_n_peaks" in merged.columns else pd.Series(np.nan, index=merged.index)
    evidence_flags: list[pd.Series] = []
    for peaks in (g_peaks, v_peaks):
        numeric = pd.to_numeric(peaks, errors="coerce")
        if bool(numeric.notna().any()):
            evidence_flags.append(numeric.gt(0).astype("boolean").mask(numeric.isna() | numeric.lt(0), pd.NA))
    for column in ("g_bayes_dip_significant", "v_bayes_dip_significant"):
        if column in merged.columns:
            evidence_flags.append(_strict_boolean_series(merged[column]))
    detected = pd.Series(pd.NA, index=merged.index, dtype="boolean")
    if evidence_flags:
        evidence = pd.concat(evidence_flags, axis=1).astype("boolean")
        detected.loc[evidence.eq(True).any(axis=1)] = True
        detected.loc[evidence.eq(False).all(axis=1)] = False
    successful = merged["trial_status"].astype("string").eq("ok")
    if bool(detected.loc[successful].isna().any()):
        raise ValueError("Successful reproduction rows contain an unknown detection decision")
    detected.loc[~successful] = pd.NA
    merged["detected"] = detected
    for column in (
        "g_bayes_dip_significant",
        "v_bayes_dip_significant",
        "g_bayes_jump_significant",
        "v_bayes_jump_significant",
        "g_used_sigma_eff",
        "v_used_sigma_eff",
    ):
        if column in merged.columns:
            parsed = _strict_boolean_series(merged[column])
            parsed.loc[~successful] = pd.NA
            merged[column] = parsed
    merged.loc[~successful, "rejection_reason"] = merged.loc[~successful, "trial_status"].astype(str)

    if run_filter and "failed_any" in merged.columns:
        failed_any = _strict_boolean_series(merged["failed_any"])
        pipeline_passed = detected & ~failed_any
        pipeline_passed.loc[~successful | failed_any.isna()] = pd.NA
        merged["pipeline_passed"] = pipeline_passed.astype("boolean")
    else:
        merged["pipeline_passed"] = detected.astype("boolean")

    def format_detection(row: pd.Series) -> str:
        if str(row.get("trial_status")) != "ok":
            error = row.get("evaluation_error")
            return f"unknown ({row.get('trial_status')}{': ' + str(error) if pd.notna(error) and error else ''})"
        if row.get("detected") is not True and not isinstance(row.get("detected"), np.bool_):
            return "—"
        if not bool(row.get("detected")):
            return "—"
        parts = [f"mag_bin={row.get('mag_bin', '')}"]

        if "g_n_peaks" in row and pd.notna(row["g_n_peaks"]):
            parts.append(f"g_peaks={int(row['g_n_peaks'])}")
        if "v_n_peaks" in row and pd.notna(row["v_n_peaks"]):
            parts.append(f"v_peaks={int(row['v_n_peaks'])}")

        def bayes_part(prefix: str) -> str | None:
            sig_value = row.get(f"{prefix}_bayes_dip_significant", pd.NA)
            sig = bool(sig_value) if pd.notna(sig_value) else False
            if not sig:
                return None
            bf = row.get(f"{prefix}_bayes_dip_bayes_factor")
            bf_str = f"{float(bf):.3f}" if pd.notna(bf) else "nan"
            n_dips = int(row.get(f"{prefix}_bayes_n_dips", 0))
            mxlog = row.get(f"{prefix}_bayes_dip_max_logbf")
            if pd.notna(mxlog):
                return f"{prefix}_bayes_dip (maxlogBF={float(mxlog):.2f}, BF={bf_str}, n={n_dips})"
            return f"{prefix}_bayes_dip (BF={bf_str}, n={n_dips})"

        for pref in ("g", "v"):
            part = bayes_part(pref)
            if part:
                parts.append(part)

        return "; ".join(parts)

    merged["detection_details"] = merged.apply(format_detection, axis=1)

    if extra_columns:
        cols = [c for c in extra_columns if c in merged.columns]
    else:
        cols = []

    ordered_cols = _ordered_reproduction_columns(merged, cols)
    return merged[ordered_cols]


__all__ = [
    "brayden_candidates",
    "tzanidakis_candidates",
    "build_reproduction_report",
]


def get_non_default_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> dict:
    """
    Compare parsed args to parser defaults and return only non-default values.
    """
    non_defaults = {}
    for action in parser._actions:
        if action.dest == "help":
            continue
        default = action.default
        value = getattr(args, action.dest, None)
        # Skip if value equals default
        if value != default:
            non_defaults[action.dest] = value
    return non_defaults


def generate_subdir_name(non_default_args: dict) -> str:
    """
    Generate a subdirectory name based on non-default arguments.
    Format: YYYYMMDD_HHMMSS[_flag1=val1_flag2=val2...]
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Filter out args that shouldn't affect naming
    filtered_args = {k: v for k, v in non_default_args.items() if k not in {"verbose"}}

    if not filtered_args:
        return timestamp

    # Build suffix from filtered args (abbreviated)
    parts = []
    # Prioritize certain flags for the directory name
    priority_keys = [
        "trigger_mode", "logbf_threshold_dip", "p_points",
        "baseline_func", "run_min_points", "accepted_morphologies",
        "candidates", "input",
    ]

    for key in priority_keys:
        if key in filtered_args:
            val = filtered_args[key]
            # Abbreviate key names
            short_key = key.replace("bayes_", "").replace("threshold_", "thr_")
            short_key = short_key.replace("logbf_", "bf_").replace("_", "")
            # Abbreviate values
            if isinstance(val, bool):
                val_str = "1" if val else "0"
            elif isinstance(val, float):
                val_str = f"{val:.2g}".replace(".", "p")
            elif isinstance(val, Path):
                val_str = val.stem[:20]
            elif isinstance(val, str):
                val_str = Path(val).stem[:20] if "/" in val or "\\" in val else val[:15]
            else:
                val_str = str(val)[:15]
            parts.append(f"{short_key}={val_str}")

    # Add remaining filtered args (up to a limit)
    for key, val in filtered_args.items():
        if key not in priority_keys and len(parts) < 8:
            short_key = key.replace("_", "")[:12]
            if isinstance(val, bool):
                val_str = "1" if val else "0"
            elif isinstance(val, float):
                val_str = f"{val:.2g}".replace(".", "p")
            elif isinstance(val, Path):
                val_str = val.stem[:15]
            elif isinstance(val, str):
                val_str = Path(val).stem[:15] if "/" in val or "\\" in val else val[:10]
            else:
                val_str = str(val)[:10]
            parts.append(f"{short_key}={val_str}")

    suffix = "_".join(parts)
    # Limit total length
    if len(suffix) > 150:
        suffix = suffix[:150]

    return f"{timestamp}_{suffix}"


def generate_log_filename(non_default_args: dict) -> str:
    """
    Generate a log filename based on non-default arguments.
    Format: reproduction_YYYYMMDD_HHMMSS[_flag1=val1_flag2=val2...].log
    """
    subdir_name = generate_subdir_name(non_default_args)
    return f"reproduction_{subdir_name}.log"


class TeeOutput:
    """Capture stdout/stderr to both terminal and a string buffer."""

    def __init__(self, original_stream):
        self.original = original_stream
        self.buffer = io.StringIO()

    def write(self, text):
        self.original.write(text)
        self.buffer.write(text)

    def flush(self):
        self.original.flush()

    def getvalue(self):
        return self.buffer.getvalue()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run targeted reproduction search on events.py candidates and summarize results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run on events.py output Parquet (uses 'path' column directly)
  malca reproduce --candidates output/strong_candidates_12_12.5.parquet

  # With manifest data
  malca reproduce --candidates candidates.parquet --manifest manifest.parquet

  # With SkyPatrol CSV files
  malca reproduce --candidates candidates.parquet --skypatrol-dir input/skypatrol2
""",
    )
    g_input = parser.add_argument_group("Input")
    g_manifest_tag = parser.add_argument_group("Manifest & tag")
    g_filter = parser.add_argument_group("Filter")
    g_detection = parser.add_argument_group("Detection")
    g_postprocess = parser.add_argument_group("Postprocess")
    g_output = parser.add_argument_group("Output")
    g_general = parser.add_argument_group("General")

    g_detection.add_argument(
        "--trigger-mode",
        choices=("posterior_prob", "logbf"),
        default=TRIGGER_MODE,
        help="Event triggering statistic.",
    )
    g_detection.add_argument("--significance-threshold", type=float, default=SIGNIFICANCE_THRESHOLD)
    g_detection.add_argument("--logbf-threshold-dip", type=float, default=LOGBF_THRESHOLD_DIP)
    g_detection.add_argument("--logbf-threshold-jump", type=float, default=LOGBF_THRESHOLD_JUMP)
    g_detection.add_argument("--p-points", type=int, default=P_POINTS)
    g_detection.add_argument("--p-min-dip", type=float, default=None)
    g_detection.add_argument("--p-max-dip", type=float, default=None)
    g_detection.add_argument("--p-min-jump", type=float, default=None)
    g_detection.add_argument("--p-max-jump", type=float, default=None)
    g_detection.add_argument("--mag-points", type=int, default=MAG_POINTS)
    g_detection.add_argument("--mag-min-dip", type=float, default=None)
    g_detection.add_argument("--mag-max-dip", type=float, default=None)
    g_detection.add_argument("--mag-min-jump", type=float, default=None)
    g_detection.add_argument("--mag-max-jump", type=float, default=None)
    g_detection.add_argument(
        "--baseline-func",
        choices=("gp", "gp_masked", "global_median", "per_camera_median"),
        default=BASELINE_FUNC,
    )
    g_detection.add_argument("--baseline-s0", type=float, default=BASELINE_S0)
    g_detection.add_argument("--baseline-w0", type=float, default=BASELINE_W0)
    g_detection.add_argument("--baseline-q", type=float, default=BASELINE_Q)
    g_detection.add_argument("--baseline-jitter", type=float, default=BASELINE_JITTER)
    g_detection.add_argument("--baseline-sigma-floor", type=float, default=None)
    g_detection.add_argument("--run-min-points", type=int, default=RUN_MIN_POINTS)
    g_detection.add_argument(
        "--run-max-gap-points",
        dest="run_max_gap_points",
        type=int,
        default=RUN_MAX_GAP_POINTS,
    )
    g_detection.add_argument("--run-max-gap-days", type=float, default=None)
    g_detection.add_argument("--run-min-duration-days", type=float, default=0.0)
    g_detection.add_argument("--min-mag-offset", type=float, default=MIN_MAG_OFFSET)

    g_output.add_argument(
        "--output-dir",
        dest="out_dir",
        default=str(DEFAULT_OUTPUT_DIR / "plots" / "reproduction"),
        help="Directory for peak_results output",
    )
    g_output.add_argument(
        "--output-format",
        dest="out_format",
        choices=("parquet",),
        default="parquet",
        help="Structured output format (Parquet)",
    )
    g_output.add_argument(
        "--plot-format",
        choices=("png", "pdf"),
        default="pdf",
        help="Plot output format (default: png)",
    )
    g_output.add_argument(
        "--log-format",
        choices=("text", "parquet"),
        default="text",
        help="Log output format: text (human-readable) or parquet (structured data).",
    )
    g_general.add_argument(
        "--workers",
        type=int,
        default=WORKERS,
        help="ProcessPool worker count.",
    )
    g_general.add_argument(
        "--chunk-size",
        type=int,
        default=REPRODUCE_CHUNK_SIZE,
        help="Rows per worker chunk",
    )
    g_general.add_argument(
        "--metrics-dip-threshold",
        type=float,
        default=0.2,
        help="Dip threshold for run_metrics",
    )
    g_general.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print debug info",
    )
    add_config_args(g_general)

    g_manifest_tag.add_argument(
        "--skip-tags",
        action="store_true",
        help="Skip tagging step (sparse LC, multi-camera, VSX)",
    )
    g_manifest_tag.add_argument(
        "--min-time-span",
        type=float,
        default=MIN_TIME_SPAN,
        help="Min time span in days for sparse LC filter (default: 100)",
    )
    g_manifest_tag.add_argument(
        "--min-points-per-day",
        type=float,
        default=MIN_POINTS_PER_DAY,
        help="Min cadence for sparse LC filter (default: 0.05)",
    )
    g_manifest_tag.add_argument(
        "--min-cameras",
        type=int,
        default=MIN_CAMERAS,
        help="Min cameras required for multi-camera filter (default: 2)",
    )
    g_manifest_tag.add_argument(
        "--skip-vsx",
        action="store_true",
        help="Skip VSX known variable filter",
    )
    g_manifest_tag.add_argument(
        "--vsx-catalog",
        type=Path,
        default=VSX_RAW_CATALOG_PATH,
        help="Path to VSX catalog CSV",
    )
    g_manifest_tag.add_argument(
        "--vsx-max-sep",
        type=float,
        default=VSX_MAX_SEP_ARCSEC,
        help="Max separation for VSX match in arcsec (default: 3.0)",
    )

    g_postprocess.add_argument(
        "--run-classify",
        action="store_true",
        help="Run classification (EB/CV/starspot rejection, YSO classification)",
    )
    g_postprocess.add_argument(
        "--run-characterize",
        action="store_true",
        help="Run characterization (Gaia DR3 query for RUWE, stellar params, 2MASS/WISE photometry)",
    )
    g_postprocess.add_argument(
        "--gaia-cache",
        type=Path,
        default=None,
        help="Path to Gaia query cache file (parquet) for faster repeated runs",
    )
    g_postprocess.add_argument(
        "--index-file",
        type=Path,
        default=Path("output/asassn_index_masked_concat_cleaned_20250919_154524_brotli.parquet"),
        help="Path to ASAS-SN index file with gaia_id, ra_deg, dec_deg columns",
    )
    g_postprocess.add_argument(
        "--run-dust",
        action="store_true",
        help="Run 3D dust extinction correction (requires dustmaps3d)",
    )
    g_filter.add_argument(
        "--run-filter",
        dest="run_filter",
        action="store_true",
        help="Apply candidate filters (Bayes factor, run robustness, morphology)",
    )
    g_filter.add_argument(
        "--min-bayes-factor",
        type=float,
        default=MIN_BAYES_FACTOR,
        help="Min Bayes factor for filter stage (default: 10.0)",
    )
    g_filter.add_argument(
        "--filter-min-run-cameras",
        dest="filter_min_run_cameras",
        type=int,
        default=POST_FILTER_MIN_RUN_CAMERAS,
        help="Min cameras for run robustness filter (default: 2)",
    )
    g_filter.add_argument(
        "--filter-min-run-points",
        dest="filter_min_run_points",
        type=int,
        default=POST_FILTER_MIN_RUN_POINTS,
        help="Min points per run for robustness filter (default: 2)",
    )
    g_postprocess.add_argument(
        "--run-postprocess",
        action="store_true",
        help="Generate diagnostic plots for candidates",
    )
    g_postprocess.add_argument(
        "--max-plots",
        type=int,
        default=None,
        help="Limit number of plots generated (default: no limit)",
    )
    g_postprocess.add_argument(
        "--run-enrich",
        action="store_true",
        help="Enrich candidates with comprehensive light curve stats",
    )
    g_postprocess.add_argument(
        "--enrich-compute-ls",
        action="store_true",
        help="Include Lomb-Scargle periodogram in enrichment (expensive)",
    )
    g_filter.add_argument(
        "--accepted-morphologies",
        type=str,
        default="gaussian,paczynski,fred",
        help="Comma-separated list of accepted morphologies (default: gaussian,paczynski). Use 'all' to accept all morphologies including noise.",
    )

    g_input.add_argument(
        "--skypatrol-dir",
        default="input/skypatrol2",
        help="Directory with SkyPatrol CSV files (<source_id>.csv)",
    )
    g_input.add_argument(
        "--manifest",
        default=None,
        help="Path to lc_manifest Parquet for targeted reproduction",
    )
    g_input.add_argument(
        "--candidates",
        default=None,
        help="Candidate spec (built-in list name or path to Parquet file from events.py).",
    )
    g_input.add_argument(
        "--path-prefix",
        default=None,
        help="Path prefix to rewrite for candidates with a 'path' column (e.g. /data/poohbah/1/assassin/rowan.90/lcsv2).",
    )
    g_input.add_argument(
        "--path-root",
        default=None,
        help="Local root that replaces --path-prefix for candidates with a 'path' column.",
    )

    parser.set_defaults(**REPRODUCE_CONFIG_DEFAULTS)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_parser()
    args = parse_args_with_config(
        parser,
        command="reproduce",
        argv=list(argv) if argv is not None else None,
        valid_keys=namespace_keys(parser, REPRODUCE_CONFIG_DEFAULTS),
        path_keys={
            "out_dir",
            "gaia_cache",
            "index_file",
            "manifest",
            "path_root",
            "skypatrol_dir",
            "vsx_catalog",
        },
    )

    # Set up logging
    log_dir = DEFAULT_OUTPUT_DIR / "logs" / "reproduction"
    log_dir.mkdir(parents=True, exist_ok=True)

    non_default_args = get_non_default_args(args, parser)
    subdir_name = generate_subdir_name(non_default_args)
    log_filename = f"reproduction_{subdir_name}.log"

    # Determine log file extension based on format
    log_ext = ".parquet" if args.log_format == "parquet" else ".log"
    log_filename_base = log_filename.rsplit(".", 1)[0] if "." in log_filename else log_filename
    log_path = log_dir / f"{log_filename_base}{log_ext}"

    # Compute plot output directory (subdirectory based on non-default args)
    plot_base_dir = Path(args.out_dir)
    plot_out_dir = plot_base_dir / subdir_name
    plot_out_dir.mkdir(parents=True, exist_ok=True)

    # Capture stdout/stderr
    tee_stdout = TeeOutput(sys.stdout)
    tee_stderr = TeeOutput(sys.stderr)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = tee_stdout
    sys.stderr = tee_stderr

    report = None
    try:
        # Log the full command
        if argv is not None:
            cmd_str = f"malca reproduce {' '.join(str(a) for a in argv)}"
        else:
            cmd_str = f"malca reproduce {' '.join(sys.orig_argv[1:]) if hasattr(sys, 'orig_argv') else '(unknown)'}"
        print(f"Command: {cmd_str}")
        print(f"Log file: {log_path}")
        print(f"Plot dir: {plot_out_dir}")
        print(f"Non-default args: {non_default_args}")
        print()

        report = _main_impl(args, plot_out_dir=plot_out_dir)

    finally:
        # Restore stdout/stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr

        # Write log file based on format
        if args.log_format == "parquet" and report is not None:
            write_parquet_table(report, log_path)
            print(f"\nParquet log saved to: {log_path}")
        else:
            # Text format: save captured stdout/stderr
            log_content = tee_stdout.getvalue()
            stderr_content = tee_stderr.getvalue()
            if stderr_content:
                log_content += "\n\n=== STDERR ===\n" + stderr_content

            with open(log_path, "w") as f:
                f.write(log_content)

            print(f"\nLog saved to: {log_path}")


def _main_impl(args: argparse.Namespace, plot_out_dir: Path | None = None) -> pd.DataFrame:
    """Main implementation, called by main() with logging wrapper. Returns the report DataFrame."""
    candidates_spec = args.candidates
    candidate_data = resolve_candidates(candidates_spec)

    # Use provided plot_out_dir or fall back to args.out_dir
    out_dir = plot_out_dir if plot_out_dir is not None else Path(args.out_dir)

    significance_threshold = float(args.significance_threshold) if args.significance_threshold is not None else None
    if args.verbose:
        if args.trigger_mode == "posterior_prob":
            print(
                "[DEBUG] Using posterior-probability triggering: "
                f"significance_threshold={significance_threshold}"
            )
        else:
            print(
                "[DEBUG] Using logBF triggering: "
                f"dip_thr={args.logbf_threshold_dip}, jump_thr={args.logbf_threshold_jump}"
            )

    # Parse accepted morphologies
    if args.accepted_morphologies.lower() == "all":
        accepted_morphologies = {"gaussian", "skew_gaussian", "paczynski", "fred", "noise", "none"}
    else:
        accepted_morphologies = {m.strip().lower() for m in args.accepted_morphologies.split(",")}
    if args.verbose:
        print(f"[DEBUG] Accepted morphologies: {accepted_morphologies}")

    report = build_reproduction_report(
        candidates=candidate_data,
        out_dir=out_dir,
        out_format=args.out_format,
        plot_format=args.plot_format,
        n_workers=args.workers,
        chunk_size=args.chunk_size,
        metrics_dip_threshold=args.metrics_dip_threshold,
        significance_threshold=significance_threshold,
        p_points=args.p_points,
        trigger_mode=args.trigger_mode,
        logbf_threshold_dip=args.logbf_threshold_dip,
        logbf_threshold_jump=args.logbf_threshold_jump,
        # Probability grid bounds
        p_min_dip=args.p_min_dip,
        p_max_dip=args.p_max_dip,
        p_min_jump=args.p_min_jump,
        p_max_jump=args.p_max_jump,
        # Magnitude grid
        mag_points=args.mag_points,
        mag_min_dip=args.mag_min_dip,
        mag_max_dip=args.mag_max_dip,
        mag_min_jump=args.mag_min_jump,
        mag_max_jump=args.mag_max_jump,
        # Baseline function
        baseline_func=args.baseline_func,
        # Baseline kwargs
        baseline_s0=args.baseline_s0,
        baseline_w0=args.baseline_w0,
        baseline_q=args.baseline_q,
        baseline_jitter=args.baseline_jitter,
        baseline_sigma_floor=args.baseline_sigma_floor,
        # Run confirmation filters
        run_min_points=args.run_min_points,
        max_gap_points=args.run_max_gap_points,
        run_max_gap_days=args.run_max_gap_days,
        run_min_duration_days=args.run_min_duration_days,
        skypatrol_dir=args.skypatrol_dir,
        path_prefix=args.path_prefix,
        path_root=args.path_root,
        manifest_path=args.manifest,
        method="bayes",
        verbose=args.verbose,
        # Filter parameters
        skip_tags=args.skip_tags,
        min_time_span=args.min_time_span,
        min_points_per_day=args.min_points_per_day,
        min_cameras=args.min_cameras,
        skip_vsx=args.skip_vsx,
        vsx_catalog=args.vsx_catalog,
        vsx_max_sep=args.vsx_max_sep,
        min_mag_offset=args.min_mag_offset,
        run_filter=args.run_filter,
        filter_min_run_cameras=args.filter_min_run_cameras,
        filter_min_run_points=args.filter_min_run_points,
        filter_min_bayes_factor=args.min_bayes_factor,
        run_postprocess=args.run_postprocess,
        max_plots=args.max_plots,
        run_enrich=args.run_enrich,
        enrich_compute_ls=args.enrich_compute_ls,
        # Morphology filter
        accepted_morphologies=accepted_morphologies,
        # Post-processing: classification and characterization
        run_classify=args.run_classify,
        run_characterize=args.run_characterize,
        gaia_cache=args.gaia_cache,
        index_file=args.index_file,
        run_dust=args.run_dust,
    )

    # Print filtering summary
    print("\n" + "=" * 60)
    print("FILTERING SUMMARY")
    print("=" * 60)
    print(f"Plot format:          {args.plot_format}")
    print(f"Baseline function:    {args.baseline_func}")
    print(f"Trigger mode:         {args.trigger_mode}")
    if args.trigger_mode == "posterior_prob":
        print(f"Posterior threshold:  {args.significance_threshold}")
    else:
        print(f"logBF thresholds:     dip={args.logbf_threshold_dip}, jump={args.logbf_threshold_jump}")
    print("Sigma_eff:            enabled (mandatory)")
    print(f"Mag grid points:      {args.mag_points}")
    if args.p_min_dip is not None or args.p_max_dip is not None:
        print(f"P-grid (dip):         min={args.p_min_dip}, max={args.p_max_dip}")
    if args.p_min_jump is not None or args.p_max_jump is not None:
        print(f"P-grid (jump):        min={args.p_min_jump}, max={args.p_max_jump}")
    tags_applied = bool(args.manifest) and not args.skip_tags
    if tags_applied:
        print("Tags:                 APPLIED")
    else:
        print("Tags:                 SKIPPED (no manifest)")
    if not args.skip_tags:
        print(f"  - Sparse LC filter:   min_time_span={args.min_time_span}d, min_cadence={args.min_points_per_day}/d")
        print(f"  - Multi-camera:       min_cameras={args.min_cameras}")
        print(f"  - VSX filter:         {'APPLIED' if not args.skip_vsx else 'SKIPPED'}")
    print(f"Signal amplitude:     {'APPLIED (min_mag_offset=' + str(args.min_mag_offset) + ')' if args.min_mag_offset > 0 else 'DISABLED'}")
    print(f"Run confirmation:     min_points={args.run_min_points}, max_gap_points={args.run_max_gap_points}")
    if args.run_max_gap_days is not None:
        print(f"                      max_gap_days={args.run_max_gap_days}")
    if args.run_min_duration_days is not None:
        print(f"                      min_duration_days={args.run_min_duration_days}")
    print(f"Morphology filter:    accepted={{{', '.join(sorted(accepted_morphologies))}}}")
    filter_state = "APPLIED" if args.run_filter else "SKIPPED"
    print(f"Filters:              {filter_state}")
    if args.run_postprocess:
        print(f"Postprocess plots:    {'ENABLED' if args.run_filter else 'SKIPPED (requires filter)'}")
    if args.run_enrich:
        print(f"Enrichment:           {'ENABLED' if args.run_filter else 'SKIPPED (requires filter)'}")
    print("=" * 60 + "\n")

    columns = [
        "source",
        "source_id",
        "category",
        "mag_bin",
        "detected",
        "rejection_reason",
        "detection_details",
        "g_n_peaks",
        "v_n_peaks",
        "g_bayes_dip_significant",
        "v_bayes_dip_significant",
        "g_bayes_n_dips",
        "v_bayes_n_dips",
        "g_bayes_dip_max_prob",
        "v_bayes_dip_max_prob",
        "g_bayes_dip_max_logbf",
        "v_bayes_dip_max_logbf",
        "g_bayes_dip_bayes_factor",
        "v_bayes_dip_bayes_factor",
    ]
    existing = [c for c in columns if c in report.columns]
    print(report[existing].to_string(index=False))

    return report


if __name__ == "__main__":
    main()
