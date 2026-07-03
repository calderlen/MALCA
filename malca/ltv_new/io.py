"""Input helpers for the standalone LTV evidence pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from malca.config import SKYPATROL_JD_OFFSET
from malca.io.table_io import read_parquet_table
from malca.ltv_new.likelihood import LightCurveData


DAT_COLUMNS = ["JD", "mag", "error", "good_bad", "camera", "v_g_band", "saturated", "cam_field"]
PATH_COLUMNS = ("lc_path", "dat_path", "path")
ID_COLUMNS = ("target_id", "candidate_id", "asas_sn_id", "source_id", "id")


@dataclass(frozen=True)
class LightCurveJob:
    target_id: str
    path: Path


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return read_parquet_table(path)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    if suffix.startswith(".dat"):
        return pd.read_csv(path, header=None, names=DAT_COLUMNS, sep=r"\s+")
    return pd.read_csv(path)


def _first_existing(columns: list[str] | pd.Index, candidates: tuple[str, ...]) -> str | None:
    lookup = {str(col).lower(): str(col) for col in columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def _normalize_band(series: pd.Series | None, n: int) -> np.ndarray:
    if series is None:
        return np.zeros(n, dtype=int)
    if pd.api.types.is_numeric_dtype(series):
        return (pd.to_numeric(series, errors="coerce").fillna(0).astype(int).to_numpy() != 0).astype(int)
    lowered = series.astype("string").str.strip().str.lower()
    return lowered.isin({"v", "1", "true", "v-band", "johnson-v"}).fillna(False).astype(int).to_numpy()


def _clean_ltv_new_lc(df: pd.DataFrame) -> pd.DataFrame:
    mask = np.ones(len(df), dtype=bool)
    mask &= pd.to_numeric(df["saturated"], errors="coerce").fillna(0).astype(int).to_numpy() == 0
    for col in ("JD", "mag", "error"):
        mask &= pd.to_numeric(df[col], errors="coerce").notna().to_numpy()
    mask &= pd.to_numeric(df["error"], errors="coerce").to_numpy(dtype=float) > 0.0
    out = df.loc[mask].copy()
    for col in ("JD", "mag", "error"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["v_g_band"] = pd.to_numeric(out["v_g_band"], errors="coerce").fillna(0).astype(int)
    return out.sort_values("JD").reset_index(drop=True)


def table_is_manifest(df: pd.DataFrame) -> bool:
    path_col = _first_existing(df.columns, PATH_COLUMNS)
    has_lc_columns = _first_existing(df.columns, ("JD", "jd", "time", "mjd")) is not None and "mag" in {
        str(c).lower() for c in df.columns
    }
    return path_col is not None and not has_lc_columns


def load_light_curve(path: str | Path, *, target_id: str | None = None) -> LightCurveData:
    path = Path(path).expanduser()
    df = _read_table(path)
    jd_col = _first_existing(df.columns, ("JD", "jd", "time", "mjd"))
    mag_col = _first_existing(df.columns, ("mag", "magnitude"))
    err_col = _first_existing(df.columns, ("error", "mag_err", "magerr", "err"))
    if jd_col is None or mag_col is None or err_col is None:
        raise ValueError(f"Light curve missing JD/mag/error columns: {path}")

    out = pd.DataFrame(
        {
            "JD": pd.to_numeric(df[jd_col], errors="coerce"),
            "mag": pd.to_numeric(df[mag_col], errors="coerce"),
            "error": pd.to_numeric(df[err_col], errors="coerce"),
        }
    )
    if "good_bad" in df.columns:
        good = pd.to_numeric(df["good_bad"], errors="coerce").fillna(0).astype(int) == 1
        out = out.loc[good].copy()
        band_source = df.loc[good]
    else:
        band_source = df

    sat_col = _first_existing(df.columns, ("saturated", "saturated/unsaturated"))
    out["saturated"] = (
        pd.to_numeric(band_source[sat_col], errors="coerce").fillna(0).astype(int).to_numpy()
        if sat_col is not None
        else 0
    )
    band_col = _first_existing(df.columns, ("v_g_band", "v/g?", "band"))
    out["v_g_band"] = _normalize_band(band_source[band_col] if band_col is not None else None, len(out))

    if out["JD"].notna().any() and float(out["JD"].median()) < 100000.0:
        out["JD"] = out["JD"] + SKYPATROL_JD_OFFSET

    out = _clean_ltv_new_lc(out)
    if out.empty:
        raise ValueError(f"No valid cleaned rows in light curve: {path}")
    resolved_id = target_id or path.stem
    return LightCurveData(
        jd=out["JD"].to_numpy(dtype=float),
        mag=out["mag"].to_numpy(dtype=float),
        err=out["error"].to_numpy(dtype=float),
        band=out["v_g_band"].to_numpy(dtype=int),
        target_id=str(resolved_id),
    )


def iter_light_curve_jobs(input_path: str | Path) -> list[LightCurveJob]:
    path = Path(input_path).expanduser()
    if path.is_dir():
        files = sorted(
            item for item in path.iterdir() if item.is_file() and item.suffix.lower() in {".dat", ".dat2", ".dat3", ".csv", ".parquet"}
        )
        return [LightCurveJob(target_id=item.stem, path=item) for item in files]

    if path.suffix.lower() in {".parquet", ".csv"}:
        df = _read_table(path)
        if table_is_manifest(df):
            path_col = _first_existing(df.columns, PATH_COLUMNS)
            id_col = _first_existing(df.columns, ID_COLUMNS)
            assert path_col is not None
            jobs: list[LightCurveJob] = []
            for idx, row in df.iterrows():
                lc_path = Path(str(row[path_col])).expanduser()
                target_id = str(row[id_col]) if id_col is not None and pd.notna(row[id_col]) else lc_path.stem
                jobs.append(LightCurveJob(target_id=target_id, path=lc_path))
            return jobs

    return [LightCurveJob(target_id=path.stem, path=path)]
