from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from malca.config import BAD_CAMERA_SCATTER_RATIO_THRESHOLD
from malca.utils import filter_bad_cameras, read_lc_dat2
from malca.utils import read_skypatrol_csv as _read_skypatrol_csv


CAMERA_COLOR_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#393b79", "#637939", "#8c6d31", "#843c39", "#7b4173",
    "#3182bd", "#e6550d", "#31a354", "#756bb1", "#636363",
]

ASASSN_COLUMNS = [
    "JD",
    "mag",
    "error",
    "good_bad",
    "camera#",
    "v_g_band",
    "saturated",
    "cam_field",
]


def stable_camera_color(camera_label: str) -> str:
    """Return a deterministic color for a camera label across plots."""
    s = str(camera_label)
    try:
        idx = int(s) % len(CAMERA_COLOR_PALETTE)
    except Exception:
        digest = hashlib.md5(s.encode("utf-8")).hexdigest()
        idx = int(digest[:8], 16) % len(CAMERA_COLOR_PALETTE)
    return CAMERA_COLOR_PALETTE[idx]


def read_asassn_dat(dat_path: str | Path) -> pd.DataFrame:
    """Read an ASAS-SN `.dat` light curve using whitespace separation."""
    return pd.read_csv(
        dat_path,
        sep=r"\s+",
        names=ASASSN_COLUMNS,
        dtype={
            "JD": float,
            "mag": float,
            "error": float,
            "good_bad": int,
            "camera#": int,
            "v_g_band": int,
            "saturated": int,
            "cam_field": str,
        },
        comment="#",
    )


def load_lightcurve_df(
    path: str | Path,
    *,
    file_ext: str | None = None,
    filter_bad_cameras_enabled: bool = False,
    bad_camera_scatter_ratio: float = BAD_CAMERA_SCATTER_RATIO_THRESHOLD,
    return_filtered_info: bool = False,
):
    """Load a native MALCA light curve, optionally filtering bad cameras."""
    from malca.config import LIGHT_CURVE_FILE_EXTENSION
    
    lc_path = Path(path)
    suffix = lc_path.suffix.lower()
    
    # If file_ext is provided, use it to determine how to read the file
    # Otherwise, use the actual file extension
    if file_ext is None and suffix:
        # Remove the leading dot from suffix
        file_ext = suffix[1:] if suffix.startswith('.') else suffix
    elif file_ext is None:
        file_ext = LIGHT_CURVE_FILE_EXTENSION
    
    # Normalize extension format (e.g., "dat2", ".dat2" -> "dat2")
    if file_ext.startswith('.'):
        file_ext = file_ext[1:]
    
    if file_ext in ("dat2", "dat3", "dat"):
        dfg, dfv = read_lc_dat2(lc_path.stem, str(lc_path.parent), file_ext=file_ext)
        if dfg.empty and dfv.empty:
            return (pd.DataFrame(), set()) if return_filtered_info else pd.DataFrame()
        df = pd.concat([dfg, dfv], ignore_index=True)
    elif suffix == ".csv":
        df = _read_skypatrol_csv(lc_path)
    else:
        # Default to reading as ASAS-SN dat format
        df = read_asassn_dat(lc_path)

    filtered_cameras: set[int] = set()
    if filter_bad_cameras_enabled and not df.empty and "camera#" in df.columns:
        df, filtered_cameras = filter_bad_cameras(
            df,
            lc_path=str(lc_path),
            scatter_ratio_threshold=bad_camera_scatter_ratio,
        )

    if return_filtered_info:
        return df, filtered_cameras
    return df
