"""Optional post-filter stochastic feature enrichment for LTV candidates."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from malca.config.config_ltv import LTV_WORKERS
from malca.config.config_pipeline import SKYPATROL_JD_OFFSET
from malca.utils import clean_lc, read_lc_dat2


STOCHASTIC_COLUMNS = [
    "stoch_sf_ml_amplitude",
    "stoch_sf_ml_gamma",
    "stoch_iar_phi",
    "stoch_mhps_high",
    "stoch_mhps_low",
    "stoch_mhps_non_zero",
    "stoch_mhps_pn_flag",
    "stoch_mhps_ratio",
    "stoch_gp_drw_sigma",
    "stoch_gp_drw_tau",
]


def _empty_stochastic_result() -> dict[str, float]:
    return {col: np.nan for col in STOCHASTIC_COLUMNS}


def _load_stochastic_functions(include_drw: bool) -> dict[str, object]:
    try:
        from malca.stats import structure_function, iar_phi_fit, mhps

        funcs: dict[str, object] = {
            "structure_function": structure_function,
            "iar_phi_fit": iar_phi_fit,
            "mhps": mhps,
        }
        if include_drw:
            from malca.stats import fit_drw

            funcs["fit_drw"] = fit_drw
        return funcs
    except Exception as exc:  # pragma: no cover - depends on optional deps
        raise RuntimeError(
            "Could not import stochastic feature functions. "
            "Install the optional stats dependencies before using "
            "--run-stochastic-postfilter."
        ) from exc


def _compute_feature_bundle(
    jd: np.ndarray,
    mag: np.ndarray,
    err: np.ndarray,
    *,
    include_drw: bool,
) -> dict[str, float]:
    funcs = _load_stochastic_functions(include_drw)
    out = _empty_stochastic_result()

    sf_amp, sf_gamma = funcs["structure_function"](mag, jd)
    out["stoch_sf_ml_amplitude"] = float(sf_amp) if np.isfinite(sf_amp) else np.nan
    out["stoch_sf_ml_gamma"] = float(sf_gamma) if np.isfinite(sf_gamma) else np.nan

    iar_phi = funcs["iar_phi_fit"](jd, mag, err)
    out["stoch_iar_phi"] = float(iar_phi) if np.isfinite(iar_phi) else np.nan

    mhps_result = funcs["mhps"](jd, mag, err)
    for key in ("mhps_high", "mhps_low", "mhps_non_zero", "mhps_pn_flag", "mhps_ratio"):
        value = mhps_result.get(key, np.nan)
        out[f"stoch_{key}"] = float(value) if np.isfinite(value) else np.nan

    if include_drw:
        drw_sigma, drw_tau = funcs["fit_drw"](jd, mag, err)
        out["stoch_gp_drw_sigma"] = float(drw_sigma) if np.isfinite(drw_sigma) else np.nan
        out["stoch_gp_drw_tau"] = float(drw_tau) if np.isfinite(drw_tau) else np.nan

    return out


def _load_clean_g_band(lc_path_str: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lc_path = Path(lc_path_str)
    if not lc_path.exists() or lc_path.suffix != ".dat2":
        raise FileNotFoundError(f"Light curve path not found or unsupported: {lc_path}")

    asassn_id = lc_path.stem
    df_g, _df_v = read_lc_dat2(asassn_id, str(lc_path.parent))
    if df_g.empty:
        raise ValueError(f"No g-band data found for {lc_path}")

    df_g = df_g.copy()
    df_g["JD"] += SKYPATROL_JD_OFFSET
    df = clean_lc(df_g)

    try:
        target_id = int(asassn_id)
    except ValueError:
        target_id = None

    if target_id == 17181160895:
        df = df[df["JD"] >= 2.458e6].copy()

    if df.empty:
        raise ValueError(f"No valid cleaned g-band rows for {lc_path}")

    jd = df["JD"].to_numpy(dtype=float)
    mag = df["mag"].to_numpy(dtype=float)
    err = df["error"].to_numpy(dtype=float)
    return jd, mag, err


def _compute_stochastic_row(payload: tuple[object, str, bool]) -> dict[str, object]:
    row_index, lc_path_str, include_drw = payload
    out: dict[str, object] = {"_row_index": row_index, "_error": None}
    out.update(_empty_stochastic_result())

    try:
        jd, mag, err = _load_clean_g_band(lc_path_str)
        out.update(_compute_feature_bundle(jd, mag, err, include_drw=include_drw))
    except Exception as exc:
        out["_error"] = str(exc)

    return out


def add_stochastic_postfilter_features(
    df: pd.DataFrame,
    *,
    lc_path_column: str = "lc_path",
    include_drw: bool = False,
    n_workers: int = LTV_WORKERS,
    verbose: bool = False,
) -> pd.DataFrame:
    """Add optional stochastic features to already-filtered LTV candidates."""
    if lc_path_column not in df.columns:
        if verbose:
            print("[ltv-stochastic] No lc_path column; skipping stochastic post-filter stage")
        return df

    valid_mask = df[lc_path_column].notna() & (df[lc_path_column].astype(str).str.strip() != "")
    if not valid_mask.any():
        if verbose:
            print("[ltv-stochastic] No valid lc_path values; skipping stochastic post-filter stage")
        return df

    # Fail early with a clear message if optional stats dependencies are missing.
    _load_stochastic_functions(include_drw)

    out_df = df.copy()
    for col in STOCHASTIC_COLUMNS:
        if col not in out_df.columns:
            out_df[col] = np.nan

    payloads = [
        (idx, str(out_df.at[idx, lc_path_column]), bool(include_drw))
        for idx in out_df.index[valid_mask]
    ]

    if verbose:
        print(
            f"[ltv-stochastic] Computing stochastic post-filter features for "
            f"{len(payloads):,} candidates"
        )

    results: list[dict[str, object]] = []
    if int(n_workers) <= 1:
        iterator = payloads
        if verbose:
            iterator = tqdm(iterator, total=len(payloads), desc="ltv-stochastic")
        for payload in iterator:
            results.append(_compute_stochastic_row(payload))
    else:
        with ProcessPoolExecutor(max_workers=int(n_workers)) as executor:
            futures = [executor.submit(_compute_stochastic_row, payload) for payload in payloads]
            iterator = as_completed(futures)
            if verbose:
                iterator = tqdm(iterator, total=len(futures), desc="ltv-stochastic")
            for future in iterator:
                results.append(future.result())

    if not results:
        return out_df

    errors = [str(row["_error"]) for row in results if row.get("_error")]
    if errors and verbose:
        print(f"[ltv-stochastic] {len(errors)} candidates failed stochastic enrichment")
        print(f"[ltv-stochastic] First error: {errors[0]}")

    features_df = pd.DataFrame(results).set_index("_row_index")
    for col in STOCHASTIC_COLUMNS:
        if col in features_df.columns:
            out_df.loc[features_df.index, col] = features_df[col].astype(float)

    if verbose:
        n_with_sf = int(out_df["stoch_sf_ml_amplitude"].notna().sum())
        print(f"[ltv-stochastic] Added stochastic features to {n_with_sf:,} candidates")

    return out_df
