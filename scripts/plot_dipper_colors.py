import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator

from malca.enrichment.characterize import crossmatch_2mass, crossmatch_allwise
from malca.catalogs.gaia_ids import normalize_gaia_source_ids, parse_gaia_source_id
from malca.enrichment.sed_alpha import (
    SED_ALPHA_BLUE_ANCHOR_MAX_MICRON,
    SED_ALPHA_LAMBDA_MAX_MICRON,
    SED_ALPHA_LAMBDA_MIN_MICRON,
    SED_ALPHA_MIN_POINTS,
    SED_ALPHA_RED_ANCHOR_MIN_MICRON,
    SED_ALPHA_COLUMNS,
    _prepared_alpha_points,
    compute_sed_alpha_features,
)
from malca.plotting.color_color_labels import (
    LABEL_H_KS_0,
    LABEL_IRAC1_IRAC2,
    LABEL_IRAC3_IRAC4,
    LABEL_J_H_0,
    LABEL_J_KS_0,
    LABEL_KS_W2_0,
    LABEL_KS_W3_0,
    LABEL_KS_W4_0,
    LABEL_W1_0,
    LABEL_W1_W2_0,
    LABEL_W1_W4_0,
    LABEL_W2_W3_0,
    LABEL_W3_W4_0,
    color_color_mag_label,
)
from malca.plotting.extinction import add_dereddened_ir_magnitudes, dereddened_color
from malca.plotting.irac import irac_vega_magnitude, irac_vega_magnitude_error
from malca.plotting.lightcurve_publication import apply_publication_rcparams


RUN_ROOT = Path("output/runs/runs_march18_bundle_all")
RESULTS_DIR = RUN_ROOT / "results"
REVIEW_DIR = RUN_ROOT / "review"
LABELS_CSV: Path | None = RESULTS_DIR / "march18_review_cmd_dustmaps_full.csv"
REVIEW_DB = REVIEW_DIR / "review.taxonomy_filled.db"
SED_PHOTOMETRY = REVIEW_DIR / "review.taxonomy_filled_sed_photometry.parquet"
SED_EXCESS_SUMMARY: Path | None = None
SPITZER_PHOTOMETRY: Path | None = None

# Compact markers retain the error bars while reducing overlap in the colour planes.
COLOR_POINT_MARKERSIZE = 4.0

EXCESS_CLASS_STYLES = {
    "robust": ("Robust", "#b2182b"),
    "probable": ("Probable", "#ef8a62"),
    "single_band_candidate": ("Single-band candidate", "#7b3294"),
    "none": ("No excess", "#2166ac"),
    "unassessable": ("Unassessable", "#969696"),
    "not_evaluated": ("Not evaluated", "#252525"),
}

WISE_COLS = ["w1", "w1_err", "w2", "w2_err", "w3", "w3_err", "w4", "w4_err"]
TMASS_COLS = ["tmass_j", "tmass_j_err", "tmass_h", "tmass_h_err", "tmass_k", "tmass_k_err"]
TEFF_COLS = [
    "teff50",
    "teff16",
    "teff84",
    "teff_gspphot",
    "teff_gspphot_lower",
    "teff_gspphot_upper",
    "teff",
    "teff_err_lower",
    "teff_err_upper",
]
NUMERIC_COLS = [
    "ra",
    "dec",
    "A_v_3d",
    *WISE_COLS,
    *TMASS_COLS,
    *TEFF_COLS,
    "H_K",
    "H_K_err",
    "j_h",
    "j_h_err",
    "j_k",
    "j_k_err",
    "ks_w2",
    "ks_w2_err",
    "ks_w3",
    "ks_w3_err",
    "ks_w4",
    "ks_w4_err",
    "w1_w2",
    "w1_w2_err",
    "w1_w3",
    "w1_w3_err",
    "w1_w4",
    "w1_w4_err",
    "w2_w3",
    "w2_w3_err",
    "w2_w4",
    "w2_w4_err",
    "w3_w4",
    "w3_w4_err",
    "vphas_r_ha",
    "vphas_r_i",
]


apply_publication_rcparams(plt)
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
})


def _finite_numeric(value: object) -> bool:
    try:
        return np.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _first_present(payload: dict, external: dict, keys: tuple[str, ...]) -> object:
    for source in (external, payload):
        for key in keys:
            value = source.get(key)
            if _finite_numeric(value):
                return value
            if isinstance(value, str) and value.strip() and value.strip().lower() not in {"nan", "none", "<na>"}:
                return value
    return np.nan


def _to_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _finite_count(df: pd.DataFrame, col: str) -> int:
    if col not in df.columns:
        return 0
    return int(np.isfinite(pd.to_numeric(df[col], errors="coerce")).sum())


def _hypot2(a: pd.Series, b: pd.Series) -> pd.Series:
    a_num = pd.to_numeric(a, errors="coerce")
    b_num = pd.to_numeric(b, errors="coerce")
    out = np.hypot(a_num, b_num)
    return pd.Series(out, index=a.index).where(np.isfinite(a_num) & np.isfinite(b_num))


def _load_dipper_ids(conn: sqlite3.Connection) -> list[str]:
    if LABELS_CSV is not None:
        labels = pd.read_csv(LABELS_CSV, dtype={"candidate_id": str})
        dipper_ids = labels.loc[
            labels["event_class"].astype(str).eq("dipper"), "candidate_id"
        ].astype(str).tolist()
        print(f"Loaded {len(dipper_ids)} labeled dippers from {LABELS_CSV}")
        return dipper_ids

    rows = conn.execute(
        "SELECT candidate_id FROM reviews WHERE event_class = 'dipper' ORDER BY candidate_id"
    ).fetchall()
    dipper_ids = [str(row[0]) for row in rows]
    print(f"Loaded {len(dipper_ids)} labeled dippers from {REVIEW_DB}:reviews")
    return dipper_ids


def _load_dipper_payloads() -> pd.DataFrame:
    requested_columns = [
        "candidate_id",
        "payload_json",
        "ra",
        "dec",
        "A_v_3d",
        "source_id",
        "gaia_id",
        "yso_class",
        "vphas_r_ha",
        "vphas_r_i",
        "teff50",
        "teff16",
        "teff84",
        "teff_gspphot",
        "teff_gspphot_lower",
        "teff_gspphot_upper",
        *WISE_COLS,
        *TMASS_COLS,
        "H_K",
        "w1_w2",
    ]
    with sqlite3.connect(REVIEW_DB) as conn:
        dipper_ids = _load_dipper_ids(conn)
        candidate_columns = {
            str(row[1]) for row in conn.execute("PRAGMA table_info(candidates)")
        }
        selected_columns = [col for col in requested_columns if col in candidate_columns]
        candidates = pd.read_sql_query(
            f"SELECT {', '.join(selected_columns)} FROM candidates",
            conn,
        )
    candidates["candidate_id"] = candidates["candidate_id"].astype(str)
    candidate_by_id = candidates.set_index("candidate_id", drop=False)

    rows: list[dict[str, object]] = []
    missing_ids: list[str] = []
    for cid in dipper_ids:
        if str(cid) not in candidate_by_id.index:
            missing_ids.append(str(cid))
            continue
        candidate = candidate_by_id.loc[str(cid)]
        payload = json.loads(candidate.get("payload_json") or "{}")
        external = payload.get("external_stats", {}) if isinstance(payload, dict) else {}

        def structured_or_payload(column: str, *fallback_keys: str) -> object:
            value = candidate.get(column, np.nan)
            if _finite_numeric(value):
                return value
            if isinstance(value, str) and value.strip() and value.strip().lower() not in {"nan", "none", "<na>"}:
                return value
            keys = fallback_keys or (column,)
            return _first_present(payload, external, keys)

        row = {
            "candidate_id": str(cid),
            "ra": structured_or_payload("ra", "ra", "ra_deg"),
            "dec": structured_or_payload("dec", "dec", "dec_deg"),
            "A_v_3d": structured_or_payload("A_v_3d"),
            "source_id": structured_or_payload("source_id", "source_id", "gaia_id"),
            "yso_class": structured_or_payload("yso_class"),
            "vphas_r_ha": structured_or_payload("vphas_r_ha"),
            "vphas_r_i": structured_or_payload("vphas_r_i"),
            "teff50": structured_or_payload("teff50"),
            "teff16": structured_or_payload("teff16"),
            "teff84": structured_or_payload("teff84"),
            "teff_gspphot": structured_or_payload("teff_gspphot"),
            "teff_gspphot_lower": structured_or_payload("teff_gspphot_lower"),
            "teff_gspphot_upper": structured_or_payload("teff_gspphot_upper"),
        }
        for col in [*WISE_COLS, *TMASS_COLS, "H_K", "w1_w2"]:
            row[col] = structured_or_payload(col)
        rows.append(row)

    if missing_ids:
        print(f"Warning: {len(missing_ids)} labeled dippers missing from review DB: {', '.join(missing_ids)}")
    df = pd.DataFrame(rows)
    df = _to_numeric(df, NUMERIC_COLS)
    print(
        "Loaded structured candidate data: "
        f"coordinates={_finite_xy_count(df, 'ra', 'dec')}/{len(df)}, "
        f"W1-W2={_finite_xy_count(df, 'w1', 'w2')}/{len(df)}, "
        f"W1-W4={_finite_xy_count(df, 'w1', 'w4')}/{len(df)}"
    )
    return df


def _needs_refresh(df: pd.DataFrame, cols: list[str]) -> bool:
    if df.empty:
        return False
    finite_coord = np.isfinite(df["ra"]) & np.isfinite(df["dec"])
    if not finite_coord.any():
        return False
    return any(_finite_count(df.loc[finite_coord], col) < int(finite_coord.sum()) for col in cols)


def _refresh_catalog_photometry(df: pd.DataFrame, *, refresh_missing: bool) -> pd.DataFrame:
    out = df.copy()
    n_coord = int((np.isfinite(out["ra"]) & np.isfinite(out["dec"])).sum())
    print(f"Catalog refresh coordinate rows: {n_coord}/{len(out)}")

    if not refresh_missing:
        print("Using stored catalog photometry; missing-catalog refresh disabled.")
        return _to_numeric(out, NUMERIC_COLS)

    if _needs_refresh(out, WISE_COLS):
        before = {col: _finite_count(out, col) for col in WISE_COLS}
        print(f"Refreshing AllWISE photometry/errors; finite before: {before}")
        out = crossmatch_allwise(out)
        out = _to_numeric(out, WISE_COLS)
        after = {col: _finite_count(out, col) for col in WISE_COLS}
        print(f"AllWISE finite after: {after}")
    else:
        print("AllWISE photometry/errors already complete for coordinate-bearing rows.")

    if _needs_refresh(out, TMASS_COLS):
        before = {col: _finite_count(out, col) for col in TMASS_COLS}
        print(f"Refreshing 2MASS photometry/errors; finite before: {before}")
        out = crossmatch_2mass(out)
        out = _to_numeric(out, TMASS_COLS)
        after = {col: _finite_count(out, col) for col in TMASS_COLS}
        print(f"2MASS finite after: {after}")
    else:
        print("2MASS photometry/errors already complete for coordinate-bearing rows.")

    return _to_numeric(out, NUMERIC_COLS)


def _refresh_gaia_teff_bounds(df: pd.DataFrame, *, refresh_missing: bool) -> pd.DataFrame:
    out = df.copy()
    if not refresh_missing:
        print("Using stored Gaia Teff bounds; missing-catalog refresh disabled.")
        return _to_numeric(out, NUMERIC_COLS)
    if "source_id" not in out.columns:
        return out

    ids = normalize_gaia_source_ids(out["source_id"])
    if not ids:
        print("No parseable Gaia source IDs available for Teff uncertainty refresh.")
        return out
    if all(_finite_count(out, col) >= _finite_count(out, "teff_gspphot") for col in ("teff_gspphot_lower", "teff_gspphot_upper")):
        print("Gaia Teff bounds already complete for rows with Gaia Teff.")
        return _to_numeric(out, NUMERIC_COLS)

    try:
        from astroquery.gaia import Gaia
    except Exception as exc:
        print(f"Warning: cannot import astroquery Gaia for Teff bounds: {exc}")
        return _to_numeric(out, NUMERIC_COLS)

    query_ids = ",".join(ids)
    query = f"""
        SELECT source_id, teff_gspphot, teff_gspphot_lower, teff_gspphot_upper
        FROM gaiadr3.astrophysical_parameters
        WHERE source_id IN ({query_ids})
    """
    try:
        print(f"Refreshing Gaia GSP-Phot Teff bounds for {len(ids)} source IDs...")
        result = Gaia.launch_job(query).get_results().to_pandas()
    except Exception as exc:
        print(f"Warning: Gaia Teff uncertainty refresh failed: {exc}")
        return _to_numeric(out, NUMERIC_COLS)

    if result.empty:
        print("Gaia Teff uncertainty refresh returned no rows.")
        return _to_numeric(out, NUMERIC_COLS)

    result.columns = [str(col).lower() for col in result.columns]
    result["source_id"] = result["source_id"].map(parse_gaia_source_id)
    out["_gaia_source_id"] = out["source_id"].map(parse_gaia_source_id)
    result = result.dropna(subset=["source_id"]).drop_duplicates(subset=["source_id"], keep="last")
    joined = out.merge(
        result,
        left_on="_gaia_source_id",
        right_on="source_id",
        how="left",
        suffixes=("", "_gaia"),
    )

    for col in ("teff_gspphot", "teff_gspphot_lower", "teff_gspphot_upper"):
        gaia_col = f"{col}_gaia"
        if gaia_col not in joined.columns:
            continue
        current = pd.to_numeric(joined[col], errors="coerce") if col in joined else pd.Series(np.nan, index=joined.index)
        fetched = pd.to_numeric(joined[gaia_col], errors="coerce")
        joined[col] = current.where(np.isfinite(current), fetched)

    drop_cols = [col for col in joined.columns if col.endswith("_gaia") or col == "_gaia_source_id"]
    if "source_id_gaia" in joined.columns:
        drop_cols.append("source_id_gaia")
    joined = joined.drop(columns=sorted(set(drop_cols)), errors="ignore")
    print(
        "Gaia Teff finite after: "
        f"teff={_finite_count(joined, 'teff_gspphot')}, "
        f"lower={_finite_count(joined, 'teff_gspphot_lower')}, "
        f"upper={_finite_count(joined, 'teff_gspphot_upper')}"
    )
    return _to_numeric(joined, NUMERIC_COLS)


def _compute_colors(df: pd.DataFrame) -> pd.DataFrame:
    out = add_dereddened_ir_magnitudes(df)

    computed_hk = out["tmass_h"] - out["tmass_k"]
    out["H_K"] = computed_hk.where(np.isfinite(computed_hk), out["H_K"])
    out["H_K_err"] = _hypot2(out["tmass_h_err"], out["tmass_k_err"])
    out["H_K_0"] = dereddened_color(out, "tmass_h", "tmass_k")

    for left, right, color in (
        ("tmass_j", "tmass_h", "j_h"),
        ("tmass_j", "tmass_k", "j_k"),
        ("tmass_k", "w2", "ks_w2"),
        ("tmass_k", "w3", "ks_w3"),
        ("tmass_k", "w4", "ks_w4"),
    ):
        out[color] = out[left] - out[right]
        out[f"{color}_err"] = _hypot2(out[f"{left}_err"], out[f"{right}_err"])
        out[f"{color}_0"] = dereddened_color(out, left, right)

    for left, right in (("w1", "w2"), ("w1", "w3"), ("w1", "w4"), ("w2", "w3"), ("w2", "w4"), ("w3", "w4")):
        color = f"{left}_{right}"
        err = f"{color}_err"
        values = out[left] - out[right]
        out[color] = values.where(np.isfinite(values), out[color] if color in out.columns else np.nan)
        out[err] = _hypot2(out[f"{left}_err"], out[f"{right}_err"])
        out[f"{color}_0"] = dereddened_color(out, left, right)

    return _to_numeric(out, NUMERIC_COLS)


def _choose_teff(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    teff50 = pd.to_numeric(out.get("teff50"), errors="coerce")
    teff_gsp = pd.to_numeric(out.get("teff_gspphot"), errors="coerce")
    out["teff"] = teff50.where(np.isfinite(teff50), teff_gsp)
    out["teff_source"] = np.select(
        [np.isfinite(teff50), ~np.isfinite(teff50) & np.isfinite(teff_gsp)],
        ["teff50", "teff_gspphot"],
        default="missing",
    )
    teff16 = pd.to_numeric(out.get("teff16"), errors="coerce")
    teff84 = pd.to_numeric(out.get("teff84"), errors="coerce")
    gsp_lower = pd.to_numeric(out.get("teff_gspphot_lower"), errors="coerce")
    gsp_upper = pd.to_numeric(out.get("teff_gspphot_upper"), errors="coerce")

    sh_lower = out["teff"] - teff16
    sh_upper = teff84 - out["teff"]
    gsp_lower_err = out["teff"] - gsp_lower
    gsp_upper_err = gsp_upper - out["teff"]
    use_sh = out["teff_source"].eq("teff50") & np.isfinite(sh_lower) & np.isfinite(sh_upper)
    use_gsp = out["teff_source"].eq("teff_gspphot") & np.isfinite(gsp_lower_err) & np.isfinite(gsp_upper_err)
    out["teff_err_lower"] = np.nan
    out["teff_err_upper"] = np.nan
    out.loc[use_sh, "teff_err_lower"] = sh_lower[use_sh].clip(lower=0.0)
    out.loc[use_sh, "teff_err_upper"] = sh_upper[use_sh].clip(lower=0.0)
    out.loc[use_gsp, "teff_err_lower"] = gsp_lower_err[use_gsp].clip(lower=0.0)
    out.loc[use_gsp, "teff_err_upper"] = gsp_upper_err[use_gsp].clip(lower=0.0)
    return _to_numeric(out, NUMERIC_COLS)


def _load_sed_alpha(df: pd.DataFrame) -> pd.DataFrame:
    if not SED_PHOTOMETRY.exists():
        print(f"Warning: missing SED photometry parquet: {SED_PHOTOMETRY}")
        return pd.DataFrame(columns=SED_ALPHA_COLUMNS)

    sed_rows = pd.read_parquet(SED_PHOTOMETRY)
    sed_rows["candidate_id"] = sed_rows["candidate_id"].astype(str)
    dipper_rows = sed_rows[sed_rows["candidate_id"].isin(set(df["candidate_id"].astype(str)))].copy()
    candidates = df[["candidate_id", "A_v_3d"]].copy()
    alpha = compute_sed_alpha_features(candidates, dipper_rows)
    alpha["candidate_id"] = alpha["candidate_id"].astype(str)
    alpha_err = _compute_sed_alpha_uncertainties(candidates, dipper_rows)
    alpha = alpha.merge(alpha_err, on="candidate_id", how="left")
    n_ok = int((alpha["sed_alpha_status"].astype(str) == "ok").sum()) if not alpha.empty else 0
    print(f"Computed SED alpha for {n_ok}/{len(df)} dippers from {SED_PHOTOMETRY}")
    if not alpha.empty:
        print(f"SED alpha status counts: {alpha['sed_alpha_status'].fillna('NA').value_counts().to_dict()}")
        print(f"SED alpha class counts: {alpha['sed_alpha_class'].fillna('NA').value_counts().to_dict()}")
    return alpha


def _load_sed_excess_classes() -> pd.DataFrame:
    columns = ["candidate_id", "excess_class"]
    if SED_EXCESS_SUMMARY is None or not SED_EXCESS_SUMMARY.exists():
        print("No SED-excess summary supplied; all dippers will be marked not evaluated.")
        return pd.DataFrame(columns=columns)

    excess = pd.read_csv(SED_EXCESS_SUMMARY, dtype={"candidate_id": str})
    missing = [col for col in columns if col not in excess.columns]
    if missing:
        raise ValueError(
            f"SED-excess summary is missing required columns {missing}: {SED_EXCESS_SUMMARY}"
        )
    excess = excess[columns].drop_duplicates(subset="candidate_id", keep="last")
    excess["candidate_id"] = excess["candidate_id"].astype(str)
    print(
        f"Loaded {len(excess)} SED-excess classifications from {SED_EXCESS_SUMMARY}: "
        f"{excess['excess_class'].fillna('not_evaluated').value_counts().to_dict()}"
    )
    return excess


def _weighted_slope_error(x: np.ndarray, sigma_y: np.ndarray) -> float:
    if len(x) < SED_ALPHA_MIN_POINTS:
        return np.nan
    if not (np.isfinite(x).all() and np.isfinite(sigma_y).all() and (sigma_y > 0).all()):
        return np.nan
    design = np.column_stack([x, np.ones_like(x)])
    weights = 1.0 / np.square(sigma_y)
    try:
        cov = np.linalg.inv(design.T @ (weights[:, None] * design))
    except np.linalg.LinAlgError:
        return np.nan
    err = float(np.sqrt(cov[0, 0]))
    return err if np.isfinite(err) else np.nan


def _sed_alpha_error_for_candidate(sed_rows: pd.DataFrame, candidate: dict | pd.Series | None) -> float:
    points = _prepared_alpha_points(sed_rows, candidate)
    if points.empty or len(points) < SED_ALPHA_MIN_POINTS:
        return np.nan
    if not (points["lambda_micron"] <= SED_ALPHA_BLUE_ANCHOR_MAX_MICRON).any():
        return np.nan
    if not (points["lambda_micron"] >= SED_ALPHA_RED_ANCHOR_MIN_MICRON).any():
        return np.nan

    x = np.log10(points["lambda_micron"].to_numpy(dtype=float))
    if np.unique(x).size < 2:
        return np.nan

    lum = pd.to_numeric(points.get("lambda_l_lambda"), errors="coerce")
    lum_err = pd.to_numeric(points.get("lambda_l_lambda_err"), errors="coerce")
    flux = pd.to_numeric(points.get("flux_lambda"), errors="coerce")
    flux_err = pd.to_numeric(points.get("flux_lambda_err"), errors="coerce")

    if np.isfinite(lum).all() and (lum > 0).all() and np.isfinite(lum_err).all() and (lum_err > 0).all():
        rel_err = (lum_err / lum).to_numpy(dtype=float)
    elif np.isfinite(flux).all() and (flux > 0).all() and np.isfinite(flux_err).all() and (flux_err > 0).all():
        rel_err = (flux_err / flux).to_numpy(dtype=float)
    else:
        return np.nan
    sigma_log_y = rel_err / np.log(10.0)
    return _weighted_slope_error(x, sigma_log_y)


def _compute_sed_alpha_uncertainties(candidates: pd.DataFrame, sed_rows: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame(columns=["candidate_id", "sed_alpha_err"])
    rows = pd.DataFrame(sed_rows)
    candidates = candidates.copy()
    candidates["candidate_id"] = candidates["candidate_id"].astype(str)
    if rows.empty or "candidate_id" not in rows.columns:
        return pd.DataFrame({"candidate_id": candidates["candidate_id"], "sed_alpha_err": np.nan})
    rows["candidate_id"] = rows["candidate_id"].astype(str)

    out = []
    for _, candidate in candidates.iterrows():
        cid = str(candidate["candidate_id"])
        candidate_rows = rows[rows["candidate_id"] == cid]
        out.append(
            {
                "candidate_id": cid,
                "sed_alpha_err": _sed_alpha_error_for_candidate(candidate_rows, candidate),
            }
        )
    return pd.DataFrame(out)


def _plot_errorbar_points(
    ax: plt.Axes,
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    xerr: str | None,
    yerr: str | None,
    label: str,
) -> int:
    plot_df = df[np.isfinite(df[x]) & np.isfinite(df[y])].copy()
    if plot_df.empty:
        return 0

    xerr_ok = np.isfinite(plot_df[xerr]) if xerr else pd.Series(False, index=plot_df.index)
    yerr_ok = np.isfinite(plot_df[yerr]) if yerr else pd.Series(False, index=plot_df.index)
    label_used = False

    def draw(mask: pd.Series, *, draw_xerr: bool, draw_yerr: bool) -> None:
        nonlocal label_used
        sub = plot_df.loc[mask]
        if sub.empty:
            return
        ax.errorbar(
            sub[x],
            sub[y],
            xerr=sub[xerr] if draw_xerr and xerr else None,
            yerr=sub[yerr] if draw_yerr and yerr else None,
            fmt="o",
            color="k",
            ecolor="0.25",
            elinewidth=0.8,
            capsize=3,
            capthick=0.8,
            markerfacecolor="k",
            markeredgecolor="k",
            markersize=COLOR_POINT_MARKERSIZE,
            linestyle="none",
            zorder=5,
            label=label if not label_used else None,
        )
        label_used = True

    draw(xerr_ok & yerr_ok, draw_xerr=True, draw_yerr=True)
    draw(xerr_ok & ~yerr_ok, draw_xerr=True, draw_yerr=False)
    draw(~xerr_ok & yerr_ok, draw_xerr=False, draw_yerr=True)
    draw(~xerr_ok & ~yerr_ok, draw_xerr=False, draw_yerr=False)
    return len(plot_df)


def _finish_color_axis(ax: plt.Axes) -> None:
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))


def _save_color_plot(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    xerr: str,
    yerr: str,
    xlabel: str,
    ylabel: str,
    output: Path,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    annotation: str | None = None,
) -> int:
    fig, ax = plt.subplots(figsize=(6, 5))
    n = _plot_errorbar_points(ax, df, x=x, y=y, xerr=xerr, yerr=yerr, label=f"Dippers ({_finite_xy_count(df, x, y)})")
    if n == 0:
        plt.close(fig)
        raise RuntimeError(
            f"Refusing to save empty color plot {output}: no finite {x}/{y} pairs"
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if annotation:
        ax.text(
            0.98,
            0.02,
            annotation,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color="0.25",
        )
    _finish_color_axis(ax)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    print(f"Saved {output} ({n} finite points)")
    _print_missing_error_summary(df, x, y, xerr, yerr)
    return n


def _finite_xy_count(df: pd.DataFrame, x: str, y: str) -> int:
    return int((np.isfinite(df[x]) & np.isfinite(df[y])).sum())


def _print_missing_error_summary(df: pd.DataFrame, x: str, y: str, xerr: str, yerr: str) -> None:
    rows = df[np.isfinite(df[x]) & np.isfinite(df[y])]
    missing_x = int((~np.isfinite(rows[xerr])).sum()) if xerr in rows else len(rows)
    missing_y = int((~np.isfinite(rows[yerr])).sum()) if yerr in rows else len(rows)
    if missing_x or missing_y:
        print(
            f"Warning: {x} vs {y} plotted with missing uncertainties for "
            f"{missing_x} x-error rows and {missing_y} y-error rows"
        )


def _plot_vphas(df: pd.DataFrame, out_dir: Path) -> None:
    df_vphas = df.dropna(subset=["vphas_r_i", "vphas_r_ha"])
    if df_vphas.empty:
        print("No VPHAS+ data available to plot.")
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(
        df_vphas["vphas_r_i"],
        df_vphas["vphas_r_ha"],
        color="k",
        edgecolor="k",
        s=40,
        zorder=5,
        label=f"Dippers ({len(df_vphas)})",
    )
    ax.set_xlabel(color_color_mag_label(r"r", r"i", "r", "i"))
    ax.set_ylabel(color_color_mag_label(r"r", r"H\alpha", "r", "Ha"))
    _finish_color_axis(ax)
    fig.tight_layout()
    out_vphas = out_dir / "dipper_vphas_color_color.pdf"
    fig.savefig(out_vphas)
    plt.close(fig)
    print(f"Saved {out_vphas}")


def _plot_sed_alpha(summary: pd.DataFrame, out_dir: Path) -> None:
    finite = summary[np.isfinite(summary["sed_alpha"])].copy()
    if finite.empty:
        print("No finite SED alpha values available to plot.")
        return
    finite = finite.sort_values("sed_alpha").reset_index(drop=True)
    finite["ecdf"] = np.arange(1, len(finite) + 1, dtype=float) / len(finite)
    x_min = min(-3.2, float(finite["sed_alpha"].min()) - 0.15)
    x_max = max(0.6, float(finite["sed_alpha"].max()) + 0.15)

    class_specs = [
        (x_min, -1.6, "Class III/photosphere", "0.92"),
        (-1.6, -0.3, "Class II", "#ffedd5"),
        (-0.3, 0.3, "Flat", "#fff7d6"),
        (0.3, x_max, "Class I", "#fee2e2"),
    ]
    class_counts = {
        "Class III/photosphere": int((finite["sed_alpha"] < -1.6).sum()),
        "Class II": int(finite["sed_alpha"].between(-1.6, -0.3, inclusive="left").sum()),
        "Flat": int(finite["sed_alpha"].between(-0.3, 0.3, inclusive="left").sum()),
        "Class I": int((finite["sed_alpha"] >= 0.3).sum()),
    }
    unknown_count = int((~np.isfinite(summary["sed_alpha"])).sum())

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for lo, hi, _, color in class_specs:
        ax.axvspan(max(lo, x_min), min(hi, x_max), color=color, zorder=0)
    for value in (-1.6, -0.3, 0.3):
        ax.axvline(value, color="0.25", linestyle="--", linewidth=0.8, zorder=1)

    ax.step(
        finite["sed_alpha"],
        finite["ecdf"],
        where="post",
        color="0.2",
        linewidth=1.4,
        zorder=3,
    )
    point_colors = np.select(
        [
            finite["sed_alpha"] < -1.6,
            finite["sed_alpha"] < -0.3,
            finite["sed_alpha"] < 0.3,
        ],
        ["#4d4d4d", "#d97706", "#ca8a04"],
        default="#b2182b",
    )
    ax.scatter(
        finite["sed_alpha"],
        finite["ecdf"],
        s=26,
        c=point_colors,
        edgecolors="white",
        linewidths=0.35,
        zorder=4,
    )

    count_text = (
        f"Class III/photosphere: {class_counts['Class III/photosphere']}    "
        f"Class II: {class_counts['Class II']}    "
        f"Flat: {class_counts['Flat']}    Class I: {class_counts['Class I']}\n"
        f"Finite: {len(finite)}/{len(summary)}    Unknown: {unknown_count}"
    )
    ax.text(
        0.02,
        0.98,
        count_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "0.8", "alpha": 0.9},
        zorder=7,
    )

    tail = finite.nlargest(min(5, len(finite)), "sed_alpha").sort_values("sed_alpha", ascending=False)
    label_y = np.linspace(0.88, 0.52, len(tail))
    for (_, row), text_y in zip(tail.iterrows(), label_y):
        ax.annotate(
            str(row["candidate_id"]),
            xy=(float(row["sed_alpha"]), float(row["ecdf"])),
            xytext=(0.02, float(text_y)),
            textcoords="data",
            ha="left",
            va="center",
            fontsize=7.5,
            arrowprops={"arrowstyle": "-", "color": "0.35", "linewidth": 0.6},
            zorder=6,
        )

    ax.set_xlabel(r"SED $\alpha$ [2-24 $\mu$m]")
    ax.set_ylabel("Empirical cumulative fraction")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0.0, 1.02)
    ax.grid(axis="y", color="white", linewidth=0.8, alpha=0.8, zorder=1)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    fig.tight_layout()
    out_path = out_dir / "dipper_sed_alpha_summary.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(
        f"Saved {out_path} ({len(finite)} finite SED alpha values; "
        f"class counts={class_counts}, unknown={unknown_count})"
    )


def _shade_sed_alpha_bands(ax: plt.Axes, y_min: float, y_max: float, *, labels: bool) -> None:
    bands = [
        (y_min, -1.6, "Class III/photosphere", "0.92"),
        (-1.6, -0.3, "Class II", "#ffedd5"),
        (-0.3, 0.3, "Flat", "#fff7d6"),
        (0.3, y_max, "Class I", "#fee2e2"),
    ]
    for lo, hi, label, color in bands:
        ax.axhspan(max(lo, y_min), min(hi, y_max), color=color, zorder=0)
        if labels:
            y_text = (max(lo, y_min) + min(hi, y_max)) / 2
            if y_min <= y_text <= y_max:
                x_pos = 0.985 if label in {"Class II", "Class III/photosphere"} else 0.015
                ax.text(
                    x_pos,
                    y_text,
                    label,
                    transform=ax.get_yaxis_transform(),
                    ha="right" if x_pos > 0.5 else "left",
                    va="center",
                    fontsize=8,
                    zorder=2,
                )
    for value in (-1.6, -0.3, 0.3):
        ax.axhline(value, color="0.25", linestyle="--", linewidth=0.8, zorder=1)


def _draw_teff_alpha_points(ax: plt.Axes, data: pd.DataFrame, *, markersize: float) -> None:
    for _, row in data.iterrows():
        class_key = str(row.get("excess_class", "not_evaluated"))
        if class_key not in EXCESS_CLASS_STYLES:
            class_key = "not_evaluated"
        _, color = EXCESS_CLASS_STYLES[class_key]
        alpha_err = pd.to_numeric(pd.Series([row.get("sed_alpha_err")]), errors="coerce").iloc[0]
        has_alpha_err = bool(np.isfinite(alpha_err) and alpha_err > 0)
        lower = pd.to_numeric(pd.Series([row.get("teff_err_lower")]), errors="coerce").iloc[0]
        upper = pd.to_numeric(pd.Series([row.get("teff_err_upper")]), errors="coerce").iloc[0]
        has_teff_err = bool(np.isfinite(lower) and np.isfinite(upper) and lower >= 0 and upper >= 0)
        xerr = np.array([[lower], [upper]]) if has_teff_err else None
        ax.errorbar(
            [float(row["teff"])],
            [float(row["sed_alpha"])],
            xerr=xerr,
            yerr=[[alpha_err], [alpha_err]] if has_alpha_err else None,
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=0.7,
            capsize=2.5,
            capthick=0.7,
            markerfacecolor=color if has_alpha_err else "white",
            markeredgecolor=color,
            markeredgewidth=1.1,
            markersize=markersize,
            linestyle="none",
            alpha=0.9 if class_key != "unassessable" else 0.72,
            zorder=5,
        )


def _plot_teff_sed_alpha(summary: pd.DataFrame, out_dir: Path) -> None:
    finite = summary[np.isfinite(summary["sed_alpha"]) & np.isfinite(summary["teff"])].copy()
    if finite.empty:
        print("No finite paired SED alpha and Teff values available to plot.")
        return

    y_min = min(-3.1, float(finite["sed_alpha"].min()) - 0.3)
    y_max = max(0.6, float(finite["sed_alpha"].max()) + 0.3)

    fig, ax = plt.subplots(figsize=(7.6, 5.8))
    _shade_sed_alpha_bands(ax, y_min, y_max, labels=False)
    _draw_teff_alpha_points(ax, finite, markersize=6.0)
    ax.set_xlabel(r"$T_{\rm eff}$ [K]")
    ax.set_ylabel(r"SED $\alpha$ [2-24 $\mu$m]")
    ax.set_ylim(y_min, y_max)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    class_counts = finite["excess_class"].fillna("not_evaluated").value_counts()
    class_handles = []
    for key, (label, color) in EXCESS_CLASS_STYLES.items():
        count = int(class_counts.get(key, 0))
        if count == 0:
            continue
        class_handles.append(
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                markerfacecolor=color,
                markeredgecolor=color,
                markersize=6,
                label=f"{label} ({count})",
            )
        )
    uncertainty_handles = [
        Line2D([], [], marker="o", linestyle="none", markerfacecolor="0.25", markeredgecolor="0.25", markersize=6, label=r"$\alpha$ uncertainty available"),
        Line2D([], [], marker="o", linestyle="none", markerfacecolor="white", markeredgecolor="0.25", markersize=6, label=r"$\alpha$ uncertainty unavailable"),
    ]
    fig.legend(
        handles=[*class_handles, *uncertainty_handles],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=4,
        frameon=True,
        fontsize=8,
    )
    fig.subplots_adjust(left=0.12, right=0.98, top=0.97, bottom=0.22)
    out_path = out_dir / "dipper_sed_alpha_teff.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    xerr_ok = (
        np.isfinite(finite["teff_err_lower"]) & np.isfinite(finite["teff_err_upper"])
        if {"teff_err_lower", "teff_err_upper"}.issubset(finite.columns)
        else pd.Series(False, index=finite.index)
    )
    yerr_ok = (
        np.isfinite(finite["sed_alpha_err"]) & (finite["sed_alpha_err"] > 0)
        if "sed_alpha_err" in finite
        else pd.Series(False, index=finite.index)
    )
    missing_alpha = int((~np.isfinite(summary["sed_alpha"]) & np.isfinite(summary["teff"])).sum())
    missing_teff = int((np.isfinite(summary["sed_alpha"]) & ~np.isfinite(summary["teff"])).sum())
    missing_teff_err = int((~xerr_ok).sum())
    missing_alpha_err = int((~yerr_ok).sum())
    print(
        f"Saved {out_path} ({len(finite)} finite SED alpha/Teff pairs; "
        f"{missing_teff} missing Teff, {missing_alpha} missing SED alpha; "
        f"{missing_alpha_err} missing alpha error, {missing_teff_err} missing Teff error)"
    )


def _add_missing_flags(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    for col in [
        "tmass_h_err",
        "tmass_k_err",
        "w1_err",
        "w2_err",
        "w3_err",
        "w4_err",
        "H_K_err",
        "w1_w2_err",
        "w2_w3_err",
        "w1_w4_err",
        "sed_alpha_err",
        "teff_err_lower",
        "teff_err_upper",
    ]:
        if col in out.columns:
            out[f"missing_{col}"] = ~np.isfinite(pd.to_numeric(out[col], errors="coerce"))
    return out


def _default_spitzer_photometry(results_dir: Path) -> Path | None:
    """Return the newest run-local Spitzer archive cache, when present."""
    candidates = sorted(results_dir.glob("sed_archive_*/cache/catalogs/sed/spitzer.parquet"))
    return candidates[-1] if candidates else None


def _load_spitzer_irac_colors(candidate_ids: pd.Series) -> pd.DataFrame:
    """Load complete four-band IRAC colours from the optional archive cache."""
    columns = [
        "candidate_id",
        "band",
        "observed_flux_nu_jy",
        "observed_flux_nu_jy_err",
        "sep_arcsec",
    ]
    if SPITZER_PHOTOMETRY is None:
        return pd.DataFrame(columns=["candidate_id", "irac_36_45", "irac_36_45_err", "irac_58_80", "irac_58_80_err"])

    raw = pd.read_parquet(SPITZER_PHOTOMETRY)
    missing = sorted(set(columns) - set(raw.columns))
    if missing:
        raise ValueError(f"Spitzer cache is missing required columns: {missing}")
    raw["candidate_id"] = raw["candidate_id"].astype(str)
    target_ids = set(candidate_ids.astype(str))
    bands = ("IRAC1", "IRAC2", "IRAC3", "IRAC4")
    rows = raw.loc[
        raw["candidate_id"].isin(target_ids) & raw["band"].isin(bands),
        columns,
    ].copy()
    if rows.empty:
        return pd.DataFrame(columns=["candidate_id", "irac_36_45", "irac_36_45_err", "irac_58_80", "irac_58_80_err"])

    rows["sep_arcsec"] = pd.to_numeric(rows["sep_arcsec"], errors="coerce")
    rows = rows.sort_values(["candidate_id", "band", "sep_arcsec"], na_position="last")
    rows = rows.drop_duplicates(["candidate_id", "band"], keep="first")
    complete_ids = (
        rows.groupby("candidate_id")["band"].agg(set).loc[lambda values: values.map(lambda value: set(bands).issubset(value))].index
    )
    rows = rows[rows["candidate_id"].isin(complete_ids)]
    if rows.empty:
        return pd.DataFrame(columns=["candidate_id", "irac_36_45", "irac_36_45_err", "irac_58_80", "irac_58_80_err"])

    wide = rows.set_index(["candidate_id", "band"])[["observed_flux_nu_jy", "observed_flux_nu_jy_err"]].unstack("band")
    out = pd.DataFrame(index=wide.index)
    for band in bands:
        flux = pd.to_numeric(wide[("observed_flux_nu_jy", band)], errors="coerce").to_numpy(dtype=float)
        flux_err = pd.to_numeric(wide[("observed_flux_nu_jy_err", band)], errors="coerce").to_numpy(dtype=float)
        out[f"{band.lower()}_mag"] = irac_vega_magnitude(flux, band)
        out[f"{band.lower()}_mag_err"] = irac_vega_magnitude_error(flux, flux_err)
    out["irac_36_45"] = out["irac1_mag"] - out["irac2_mag"]
    out["irac_36_45_err"] = _hypot2(out["irac1_mag_err"], out["irac2_mag_err"])
    out["irac_58_80"] = out["irac3_mag"] - out["irac4_mag"]
    out["irac_58_80_err"] = _hypot2(out["irac3_mag_err"], out["irac4_mag_err"])
    return out.reset_index()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate dipper optical/IR color and SED-alpha summary plots."
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        help="Run directory. When set, labels default to the review DB and outputs to RUN_ROOT/results.",
    )
    parser.add_argument("--review-db", type=Path, help="Override the review SQLite database.")
    parser.add_argument(
        "--labels-csv",
        type=Path,
        help="Optional CSV containing candidate_id and event_class; otherwise use review DB labels.",
    )
    parser.add_argument("--sed-photometry", type=Path, help="Override the SED photometry parquet.")
    parser.add_argument(
        "--sed-excess-summary",
        type=Path,
        help="Optional candidate-level WISE excess summary used to color the Teff plot.",
    )
    parser.add_argument(
        "--spitzer-photometry",
        type=Path,
        help="Optional Spitzer archive-cache parquet used for the diagnostic IRAC colour plot.",
    )
    parser.add_argument(
        "--refresh-missing-catalog",
        action="store_true",
        help="Query remote AllWISE/2MASS services for missing stored measurements.",
    )
    return parser.parse_args()


def _configure_paths(args: argparse.Namespace) -> None:
    global RUN_ROOT, RESULTS_DIR, REVIEW_DIR, LABELS_CSV, REVIEW_DB, SED_PHOTOMETRY, SED_EXCESS_SUMMARY, SPITZER_PHOTOMETRY

    if args.run_root is not None:
        RUN_ROOT = args.run_root
        RESULTS_DIR = RUN_ROOT / "results"
        REVIEW_DIR = RUN_ROOT / "review"
        LABELS_CSV = args.labels_csv
        REVIEW_DB = args.review_db or (REVIEW_DIR / "review.db")
        default_sed = RESULTS_DIR / "sed_photometry.parquet"
        SED_PHOTOMETRY = args.sed_photometry or default_sed
        default_excess = RESULTS_DIR / "marked_dipper_seds" / "marked_dipper_sed_excess_summary.csv"
        SED_EXCESS_SUMMARY = args.sed_excess_summary or (default_excess if default_excess.exists() else None)
        SPITZER_PHOTOMETRY = args.spitzer_photometry or _default_spitzer_photometry(RESULTS_DIR)
    else:
        if args.labels_csv is not None:
            LABELS_CSV = args.labels_csv
        if args.review_db is not None:
            REVIEW_DB = args.review_db
        if args.sed_photometry is not None:
            SED_PHOTOMETRY = args.sed_photometry
        if args.sed_excess_summary is not None:
            SED_EXCESS_SUMMARY = args.sed_excess_summary
        if args.spitzer_photometry is not None:
            SPITZER_PHOTOMETRY = args.spitzer_photometry

    for input_path, label in ((REVIEW_DB, "review DB"), (SED_PHOTOMETRY, "SED photometry")):
        if not input_path.exists():
            raise FileNotFoundError(f"Missing {label}: {input_path}")
    if LABELS_CSV is not None and not LABELS_CSV.exists():
        raise FileNotFoundError(f"Missing labels CSV: {LABELS_CSV}")
    if SED_EXCESS_SUMMARY is not None and not SED_EXCESS_SUMMARY.exists():
        raise FileNotFoundError(f"Missing SED-excess summary: {SED_EXCESS_SUMMARY}")
    if SPITZER_PHOTOMETRY is not None and not SPITZER_PHOTOMETRY.exists():
        raise FileNotFoundError(f"Missing Spitzer photometry cache: {SPITZER_PHOTOMETRY}")

    print(f"Run root: {RUN_ROOT}")
    print(f"Review DB: {REVIEW_DB}")
    print(f"Labels: {LABELS_CSV if LABELS_CSV is not None else 'reviews.event_class'}")
    print(f"SED photometry: {SED_PHOTOMETRY}")
    print(f"SED-excess summary: {SED_EXCESS_SUMMARY or 'not supplied'}")
    print(f"Spitzer photometry: {SPITZER_PHOTOMETRY or 'not supplied'}")
    print(f"Results directory: {RESULTS_DIR}")


def main() -> None:
    args = _parse_args()
    _configure_paths(args)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = _load_dipper_payloads()
    df = _refresh_catalog_photometry(df, refresh_missing=args.refresh_missing_catalog)
    df = _compute_colors(df)
    df = _refresh_gaia_teff_bounds(df, refresh_missing=args.refresh_missing_catalog)
    df = _choose_teff(df)

    alpha = _load_sed_alpha(df)
    summary = df.merge(alpha, on="candidate_id", how="left")
    excess_classes = _load_sed_excess_classes()
    summary = summary.merge(excess_classes, on="candidate_id", how="left")
    summary["excess_class"] = summary["excess_class"].fillna("not_evaluated")
    if "sed_alpha_n_points" in summary:
        summary["sed_alpha_band_count"] = summary["sed_alpha_n_points"]
    summary = _add_missing_flags(summary)

    print(f"(W1-W2)_0/(H-Ks)_0 finite points: {_finite_xy_count(summary, 'w1_w2_0', 'H_K_0')}")
    print(f"(W1-W2)_0/(W2-W3)_0 finite points: {_finite_xy_count(summary, 'w1_w2_0', 'w2_w3_0')}")
    print(f"(W1-W2)_0/(W1-W4)_0 finite points: {_finite_xy_count(summary, 'w1_w2_0', 'w1_w4_0')}")
    print(f"(Ks-W4)_0/(J-Ks)_0 finite points: {_finite_xy_count(summary, 'ks_w4_0', 'j_k_0')}")
    print(f"(Ks-W2)_0/(J-Ks)_0 finite points: {_finite_xy_count(summary, 'ks_w2_0', 'j_k_0')}")
    print(f"(Ks-W2)_0/(Ks-W3)_0 finite points: {_finite_xy_count(summary, 'ks_w2_0', 'ks_w3_0')}")
    print(f"(J-H)_0/(H-Ks)_0 finite points: {_finite_xy_count(summary, 'j_h_0', 'H_K_0')}")
    print(f"(W3-W4)_0/(W1-W2)_0 finite points: {_finite_xy_count(summary, 'w3_w4_0', 'w1_w2_0')}")
    for col in ["tmass_h_err", "tmass_k_err", "w1_err", "w2_err", "w3_err", "w4_err"]:
        print(f"{col}: {_finite_count(summary, col)}/{len(summary)} finite")
    print(f"Teff finite points: {_finite_count(summary, 'teff')}/{len(summary)}")
    print(f"SED alpha uncertainty finite points: {_finite_count(summary, 'sed_alpha_err')}/{len(summary)}")
    print(f"Teff lower/upper uncertainty finite points: {_finite_count(summary, 'teff_err_lower')}/{len(summary)}")

    _plot_vphas(summary, RESULTS_DIR)
    _save_color_plot(
        summary,
        x="w1_w2_0",
        y="H_K_0",
        xerr="w1_w2_err",
        yerr="H_K_err",
        xlabel=LABEL_W1_W2_0,
        ylabel=LABEL_H_KS_0,
        output=RESULTS_DIR / "dipper_wise_color_color.pdf",
        xlim=(-0.2, 1.0),
        ylim=(0.0, 1.0),
    )
    _save_color_plot(
        summary,
        x="w1_w2_0",
        y="w2_w3_0",
        xerr="w1_w2_err",
        yerr="w2_w3_err",
        xlabel=LABEL_W1_W2_0,
        ylabel=LABEL_W2_W3_0,
        output=RESULTS_DIR / "dipper_wise_w1w2_w2w3.pdf",
    )
    _save_color_plot(
        summary,
        x="w1_w2_0",
        y="w1_w4_0",
        xerr="w1_w2_err",
        yerr="w1_w4_err",
        xlabel=LABEL_W1_W2_0,
        ylabel=LABEL_W1_W4_0,
        output=RESULTS_DIR / "dipper_wise_w1w2_w1w4.pdf",
    )
    _save_color_plot(
        summary,
        x="w1_w2_0",
        y="w1_0",
        xerr="w1_w2_err",
        yerr="w1_err",
        xlabel=LABEL_W1_W2_0,
        ylabel=LABEL_W1_0,
        output=RESULTS_DIR / "dipper_wise_w1_w1w2.pdf",
    )
    _save_color_plot(
        summary,
        x="ks_w4_0",
        y="j_k_0",
        xerr="ks_w4_err",
        yerr="j_k_err",
        xlabel=LABEL_KS_W4_0,
        ylabel=LABEL_J_KS_0,
        output=RESULTS_DIR / "dipper_2mass_wise_ksw4_jks.pdf",
    )
    _save_color_plot(
        summary,
        x="ks_w2_0",
        y="j_k_0",
        xerr="ks_w2_err",
        yerr="j_k_err",
        xlabel=LABEL_KS_W2_0,
        ylabel=LABEL_J_KS_0,
        output=RESULTS_DIR / "dipper_2mass_wise_ksw2_jks.pdf",
    )
    _save_color_plot(
        summary,
        x="ks_w2_0",
        y="ks_w3_0",
        xerr="ks_w2_err",
        yerr="ks_w3_err",
        xlabel=LABEL_KS_W2_0,
        ylabel=LABEL_KS_W3_0,
        output=RESULTS_DIR / "dipper_2mass_wise_ksw2_ksw3.pdf",
    )
    _save_color_plot(
        summary,
        x="j_h_0",
        y="H_K_0",
        xerr="j_h_err",
        yerr="H_K_err",
        xlabel=LABEL_J_H_0,
        ylabel=LABEL_H_KS_0,
        output=RESULTS_DIR / "dipper_2mass_jh_hks.pdf",
    )
    _save_color_plot(
        summary,
        x="w3_w4_0",
        y="w1_w2_0",
        xerr="w3_w4_err",
        yerr="w1_w2_err",
        xlabel=LABEL_W3_W4_0,
        ylabel=LABEL_W1_W2_0,
        output=RESULTS_DIR / "dipper_wise_w3w4_w1w2.pdf",
    )
    spitzer_irac = _load_spitzer_irac_colors(summary["candidate_id"])
    print(
        "Spitzer IRAC diagnostic complete four-band points: "
        f"{_finite_xy_count(spitzer_irac, 'irac_36_45', 'irac_58_80')}/{len(summary)}"
    )
    if not spitzer_irac.empty:
        _save_color_plot(
            spitzer_irac,
            x="irac_36_45",
            y="irac_58_80",
            xerr="irac_36_45_err",
            yerr="irac_58_80_err",
            xlabel=LABEL_IRAC1_IRAC2,
            ylabel=LABEL_IRAC3_IRAC4,
            output=RESULTS_DIR / "dipper_spitzer_irac_color_color_diagnostic.pdf",
        )
    _plot_sed_alpha(summary, RESULTS_DIR)
    _plot_teff_sed_alpha(summary, RESULTS_DIR)

    csv_path = RESULTS_DIR / "dipper_ir_excess_summary.csv"
    summary.to_csv(csv_path, index=False)
    print(f"Saved {csv_path} ({len(summary)} rows)")


if __name__ == "__main__":
    main()
