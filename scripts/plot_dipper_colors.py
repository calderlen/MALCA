import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
from malca.plotting.lightcurve_publication import apply_publication_rcparams


RUN_ROOT = Path("output/runs/runs_march18_bundle_all")
RESULTS_DIR = RUN_ROOT / "results"
REVIEW_DIR = RUN_ROOT / "review"
LABELS_CSV = RESULTS_DIR / "march18_review_cmd_dustmaps_full.csv"
REVIEW_DB = REVIEW_DIR / "review.taxonomy_filled.db"
SED_PHOTOMETRY = REVIEW_DIR / "review.taxonomy_filled_sed_photometry.parquet"

WISE_COLS = ["w1", "w1_err", "w2", "w2_err", "w3", "w3_err", "w4", "w4_err"]
TMASS_COLS = ["tmass_h", "tmass_h_err", "tmass_k", "tmass_k_err"]
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


def _load_dipper_payloads() -> pd.DataFrame:
    labels = pd.read_csv(LABELS_CSV, dtype={"candidate_id": str})
    dipper_ids = labels.loc[
        labels["event_class"].astype(str).eq("dipper"), "candidate_id"
    ].astype(str).tolist()
    print(f"Loaded {len(dipper_ids)} labeled dippers from {LABELS_CSV}")

    with sqlite3.connect(REVIEW_DB) as conn:
        payload_by_id = {
            str(cid): json.loads(payload_json or "{}")
            for cid, payload_json in conn.execute("SELECT candidate_id, payload_json FROM candidates")
        }

    rows: list[dict[str, object]] = []
    missing_ids: list[str] = []
    for cid in dipper_ids:
        payload = payload_by_id.get(str(cid))
        if payload is None:
            missing_ids.append(str(cid))
            continue
        external = payload.get("external_stats", {}) if isinstance(payload, dict) else {}
        row = {
            "candidate_id": str(cid),
            "ra": _first_present(payload, external, ("ra", "ra_deg")),
            "dec": _first_present(payload, external, ("dec", "dec_deg")),
            "A_v_3d": _first_present(payload, external, ("A_v_3d",)),
            "source_id": _first_present(payload, external, ("source_id", "gaia_id")),
            "yso_class": external.get("yso_class", payload.get("yso_class")),
            "vphas_r_ha": _first_present(payload, external, ("vphas_r_ha",)),
            "vphas_r_i": _first_present(payload, external, ("vphas_r_i",)),
            "teff50": _first_present(payload, external, ("teff50",)),
            "teff16": _first_present(payload, external, ("teff16",)),
            "teff84": _first_present(payload, external, ("teff84",)),
            "teff_gspphot": _first_present(payload, external, ("teff_gspphot",)),
            "teff_gspphot_lower": _first_present(payload, external, ("teff_gspphot_lower",)),
            "teff_gspphot_upper": _first_present(payload, external, ("teff_gspphot_upper",)),
        }
        for col in [*WISE_COLS, *TMASS_COLS, "H_K", "w1_w2"]:
            row[col] = _first_present(payload, external, (col,))
        rows.append(row)

    if missing_ids:
        print(f"Warning: {len(missing_ids)} labeled dippers missing from review DB: {', '.join(missing_ids)}")
    df = pd.DataFrame(rows)
    return _to_numeric(df, NUMERIC_COLS)


def _needs_refresh(df: pd.DataFrame, cols: list[str]) -> bool:
    if df.empty:
        return False
    finite_coord = np.isfinite(df["ra"]) & np.isfinite(df["dec"])
    if not finite_coord.any():
        return False
    return any(_finite_count(df.loc[finite_coord], col) < int(finite_coord.sum()) for col in cols)


def _refresh_catalog_photometry(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    n_coord = int((np.isfinite(out["ra"]) & np.isfinite(out["dec"])).sum())
    print(f"Catalog refresh coordinate rows: {n_coord}/{len(out)}")

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


def _refresh_gaia_teff_bounds(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
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
    out = df.copy()

    computed_hk = out["tmass_h"] - out["tmass_k"]
    out["H_K"] = computed_hk.where(np.isfinite(computed_hk), out["H_K"])
    out["H_K_err"] = _hypot2(out["tmass_h_err"], out["tmass_k_err"])

    for left, right in (("w1", "w2"), ("w1", "w3"), ("w1", "w4"), ("w2", "w3"), ("w2", "w4"), ("w3", "w4")):
        color = f"{left}_{right}"
        err = f"{color}_err"
        values = out[left] - out[right]
        out[color] = values.where(np.isfinite(values), out[color] if color in out.columns else np.nan)
        out[err] = _hypot2(out[f"{left}_err"], out[f"{right}_err"])

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
            markersize=6,
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
    ax.legend(loc="best")
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
) -> int:
    fig, ax = plt.subplots(figsize=(6, 5))
    n = _plot_errorbar_points(ax, df, x=x, y=y, xerr=xerr, yerr=yerr, label=f"Dippers ({_finite_xy_count(df, x, y)})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
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
    ax.set_xlabel(r"$r - i$ [mag]")
    ax.set_ylabel(r"$r - H\alpha$ [mag]")
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
    y_min = min(-3.1, float(finite["sed_alpha"].min()) - 0.3)
    y_max = max(0.6, float(finite["sed_alpha"].max()) + 0.3)

    fig, ax = plt.subplots(figsize=(8, 4.6))
    bands = [
        (y_min, -1.6, "Class III/photosphere", "0.92"),
        (-1.6, -0.3, "Class II", "#ffedd5"),
        (-0.3, 0.3, "Flat", "#fff7d6"),
        (0.3, y_max, "Class I", "#fee2e2"),
    ]
    for lo, hi, label, color in bands:
        ax.axhspan(max(lo, y_min), min(hi, y_max), color=color, zorder=0)
        y_text = (max(lo, y_min) + min(hi, y_max)) / 2
        if y_min <= y_text <= y_max:
            ax.text(0.99, y_text, label, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=8)

    xvals = np.arange(len(finite))
    ax.errorbar(
        xvals,
        finite["sed_alpha"],
        fmt="o",
        color="k",
        ecolor="0.25",
        elinewidth=0.8,
        capsize=3,
        capthick=0.8,
        markerfacecolor="k",
        markeredgecolor="k",
        markersize=5,
        linestyle="none",
        zorder=5,
    )
    ax.axhline(-1.6, color="k", linestyle="--", linewidth=0.5)
    ax.axhline(-0.3, color="k", linestyle="--", linewidth=0.5)
    ax.axhline(0.3, color="k", linestyle="--", linewidth=0.5)
    ax.set_ylabel(r"SED $\alpha$ [2-24 $\mu$m]")
    ax.set_xlabel("Dipper candidate")
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(xvals)
    ax.set_xticklabels(finite["candidate_id"].astype(str), rotation=90, fontsize=7)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    fig.tight_layout()
    out_path = out_dir / "dipper_sed_alpha_summary.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path} ({len(finite)} finite SED alpha values)")


def _plot_teff_sed_alpha(summary: pd.DataFrame, out_dir: Path) -> None:
    finite = summary[np.isfinite(summary["sed_alpha"]) & np.isfinite(summary["teff"])].copy()
    if finite.empty:
        print("No finite paired SED alpha and Teff values available to plot.")
        return

    x_min = max(0.0, float(finite["teff"].min()) - 350.0)
    x_max = float(finite["teff"].max()) + 350.0
    y_min = min(-3.1, float(finite["sed_alpha"].min()) - 0.3)
    y_max = max(0.6, float(finite["sed_alpha"].max()) + 0.3)

    fig, ax = plt.subplots(figsize=(6, 5))
    bands = [
        (y_min, -1.6, "Class III/photosphere", "0.92"),
        (-1.6, -0.3, "Class II", "#ffedd5"),
        (-0.3, 0.3, "Flat", "#fff7d6"),
        (0.3, y_max, "Class I", "#fee2e2"),
    ]
    for lo, hi, label, color in bands:
        ax.axhspan(max(lo, y_min), min(hi, y_max), color=color, zorder=0)
        y_text = (max(lo, y_min) + min(hi, y_max)) / 2
        if y_min <= y_text <= y_max:
            ax.text(
                0.98,
                y_text,
                label,
                transform=ax.get_yaxis_transform(),
                ha="right",
                va="center",
                fontsize=8,
            )

    xerr_ok = (
        np.isfinite(finite["teff_err_lower"]) & np.isfinite(finite["teff_err_upper"])
        if {"teff_err_lower", "teff_err_upper"}.issubset(finite.columns)
        else pd.Series(False, index=finite.index)
    )
    yerr_ok = np.isfinite(finite["sed_alpha_err"]) if "sed_alpha_err" in finite else pd.Series(False, index=finite.index)
    label_used = False

    def draw(mask: pd.Series, *, draw_xerr: bool, draw_yerr: bool) -> None:
        nonlocal label_used
        sub = finite.loc[mask]
        if sub.empty:
            return
        xerr = None
        if draw_xerr:
            xerr = np.vstack([sub["teff_err_lower"].to_numpy(dtype=float), sub["teff_err_upper"].to_numpy(dtype=float)])
        ax.errorbar(
            sub["teff"],
            sub["sed_alpha"],
            xerr=xerr,
            yerr=sub["sed_alpha_err"] if draw_yerr else None,
            fmt="o",
            color="k",
            ecolor="0.25",
            elinewidth=0.8,
            capsize=3,
            capthick=0.8,
            markerfacecolor="k",
            markeredgecolor="k",
            markersize=6,
            linestyle="none",
            zorder=5,
            label=f"Dippers ({len(finite)})" if not label_used else None,
        )
        label_used = True

    draw(xerr_ok & yerr_ok, draw_xerr=True, draw_yerr=True)
    draw(xerr_ok & ~yerr_ok, draw_xerr=True, draw_yerr=False)
    draw(~xerr_ok & yerr_ok, draw_xerr=False, draw_yerr=True)
    draw(~xerr_ok & ~yerr_ok, draw_xerr=False, draw_yerr=False)
    for value in (-1.6, -0.3, 0.3):
        ax.axhline(value, color="k", linestyle="--", linewidth=0.5, zorder=1)
    ax.set_xlabel(r"$T_{\rm eff}$ [K]")
    ax.set_ylabel(r"SED $\alpha$ [2-24 $\mu$m]")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.legend(loc="best")
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    fig.tight_layout()
    out_path = out_dir / "dipper_sed_alpha_teff.pdf"
    fig.savefig(out_path)
    plt.close(fig)
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


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = _load_dipper_payloads()
    df = _refresh_catalog_photometry(df)
    df = _compute_colors(df)
    df = _refresh_gaia_teff_bounds(df)
    df = _choose_teff(df)

    alpha = _load_sed_alpha(df)
    summary = df.merge(alpha, on="candidate_id", how="left")
    if "sed_alpha_n_points" in summary:
        summary["sed_alpha_band_count"] = summary["sed_alpha_n_points"]
    summary = _add_missing_flags(summary)

    print(f"W1-W2/H-K finite points: {_finite_xy_count(summary, 'w1_w2', 'H_K')}")
    print(f"W1-W2/W2-W3 finite points: {_finite_xy_count(summary, 'w1_w2', 'w2_w3')}")
    print(f"W1-W2/W1-W4 finite points: {_finite_xy_count(summary, 'w1_w2', 'w1_w4')}")
    for col in ["tmass_h_err", "tmass_k_err", "w1_err", "w2_err", "w3_err", "w4_err"]:
        print(f"{col}: {_finite_count(summary, col)}/{len(summary)} finite")
    print(f"Teff finite points: {_finite_count(summary, 'teff')}/{len(summary)}")
    print(f"SED alpha uncertainty finite points: {_finite_count(summary, 'sed_alpha_err')}/{len(summary)}")
    print(f"Teff lower/upper uncertainty finite points: {_finite_count(summary, 'teff_err_lower')}/{len(summary)}")

    _plot_vphas(summary, RESULTS_DIR)
    _save_color_plot(
        summary,
        x="w1_w2",
        y="H_K",
        xerr="w1_w2_err",
        yerr="H_K_err",
        xlabel=r"$W_1 - W_2$ [mag]",
        ylabel=r"$H - K_s$ [mag]",
        output=RESULTS_DIR / "dipper_wise_color_color.pdf",
        xlim=(-0.2, 1.0),
        ylim=(0.0, 1.0),
    )
    _save_color_plot(
        summary,
        x="w1_w2",
        y="w2_w3",
        xerr="w1_w2_err",
        yerr="w2_w3_err",
        xlabel=r"$W_1 - W_2$ [mag]",
        ylabel=r"$W_2 - W_3$ [mag]",
        output=RESULTS_DIR / "dipper_wise_w1w2_w2w3.pdf",
    )
    _save_color_plot(
        summary,
        x="w1_w2",
        y="w1_w4",
        xerr="w1_w2_err",
        yerr="w1_w4_err",
        xlabel=r"$W_1 - W_2$ [mag]",
        ylabel=r"$W_1 - W_4$ [mag]",
        output=RESULTS_DIR / "dipper_wise_w1w2_w1w4.pdf",
    )
    _plot_sed_alpha(summary, RESULTS_DIR)
    _plot_teff_sed_alpha(summary, RESULTS_DIR)

    csv_path = RESULTS_DIR / "dipper_ir_excess_summary.csv"
    summary.to_csv(csv_path, index=False)
    print(f"Saved {csv_path} ({len(summary)} rows)")


if __name__ == "__main__":
    main()
