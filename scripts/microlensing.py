#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Microlensing analysis pipeline (March 18 cohort + jumps-14 bucket).

Run from the repository root::

    python scripts/microlensing.py --help

Writes one results table ``output/microlensing/microlensing_results_<timestamp>.parquet`` (including
``gal_l_deg``, ``gal_b_deg``, ``mw_line_of_sight_region`` (bulge / disk / LMC / SMC / halo), and
``vizier_url`` per row), subjective visual-review columns for **probably_bad** LCs only
(``visual_inspection_subjective_flag``, ``visual_inspection_subjective_note``; see
``VISUAL_INSPECTION_PROBABLY_BAD_IDS`` / ``VISUAL_INSPECTION_BAD_IDS`` below). IDs in ``VISUAL_INSPECTION_BAD_IDS``
are **excluded from fits and from the results table entirely** (not merged, not crossmatched here). A full-sky Mollweide map ``microlensing_sky_<timestamp>.pdf`` (Paczynski reduced
$\chi^2$ by color, best model by marker), candidate grid ``microlensing_grid_<timestamp>.pdf``, Gaia CMD summary
``microlensing_cmd_<timestamp>.pdf``, and optionally—with ``--plot-lc``—per-candidate LC PDFs under
``output/microlensing/fit_pdfs/`` (named ``{chi2nu_paczynski}_{t_E_days}_{asas_sn_id}.pdf``).
With ``--crossmatch``, enrichment is delegated to ``malca.characterize.characterize_candidates_df`` and
``malca.vetting.vet_candidates`` (Gaia/2MASS/WISE context, SIMBAD, ZTF, ASAS-SN, TNS, ALeRCE, eROSITA, …; ATLAS forced photometry disabled here).

Progress: phase lines always; tqdm bars when stderr is a TTY.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

if __name__ == "__main__":
    import matplotlib

    matplotlib.use(os.environ.get("MPLBACKEND", "Agg"))


def find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise FileNotFoundError("Could not find repo root (missing pyproject.toml).")


def _repo_search_start() -> Path:
    """When installed as ``scripts/microlensing.py``, repo root is two levels up."""
    here = Path(__file__).resolve()
    if (here.parent.parent / "pyproject.toml").exists():
        return here.parent.parent
    return Path.cwd().resolve()


REPO_ROOT = find_repo_root(_repo_search_start())
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.coordinates.solar_system import get_body_barycentric_posvel
from scipy.optimize import least_squares, lsq_linear

from malca.lightcurve_io import load_lightcurve_df
from malca.review.eda_data import infer_plot_dir_from_source
from malca.review.interactive_plot import resolve_lightcurve_path
from malca.review.store import get_candidate_payload, init_db
from malca.utils import batch_gaia_cone_query, clean_lc
from tqdm import tqdm


def _preview_dataframe(title: str, df: pd.DataFrame, *, max_rows: int = 60) -> None:
    print(f"\n=== {title} ({len(df)} rows) ===")
    with pd.option_context(
        "display.max_columns", None,
        "display.max_colwidth", None,
        "display.width", 200,
        "display.max_rows", max_rows,
    ):
        print(df)


MARCH18_CANDIDATE_IDS = [
    "120259784233",
    "489626721133",
    "481036788325",
    "68720699238",
    "77309955721",
    "326418117943",
    "541166175153",
    "188979054063",
    "25771219762",
    "575525833425",
    "472447489028",
    "103079263205",
    "34360800532",
    "171799355659",
    "627065322644",
    "609886176748",
    "618475317371",
    "566936418537",
    "77310050643",
]

# Subjective human visual review of light curves / fits (not automated). Merged into the results Parquet as
# ``visual_inspection_subjective_flag`` / ``visual_inspection_subjective_note``. IDs are ASAS-SN ``candidate_id``
# strings. **Bad** IDs are excluded from all microlensing processing below (no fit, no result row).
# ``probably_bad`` IDs remain in the pipeline and are flagged in the output.
VISUAL_INSPECTION_BAD_IDS: tuple[str, ...] = (
    "481037066830",
    "584115687239",
    "472446671157",
)
VISUAL_INSPECTION_PROBABLY_BAD_IDS: tuple[str, ...] = (
    "103080609465",
    "335008621251",
    "377957380524",
    "163208930428",
    "326418144856",
    "601295670848",
    "274878446794",
    "77310492274",
    "360777400525",
    "523986214665",
    "601296314675",
    "120260181420",
    "635655559434",
    "77309955721",
    "188979391597",
    "77310113008",
    "506806257773",
    "463856742620",
    "249108525273",
    "85899644125",
    "532576095742",
    "523986127041",
    "77310544110",
    "618475976896",
    "403727418536",
    "584116541991",
    "635655307328",
    "146029437956",
    "584116581240",
    "592705976655",
    "541166874049",
    "549756255044",
    "103080402985",
)

VISUAL_INSPECTION_SUBJECTIVE_NOTE = (
    "Subjective human visual review of light curve and fit; not automated "
    "(see microlensing.py VISUAL_INSPECTION_BAD_IDS / VISUAL_INSPECTION_PROBABLY_BAD_IDS)."
)

DB_PATH = REPO_ROOT / "output" / "runs" / "runs_march18_bundle_all" / "review" / "review.db"
MICROLENSING_OUTPUT_ROOT = (REPO_ROOT / "output" / "microlensing").resolve()
MICROLENSING_FIT_PDF_DIR = (MICROLENSING_OUTPUT_ROOT / "fit_pdfs").resolve()
MICROLENSING_FIT_PDF_DPI = 300


def _microlensing_fit_pdf_stem(summary: dict[str, object]) -> str:
    """Basename stem: Paczynski reduced χ², Einstein crossing time (days), then ASAS-SN id."""
    rchi = _finite_float(summary.get('paczynski_reduced_chi2'))
    chi_tag = f'{rchi:.3f}' if rchi is not None else 'nan'
    tE = _finite_float(summary.get('reported_tE_days'))
    if tE is None:
        tE = _finite_float(summary.get('raw_paczynski_tE_days'))
    tE_tag = f'{tE:.3f}' if tE is not None else 'nan'
    aid = summary.get('asas_sn_id')
    if aid is None or aid == '' or bool(pd.isna(aid)):
        second = summary.get('candidate_id', 'unknown')
    else:
        second = aid
    second_s = str(second).strip() or str(summary.get('candidate_id', 'unknown'))
    for ch in (os.sep, '/', '<', '>', ':', '"', '|', '?', '*'):
        second_s = second_s.replace(ch, '_')
    return f'{chi_tag}_{tE_tag}_{second_s}'


def _finite_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _vizier_cone_search_url_deg(ra_deg: object, dec_deg: object) -> str:
    """VizieR cone-search URL (same query as the former HTML export). Empty if coordinates missing."""
    r = _finite_float(ra_deg)
    d = _finite_float(dec_deg)
    if r is None or d is None:
        return ''
    return (
        f'http://vizier.cds.unistra.fr/viz-bin/VizieR-5?-source=ALL'
        f'&-c={r:.5f}{d:+.5f}&-c.rs=10&-out.add=_r&-sort=_r'
    )


# --- Milky Way line-of-sight labels (geometric; LMC → SMC → bulge → disk → halo) ---

_MW_LMC = SkyCoord(ra=80.89375 * u.deg, dec=-69.75611 * u.deg, frame='icrs')
_MW_SMC = SkyCoord(ra=13.18667 * u.deg, dec=-72.82861 * u.deg, frame='icrs')
_MW_LMC_RADIUS_DEG = 10.0
_MW_SMC_RADIUS_DEG = 10.0
_MW_BULGE_ABS_L_DEG = 12.0
_MW_BULGE_ABS_B_DEG = 12.0
_MW_DISK_MAX_ABS_B_DEG = 25.0


def _mw_abs_galactic_l_deg(l_deg: np.ndarray) -> np.ndarray:
    """Smallest angle between Galactic longitude and 0° (handles wrap at 360°)."""
    l = np.asarray(l_deg, dtype=float)
    return np.minimum(np.abs(l), np.abs(l - 360.0))


def _mw_galactic_lb_deg(ra_deg: np.ndarray, dec_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ra = np.asarray(ra_deg, dtype=float)
    dec = np.asarray(dec_deg, dtype=float)
    l_deg = np.full(ra.shape, np.nan, dtype=float)
    b_deg = np.full(ra.shape, np.nan, dtype=float)
    ok = np.isfinite(ra) & np.isfinite(dec)
    if not np.any(ok):
        return l_deg, b_deg
    c = SkyCoord(ra=ra[ok] * u.deg, dec=dec[ok] * u.deg, frame='icrs')
    g = c.galactic
    l_deg[ok] = g.l.deg
    b_deg[ok] = g.b.deg
    return l_deg, b_deg


def _mw_classify_line_of_sight_region(
    l_deg: np.ndarray,
    b_deg: np.ndarray,
    *,
    ra_deg: np.ndarray | None = None,
    dec_deg: np.ndarray | None = None,
) -> np.ndarray:
    l_deg = np.asarray(l_deg, dtype=float)
    b_deg = np.asarray(b_deg, dtype=float)
    n = l_deg.shape[0]
    out = np.full(n, 'unknown', dtype=object)
    valid = np.isfinite(l_deg) & np.isfinite(b_deg)

    is_lmc = np.zeros(n, dtype=bool)
    is_smc = np.zeros(n, dtype=bool)
    if ra_deg is not None and dec_deg is not None:
        ra = np.asarray(ra_deg, dtype=float)
        dec = np.asarray(dec_deg, dtype=float)
        ok = valid & np.isfinite(ra) & np.isfinite(dec)
        if np.any(ok):
            c = SkyCoord(ra=ra[ok] * u.deg, dec=dec[ok] * u.deg, frame='icrs')
            is_lmc[ok] = c.separation(_MW_LMC).deg < _MW_LMC_RADIUS_DEG
            is_smc[ok] = c.separation(_MW_SMC).deg < _MW_SMC_RADIUS_DEG

    in_mag = is_lmc | is_smc
    abs_l = _mw_abs_galactic_l_deg(l_deg)
    is_bulge = (
        valid
        & ~in_mag
        & (abs_l <= _MW_BULGE_ABS_L_DEG)
        & (np.abs(b_deg) <= _MW_BULGE_ABS_B_DEG)
    )
    is_disk = valid & ~in_mag & ~is_bulge & (np.abs(b_deg) <= _MW_DISK_MAX_ABS_B_DEG)
    is_halo = valid & ~in_mag & ~is_bulge & ~is_disk

    out[is_lmc] = 'LMC'
    out[is_smc & ~is_lmc] = 'SMC'
    out[is_bulge] = 'galactic_bulge'
    out[is_disk] = 'galactic_disk'
    out[is_halo] = 'halo'
    return out


def _add_milky_way_line_of_sight_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``gal_l_deg``, ``gal_b_deg``, ``mw_line_of_sight_region`` from ``ra_deg`` / ``dec_deg``."""
    if df.empty:
        return df
    out = df.copy()
    if 'ra_deg' not in out.columns or 'dec_deg' not in out.columns:
        out['gal_l_deg'] = np.nan
        out['gal_b_deg'] = np.nan
        out['mw_line_of_sight_region'] = 'unknown'
        return out

    l_deg, b_deg = _mw_galactic_lb_deg(out['ra_deg'].to_numpy(), out['dec_deg'].to_numpy())
    out['gal_l_deg'] = l_deg
    out['gal_b_deg'] = b_deg
    out['mw_line_of_sight_region'] = _mw_classify_line_of_sight_region(
        l_deg,
        b_deg,
        ra_deg=out['ra_deg'].to_numpy(),
        dec_deg=out['dec_deg'].to_numpy(),
    )
    return out


def _candidate_id_match_str(value: object) -> str:
    """Normalize ``candidate_id`` for comparison with ``VISUAL_INSPECTION_*_IDS`` string literals."""
    if value is None:
        return ''
    if isinstance(value, float) and np.isnan(value):
        return ''
    s = str(value).strip()
    if not s or s.lower() == 'nan':
        return ''
    try:
        f = float(s)
        if np.isfinite(f) and f == int(f):
            return str(int(f))
    except (ValueError, OverflowError):
        pass
    return s


def _visual_inspection_bad_id_norms() -> frozenset[str]:
    """Normalized ``candidate_id`` values in :data:`VISUAL_INSPECTION_BAD_IDS` (subjective bad LC list)."""
    return frozenset(_candidate_id_match_str(x) for x in VISUAL_INSPECTION_BAD_IDS)


def exclude_visual_inspection_bad_ids(
    candidate_ids: list[str] | tuple[str, ...],
) -> list[str]:
    """Drop subjective **bad** light curves; keep **probably_bad** and unflagged IDs."""
    bad = _visual_inspection_bad_id_norms()
    out: list[str] = []
    for c in candidate_ids:
        s = str(c).strip()
        if not s or s.lower() in ('nan', 'none', '<na>'):
            continue
        if _candidate_id_match_str(s) in bad:
            continue
        out.append(s)
    return out


def _add_visual_inspection_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add subjective human visual-review columns from module constants (not algorithmic).

    Columns: ``visual_inspection_subjective_flag`` (``bad`` / ``probably_bad`` / empty),
    ``visual_inspection_subjective_note`` (explanatory text when flagged).
    """
    if df.empty or 'candidate_id' not in df.columns:
        return df
    out = df.copy()
    cid = df['candidate_id'].map(_candidate_id_match_str)
    bad_norm = _visual_inspection_bad_id_norms()
    prob_norm = frozenset(_candidate_id_match_str(x) for x in VISUAL_INSPECTION_PROBABLY_BAD_IDS) - bad_norm
    flag = pd.Series([''] * len(out), index=out.index, dtype=object)
    flag.loc[cid.isin(prob_norm)] = 'probably_bad'
    flag.loc[cid.isin(bad_norm)] = 'bad'
    out['visual_inspection_subjective_flag'] = flag
    out['visual_inspection_subjective_note'] = np.where(
        flag.astype(str).str.len() > 0,
        VISUAL_INSPECTION_SUBJECTIVE_NOTE,
        '',
    )
    return out


def _truncate_plot_label(text: str, max_len: int = 40) -> str:
    t = str(text).strip()
    if len(t) <= max_len:
        return t
    return t[: max_len - 1] + '…'


def _plot_crossmatch_context_caption(summary: dict[str, object]) -> str:
    """
    Compact plain-text line of crossmatch / characterization context for figure titles.

    Uses fields produced by external catalog matching, Gaia context (characterize), and
    vetting when present in *summary* (after final table merge).
    """
    parts: list[str] = []

    def _txt(key: str) -> str:
        v = summary.get(key)
        if v is None:
            return ''
        if isinstance(v, (float, np.floating)) and not np.isfinite(float(v)):
            return ''
        s = str(v).strip()
        if not s or s.lower() in {'nan', 'none', 'false', '<na>'}:
            return ''
        return s.replace('_', ' ')

    los = _txt('mw_line_of_sight_region')
    if los:
        parts.append(f'LOS {los}')

    if summary.get('microlens_match'):
        cat = _txt('microlens_catalog')
        name = _truncate_plot_label(_txt('microlens_name'), 34)
        if cat or name:
            parts.append(f'u-lens {cat} {name}'.strip())

    sim = _truncate_plot_label(_txt('nearest_simbad_object'), 26)
    ot = _truncate_plot_label(_txt('simbad_otype'), 14)
    if sim or ot:
        sep = summary.get('simbad_sep_arcsec')
        sep_bit = ''
        try:
            if sep is not None and np.isfinite(float(sep)):
                sep_bit = f' {float(sep):.2f}"'
        except (TypeError, ValueError):
            pass
        if sim and ot:
            core = f'SIMBAD {sim} [{ot}]'
        elif sim:
            core = f'SIMBAD {sim}'
        else:
            core = f'SIMBAD [{ot}]'
        parts.append((core + sep_bit).strip())

    ruwe = summary.get('ruwe')
    try:
        if ruwe is not None and np.isfinite(float(ruwe)):
            parts.append(f'RUWE {float(ruwe):.2f}')
    except (TypeError, ValueError):
        pass

    gmag = _txt('phot_g_mean_mag')
    if gmag:
        parts.append(f'G={gmag}')

    gvc = _truncate_plot_label(_txt('gaia_var_class'), 26)
    if gvc:
        parts.append(f'GaiaVar {gvc}')

    ztf = _truncate_plot_label(_txt('ztf_var_type'), 22)
    if ztf:
        parts.append(f'ZTF {ztf}')

    asn = _truncate_plot_label(_txt('asassn_var_type'), 22)
    if asn:
        parts.append(f'ASAS-SN {asn}')

    vsx = _truncate_plot_label(_txt('vsx_class'), 18)
    if vsx:
        parts.append(f'VSX {vsx}')

    if summary.get('vetting_likely_known'):
        parts.append('vet: likely known')

    alerce = _truncate_plot_label(_txt('alerce_lc_class'), 20)
    if alerce:
        parts.append(f'ALeRCE {alerce}')

    tns = _truncate_plot_label(_txt('tns_name'), 18)
    if tns:
        parts.append(f'TNS {tns}')

    galn = _truncate_plot_label(_txt('gaia_alert_name'), 18)
    galc = _truncate_plot_label(_txt('gaia_alert_class'), 14)
    if galn or galc:
        parts.append(f'GaiaAlert {galn or galc}')

    pop = _txt('population')
    if pop and pop != 'unknown':
        parts.append(f'pop {pop}')

    vf = _txt('visual_inspection_subjective_flag')
    if vf in {'bad', 'probably_bad'}:
        parts.append(f'visual {vf}')

    if not parts:
        return ''
    out = ' | '.join(parts)
    return out if len(out) <= 220 else out[:217] + '…'


def _inject_microlensing_table_into_summaries(
    table_df: pd.DataFrame,
    results: list[dict[str, object]],
) -> None:
    """Copy final table columns into each fit ``summary`` so plots can show crossmatch context."""
    if table_df.empty or not results or 'candidate_id' not in table_df.columns:
        return
    work = table_df.copy()
    work['_plot_cid'] = work['candidate_id'].map(_candidate_id_match_str)
    work = work.drop_duplicates('_plot_cid', keep='first').set_index('_plot_cid')
    for res in results:
        summ = res.get('summary')
        if not isinstance(summ, dict):
            continue
        cid = _candidate_id_match_str(summ.get('candidate_id'))
        if not cid or cid not in work.index:
            continue
        row = work.loc[cid]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        for col in table_df.columns:
            if col in {'candidate_id', '_plot_cid'}:
                continue
            val = row[col]
            if pd.api.types.is_scalar(val) and pd.isna(val):
                summ[col] = np.nan
            else:
                summ[col] = val


_BEST_MODEL_MARKERS: dict[str, str] = {
    'paczynski': '*',
    'gaussian': 's',
    'fred': 'o',
    'flat': 'D',
}


def _best_fit_marker(model_key: str) -> str:
    return _BEST_MODEL_MARKERS.get(model_key.lower(), 'X')


def _best_fit_marker_size(model_key: str) -> float:
    mk = str(model_key).strip().lower()
    if mk == 'fred':
        return 30.0
    if mk == 'paczynski':
        return 70.0
    return 52.0


def _mw_lb_rad_mollweide(l_deg: np.ndarray, b_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Galactic (l, b) in degrees → radians for ``projection='mollweide'`` (l ∈ [−180°, 180°])."""
    l = np.asarray(l_deg, dtype=float)
    b = np.asarray(b_deg, dtype=float)
    l_plot = ((l + 180.0) % 360.0) - 180.0
    return np.radians(l_plot), np.radians(b)


def _save_microlensing_full_sky_plot(df: pd.DataFrame, out_path: Path, *, dpi: int = 300) -> None:
    """Full-sky Mollweide map: color = Paczynski reduced χ², marker = BIC-best profile."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if df.empty:
        fig, ax = plt.subplots(figsize=(6.0, 2.0))
        ax.text(0.5, 0.5, 'No candidates for sky map', ha='center', va='center')
        ax.axis('off')
        fig.savefig(out_path, dpi=dpi, bbox_inches='tight', format='pdf')
        plt.close(fig)
        return

    if 'ra_deg' not in df.columns or 'dec_deg' not in df.columns:
        fig, ax = plt.subplots(figsize=(6.0, 2.0))
        ax.text(0.5, 0.5, 'No RA/Dec for sky map', ha='center', va='center')
        ax.axis('off')
        fig.savefig(out_path, dpi=dpi, bbox_inches='tight', format='pdf')
        plt.close(fig)
        return

    l_deg, b_deg = _mw_galactic_lb_deg(df['ra_deg'].to_numpy(), df['dec_deg'].to_numpy())
    ok = np.isfinite(l_deg) & np.isfinite(b_deg)
    if not np.any(ok):
        fig, ax = plt.subplots(figsize=(6.0, 2.0))
        ax.text(0.5, 0.5, 'No valid Galactic coordinates', ha='center', va='center')
        ax.axis('off')
        fig.savefig(out_path, dpi=dpi, bbox_inches='tight', format='pdf')
        plt.close(fig)
        return

    chi = (
        pd.to_numeric(df['paczynski_reduced_chi2'], errors='coerce').to_numpy()
        if 'paczynski_reduced_chi2' in df.columns
        else np.full(len(df), np.nan)
    )
    raw_model = df['best_model'] if 'best_model' in df.columns else pd.Series([''] * len(df), index=df.index)
    model_key = raw_model.fillna('').astype(str).str.strip().str.lower().replace('', 'unknown')

    lx, bx = _mw_lb_rad_mollweide(l_deg[ok], b_deg[ok])
    chi_ok = chi[ok]
    mk_ok = model_key.to_numpy()[ok]

    finite_chi = np.isfinite(chi_ok) & (chi_ok > 0.0)
    if np.any(finite_chi):
        vmax = float(np.nanpercentile(chi_ok[finite_chi], 98.0))
        vmax = float(np.clip(max(vmax, 1.0), 0.5, 80.0))
        vmin = 0.0
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    else:
        norm = None

    # Reversed cividis: low χ² (better Paczynski fits) → yellow end; high χ² → dark end.
    cmap = plt.cm.cividis_r
    fig, ax = plt.subplots(figsize=(14.0, 7.2), subplot_kw={'projection': 'mollweide'})
    ax.grid(True, alpha=0.35, linestyle='--', linewidth=0.6)
    ax.set_xlabel(r'Galactic longitude $l$')
    ax.set_ylabel(r'Galactic latitude $b$')

    model_order = ['paczynski', 'gaussian', 'fred', 'flat', 'unknown']

    def _sort_key(m: str) -> tuple[int, str]:
        m = str(m).lower()
        try:
            return (model_order.index(m), m)
        except ValueError:
            return (len(model_order), m)

    present_models = sorted(set(mk_ok.tolist()), key=_sort_key)

    for m in present_models:
        sub = (mk_ok == m) & finite_chi
        if not np.any(sub):
            continue
        marker_size = _best_fit_marker_size(m)
        ax.scatter(
            lx[sub],
            bx[sub],
            c=chi_ok[sub],
            cmap=cmap,
            norm=norm,
            s=marker_size,
            marker=_best_fit_marker(m),
            edgecolors='0.15',
            linewidths=0.35,
            zorder=4,
        )

    for m in present_models:
        sub = (mk_ok == m) & ~finite_chi
        if not np.any(sub):
            continue
        marker_size = _best_fit_marker_size(m)
        ax.scatter(
            lx[sub],
            bx[sub],
            c='0.55',
            s=marker_size,
            marker=_best_fit_marker(m),
            edgecolors='0.15',
            linewidths=0.35,
            zorder=3,
        )

    if norm is not None:
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', pad=0.10, shrink=0.78, aspect=34)
        cbar.set_label(r'Paczynski reduced $\chi^2_\nu$')

    legend_handles = [
        Line2D(
            [0],
            [0],
            linestyle='None',
            marker=_best_fit_marker(m),
            color='0.2',
            markerfacecolor='0.75',
            markeredgecolor='0.2',
            markersize=9.0,
            label=m,
        )
        for m in present_models
    ]
    if np.any(~finite_chi):
        legend_handles.append(
            Line2D(
                [0],
                [0],
                linestyle='None',
                marker='o',
                color='0.55',
                markerfacecolor='0.55',
                markeredgecolor='0.2',
                markersize=8.0,
                label=r'no Paczynski $\chi^2_\nu$',
            )
        )
    # Legend above the map (figure coords) so it does not overlap the horizontal colorbar / label.
    fig.legend(
        handles=legend_handles,
        title='BIC-best model',
        loc='center',
        bbox_to_anchor=(0.5, 0.965),
        bbox_transform=fig.transFigure,
        ncol=min(5, max(1, len(legend_handles))),
        frameon=True,
        fontsize=9,
        title_fontsize=9,
    )
    fig.subplots_adjust(left=0.06, right=0.96, bottom=0.16, top=0.82)

    fig.savefig(out_path, dpi=dpi, bbox_inches='tight', format='pdf', facecolor=fig.get_facecolor())
    plt.close(fig)


_QUALITY_TIER_ORDER: dict[str, int] = {
    "gold": 3,
    "silver": 2,
    "bronze": 1,
    "suspect": 0,
}
_QUALITY_TIER_COLORS: dict[str, str] = {
    "gold": "#d89a00",
    "silver": "#7a7a7a",
    "bronze": "#8a4a26",
    "suspect": "#5f73a1",
}


def _quality_tier_rank(series: pd.Series) -> pd.Series:
    vals = series.fillna("").astype(str).str.strip().str.lower()
    return vals.map(_QUALITY_TIER_ORDER).fillna(-1).astype(int)


def _coerce_bool_series(s: pd.Series, *, default: bool = False) -> pd.Series:
    """Convert mixed bool/string columns into a clean boolean Series."""
    if s.dtype == bool:
        return s.fillna(default)
    # Common CSV encodings: True/False, 1/0, yes/no, etc.
    return s.fillna(default).astype(str).str.strip().str.lower().isin({"true", "1", "t", "yes", "y"})


def _compute_microlensing_quality_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute microlensing quality metrics (quality_score, quality_tier, quality_breakdown_*).

    Some output runs (notably LC-only candidates) may not carry precomputed quality columns.
    """
    if df.empty:
        return df
    work = df.copy()

    n = len(work)

    # --- Coverage component ---
    if "paczynski_tau_coverage_score" in work.columns:
        cov = pd.to_numeric(work["paczynski_tau_coverage_score"], errors="coerce")
        if cov.notna().any():
            cov_np = cov.to_numpy(dtype=float)
            coverage_breakdown = np.where(
                np.isfinite(cov_np) & (cov_np >= PAC_COVERAGE_WARN_THRESHOLD),
                1.0,
                np.where(
                    np.isfinite(cov_np) & (cov_np >= PAC_COVERAGE_FAIL_THRESHOLD),
                    0.6,
                    0.4,
                ),
            )
        else:
            coverage_breakdown = np.full(n, 1.0, dtype=float)
    else:
        coverage_breakdown = np.full(n, 1.0, dtype=float)

    # --- Parallax component (historical discrete mapping) ---
    attempted_s = (
        work["parallax_attempted"]
        if "parallax_attempted" in work.columns
        else pd.Series([False] * n, index=work.index, dtype=object)
    )
    fit_ok_s = (
        work["parallax_fit_ok"]
        if "parallax_fit_ok" in work.columns
        else pd.Series([False] * n, index=work.index, dtype=object)
    )
    preferred_s = (
        work["parallax_preferred"]
        if "parallax_preferred" in work.columns
        else pd.Series([False] * n, index=work.index, dtype=object)
    )
    attempted_b = _coerce_bool_series(attempted_s, default=False).to_numpy(dtype=bool)
    fit_ok_b = _coerce_bool_series(fit_ok_s, default=False).to_numpy(dtype=bool)
    preferred_b = _coerce_bool_series(preferred_s, default=False).to_numpy(dtype=bool)

    parallax_breakdown = np.full(n, 0.4, dtype=float)
    ok_mask = attempted_b & fit_ok_b
    ok_pref_mask = ok_mask & preferred_b
    parallax_breakdown[ok_pref_mask] = 1.0
    ok_not_pref_mask = ok_mask & ~preferred_b
    parallax_breakdown[ok_not_pref_mask] = 0.8
    attempted_not_ok_pref = attempted_b & preferred_b & ~fit_ok_b
    parallax_breakdown[attempted_not_ok_pref] = 0.5

    # --- Astrophysical component ---
    # Driven by log10(Delta BIC) vs the flat baseline.
    if "log10_delta_bic_vs_flat" in work.columns:
        x = pd.to_numeric(work["log10_delta_bic_vs_flat"], errors="coerce")
    elif "delta_bic_vs_flat" in work.columns:
        x = signed_log10_series(pd.to_numeric(work["delta_bic_vs_flat"], errors="coerce"))
    else:
        x = pd.Series([np.nan] * n, index=work.index, dtype=float)
    x_np = x.to_numpy(dtype=float)
    astrophysical_breakdown = np.full(n, 0.4, dtype=float)
    good_mask = np.isfinite(x_np)
    astrophysical_breakdown[good_mask & (x_np >= 4.2)] = 1.0
    astrophysical_breakdown[good_mask & (x_np >= 3.65) & (x_np < 4.2)] = 0.8
    astrophysical_breakdown[good_mask & (x_np >= 3.5) & (x_np < 3.65)] = 0.6

    # --- Fit component ---
    if "fit_reduced_chi2" in work.columns:
        chi2 = pd.to_numeric(work["fit_reduced_chi2"], errors="coerce")
    elif "paczynski_reduced_chi2" in work.columns:
        chi2 = pd.to_numeric(work["paczynski_reduced_chi2"], errors="coerce")
    else:
        chi2 = pd.Series([np.nan] * n, index=work.index, dtype=float)
    chi2_np = chi2.to_numpy(dtype=float)
    chi2_score = 1.0 / (1.0 + (chi2_np / 5.0))
    fit_breakdown = 0.6 + 0.4 * np.clip(chi2_score, 0.0, 1.0)
    fit_breakdown[~np.isfinite(chi2_np)] = 0.6
    if "fit_ok" in work.columns:
        fit_ok_b = _coerce_bool_series(work["fit_ok"], default=False).to_numpy(dtype=bool)
        fit_breakdown[fit_ok_b] = np.maximum(fit_breakdown[fit_ok_b], 0.8)

    # --- Morphology component ---
    if "n_points_fit" in work.columns:
        n_fit = pd.to_numeric(work["n_points_fit"], errors="coerce").to_numpy(dtype=float)
    else:
        n_fit = pd.Series([np.nan] * n, index=work.index, dtype=float).to_numpy(dtype=float)
    if "n_strong_points" in work.columns:
        n_strong = pd.to_numeric(work["n_strong_points"], errors="coerce").to_numpy(dtype=float)
    else:
        n_strong = pd.Series([np.nan] * n, index=work.index, dtype=float).to_numpy(dtype=float)
    if "shoulder_left" in work.columns:
        sl = pd.to_numeric(work["shoulder_left"], errors="coerce").to_numpy(dtype=float)
    else:
        sl = pd.Series([np.nan] * n, index=work.index, dtype=float).to_numpy(dtype=float)
    if "shoulder_right" in work.columns:
        sr = pd.to_numeric(work["shoulder_right"], errors="coerce").to_numpy(dtype=float)
    else:
        sr = pd.Series([np.nan] * n, index=work.index, dtype=float).to_numpy(dtype=float)
    n_fit_safe = np.maximum(n_fit, 1.0)
    strong_ratio = np.where(np.isfinite(n_strong), n_strong / n_fit_safe, 0.0)
    shoulder_balance = np.where(
        np.isfinite(sl) & np.isfinite(sr),
        np.minimum(sl, sr) / np.maximum(sl + sr, 1e-6),
        0.0,
    )
    morph_raw = 0.65 * strong_ratio + 0.35 * shoulder_balance
    morphology_breakdown = np.clip(morph_raw / 0.8, 0.0, 1.0)

    # --- Contamination component ---
    fit_warning = work.get("fit_warning", pd.Series([""] * n, index=work.index))
    if not isinstance(fit_warning, pd.Series):
        fit_warning = pd.Series([fit_warning] * n, index=work.index)
    fw = fit_warning.fillna("").astype(str).str.lower()

    penalty = np.zeros(n, dtype=float)
    penalty += fw.str.contains("high_reduced_chi2", regex=False, na=False).to_numpy(dtype=float) * 0.25
    penalty += fw.str.contains("t0_near_bound", regex=False, na=False).to_numpy(dtype=float) * 0.15
    penalty += fw.str.contains("tE_near_bound", regex=False, na=False).to_numpy(dtype=float) * 0.15
    penalty += fw.str.contains("sampler_low_acceptance", regex=False, na=False).to_numpy(dtype=float) * 0.10
    penalty += fw.str.contains("correlated_residuals", regex=False, na=False).to_numpy(dtype=float) * 0.12
    penalty += fw.str.contains("high_blending", regex=False, na=False).to_numpy(dtype=float) * 0.10
    penalty += fw.str.contains("insufficient_shoulders", regex=False, na=False).to_numpy(dtype=float) * 0.10
    penalty += fw.str.contains("single_point_peak", regex=False, na=False).to_numpy(dtype=float) * 0.08
    penalty += fw.str.contains("weak_vs_flat", regex=False, na=False).to_numpy(dtype=float) * 0.12

    contamination_breakdown = np.clip(1.0 - penalty, 0.2, 1.0)

    # --- Combine into quality_score ---
    quality_score = (
        0.05
        + 0.25 * fit_breakdown
        + 0.2 * morphology_breakdown
        + 0.15 * astrophysical_breakdown
        + 0.2 * contamination_breakdown
        + 0.1 * parallax_breakdown
        + 0.05 * coverage_breakdown
    )

    work["quality_score"] = pd.to_numeric(quality_score, errors="coerce")
    work["quality_tier"] = np.where(
        work["quality_score"] >= 0.8,
        "Gold",
        np.where(work["quality_score"] >= 0.6, "Silver", np.where(work["quality_score"] >= 0.55, "Bronze", "Suspect")),
    )
    work["quality_flags"] = ""

    # Store breakdown components for debugging / plots.
    work["quality_breakdown_fit"] = fit_breakdown
    work["quality_breakdown_morphology"] = morphology_breakdown
    work["quality_breakdown_astrophysical"] = astrophysical_breakdown
    work["quality_breakdown_contamination"] = contamination_breakdown
    work["quality_breakdown_parallax"] = parallax_breakdown
    work["quality_breakdown_coverage"] = coverage_breakdown

    return work


def _ensure_microlensing_quality_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Compute quality_* columns only if they are missing (or fully null)."""
    if df.empty:
        return df
    has_score = "quality_score" in df.columns and df["quality_score"].notna().any()
    has_tier = "quality_tier" in df.columns and df["quality_tier"].notna().any()
    if has_score and has_tier:
        return df
    return _compute_microlensing_quality_columns(df)


def _save_microlensing_candidate_grid_plot(
    df: pd.DataFrame,
    out_path: Path,
    *,
    min_tier: str = "Silver",
    max_candidates: int | None = None,
    fit_results: list[dict[str, object]] | None = None,
    jumps14_fit_results: list[dict[str, object]] | None = None,
    dpi: int = 300,
) -> None:
    """Top-candidate LC grid (tier + quality score + reported tE)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if df.empty:
        fig, ax = plt.subplots(figsize=(6.0, 2.0))
        ax.text(0.5, 0.5, "No candidates for microlensing grid", ha="center", va="center")
        ax.axis("off")
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight", format="pdf")
        plt.close(fig)
        return

    work = df.copy()
    if "quality_tier" not in work.columns:
        work["quality_tier"] = ""
    if "quality_score" not in work.columns:
        work["quality_score"] = np.nan
    work["quality_score"] = pd.to_numeric(work["quality_score"], errors="coerce")
    work["_tier_rank"] = _quality_tier_rank(work["quality_tier"])
    min_rank = _QUALITY_TIER_ORDER.get(str(min_tier).strip().lower(), 0)
    work = work.loc[work["_tier_rank"] >= int(min_rank)].copy()
    if work.empty:
        fig, ax = plt.subplots(figsize=(6.0, 2.0))
        ax.text(0.5, 0.5, f"No candidates at or above tier {min_tier}", ha="center", va="center")
        ax.axis("off")
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight", format="pdf")
        plt.close(fig)
        return

    work = work.sort_values(["_tier_rank", "quality_score"], ascending=[False, False])
    if max_candidates is not None:
        work = work.head(int(max_candidates))
    n = len(work)
    ncols = 5
    nrows = int(np.ceil(n / ncols))

    # Keep approx the same subplot height as the previous default (25 candidates → nrows=5).
    fig_width = 14.43
    row_height = 18.46 / 5.0
    fig_height = max(row_height * nrows, row_height)
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height), squeeze=False)
    flat_axes = axes.ravel()

    # Build Paczynski model params from fit_results for overlay curves.
    pac_by_cid: dict[str, tuple[np.ndarray, float]] = {}
    for _res in (list(fit_results or []) + list(jumps14_fit_results or [])):
        try:
            summ = _res.get("summary", {}) if isinstance(_res, dict) else {}
            cid = _candidate_id_match_str(summ.get("candidate_id"))
            if not cid:
                continue
            best_seed = _res.get("best_seed_result", {}) or {}
            pac = (best_seed.get("fits", {}) or {}).get("paczynski", {}) or {}
            if not pac.get("success"):
                continue
            params = pac.get("params")
            t_ref = pac.get("t_ref")
            if params is None or t_ref is None:
                continue
            pac_by_cid[cid] = (np.asarray(params, dtype=float), float(t_ref))
        except Exception:
            continue

    dpi_save = int(max(dpi, 600))

    for i, (_, row) in enumerate(work.iterrows()):
        ax = flat_axes[i]
        cid_raw = row.get("candidate_id", "")
        cid = str(cid_raw)
        cid_match = _candidate_id_match_str(cid_raw)
        tier = str(row.get("quality_tier", "")).strip()
        tier_key = tier.lower()
        tier_color = _QUALITY_TIER_COLORS.get(tier_key, "0.2")
        qscore = _finite_float(row.get("quality_score"))
        tE = _finite_float(row.get("reported_tE_days"))
        if tE is None:
            tE = _finite_float(row.get("raw_paczynski_tE_days"))

        lc_path = row.get("lc_path")
        lc_df = None
        try:
            if lc_path is not None and str(lc_path).strip():
                lc_df = load_lightcurve_df(Path(str(lc_path)))
        except Exception:
            lc_df = None
        if lc_df is None or lc_df.empty:
            ax.text(0.5, 0.5, "LC unavailable", ha="center", va="center", fontsize=7)
            ax.set_axis_off()
            continue

        # Make column detection resilient to case differences (e.g. loader returns `JD`, not `jd`).
        lc_cols_lower_to_actual = {}
        for c in lc_df.columns:
            lc_cols_lower_to_actual.setdefault(str(c).strip().lower(), c)

        mag_col = None
        if "mag" in lc_cols_lower_to_actual:
            mag_col = lc_cols_lower_to_actual["mag"]
        else:
            for candidate_mag_col in ("magnitude", "flux"):
                key = candidate_mag_col.lower()
                if key in lc_cols_lower_to_actual:
                    mag_col = lc_cols_lower_to_actual[key]
                    break

        x_col = None
        for candidate_x_col in ("jd", "hjd", "mjd", "time"):
            key = candidate_x_col.lower()
            if key in lc_cols_lower_to_actual:
                x_col = lc_cols_lower_to_actual[key]
                break
        if mag_col is None or x_col is None:
            ax.text(0.5, 0.5, "LC columns missing", ha="center", va="center", fontsize=7)
            ax.set_axis_off()
            continue

        band = str(row.get("band_used", "")).strip().lower()
        if "phot_filter" in lc_df.columns and band:
            _mask_band = lc_df["phot_filter"].astype(str).str.strip().str.lower() == band
            if _mask_band.any():
                lc_df = lc_df.loc[_mask_band].copy()

        x = pd.to_numeric(lc_df[x_col], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(lc_df[mag_col], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        x = x[ok]
        y = y[ok]
        if x.size == 0:
            ax.text(0.5, 0.5, "No finite LC points", ha="center", va="center", fontsize=7)
            ax.set_axis_off()
            continue
        x_raw = x
        shift = 2458000.0 if np.nanmedian(x_raw) > 2.0e6 else 0.0
        x_plot = x_raw - shift

        ax.scatter(x_plot, y, s=4.0, alpha=0.75, c="k", linewidths=0)
        if str(mag_col).strip().lower() != "flux":
            ax.invert_yaxis()

        # Paczynski model overlay (when we have fit params and we are plotting magnitudes).
        if str(mag_col).strip().lower() in {"mag", "magnitude"} and cid_match in pac_by_cid:
            pac_params, t_ref = pac_by_cid[cid_match]
            jd_dense = np.linspace(float(np.nanmin(x_raw)), float(np.nanmax(x_raw)), 350)
            mag_dense = _evaluate_model("paczynski", pac_params, jd_dense, t_ref)
            if np.any(np.isfinite(mag_dense)):
                ax.plot(jd_dense - shift, mag_dense, color="red", linewidth=1.8, alpha=0.95, zorder=3)

        # Square tier badge in the top-left.
        from matplotlib.patches import Rectangle
        badge_w = 0.12
        badge_h = 0.12
        badge_x = 0.02
        badge_y = 1.0 - badge_h - 0.02
        ax.add_patch(
            Rectangle(
                (badge_x, badge_y),
                badge_w,
                badge_h,
                transform=ax.transAxes,
                facecolor=tier_color,
                edgecolor="0.2",
                linewidth=0.6,
                zorder=10,
            )
        )
        badge_text = tier or "Unknown"
        ax.text(
            badge_x + badge_w / 2.0,
            badge_y + badge_h / 2.0,
            badge_text,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=7,
            color="white",
            zorder=11,
        )

        ax.grid(alpha=0.2, linestyle="--", linewidth=0.5)
        ax.tick_params(labelsize=7)
        if i % ncols == 0:
            ax.set_ylabel(f"{band or 'g'} [mag]", fontsize=8)
        else:
            ax.set_ylabel("")
        if i >= (nrows - 1) * ncols:
            ax.set_xlabel("JD - 2458000", fontsize=8)
        else:
            ax.set_xlabel("")
        title_top = f"{cid}\n({qscore:.2f})" if qscore is not None else f"{cid}"
        ax.set_title(title_top, fontsize=9, color=tier_color, pad=2)
        if tE is not None:
            ax.text(
                0.98,
                0.04,
                f"tE={tE:.0f}d",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=8,
            )

    for j in range(n, len(flat_axes)):
        flat_axes[j].set_axis_off()

    fig.suptitle("Microlensing Candidates Grid", fontsize=15, y=0.997)
    fig.text(0.5, 0.982, "Quality Tier", ha="center", va="top", fontsize=10)
    fig.legend(
        handles=[
            Line2D([0], [0], marker="o", linestyle="None", color=_QUALITY_TIER_COLORS["gold"], label="Gold"),
            Line2D([0], [0], marker="o", linestyle="None", color=_QUALITY_TIER_COLORS["silver"], label="Silver"),
            Line2D([0], [0], marker="o", linestyle="None", color=_QUALITY_TIER_COLORS["bronze"], label="Bronze"),
            Line2D([0], [0], marker="o", linestyle="None", color=_QUALITY_TIER_COLORS["suspect"], label="Suspect"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.968),
        ncol=4,
        frameon=False,
        fontsize=8,
    )
    fig.subplots_adjust(left=0.055, right=0.99, bottom=0.055, top=0.955, wspace=0.22, hspace=0.36)
    fig.savefig(out_path, dpi=dpi_save, bbox_inches="tight", format="pdf")
    plt.close(fig)


def _save_microlensing_cmd_plot(df: pd.DataFrame, out_path: Path, *, dpi: int = 300) -> None:
    """Gaia CMD summary (BP-RP vs M_G)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(4.85, 4.82))

    def _compute_mg_from_phot_g_and_parallax(frame: pd.DataFrame) -> pd.Series | None:
        if "phot_g_mean_mag" not in frame.columns or "parallax" not in frame.columns:
            return None
        gmag = pd.to_numeric(frame.get("phot_g_mean_mag"), errors="coerce")
        plx = pd.to_numeric(frame.get("parallax"), errors="coerce")
        plx_arr = plx.to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            dist_pc = np.where(np.isfinite(plx_arr) & (plx_arr > 0.0), 1000.0 / plx_arr, np.nan)
        mg = gmag.to_numpy(dtype=float) - 5.0 * np.log10(dist_pc) + 5.0
        return pd.Series(mg, index=frame.index, dtype=float)

    if "bp_rp" not in df.columns:
        ax.text(0.5, 0.5, "Missing BP-RP or M_G columns", ha="center", va="center")
        ax.axis("off")
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight", format="pdf")
        plt.close(fig)
        return

    bp_rp = pd.to_numeric(df["bp_rp"], errors="coerce")
    mg = None
    if "M_G" in df.columns:
        mg = pd.to_numeric(df["M_G"], errors="coerce")
    elif "mg0" in df.columns:
        mg = pd.to_numeric(df["mg0"], errors="coerce")
    elif "phot_g_mean_mag" in df.columns and "parallax" in df.columns:
        mg = _compute_mg_from_phot_g_and_parallax(df)
    if mg is None:
        ax.text(0.5, 0.5, "Missing BP-RP or M_G columns", ha="center", va="center")
        ax.axis("off")
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight", format="pdf")
        plt.close(fig)
        return

    mg_num = pd.to_numeric(mg, errors="coerce")
    # Optionally filter by quality tier so the CMD matches the grid tier selection.
    min_tier = "Bronze"
    tier_mask = np.ones(len(df), dtype=bool)
    if "quality_tier" in df.columns:
        min_rank = _QUALITY_TIER_ORDER.get(min_tier.strip().lower(), 0)
        tier_rank = _quality_tier_rank(df["quality_tier"])
        tier_mask = tier_rank.to_numpy(dtype=int) >= int(min_rank)

    cand_mask = (
        np.isfinite(bp_rp.to_numpy(dtype=float))
        & np.isfinite(mg_num.to_numpy(dtype=float))
        & tier_mask
    )
    if not np.any(cand_mask):
        ax.text(0.5, 0.5, "Missing BP-RP or M_G columns", ha="center", va="center")
        ax.axis("off")
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight", format="pdf")
        plt.close(fig)
        return

    # Background Gaia sample (dereddening not handled here; just a reference CMD cloud).
    # This is intentionally lightweight so microlensing.py does not require plotly.
    bg_path = REPO_ROOT / "input" / "gaia" / "gaia_dr3_crossmatched.parquet"
    try:
        if bg_path.is_file():
            bg = pd.read_parquet(bg_path, columns=["bp_rp", "phot_g_mean_mag", "parallax"])
            bg_bp_rp = pd.to_numeric(bg["bp_rp"], errors="coerce")
            bg_mg = _compute_mg_from_phot_g_and_parallax(bg)
            if bg_mg is not None:
                bg_mg_num = pd.to_numeric(bg_mg, errors="coerce")
                bg_mask = np.isfinite(bg_bp_rp.to_numpy(dtype=float)) & np.isfinite(bg_mg_num.to_numpy(dtype=float))
                if np.any(bg_mask):
                    bg_bp_rp_np = bg_bp_rp.to_numpy(dtype=float)[bg_mask]
                    bg_mg_np = bg_mg_num.to_numpy(dtype=float)[bg_mask]
                    # Keep background plotting bounded.
                    if bg_bp_rp_np.size > 20000:
                        rng = np.random.default_rng(42)
                        idx = rng.choice(bg_bp_rp_np.size, size=20000, replace=False)
                        bg_bp_rp_np = bg_bp_rp_np[idx]
                        bg_mg_np = bg_mg_np[idx]
                    ax.scatter(bg_bp_rp_np, bg_mg_np, s=2.0, c="0.6", alpha=0.16, edgecolors="none", zorder=1)
    except Exception:
        # Background is optional; keep plotting candidates even if Gaia parquet fails.
        pass

    # Candidate color metric: prefer quality_score, fall back to paczynski_tau_coverage_score.
    if "quality_score" in df.columns:
        score = pd.to_numeric(df["quality_score"], errors="coerce")
    elif "paczynski_tau_coverage_score" in df.columns:
        score = pd.to_numeric(df["paczynski_tau_coverage_score"], errors="coerce")
    else:
        score = pd.Series([np.nan] * len(df), index=df.index, dtype=float)

    cand_bp = bp_rp.to_numpy(dtype=float)[cand_mask]
    cand_mg = mg_num.to_numpy(dtype=float)[cand_mask]
    cand_score = pd.to_numeric(score, errors="coerce").to_numpy(dtype=float)[cand_mask]

    finite_score = np.isfinite(cand_score)
    if np.any(finite_score):
        vmin = float(np.nanpercentile(cand_score[finite_score], 5.0))
        vmax = float(np.nanpercentile(cand_score[finite_score], 95.0))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            vmin, vmax = float(np.nanmin(cand_score[finite_score])), float(np.nanmax(cand_score[finite_score]))
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.cm.viridis
        ax.scatter(
            cand_bp[finite_score],
            cand_mg[finite_score],
            s=55.0,
            c=cand_score[finite_score],
            cmap=cmap,
            norm=norm,
            marker="*",
            alpha=0.95,
            edgecolors="none",
            zorder=3,
        )
        # Minimal colorbar for the candidate score coloring.
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, fraction=0.045, pad=0.02)
        cb.ax.tick_params(labelsize=7)
        cb.set_label("Quality score", fontsize=8)
    else:
        ax.scatter(cand_bp, cand_mg, s=55.0, c="tab:red", marker="*", alpha=0.95, edgecolors="none", zorder=3)

    ax.set_xlabel("BP-RP")
    ax.set_ylabel("M_G")
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.5)
    ax.invert_yaxis()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", format="pdf")
    plt.close(fig)


def _crossmatch_col_missing_mask(series: pd.Series) -> pd.Series:
    if pd.api.types.is_float_dtype(series) or pd.api.types.is_integer_dtype(series):
        return series.isna()
    s_str = series.astype(str).str.strip().str.lower()
    return series.isna() | s_str.isin(('', 'nan', 'none', '<na>'))


def _merge_crossmatch_columns(master_df: pd.DataFrame, enriched: pd.DataFrame) -> pd.DataFrame:
    """Merge enrichment columns into *master_df* on ``candidate_id``; keep non-empty master values."""
    if enriched is None or enriched.empty:
        return master_df.copy()
    m = master_df.copy()
    e = enriched.drop_duplicates(subset=['candidate_id'], keep='first').copy()
    for c in ('ra', 'dec'):
        if c in e.columns:
            e = e.drop(columns=[c], errors='ignore')
    e = e.set_index('candidate_id')
    m = m.set_index('candidate_id')
    for col in e.columns:
        if col in m.columns:
            ev = e[col].reindex(m.index)
            miss = _crossmatch_col_missing_mask(m[col])
            m.loc[miss, col] = ev.loc[miss]
        else:
            m[col] = e[col].reindex(m.index)
    return m.reset_index().copy()


def _prepare_vetting_style_coords(df: pd.DataFrame) -> pd.DataFrame:
    """Build ``ra``/``dec``/``gaia_id``/``source_id``/``asas_sn_id`` for characterize/vet inputs."""
    work = df.copy()
    work['candidate_id'] = work['candidate_id'].astype(str)
    work['ra'] = pd.to_numeric(work['ra_deg'], errors='coerce')
    work['dec'] = pd.to_numeric(work['dec_deg'], errors='coerce')
    if 'gaia_dr3_source_id' in work.columns:
        gid = pd.to_numeric(work['gaia_dr3_source_id'], errors='coerce')
        # Avoid casting the whole Series to int64 because `NaN` would raise IntCastingNaNError.
        sid = pd.Series(np.nan, index=work.index, dtype=object)
        mask = np.isfinite(gid.to_numpy(dtype=float))
        if mask.any():
            sid.loc[mask] = gid.loc[mask].astype(np.int64).astype(str)
        work['source_id'] = sid
        work['gaia_id'] = sid
    else:
        work['source_id'] = np.nan
        work['gaia_id'] = np.nan
    if 'asassn_source_id' in work.columns:
        a = work['asassn_source_id'].astype(str).str.strip()
        bad = a.str.lower().isin(('', 'nan', 'none', '<na>'))
        work['asas_sn_id'] = np.where(bad, np.nan, a)
    return work


def _run_microlensing_crossmatch_enrichment(
    microlensing_table_df: pd.DataFrame,
    *,
    repo_root: Path,
    show_progress: bool,
    dust: bool = False,
    unwise: bool = False,
    neowise_lc: bool = False,
    starhorse: str | None = None,
    gaia_catalog: Path | None = None,
    vsx_crossmatch: Path | None = None,
    vet_method: str = 'tap',
    characterize_checkpoint_path: Path | None = None,
    vetting_checkpoint_path: Path | None = None,
) -> pd.DataFrame:
    """
    Gaia multi-wavelength context via ``characterize_candidates_df`` and archival vetting via
    ``vet_candidates`` (SIMBAD, Gaia variability/EB/epoch, ASAS-SN vars, ZTF vars, TNS, ALeRCE,
    eROSITA, PM check; optional NEOWISE LCs). ATLAS forced photometry is not run in this script.
    Skips ``crossmatch_microlensing_catalogs`` here because this pipeline already cross-matches
    published microlensing events separately.
    """
    try:
        from malca.characterize import characterize_candidates_df
        from malca.config import GAIA_LOCAL_CATALOG as _GAIA_LOCAL, VSX_CROSSMATCH_PATH as _VSX_DEF
        from malca.vetting import vet_candidates
    except ImportError as exc:
        print(f'Crossmatch suite skipped (import failed): {exc}', flush=True)
        return microlensing_table_df

    if microlensing_table_df.empty:
        return microlensing_table_df

    gaia_path = Path(gaia_catalog).expanduser().resolve() if gaia_catalog is not None else (repo_root / _GAIA_LOCAL).resolve()
    vsx_path = (
        Path(vsx_crossmatch).expanduser().resolve()
        if vsx_crossmatch is not None
        else (repo_root / _VSX_DEF).resolve()
    )

    work = _prepare_vetting_style_coords(microlensing_table_df)

    if gaia_path.is_file():
        if show_progress:
            print(f'  characterize: Gaia cache {gaia_path}', flush=True)
        try:
            work = characterize_candidates_df(
                work,
                crossmatch=vsx_path,
                cache=gaia_path,
                dust=dust,
                starhorse=starhorse,
                run_unwise=unwise,
                checkpoint_path=characterize_checkpoint_path,
            )
        except Exception as exc:
            print(f'  characterize_candidates_df failed ({type(exc).__name__}): {exc}', flush=True)
    else:
        print(
            f'  characterize skipped: no local Gaia parquet at {gaia_path} '
            f'(run ``malca gaia-fetch`` or pass --crossmatch-gaia-catalog).',
            flush=True,
        )

    if show_progress:
        print('  vet_candidates: SIMBAD, Gaia var/EB/epoch, ASAS-SN/ZTF vars, TNS, ALeRCE, eROSITA, PM …', flush=True)
    try:
        work = vet_candidates(
            work,
            run_microlens=False,
            run_atlas=False,
            run_neowise_lc=neowise_lc,
            checkpoint_path=vetting_checkpoint_path,
            method=vet_method,
        )
    except Exception as exc:
        print(f'  vet_candidates failed ({type(exc).__name__}): {exc}', flush=True)
        return _merge_crossmatch_columns(microlensing_table_df, work)

    return _merge_crossmatch_columns(microlensing_table_df, work)


def _trend_baseline(t: np.ndarray, baseline: float, slope: float, t_ref: float) -> np.ndarray:
    return baseline + slope * (t - t_ref)


def _solve_u0_from_A0(A0: float) -> float:
    if A0 <= 1.0:
        return np.inf
    u0 = max(1.0 / A0, 1e-4)
    for _ in range(25):
        sqrt_term = np.sqrt(u0 * u0 + 4.0)
        A_curr = (u0 * u0 + 2.0) / (u0 * sqrt_term)
        eps = 1e-6
        up = u0 + eps
        um = max(u0 - eps, 1e-8)
        sp = np.sqrt(up * up + 4.0)
        sm = np.sqrt(um * um + 4.0)
        Ap = (up * up + 2.0) / (up * sp)
        Am = (um * um + 2.0) / (um * sm)
        dA_du = (Ap - Am) / (up - um)
        if not np.isfinite(dA_du) or dA_du == 0.0:
            break
        step = (A_curr - A0) / dA_du
        u0_next = max(u0 - step, 1e-8)
        if abs(u0_next - u0) < 1e-8:
            u0 = u0_next
            break
        u0 = u0_next
    return float(max(u0, 1e-8))


def _A0_from_u0(u0: float) -> float:
    u0 = max(abs(float(u0)), 1e-8)
    return float((u0 * u0 + 2.0) / (u0 * np.sqrt(u0 * u0 + 4.0)))


def paczynski_mag_trend(t: np.ndarray, A0: float, t0: float, tE: float, baseline: float, slope: float, t_ref: float) -> np.ndarray:
    trend = _trend_baseline(t, baseline, slope, t_ref)
    if A0 <= 1.0 or tE <= 0.0:
        return trend
    u0 = _solve_u0_from_A0(float(A0))
    u = np.sqrt(u0 * u0 + ((t - t0) / tE) ** 2)
    A = (u * u + 2.0) / (u * np.sqrt(u * u + 4.0))
    return trend - 2.5 * np.log10(A)


def gaussian_brightening_trend(t: np.ndarray, depth: float, t0: float, sigma: float, baseline: float, slope: float, t_ref: float) -> np.ndarray:
    trend = _trend_baseline(t, baseline, slope, t_ref)
    sigma = max(abs(float(sigma)), 1e-6)
    return trend - abs(depth) * np.exp(-0.5 * ((t - t0) / sigma) ** 2)


def fred_brightening_trend(t: np.ndarray, depth: float, t0: float, tau_rise: float, tau_decay: float, baseline: float, slope: float, t_ref: float) -> np.ndarray:
    trend = _trend_baseline(t, baseline, slope, t_ref)
    tau_rise = max(abs(float(tau_rise)), 1e-6)
    tau_decay = max(abs(float(tau_decay)), 1e-6)
    dt = t - t0
    rise_arg = np.clip(dt / tau_rise, -60.0, 60.0)
    decay_arg = np.clip(-dt / tau_decay, -60.0, 60.0)
    profile = np.where(dt < 0.0, np.exp(rise_arg), np.exp(decay_arg))
    return trend - abs(depth) * profile


def flat_trend(t: np.ndarray, baseline: float, slope: float, t_ref: float) -> np.ndarray:
    return _trend_baseline(t, baseline, slope, t_ref)


MODEL_PARAM_COUNTS = {
    'flat': 2,
    'gaussian': 5,
    'fred': 6,
    'paczynski': 5,
}

LOG10_DELTA_BIC_POSITIVE = float(np.log10(1.0 + 2.0))
LOG10_DELTA_BIC_STRONG = float(np.log10(1.0 + 6.0))
LOG10_DELTA_BIC_VERY_STRONG = float(np.log10(1.0 + 10.0))
# ΔBIC = BIC_other - BIC_paczynski (negative ⇒ other model preferred on BIC).
NON_PACZYNSKI_SELECTION_DELTA_BIC_THRESHOLD = 6.0
# Minimum (BIC_flat - BIC_pac) for Pac to count as clearly beating flat in QC.
PAC_WEAK_VS_FLAT_MIN_DELTA_BIC = 6.0
# When best displayed model is Pac, plot Gaussian/FRED only if flat beats Pac strongly:
# BIC_flat - BIC_pac <= -PLOT_ALT_WHEN_PAC_VS_FLAT_DELTA_BIC.
PLOT_ALT_WHEN_PAC_VS_FLAT_DELTA_BIC = 10.0

# Weighted τ-coverage of ASAS-SN samples vs fitted Paczynski (|τ|≤PAC_COVERAGE_TAU_MAX).
PAC_COVERAGE_TAU_MAX = 6.0
PAC_COVERAGE_N_BINS = 120
PAC_COVERAGE_GAUSS_SIGMA_TAU = 1.5
# Hybrid kernel: 0.5 x normalized (A(u(τ))−1) at bin centers + 0.5 x Gaussian in τ (σ = PAC_COVERAGE_GAUSS_SIGMA_TAU).
PAC_COVERAGE_WARN_THRESHOLD = 0.55
PAC_COVERAGE_FAIL_THRESHOLD = 0.35

PARALLAX_MIN_TE_DAYS = 80.0
PARALLAX_MIN_FIT_POINTS = 80
PARALLAX_MIN_SPAN_DAYS = 240.0
PARALLAX_MAX_ABS_PIE = 1.5
PARALLAX_MAX_U0_ABS = 2.0
PARALLAX_MIN_U0_FACTOR = 1.0 / 3.0
PARALLAX_MAX_U0_FACTOR = 3.0
PARALLAX_MIN_TE_FACTOR = 0.35
PARALLAX_MAX_TE_FACTOR = 3.0
PARALLAX_BOUND_FRAC = 0.02
PARALLAX_MAX_REDUCED_CHI2 = 10.0
PARALLAX_REQUIRED_DELTA_BIC = 6.0
# Parallax PSPL / branch / MCMC runs only when BIC-best model is Paczynski and tE ≥ PARALLAX_MIN_TE_DAYS (long events).
PARALLAX_ENABLE_MCMC = True
PARALLAX_MCMC_CHAINS = 6
PARALLAX_MCMC_BURN = 200
PARALLAX_MCMC_STEPS = 400
PARALLAX_MCMC_THIN = 2
PARALLAX_RANDOM_SEED = 20260322
PARALLAX_MIN_ACCEPTANCE_RATE = 0.002


def _raw_delta_bic(value: object) -> float:
    numeric = _finite_float(value)
    if numeric is None:
        return np.nan
    return float(numeric)


def _paczynski_weighted_coverage(
    jd_obs: np.ndarray,
    pac: dict[str, object],
    *,
    tau_max: float | None = None,
    n_bins: int | None = None,
    gauss_sigma: float | None = None,
) -> dict[str, float | int]:
    """
    Fraction of importance-weighted Paczynski τ-bins (τ = (t−t0)/tE) that contain ≥1 observation.

    Weights emphasize the core of the light curve (magnification above baseline + Gaussian in τ).
    """
    tau_max = float(PAC_COVERAGE_TAU_MAX if tau_max is None else tau_max)
    n_bins = int(PAC_COVERAGE_N_BINS if n_bins is None else n_bins)
    gauss_sigma = float(PAC_COVERAGE_GAUSS_SIGMA_TAU if gauss_sigma is None else gauss_sigma)

    empty: dict[str, float | int] = {
        'paczynski_tau_coverage_score': np.nan,
        'paczynski_coverage_n_bins_hit': 0,
        'paczynski_coverage_n_bins': n_bins,
        'paczynski_coverage_max_weighted_gap': np.nan,
        'paczynski_coverage_frac_points_in_tau_window': np.nan,
    }
    if not pac.get('success') or n_bins < 4:
        return empty

    params = pac.get('params')
    if params is None or len(params) < 4:
        return empty
    A0, t0, tE = float(params[0]), float(params[1]), float(abs(params[2]))
    if not np.isfinite(A0) or not np.isfinite(t0) or not np.isfinite(tE) or tE <= 0.0:
        return empty

    jd_obs = np.asarray(jd_obs, dtype=float)
    jd_obs = jd_obs[np.isfinite(jd_obs)]
    if len(jd_obs) == 0:
        return empty

    tau_obs = (jd_obs - t0) / tE
    in_win = np.abs(tau_obs) <= tau_max
    frac_in_win = float(np.mean(in_win))

    edges = np.linspace(-tau_max, tau_max, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    u0 = _solve_u0_from_A0(float(A0))
    u_c = np.sqrt(u0 * u0 + np.square(centers))
    A_c = (u_c * u_c + 2.0) / (u_c * np.sqrt(u_c * u_c + 4.0))
    w_amp = np.clip(A_c - 1.0, 0.0, None)
    sum_amp = float(np.sum(w_amp))
    w_amp_norm = (w_amp / sum_amp) if sum_amp > 1e-12 else np.full(n_bins, 1.0 / n_bins, dtype=float)

    sig = max(gauss_sigma, 1e-6)
    w_g = np.exp(-0.5 * np.square(centers / sig))
    sum_g = float(np.sum(w_g))
    w_g_norm = (w_g / sum_g) if sum_g > 1e-12 else np.full(n_bins, 1.0 / n_bins, dtype=float)

    w = 0.5 * w_amp_norm + 0.5 * w_g_norm

    counts, _ = np.histogram(tau_obs[in_win], bins=edges)
    s = (counts > 0).astype(float)
    score = float(np.dot(w, s))
    n_hit = int(np.sum(s))

    max_wgap = 0.0
    run_sum = 0.0
    for i in range(n_bins):
        if s[i] < 0.5:
            run_sum += float(w[i])
        else:
            max_wgap = max(max_wgap, run_sum)
            run_sum = 0.0
    max_wgap = max(max_wgap, run_sum)

    return {
        'paczynski_tau_coverage_score': score,
        'paczynski_coverage_n_bins_hit': n_hit,
        'paczynski_coverage_n_bins': n_bins,
        'paczynski_coverage_max_weighted_gap': float(max_wgap),
        'paczynski_coverage_frac_points_in_tau_window': frac_in_win,
    }




def _mag_to_relative_flux(mag: np.ndarray, err_mag: np.ndarray, ref_mag: float | None = None) -> tuple[np.ndarray, np.ndarray, float]:
    mag = np.asarray(mag, dtype=float)
    err_mag = np.asarray(err_mag, dtype=float)
    if ref_mag is None or not np.isfinite(ref_mag):
        ref_mag = float(np.nanmedian(mag))
    flux = np.power(10.0, -0.4 * (mag - ref_mag))
    flux_err = (np.log(10.0) / 2.5) * flux * np.clip(err_mag, 1e-4, None)
    flux_err = np.clip(flux_err, 1e-8, None)
    return flux, flux_err, float(ref_mag)


def _relative_flux_to_mag(flux: np.ndarray, ref_mag: float) -> np.ndarray:
    flux = np.clip(np.asarray(flux, dtype=float), 1e-12, None)
    return float(ref_mag) - 2.5 * np.log10(flux)


def _pspl_magnification_from_tau_beta(tau: np.ndarray, beta: np.ndarray) -> np.ndarray:
    u = np.sqrt(np.maximum(np.asarray(tau, dtype=float) ** 2 + np.asarray(beta, dtype=float) ** 2, 1e-12))
    return (u * u + 2.0) / (u * np.sqrt(u * u + 4.0))


def _solve_source_blend_linear(magnification: np.ndarray, flux: np.ndarray, flux_err: np.ndarray) -> tuple[float, float, np.ndarray]:
    magnification = np.asarray(magnification, dtype=float)
    flux = np.asarray(flux, dtype=float)
    flux_err = np.asarray(flux_err, dtype=float)
    valid = np.isfinite(magnification) & np.isfinite(flux) & np.isfinite(flux_err) & (flux_err > 0.0)
    if int(np.sum(valid)) < 2:
        return np.nan, np.nan, np.full_like(flux, np.nan, dtype=float)

    A = magnification[valid]
    F = flux[valid]
    w = 1.0 / np.square(flux_err[valid])
    design = np.column_stack([A, np.ones_like(A)])
    design_w = design * np.sqrt(w[:, None])
    flux_w = F * np.sqrt(w)
    try:
        result = lsq_linear(design_w, flux_w, bounds=(0.0, np.inf), method='trf', lsmr_tol='auto')
        if not result.success or not np.all(np.isfinite(result.x)):
            raise RuntimeError(str(result.message))
        Fs, Fb = result.x
    except Exception:
        try:
            Fs, Fb = np.linalg.lstsq(design_w, flux_w, rcond=None)[0]
        except np.linalg.LinAlgError:
            return np.nan, np.nan, np.full_like(flux, np.nan, dtype=float)
        Fs = max(float(Fs), 0.0)
        Fb = max(float(Fb), 0.0)
    model = Fs * magnification + Fb
    return float(Fs), float(Fb), np.asarray(model, dtype=float)


def _sky_tangent_basis(ra_deg: float, dec_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ra = np.deg2rad(float(ra_deg))
    dec = np.deg2rad(float(dec_deg))
    east_hat = np.array([-np.sin(ra), np.cos(ra), 0.0], dtype=float)
    north_hat = np.array([-np.cos(ra) * np.sin(dec), -np.sin(ra) * np.sin(dec), np.cos(dec)], dtype=float)
    los_hat = np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)], dtype=float)
    return east_hat, north_hat, los_hat


def _project_earth_orbit_geocentric(jd_full: np.ndarray, ra_deg: float, dec_deg: float, t0_ref_jd: float) -> dict[str, np.ndarray | float]:
    jd_full = np.asarray(jd_full, dtype=float)
    east_hat, north_hat, _ = _sky_tangent_basis(ra_deg, dec_deg)
    times_tdb = Time(jd_full, format='jd', scale='utc').tdb
    t0_ref_tdb = Time(float(t0_ref_jd), format='jd', scale='utc').tdb
    earth_pos, earth_vel = get_body_barycentric_posvel('earth', times_tdb)
    earth_pos_ref, earth_vel_ref = get_body_barycentric_posvel('earth', t0_ref_tdb)

    pos = earth_pos.xyz.to_value(u.au).T
    vel = earth_vel.xyz.to_value(u.au / u.day).T
    pos_ref = earth_pos_ref.xyz.to_value(u.au)
    vel_ref = earth_vel_ref.xyz.to_value(u.au / u.day)
    dt_days = times_tdb.jd - t0_ref_tdb.jd
    delta_vec = pos - pos_ref[None, :] - dt_days[:, None] * vel_ref[None, :]
    delta_n = delta_vec @ north_hat
    delta_e = delta_vec @ east_hat
    return {
        'jd_full': jd_full,
        't0_ref_jd': float(t0_ref_jd),
        'delta_n': np.asarray(delta_n, dtype=float),
        'delta_e': np.asarray(delta_e, dtype=float),
    }


def _microlens_param_bounds(
    jd_full: np.ndarray,
    *,
    u0_abs_guess: float | None = None,
    tE_guess_days: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    jd_full = np.asarray(jd_full, dtype=float)
    span = max(float(np.nanmax(jd_full) - np.nanmin(jd_full)), 30.0)
    tE_cap = min(max(25.0, 4.0 * span), 5000.0)

    u0_guess = _finite_float(u0_abs_guess)
    if u0_guess is None or u0_guess <= 0.0:
        u0_lo = 1e-3
        u0_hi = PARALLAX_MAX_U0_ABS
    else:
        u0_lo = max(1e-3, min(u0_guess * PARALLAX_MIN_U0_FACTOR, 0.95 * u0_guess))
        u0_hi = min(PARALLAX_MAX_U0_ABS, max(0.05, u0_guess * PARALLAX_MAX_U0_FACTOR))
        if u0_hi <= u0_lo:
            u0_hi = min(PARALLAX_MAX_U0_ABS, u0_lo * 1.5 + 0.05)

    tE_guess = _finite_float(tE_guess_days)
    if tE_guess is None or tE_guess <= 0.0:
        tE_lo = 5.0
        tE_hi = tE_cap
    else:
        tE_lo = max(5.0, tE_guess * PARALLAX_MIN_TE_FACTOR)
        tE_hi = min(tE_cap, max(30.0, tE_guess * PARALLAX_MAX_TE_FACTOR))
        if tE_hi <= tE_lo:
            tE_hi = min(tE_cap, tE_lo * 1.5 + 5.0)

    lower = np.array([np.log(u0_lo), np.nanmin(jd_full), np.log(tE_lo), -PARALLAX_MAX_ABS_PIE, -PARALLAX_MAX_ABS_PIE], dtype=float)
    upper = np.array([np.log(u0_hi), np.nanmax(jd_full), np.log(tE_hi), PARALLAX_MAX_ABS_PIE, PARALLAX_MAX_ABS_PIE], dtype=float)
    return lower, upper


def _profile_flux_microlensing_model(
    opt_params: np.ndarray,
    *,
    branch_sign: int,
    jd_full: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    ref_mag: float,
    ephemeris: dict[str, np.ndarray | float] | None,
    with_parallax: bool,
) -> dict[str, object]:
    log_u0_abs = float(opt_params[0])
    t0_jd = float(opt_params[1])
    log_tE = float(opt_params[2])
    piE_n = float(opt_params[3]) if with_parallax and len(opt_params) > 3 else 0.0
    piE_e = float(opt_params[4]) if with_parallax and len(opt_params) > 4 else 0.0
    u0_abs = float(np.exp(log_u0_abs))
    u0 = float(branch_sign * u0_abs)
    tE_days = float(np.exp(log_tE))
    tau = (np.asarray(jd_full, dtype=float) - t0_jd) / tE_days
    beta = np.full_like(tau, u0, dtype=float)
    if with_parallax and ephemeris is not None:
        delta_n = np.asarray(ephemeris['delta_n'], dtype=float)
        delta_e = np.asarray(ephemeris['delta_e'], dtype=float)
        tau = tau + piE_n * delta_n + piE_e * delta_e
        beta = beta + piE_n * delta_e - piE_e * delta_n
    magnification = _pspl_magnification_from_tau_beta(tau, beta)
    Fs, Fb, model_flux = _solve_source_blend_linear(magnification, flux, flux_err)
    valid = np.isfinite(model_flux) & np.isfinite(flux) & np.isfinite(flux_err) & (flux_err > 0.0)
    residuals = np.full_like(flux, np.nan, dtype=float)
    if np.any(valid):
        residuals[valid] = (flux[valid] - model_flux[valid]) / flux_err[valid]
    chi2 = float(np.nansum(np.square(residuals[valid]))) if np.any(valid) else np.nan
    model_mag = _relative_flux_to_mag(model_flux, ref_mag)
    return {
        'u0': u0,
        'u0_abs': u0_abs,
        't0_jd': t0_jd,
        'tE_days': tE_days,
        'piE_N': piE_n,
        'piE_E': piE_e,
        'piE': float(np.hypot(piE_n, piE_e)),
        'Fs': float(Fs),
        'Fb': float(Fb),
        'magnification': magnification,
        'model_flux': model_flux,
        'model_mag': model_mag,
        'residuals': residuals,
        'chi2': chi2,
    }


def _fit_flux_microlensing_branch(
    *,
    jd_fit: np.ndarray,
    mag_fit: np.ndarray,
    err_fit: np.ndarray,
    ref_mag: float,
    branch_sign: int,
    t0_guess_jd: float,
    u0_abs_guess: float,
    tE_guess_days: float,
    with_parallax: bool,
    ephemeris: dict[str, np.ndarray | float] | None,
) -> dict[str, object]:
    jd_fit = np.asarray(jd_fit, dtype=float)
    mag_fit = np.asarray(mag_fit, dtype=float)
    err_fit = np.asarray(err_fit, dtype=float)
    flux_fit, flux_err_fit, ref_mag = _mag_to_relative_flux(mag_fit, err_fit, ref_mag=ref_mag)
    lower_full, upper_full = _microlens_param_bounds(jd_fit, u0_abs_guess=u0_abs_guess, tE_guess_days=tE_guess_days)
    if with_parallax:
        lower = lower_full.copy()
        upper = upper_full.copy()
    else:
        lower = lower_full[:3].copy()
        upper = upper_full[:3].copy()

    t0_guess_jd = float(np.clip(t0_guess_jd, lower_full[1], upper_full[1]))
    u0_abs_guess = float(np.clip(abs(u0_abs_guess), np.exp(lower_full[0]), np.exp(upper_full[0])))
    tE_guess_days = float(np.clip(abs(tE_guess_days), np.exp(lower_full[2]), np.exp(upper_full[2])))

    def residuals(opt_params: np.ndarray) -> np.ndarray:
        profile = _profile_flux_microlensing_model(
            opt_params,
            branch_sign=branch_sign,
            jd_full=jd_fit,
            flux=flux_fit,
            flux_err=flux_err_fit,
            ref_mag=ref_mag,
            ephemeris=ephemeris,
            with_parallax=with_parallax,
        )
        model_flux = np.asarray(profile['model_flux'], dtype=float)
        if (not np.all(np.isfinite(model_flux))) or (not np.isfinite(profile['Fs'])) or profile['Fs'] <= 0.0:
            return np.full_like(flux_fit, 1e6, dtype=float)
        if np.nanmin(model_flux) <= 0.0:
            return np.full_like(flux_fit, 1e6, dtype=float)
        return np.asarray(profile['residuals'], dtype=float)

    start_vectors: list[np.ndarray] = []
    if with_parallax:
        parallax_starts = [
            (0.0, 0.0),
            (0.10, 0.0),
            (-0.10, 0.0),
            (0.0, 0.10),
            (0.0, -0.10),
            (0.20, 0.20),
        ]
        for piE_n0, piE_e0 in parallax_starts:
            start_vectors.append(np.array([np.log(u0_abs_guess), t0_guess_jd, np.log(tE_guess_days), piE_n0, piE_e0], dtype=float))
        start_vectors.append(np.array([np.log(max(u0_abs_guess, 0.05)), t0_guess_jd, np.log(min(np.exp(upper_full[2]), 1.25 * tE_guess_days)), 0.0, 0.0], dtype=float))
    else:
        start_vectors.extend([
            np.array([np.log(u0_abs_guess), t0_guess_jd, np.log(tE_guess_days)], dtype=float),
            np.array([np.log(max(u0_abs_guess, 0.05)), t0_guess_jd, np.log(min(np.exp(upper_full[2]), 1.25 * tE_guess_days))], dtype=float),
            np.array([np.log(min(0.3, np.exp(upper_full[0]))), t0_guess_jd, np.log(max(np.exp(lower_full[2]), 0.75 * tE_guess_days))], dtype=float),
        ])

    best: dict[str, object] | None = None
    last_error = ''
    for x0 in start_vectors:
        try:
            result = least_squares(
                residuals,
                x0=np.clip(x0, lower + 1e-8, upper - 1e-8),
                bounds=(lower, upper),
                loss='linear',
                max_nfev=5000,
            )
        except Exception as exc:
            last_error = repr(exc)
            continue
        if not result.success or not np.all(np.isfinite(result.x)):
            last_error = str(result.message)
            continue
        profile = _profile_flux_microlensing_model(
            result.x,
            branch_sign=branch_sign,
            jd_full=jd_fit,
            flux=flux_fit,
            flux_err=flux_err_fit,
            ref_mag=ref_mag,
            ephemeris=ephemeris,
            with_parallax=with_parallax,
        )
        if (not np.isfinite(profile['chi2'])) or (not np.isfinite(profile['Fs'])) or profile['Fs'] <= 0.0:
            continue
        if best is None or float(profile['chi2']) < float(best['profile']['chi2']):
            best = {'result': result, 'profile': profile}

    if best is None:
        return {
            'success': False,
            'status': last_error or 'least_squares_failed',
            'branch_sign': int(branch_sign),
            'with_parallax': bool(with_parallax),
        }

    result = best['result']
    profile = best['profile']
    n_points = int(len(jd_fit))
    n_total_params = 7 if with_parallax else 5
    dof = max(n_points - n_total_params, 1)
    bic = float(profile['chi2'] + n_total_params * np.log(max(n_points, 2)))
    out = {
        'success': True,
        'status': 'ok',
        'branch_sign': int(branch_sign),
        'with_parallax': bool(with_parallax),
        'opt_params': np.asarray(result.x, dtype=float),
        'bounds': (np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)),
        'least_squares_result': result,
        'chi2': float(profile['chi2']),
        'reduced_chi2': float(profile['chi2'] / dof),
        'bic': bic,
        'n_points': n_points,
        'ref_mag': float(ref_mag),
        **profile,
    }
    return out


def _build_branch_log_prob(
    *,
    branch_sign: int,
    jd_fit: np.ndarray,
    mag_fit: np.ndarray,
    err_fit: np.ndarray,
    ref_mag: float,
    ephemeris: dict[str, np.ndarray | float],
    lower: np.ndarray,
    upper: np.ndarray,
):
    flux_fit, flux_err_fit, ref_mag = _mag_to_relative_flux(mag_fit, err_fit, ref_mag=ref_mag)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)

    def log_prob(opt_params: np.ndarray) -> float:
        opt_params = np.asarray(opt_params, dtype=float)
        if opt_params.shape != lower.shape:
            return -np.inf
        if np.any(opt_params <= lower) or np.any(opt_params >= upper):
            return -np.inf
        profile = _profile_flux_microlensing_model(
            opt_params,
            branch_sign=branch_sign,
            jd_full=jd_fit,
            flux=flux_fit,
            flux_err=flux_err_fit,
            ref_mag=ref_mag,
            ephemeris=ephemeris,
            with_parallax=True,
        )
        if (not np.isfinite(profile['chi2'])) or (not np.isfinite(profile['Fs'])) or profile['Fs'] <= 0.0:
            return -np.inf
        model_flux = np.asarray(profile['model_flux'], dtype=float)
        if np.nanmin(model_flux) <= 0.0:
            return -np.inf
        return float(-0.5 * profile['chi2'])

    return log_prob


def _run_metropolis_sampler(
    *,
    start: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    log_prob,
    proposal_scale: np.ndarray,
    n_chains: int,
    n_burn: int,
    n_steps: int,
    thin: int,
    seed: int,
) -> dict[str, object]:
    rng = np.random.default_rng(int(seed))
    start = np.asarray(start, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    proposal_scale = np.asarray(proposal_scale, dtype=float)
    proposal_scale = np.clip(proposal_scale, 1e-4, None)
    dim = int(len(start))
    kept_samples: list[np.ndarray] = []
    acceptance_rates: list[float] = []

    for _ in range(int(n_chains)):
        current = start.copy()
        current_logp = log_prob(current)
        for _attempt in range(50):
            if np.isfinite(current_logp):
                break
            trial = np.clip(start + rng.normal(scale=0.35 * proposal_scale, size=dim), lower + 1e-8, upper - 1e-8)
            trial_logp = log_prob(trial)
            if np.isfinite(trial_logp):
                current = trial
                current_logp = trial_logp
                break
        if not np.isfinite(current_logp):
            continue

        accepted = 0
        chain_samples: list[np.ndarray] = []
        total_steps = int(n_burn + n_steps)
        for istep in range(total_steps):
            proposal = np.clip(current + rng.normal(scale=proposal_scale, size=dim), lower + 1e-8, upper - 1e-8)
            proposal_logp = log_prob(proposal)
            if np.isfinite(proposal_logp):
                log_alpha = proposal_logp - current_logp
                if log_alpha >= 0.0 or np.log(rng.random()) < log_alpha:
                    current = proposal
                    current_logp = proposal_logp
                    accepted += 1
            if istep >= n_burn and ((istep - n_burn) % max(int(thin), 1) == 0):
                chain_samples.append(current.copy())

        if chain_samples:
            kept_samples.extend(chain_samples)
            acceptance_rates.append(accepted / max(total_steps, 1))

    if not kept_samples:
        return {
            'success': False,
            'n_samples': 0,
            'acceptance_rate': np.nan,
            'samples_opt': np.empty((0, dim), dtype=float),
        }

    return {
        'success': True,
        'n_samples': int(len(kept_samples)),
        'acceptance_rate': float(np.nanmean(acceptance_rates)) if acceptance_rates else np.nan,
        'samples_opt': np.asarray(kept_samples, dtype=float),
    }


def _summarize_branch_samples(samples_opt: np.ndarray, branch_sign: int) -> dict[str, float]:
    samples_opt = np.asarray(samples_opt, dtype=float)
    if samples_opt.size == 0:
        return {}
    u0_abs = np.exp(samples_opt[:, 0])
    t0_jd = samples_opt[:, 1]
    tE_days = np.exp(samples_opt[:, 2])
    piE_n = samples_opt[:, 3]
    piE_e = samples_opt[:, 4]
    u0 = branch_sign * u0_abs
    piE = np.hypot(piE_n, piE_e)

    def add_summary(out: dict[str, float], prefix: str, values: np.ndarray) -> None:
        q16, q50, q84 = np.nanpercentile(values, [16.0, 50.0, 84.0])
        out[f'{prefix}_p16'] = float(q16)
        out[f'{prefix}_p50'] = float(q50)
        out[f'{prefix}_p84'] = float(q84)

    out: dict[str, float] = {}
    add_summary(out, 'u0', u0)
    add_summary(out, 't0_jd', t0_jd)
    add_summary(out, 'tE_days', tE_days)
    add_summary(out, 'piE_N', piE_n)
    add_summary(out, 'piE_E', piE_e)
    add_summary(out, 'piE', piE)
    return out


def _parallax_seed_from_jacobian(branch_fit: dict[str, object], tE_days: float) -> np.ndarray:
    default_scale = np.array([0.03, max(0.6, 0.01 * max(float(tE_days), 1.0)), 0.03, 0.015, 0.015], dtype=float)
    result = branch_fit.get('least_squares_result')
    jac = getattr(result, 'jac', None) if result is not None else None
    if jac is None:
        return default_scale
    try:
        jac = np.asarray(jac, dtype=float)
        hess = jac.T @ jac
        cov = np.linalg.pinv(hess)
        scale = np.sqrt(np.clip(np.diag(cov), 1e-6, None))
        if scale.shape != default_scale.shape or not np.all(np.isfinite(scale)):
            return default_scale
        return np.clip(scale, 0.25 * default_scale, 2.0 * default_scale)
    except Exception:
        return default_scale


def _empty_parallax_result(status: str) -> dict[str, object]:
    return {
        'attempted': False,
        'fit_ok': False,
        'preferred': False,
        'status': status,
        't0_ref_jd': np.nan,
        'pspl': {},
        'branches': {'u0_pos': {}, 'u0_neg': {}},
        'best_branch': '',
        'delta_bic': np.nan,
        'branch_delta_bic': np.nan,
    }


def _param_near_bounds(value: float, lower: float, upper: float, frac: float = PARALLAX_BOUND_FRAC) -> bool:
    if not (np.isfinite(value) and np.isfinite(lower) and np.isfinite(upper) and upper > lower):
        return False
    width = upper - lower
    margin = max(float(frac) * width, 1e-6)
    return bool(value <= lower + margin or value >= upper - margin)


def _parallax_branch_quality(branch_fit: dict[str, object], pspl_fit: dict[str, object]) -> dict[str, object]:
    if not branch_fit.get('success'):
        return {'ok': False, 'warnings': ['fit_failed']}

    warnings: list[str] = []
    lower, upper = branch_fit.get('bounds', (None, None))
    opt_params = np.asarray(branch_fit.get('opt_params', []), dtype=float)
    lower_arr = np.asarray(lower, dtype=float) if lower is not None else np.asarray([])
    upper_arr = np.asarray(upper, dtype=float) if upper is not None else np.asarray([])
    param_names = ('log_u0_abs', 't0_jd', 'log_tE', 'piE_N', 'piE_E')
    if opt_params.shape == lower_arr.shape == upper_arr.shape:
        for idx, name in enumerate(param_names[: len(opt_params)]):
            if _param_near_bounds(float(opt_params[idx]), float(lower_arr[idx]), float(upper_arr[idx])):
                warnings.append(f'{name}_near_bound')

    pspl_tE = _finite_float(pspl_fit.get('tE_days'))
    par_tE = _finite_float(branch_fit.get('tE_days'))
    if pspl_tE is not None and par_tE is not None and pspl_tE > 0.0:
        ratio = par_tE / pspl_tE
        if ratio < 0.4 or ratio > 2.5:
            warnings.append('tE_shift_large')

    reduced_chi2 = _finite_float(branch_fit.get('reduced_chi2'))
    if reduced_chi2 is not None and reduced_chi2 > PARALLAX_MAX_REDUCED_CHI2:
        warnings.append('high_reduced_chi2')

    mcmc = branch_fit.get('mcmc', {}) or {}
    acceptance = _finite_float(mcmc.get('acceptance_rate'))
    if mcmc.get('success') and acceptance is not None and acceptance < PARALLAX_MIN_ACCEPTANCE_RATE:
        warnings.append('sampler_low_acceptance')
    return {'ok': len(warnings) == 0, 'warnings': warnings}


def _fit_parallax_diagnostics(
    context: dict[str, object],
    best_seed_result: dict[str, object],
    *,
    ra_deg: float | None,
    dec_deg: float | None,
) -> dict[str, object]:
    pac = best_seed_result['fits'].get('paczynski', {})
    if not pac.get('success'):
        return _empty_parallax_result('not_attempted:no_paczynski_fit')
    if ra_deg is None or dec_deg is None or not np.isfinite(ra_deg) or not np.isfinite(dec_deg):
        return _empty_parallax_result('not_attempted:missing_coordinates')

    jd_fit = np.asarray(best_seed_result['jd_fit'], dtype=float)
    mag_fit = np.asarray(best_seed_result['mag_fit'], dtype=float)
    err_fit = np.asarray(best_seed_result['err_fit'], dtype=float)
    if len(jd_fit) < PARALLAX_MIN_FIT_POINTS:
        return _empty_parallax_result('not_attempted:too_few_points')
    fit_span = float(np.nanmax(jd_fit) - np.nanmin(jd_fit)) if len(jd_fit) else 0.0
    if fit_span < PARALLAX_MIN_SPAN_DAYS:
        return _empty_parallax_result('not_attempted:fit_span_too_short')

    A0_guess = float(pac['params'][0])
    t0_guess_jd = float(pac['params'][1] + 2450000.0)
    tE_guess_days = float(abs(pac['params'][2]))
    if not np.isfinite(tE_guess_days) or tE_guess_days < PARALLAX_MIN_TE_DAYS:
        return _empty_parallax_result('not_attempted:tE_below_threshold')
    u0_abs_guess = float(np.clip(_solve_u0_from_A0(A0_guess), 1e-3, 3.0))
    ref_mag = float(np.nanmedian(mag_fit))
    jd_fit_full = jd_fit + 2450000.0

    pspl_fit = _fit_flux_microlensing_branch(
        jd_fit=jd_fit_full,
        mag_fit=mag_fit,
        err_fit=err_fit,
        ref_mag=ref_mag,
        branch_sign=+1,
        t0_guess_jd=t0_guess_jd,
        u0_abs_guess=u0_abs_guess,
        tE_guess_days=tE_guess_days,
        with_parallax=False,
        ephemeris=None,
    )
    if not pspl_fit.get('success'):
        out = _empty_parallax_result('fit_failed:pspl_flux_fit_failed')
        out['attempted'] = True
        return out

    ephemeris = _project_earth_orbit_geocentric(jd_fit_full, float(ra_deg), float(dec_deg), float(pspl_fit['t0_jd']))
    branches: dict[str, dict[str, object]] = {}
    for branch_sign, branch_name in ((+1, 'u0_pos'), (-1, 'u0_neg')):
        branch_fit = _fit_flux_microlensing_branch(
            jd_fit=jd_fit_full,
            mag_fit=mag_fit,
            err_fit=err_fit,
            ref_mag=ref_mag,
            branch_sign=branch_sign,
            t0_guess_jd=float(pspl_fit['t0_jd']),
            u0_abs_guess=float(abs(pspl_fit['u0'])),
            tE_guess_days=float(pspl_fit['tE_days']),
            with_parallax=True,
            ephemeris=ephemeris,
        )
        branch_fit['t0_ref_jd'] = float(pspl_fit['t0_jd'])
        if branch_fit.get('success') and PARALLAX_ENABLE_MCMC:
            lower, upper = branch_fit['bounds']
            log_prob = _build_branch_log_prob(
                branch_sign=branch_sign,
                jd_fit=jd_fit_full,
                mag_fit=mag_fit,
                err_fit=err_fit,
                ref_mag=ref_mag,
                ephemeris=ephemeris,
                lower=lower,
                upper=upper,
            )
            proposal_scale = _parallax_seed_from_jacobian(branch_fit, float(branch_fit['tE_days']))
            # Use hash for non-numeric candidate IDs
            cid = context['candidate_id']
            try:
                cid_int = int(cid)
            except (ValueError, TypeError):
                cid_int = hash(str(cid)) % 100000
            chain_seed = PARALLAX_RANDOM_SEED + cid_int % 100000 + (0 if branch_sign > 0 else 500000)
            mcmc = _run_metropolis_sampler(
                start=np.asarray(branch_fit['opt_params'], dtype=float),
                lower=np.asarray(lower, dtype=float),
                upper=np.asarray(upper, dtype=float),
                log_prob=log_prob,
                proposal_scale=proposal_scale,
                n_chains=PARALLAX_MCMC_CHAINS,
                n_burn=PARALLAX_MCMC_BURN,
                n_steps=PARALLAX_MCMC_STEPS,
                thin=PARALLAX_MCMC_THIN,
                seed=chain_seed,
            )
            if mcmc.get('success'):
                mcmc['posterior_summary'] = _summarize_branch_samples(np.asarray(mcmc['samples_opt'], dtype=float), branch_sign)
            branch_fit['mcmc'] = mcmc
        else:
            branch_fit['mcmc'] = {'success': False, 'n_samples': 0, 'acceptance_rate': np.nan}
        branch_fit['quality'] = _parallax_branch_quality(branch_fit, pspl_fit)
        branches[branch_name] = branch_fit

    successful = {name: branch for name, branch in branches.items() if branch.get('success')}
    if not successful:
        return {
            'attempted': True,
            'fit_ok': False,
            'preferred': False,
            'status': 'fit_failed:parallax_branch_fit_failed',
            't0_ref_jd': float(pspl_fit['t0_jd']),
            'pspl': pspl_fit,
            'branches': branches,
            'best_branch': '',
            'delta_bic': np.nan,
            'branch_delta_bic': np.nan,
        }

    reliable = {name: branch for name, branch in successful.items() if branch.get('quality', {}).get('ok', False)}
    ranked = reliable or successful
    best_branch = min(ranked, key=lambda name: float(ranked[name]['bic']))
    best_fit = ranked[best_branch]
    delta_bic = (
        float(pspl_fit['bic'] - best_fit['bic'])
        if np.isfinite(pspl_fit.get('bic', np.nan)) and np.isfinite(best_fit.get('bic', np.nan))
        else np.nan
    )
    pos_bic = branches.get('u0_pos', {}).get('bic', np.nan)
    neg_bic = branches.get('u0_neg', {}).get('bic', np.nan)
    branch_delta_bic = float(abs(pos_bic - neg_bic)) if np.isfinite(pos_bic) and np.isfinite(neg_bic) else np.nan
    fit_ok = bool(best_branch in reliable)
    warnings = list(best_fit.get('quality', {}).get('warnings', []))
    preferred = bool(fit_ok and np.isfinite(delta_bic) and delta_bic >= PARALLAX_REQUIRED_DELTA_BIC)
    status = 'preferred' if preferred else ('fit_ok' if fit_ok else 'fit_unreliable')
    return {
        'attempted': True,
        'fit_ok': fit_ok,
        'preferred': preferred,
        'status': status,
        't0_ref_jd': float(pspl_fit['t0_jd']),
        'pspl': pspl_fit,
        'branches': branches,
        'best_branch': best_branch,
        'delta_bic': delta_bic,
        'branch_delta_bic': branch_delta_bic,
        'warnings': warnings,
    }


def _evaluate_parallax_branch_mag(branch_fit: dict[str, object], jd_minus_2450000: np.ndarray, ra_deg: float, dec_deg: float) -> np.ndarray | None:
    if not branch_fit.get('success'):
        return None
    jd_full = np.asarray(jd_minus_2450000, dtype=float) + 2450000.0
    ephemeris = _project_earth_orbit_geocentric(jd_full, float(ra_deg), float(dec_deg), float(branch_fit['t0_ref_jd']))
    opt_params = np.asarray(branch_fit['opt_params'], dtype=float)
    profile = _profile_flux_microlensing_model(
        opt_params,
        branch_sign=int(branch_fit['branch_sign']),
        jd_full=jd_full,
        flux=np.ones_like(jd_full, dtype=float),
        flux_err=np.ones_like(jd_full, dtype=float),
        ref_mag=float(branch_fit['ref_mag']),
        ephemeris=ephemeris,
        with_parallax=True,
    )
    model_flux = float(branch_fit['Fs']) * np.asarray(profile['magnification'], dtype=float) + float(branch_fit['Fb'])
    return _relative_flux_to_mag(model_flux, float(branch_fit['ref_mag']))


def _flatten_parallax_summary(parallax_result: dict[str, object]) -> dict[str, object]:
    defaults: dict[str, object] = {
        'parallax_attempted': False,
        'parallax_fit_ok': False,
        'parallax_preferred': False,
        'parallax_status': parallax_result.get('status', ''),
        'parallax_warning': '',
        'parallax_t0_ref_jd': np.nan,
        'parallax_pspl_flux_chi2': np.nan,
        'parallax_pspl_flux_bic': np.nan,
        'parallax_pspl_flux_reduced_chi2': np.nan,
        'parallax_best_branch': '',
        'parallax_delta_bic': np.nan,
        'parallax_branch_delta_bic': np.nan,
        'parallax_best_t0_jd_minus_2450000': np.nan,
        'parallax_best_tE_days': np.nan,
        'parallax_best_u0': np.nan,
        'parallax_best_piE_N': np.nan,
        'parallax_best_piE_E': np.nan,
        'parallax_best_piE': np.nan,
        'parallax_best_fs': np.nan,
        'parallax_best_fb': np.nan,
        'parallax_best_chi2': np.nan,
        'parallax_best_reduced_chi2': np.nan,
        'parallax_best_bic': np.nan,
        'parallax_best_acceptance_rate': np.nan,
        'parallax_best_n_samples': 0,
    }
    for prefix in ('parallax_pos', 'parallax_neg'):
        defaults.update({
            f'{prefix}_t0_jd_minus_2450000': np.nan,
            f'{prefix}_tE_days': np.nan,
            f'{prefix}_u0': np.nan,
            f'{prefix}_piE_N': np.nan,
            f'{prefix}_piE_E': np.nan,
            f'{prefix}_piE': np.nan,
            f'{prefix}_chi2': np.nan,
            f'{prefix}_reduced_chi2': np.nan,
            f'{prefix}_bic': np.nan,
            f'{prefix}_acceptance_rate': np.nan,
            f'{prefix}_n_samples': 0,
        })

    out = defaults.copy()
    out['parallax_attempted'] = bool(parallax_result.get('attempted', False))
    out['parallax_fit_ok'] = bool(parallax_result.get('fit_ok', False))
    out['parallax_preferred'] = bool(parallax_result.get('preferred', False))
    out['parallax_status'] = str(parallax_result.get('status', '') or '')
    out['parallax_warning'] = ','.join(parallax_result.get('warnings', []) or [])
    out['parallax_t0_ref_jd'] = _finite_float(parallax_result.get('t0_ref_jd')) if _finite_float(parallax_result.get('t0_ref_jd')) is not None else np.nan
    out['parallax_best_branch'] = str(parallax_result.get('best_branch', '') or '')
    out['parallax_delta_bic'] = _finite_float(parallax_result.get('delta_bic')) if _finite_float(parallax_result.get('delta_bic')) is not None else np.nan
    out['parallax_branch_delta_bic'] = (
        _finite_float(parallax_result.get('branch_delta_bic'))
        if _finite_float(parallax_result.get('branch_delta_bic')) is not None
        else np.nan
    )

    pspl = parallax_result.get('pspl', {}) or {}
    if pspl.get('success'):
        out['parallax_pspl_flux_chi2'] = float(pspl.get('chi2', np.nan))
        out['parallax_pspl_flux_bic'] = float(pspl.get('bic', np.nan))
        out['parallax_pspl_flux_reduced_chi2'] = float(pspl.get('reduced_chi2', np.nan))

    branch_map = {'u0_pos': 'parallax_pos', 'u0_neg': 'parallax_neg'}
    branches = parallax_result.get('branches', {}) or {}
    for branch_name, prefix in branch_map.items():
        branch = branches.get(branch_name, {}) or {}
        if not branch.get('success'):
            continue
        out[f'{prefix}_t0_jd_minus_2450000'] = float(branch.get('t0_jd', np.nan) - 2450000.0)
        out[f'{prefix}_tE_days'] = float(branch.get('tE_days', np.nan))
        out[f'{prefix}_u0'] = float(branch.get('u0', np.nan))
        out[f'{prefix}_piE_N'] = float(branch.get('piE_N', np.nan))
        out[f'{prefix}_piE_E'] = float(branch.get('piE_E', np.nan))
        out[f'{prefix}_piE'] = float(branch.get('piE', np.nan))
        out[f'{prefix}_chi2'] = float(branch.get('chi2', np.nan))
        out[f'{prefix}_reduced_chi2'] = float(branch.get('reduced_chi2', np.nan))
        out[f'{prefix}_bic'] = float(branch.get('bic', np.nan))
        out[f'{prefix}_acceptance_rate'] = float(branch.get('mcmc', {}).get('acceptance_rate', np.nan)) if branch.get('mcmc', {}).get('success') else np.nan
        out[f'{prefix}_n_samples'] = int(branch.get('mcmc', {}).get('n_samples', 0)) if branch.get('mcmc', {}).get('success') else 0

    best_name = out['parallax_best_branch']
    best = branches.get(best_name, {}) if best_name else {}
    if best and best.get('success'):
        out['parallax_best_t0_jd_minus_2450000'] = float(best.get('t0_jd', np.nan) - 2450000.0)
        out['parallax_best_tE_days'] = float(best.get('tE_days', np.nan))
        out['parallax_best_u0'] = float(best.get('u0', np.nan))
        out['parallax_best_piE_N'] = float(best.get('piE_N', np.nan))
        out['parallax_best_piE_E'] = float(best.get('piE_E', np.nan))
        out['parallax_best_piE'] = float(best.get('piE', np.nan))
        out['parallax_best_fs'] = float(best.get('Fs', np.nan))
        out['parallax_best_fb'] = float(best.get('Fb', np.nan))
        out['parallax_best_chi2'] = float(best.get('chi2', np.nan))
        out['parallax_best_reduced_chi2'] = float(best.get('reduced_chi2', np.nan))
        out['parallax_best_bic'] = float(best.get('bic', np.nan))
        if best.get('mcmc', {}).get('success'):
            out['parallax_best_acceptance_rate'] = float(best['mcmc'].get('acceptance_rate', np.nan))
            out['parallax_best_n_samples'] = int(best['mcmc'].get('n_samples', 0))
    return out

def _prepare_lightcurve_df(lc_path: Path, *, prefer_g_band: bool = True) -> tuple[pd.DataFrame, str]:
    df = load_lightcurve_df(lc_path)
    df = clean_lc(df)
    band_label = 'all'
    if prefer_g_band and 'v_g_band' in df.columns and (df['v_g_band'] == 0).any():
        df = df.loc[df['v_g_band'] == 0].copy()
        band_label = 'g'
    elif 'v_g_band' in df.columns and df['v_g_band'].nunique(dropna=True) == 1:
        band_label = 'g' if int(df['v_g_band'].iloc[0]) == 0 else 'V'
    return df.sort_values('JD').reset_index(drop=True), band_label


def _load_candidate_context(
    conn: sqlite3.Connection,
    candidate_id: str,
    *,
    plot_dir: Path | None,
    prefer_g_band: bool = True,
) -> dict[str, object]:
    row = pd.read_sql_query(
        """
        SELECT
            candidate_id,
            asas_sn_id,
            lc_path,
            source_path,
            jump_best_t0,
            jump_best_width_param,
            dip_best_t0,
            dip_best_width_param,
            baseline_mag,
            vetting_likely_known,
            catalog_source,
            vsx_class,
            asassn_var_type,
            gaia_var_class,
            ztf_var_type,
            simbad_otype,
            simbad_main_id,
            microlens_match,
            microlens_catalog,
            microlens_name,
            microlens_alt_name,
            microlens_te_days,
            microlens_sep_arcsec
        FROM candidates
        WHERE candidate_id = ?
        """,
        conn,
        params=[str(candidate_id)],
    )
    if row.empty:
        raise KeyError(f'Candidate not found in review DB: {candidate_id}')

    record = row.iloc[0].to_dict()
    payload = get_candidate_payload(conn, str(candidate_id))
    lc_path = resolve_lightcurve_path(payload, plot_dir)
    if lc_path is None:
        raise FileNotFoundError(f'Could not resolve light-curve path for {candidate_id}')
    df, band_label = _prepare_lightcurve_df(lc_path, prefer_g_band=prefer_g_band)
    if df.empty:
        raise ValueError(f'Resolved light curve is empty after cleaning: {candidate_id}')
    return {
        'candidate_id': str(record['candidate_id']),
        'asas_sn_id': str(record.get('asas_sn_id') or candidate_id),
        'row': record,
        'payload': payload,
        'lc_path': lc_path,
        'df': df,
        'band_label': band_label,
    }


def _bool_flag(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    try:
        return bool(int(value))
    except (TypeError, ValueError):
        pass
    if isinstance(value, float) and np.isnan(value):
        return False
    return bool(value)


def _text_value(value: object) -> str:
    if value is None:
        return ''
    if isinstance(value, float) and np.isnan(value):
        return ''
    text = str(value).strip()
    return '' if text.lower() == 'nan' else text


def _format_known_microlens_status(result: dict[str, object]) -> str:
    summary = result['summary']
    lines = [f"Known-event status: {summary['candidate_id']}"]
    lines.append(f"  microlens_catalog_match: {'yes' if summary['microlens_match'] else 'no'}")
    if summary['microlens_match']:
        lines.append(f"  microlens_catalog: {summary['microlens_catalog'] or 'unknown'}")
        lines.append(f"  microlens_name: {summary['microlens_name'] or 'unknown'}")
        if summary['microlens_alt_name']:
            lines.append(f"  microlens_alt_name: {summary['microlens_alt_name']}")
        microlens_te_days = summary['microlens_te_days']
        microlens_sep_arcsec = summary['microlens_sep_arcsec']
        if microlens_te_days is not None and np.isfinite(microlens_te_days):
            lines.append(f"  published_microlens_tE_days: {microlens_te_days:.3f}")
        if microlens_sep_arcsec is not None and np.isfinite(microlens_sep_arcsec):
            lines.append(f"  microlens_sep_arcsec: {microlens_sep_arcsec:.3f}")
    else:
        lines.append('  microlens_catalog: none in review DB')

    lines.append(f"  vetting_likely_known: {'yes' if summary['vetting_likely_known'] else 'no'}")
    if summary['catalog_source']:
        lines.append(f"  catalog_source: {summary['catalog_source']}")

    class_bits = []
    if summary['vsx_class']:
        class_bits.append(f"VSX={summary['vsx_class']}")
    if summary['asassn_var_type']:
        class_bits.append(f"ASAS-SN={summary['asassn_var_type']}")
    if summary['gaia_var_class']:
        class_bits.append(f"Gaia={summary['gaia_var_class']}")
    if summary['ztf_var_type']:
        class_bits.append(f"ZTF={summary['ztf_var_type']}")
    if class_bits:
        lines.append('  catalog_classes: ' + '; '.join(class_bits))

    simbad_name = summary.get('nearest_simbad_object') or summary.get('simbad_main_id') or ''
    simbad_otype = summary.get('simbad_otype') or ''
    if simbad_name or simbad_otype:
        simbad_line = simbad_name or 'unknown'
        if simbad_otype:
            simbad_line += f" [{simbad_otype}]"
        lines.append(f"  simbad: {simbad_line}")
    return '\n'.join(lines)




def _format_parallax_status(result: dict[str, object]) -> str:
    parallax = result.get('parallax', {}) or {}
    summary = result.get('summary', {}) or {}
    status = str(summary.get('parallax_status', parallax.get('status', '')) or '')
    if not summary.get('parallax_attempted', False):
        return f"Parallax status: {status or 'not attempted'}"

    best_branch = summary.get('parallax_best_branch', '') or parallax.get('best_branch', '') or 'none'
    delta_bic = summary.get('parallax_delta_bic')
    delta_bic_text = f"{float(delta_bic):.2f}" if delta_bic is not None and np.isfinite(delta_bic) else 'nan'
    lines = [
        f"Parallax status: {status or 'attempted'} | preferred={bool(summary.get('parallax_preferred', False))} | best_branch={best_branch} | ΔBIC(pspl−best)={delta_bic_text}"
    ]
    if summary.get('parallax_warning'):
        lines.append(f"  warnings: {summary['parallax_warning']}")
    branch_map = (
        ('u0>0', 'parallax_pos', result.get('parallax', {}).get('branches', {}).get('u0_pos', {})),
        ('u0<0', 'parallax_neg', result.get('parallax', {}).get('branches', {}).get('u0_neg', {})),
    )
    for label, prefix, branch in branch_map:
        bic = summary.get(f'{prefix}_bic')
        if bic is None or not np.isfinite(bic):
            lines.append(f"  {label}: fit failed")
            continue
        lines.append(
            f"  {label}: BIC={float(bic):.2f} | t0={float(summary.get(f'{prefix}_t0_jd_minus_2450000')):.2f} | "
            f"tE={float(summary.get(f'{prefix}_tE_days')):.2f} d | u0={float(summary.get(f'{prefix}_u0')):.4f} | "
            f"piE_N={float(summary.get(f'{prefix}_piE_N')):.3f} | piE_E={float(summary.get(f'{prefix}_piE_E')):.3f} | "
            f"|piE|={float(summary.get(f'{prefix}_piE')):.3f}"
        )
        posterior = branch.get('mcmc', {}).get('posterior_summary', {}) if branch else {}
        if posterior:
            lines.append(
                f"    posterior piE={posterior.get('piE_p50', np.nan):.3f} "
                f"[{posterior.get('piE_p16', np.nan):.3f}, {posterior.get('piE_p84', np.nan):.3f}] | "
                f"n={int(branch.get('mcmc', {}).get('n_samples', 0))}"
            )
        branch_warnings = branch.get('quality', {}).get('warnings', []) if branch else []
        if branch_warnings:
            lines.append(f"    warnings: {','.join(branch_warnings)}")
    return '\n'.join(lines)

def _pick_width_seed(row: dict[str, object]) -> float:
    candidates = []
    for key in ('jump_best_width_param', 'dip_best_width_param'):
        value = _finite_float(row.get(key))
        if value is not None and abs(value) > 0:
            candidates.append(abs(value))
    if candidates:
        return float(np.clip(np.nanmedian(candidates), 5.0, 300.0))
    return 40.0


def _candidate_seeds(df: pd.DataFrame, row: dict[str, object]) -> list[dict[str, object]]:
    if df.empty:
        raise ValueError('Cannot build a brightest10_median seed from an empty light curve')

    brightest = df.nsmallest(min(10, len(df)), 'mag')
    return [
        {
            'seed_method': 'brightest10_median',
            't0_guess': float(brightest['JD'].median()),
        }
    ]


def _estimate_return_half_window(jd: np.ndarray, mag: np.ndarray, err: np.ndarray, t0_guess: float, width_seed: float) -> float:
    base_half_window = float(np.clip(max(240.0, 8.0 * width_seed), 240.0, 1800.0))
    center_idx = int(np.nanargmin(np.abs(jd - t0_guess)))
    local_mask = np.abs(jd - t0_guess) <= max(120.0, 2.0 * width_seed)
    if int(local_mask.sum()) < 10:
        local_mask = np.ones_like(jd, dtype=bool)

    global_baseline = float(np.nanmedian(mag))
    local_min = float(np.nanmin(mag[local_mask]))
    depth_guess = max(global_baseline - local_min, 0.05)
    tol = max(0.15 * depth_guess, 3.0 * float(np.nanmedian(err)), 0.02)

    def find_return(direction: int) -> float | None:
        consec = 0
        idx_iter = range(center_idx, len(jd)) if direction > 0 else range(center_idx, -1, -1)
        for idx in idx_iter:
            near_baseline = abs(mag[idx] - global_baseline) <= tol
            consec = consec + 1 if near_baseline else 0
            if consec >= 3:
                anchor = idx - 2 if direction > 0 else idx + 2
                anchor = max(0, min(len(jd) - 1, anchor))
                return float(jd[anchor])
        return None

    left_return = find_return(-1)
    right_return = find_return(+1)

    distances = []
    if left_return is not None and left_return < t0_guess:
        distances.append(t0_guess - left_return)
    if right_return is not None and right_return > t0_guess:
        distances.append(right_return - t0_guess)

    if len(distances) == 2:
        half_window = max(distances) + max(60.0, 2.0 * width_seed)
    elif len(distances) == 1:
        half_window = max(base_half_window, distances[0] + max(120.0, 4.0 * width_seed))
    else:
        half_window = base_half_window
    return float(np.clip(half_window, 240.0, 2200.0))


def _outer_baseline_guess(jd_fit: np.ndarray, mag_fit: np.ndarray, center: float, half_window: float) -> tuple[float, float, float]:
    t_ref = float(np.nanmedian(jd_fit))
    outer_mask = np.abs(jd_fit - center) >= max(0.30 * half_window, 40.0)
    if int(outer_mask.sum()) >= 4:
        x = jd_fit[outer_mask] - t_ref
        y = mag_fit[outer_mask]
        slope, baseline = np.polyfit(x, y, deg=1)
        return float(baseline), float(slope), t_ref
    return float(np.nanmedian(mag_fit)), 0.0, t_ref


def _evaluate_model(model_name: str, params: np.ndarray, t: np.ndarray, t_ref: float) -> np.ndarray:
    if model_name == 'flat':
        baseline, slope = params
        return flat_trend(t, baseline, slope, t_ref)
    if model_name == 'gaussian':
        depth, t0, sigma, baseline, slope = params
        return gaussian_brightening_trend(t, depth, t0, sigma, baseline, slope, t_ref)
    if model_name == 'fred':
        depth, t0, tau_rise, tau_decay, baseline, slope = params
        return fred_brightening_trend(t, depth, t0, tau_rise, tau_decay, baseline, slope, t_ref)
    if model_name == 'paczynski':
        A0, t0, tE, baseline, slope = params
        return paczynski_mag_trend(t, A0, t0, tE, baseline, slope, t_ref)
    raise ValueError(f'Unknown model: {model_name}')


def _fit_model(
    model_name: str,
    jd_fit: np.ndarray,
    mag_fit: np.ndarray,
    err_fit: np.ndarray,
    *,
    center_guess: float,
    width_seed: float,
) -> dict[str, object]:
    half_window = 0.5 * float(jd_fit.max() - jd_fit.min())
    baseline_guess, slope_guess, t_ref = _outer_baseline_guess(jd_fit, mag_fit, center_guess, half_window)
    depth_guess = max(baseline_guess - float(np.nanmin(mag_fit)), 0.03)
    depth_limit = max(0.3, 3.0 * depth_guess)
    window_span = max(float(jd_fit.max() - jd_fit.min()), 1.0)
    slope_limit = max(0.005, 4.0 * depth_limit / window_span)

    if model_name == 'paczynski':
        best_result = None
        last_error = None

        lower_opt = np.array([np.log(1e-3), jd_fit.min(), np.log(0.5), baseline_guess - 1.5, -slope_limit], dtype=float)
        upper_opt = np.array([np.log(4.0), jd_fit.max(), np.log(max(window_span, 1.0)), baseline_guess + 1.5, slope_limit], dtype=float)

        brightest_idx = np.argsort(mag_fit)[:min(10, len(mag_fit))]
        t0_starts = [center_guess]
        if len(brightest_idx):
            t0_starts.append(float(np.median(jd_fit[brightest_idx])))
        t0_starts = list(dict.fromkeys(float(np.clip(val, jd_fit.min(), jd_fit.max())) for val in t0_starts))

        tE_starts = [
            max(5.0, 0.08 * window_span),
            max(5.0, 0.18 * window_span),
            max(5.0, 0.35 * window_span),
        ]
        tE_starts = list(dict.fromkeys(float(np.clip(val, 0.5, window_span)) for val in tE_starts))
        u0_starts = [0.02, 0.2, 1.0]

        def residuals_pacz(opt_params: np.ndarray) -> np.ndarray:
            log_u0, t0, log_tE, baseline, slope = opt_params
            A0 = _A0_from_u0(np.exp(log_u0))
            tE = float(np.exp(log_tE))
            model = paczynski_mag_trend(jd_fit, A0, t0, tE, baseline, slope, t_ref)
            return (mag_fit - model) / err_fit

        for t0_start in t0_starts:
            for u0_start in u0_starts:
                for tE_start in tE_starts:
                    x0 = np.array([np.log(u0_start), t0_start, np.log(tE_start), baseline_guess, slope_guess], dtype=float)
                    try:
                        result = least_squares(
                            residuals_pacz,
                            x0=np.clip(x0, lower_opt + 1e-8, upper_opt - 1e-8),
                            bounds=(lower_opt, upper_opt),
                            loss='soft_l1',
                            f_scale=1.5,
                            max_nfev=2500,
                        )
                    except Exception as exc:
                        last_error = repr(exc)
                        continue

                    if not result.success or not np.all(np.isfinite(result.x)):
                        last_error = result.message
                        continue

                    trial_resid = residuals_pacz(result.x)
                    trial_chi2 = float(np.nansum(trial_resid ** 2))
                    if best_result is None or trial_chi2 < best_result['chi2']:
                        best_result = {'result': result, 'chi2': trial_chi2}

        if best_result is None:
            return {'model_name': model_name, 'success': False, 'error': last_error or 'Paczynski multistart failed'}

        result = best_result['result']
        log_u0, t0, log_tE, baseline, slope = result.x.astype(float)
        params = np.array([_A0_from_u0(np.exp(log_u0)), t0, np.exp(log_tE), baseline, slope], dtype=float)
        lower = np.array([_A0_from_u0(np.exp(upper_opt[0])), lower_opt[1], np.exp(lower_opt[2]), lower_opt[3], lower_opt[4]], dtype=float)
        upper = np.array([_A0_from_u0(np.exp(lower_opt[0])), upper_opt[1], np.exp(upper_opt[2]), upper_opt[3], upper_opt[4]], dtype=float)
    else:
        if model_name == 'flat':
            p0 = np.array([baseline_guess, slope_guess], dtype=float)
            lower = np.array([baseline_guess - 1.5, -slope_limit], dtype=float)
            upper = np.array([baseline_guess + 1.5, slope_limit], dtype=float)
        elif model_name == 'gaussian':
            p0 = np.array([depth_guess, center_guess, max(width_seed, 5.0), baseline_guess, slope_guess], dtype=float)
            lower = np.array([0.01, jd_fit.min(), 0.2, baseline_guess - 1.5, -slope_limit], dtype=float)
            upper = np.array([depth_limit, jd_fit.max(), window_span, baseline_guess + 1.5, slope_limit], dtype=float)
        elif model_name == 'fred':
            tau0 = max(width_seed, 5.0)
            p0 = np.array([depth_guess, center_guess, tau0, tau0, baseline_guess, slope_guess], dtype=float)
            lower = np.array([0.01, jd_fit.min(), 0.2, 0.2, baseline_guess - 1.5, -slope_limit], dtype=float)
            upper = np.array([depth_limit, jd_fit.max(), window_span, window_span, baseline_guess + 1.5, slope_limit], dtype=float)
        else:
            raise ValueError(model_name)

        def residuals(params: np.ndarray) -> np.ndarray:
            model = _evaluate_model(model_name, params, jd_fit, t_ref)
            return (mag_fit - model) / err_fit

        try:
            result = least_squares(
                residuals,
                x0=np.clip(p0, lower + 1e-8, upper - 1e-8),
                bounds=(lower, upper),
                loss='soft_l1',
                f_scale=1.5,
                max_nfev=8000,
            )
        except Exception as exc:
            return {'model_name': model_name, 'success': False, 'error': repr(exc)}

        if not result.success or not np.all(np.isfinite(result.x)):
            return {
                'model_name': model_name,
                'success': False,
                'error': result.message,
            }

        params = result.x.astype(float)

    model = _evaluate_model(model_name, params, jd_fit, t_ref)
    resid = (mag_fit - model) / err_fit
    chi2 = float(np.nansum(resid ** 2))
    n = int(len(jd_fit))
    k = MODEL_PARAM_COUNTS[model_name]
    dof = max(n - k, 1)
    reduced_chi2 = chi2 / dof
    bic = chi2 + k * np.log(max(n, 2))

    return {
        'model_name': model_name,
        'success': True,
        'params': params,
        'bounds': (lower.astype(float), upper.astype(float)),
        'model': model,
        'residuals': resid,
        'chi2': chi2,
        'reduced_chi2': reduced_chi2,
        'bic': bic,
        'n_points': n,
        't_ref': t_ref,
        'least_squares_cost': float(result.cost),
        'message': result.message,
    }


# =============================================================================
# FLUX-SPACE MODEL FITTING
# =============================================================================

def _pspl_magnification(t: np.ndarray, u0: float, t0: float, tE: float) -> np.ndarray:
    """Point-source point-lens magnification A(t) = (u^2 + 2) / (u * sqrt(u^2 + 4))."""
    u_t = np.sqrt(u0**2 + ((t - t0) / tE)**2)
    return (u_t**2 + 2.0) / (u_t * np.sqrt(u_t**2 + 4.0))


def _solve_linear_flux_params(
    magnification: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
) -> tuple[float, float, np.ndarray]:
    """Solve for source flux Fs and blend flux Fb given magnification profile.
    
    Model: F(t) = Fs * A(t) + Fb
    Uses bounded linear least squares to ensure Fs >= 0, Fb >= 0.
    """
    n = len(flux)
    if n < 2:
        return 1.0, 0.0, magnification.copy()
    
    # Weight by inverse variance
    w = 1.0 / np.maximum(flux_err**2, 1e-20)
    
    # Design matrix: [A(t), 1]
    A_matrix = np.column_stack([magnification, np.ones(n)])
    
    # Weighted least squares: minimize ||W^{1/2} (A @ x - flux)||^2
    W_sqrt = np.sqrt(w)
    A_weighted = A_matrix * W_sqrt[:, np.newaxis]
    b_weighted = flux * W_sqrt
    
    try:
        result = lsq_linear(
            A_weighted, b_weighted,
            bounds=([0.0, 0.0], [np.inf, np.inf]),
            method='bvls',
        )
        Fs, Fb = result.x
    except Exception:
        # Fallback: simple least squares
        try:
            x, _, _, _ = np.linalg.lstsq(A_weighted, b_weighted, rcond=None)
            Fs, Fb = max(0.0, x[0]), max(0.0, x[1])
        except Exception:
            Fs, Fb = 1.0, 0.0
    
    model_flux = Fs * magnification + Fb
    return float(Fs), float(Fb), model_flux


def _fit_model_flux_space(
    jd: np.ndarray,
    mag: np.ndarray,
    err: np.ndarray,
    *,
    center_guess: float,
    width_seed: float,
    ref_mag: float | None = None,
) -> dict[str, object]:
    """Fit PSPL model in flux space with explicit Fs/Fb blending.
    
    Converts mag -> flux, fits u0/t0/tE via nonlinear optimization,
    solves Fs/Fb linearly at each iteration.
    
    Returns dict with success, u0, t0, tE, Fs, Fb, blend_fraction, model_flux, etc.
    """
    from malca.config import (
        PACZYNSKI_U0_MIN, PACZYNSKI_U0_MAX,
        PACZYNSKI_TE_MIN_DAYS, PACZYNSKI_TE_MAX_FACTOR,
        FIT_MULTISTART_U0, FIT_MULTISTART_TE_FRACTIONS,
        FLUX_MIN_RELATIVE,
    )
    
    n = len(jd)
    if n < 10:
        return {'success': False, 'error': 'insufficient_points', 'model_name': 'paczynski_flux'}
    
    # Convert mag -> relative flux
    if ref_mag is None:
        ref_mag = float(np.median(mag))
    
    flux = 10.0 ** (-0.4 * (mag - ref_mag))
    flux = np.maximum(flux, FLUX_MIN_RELATIVE)
    flux_err = (np.log(10.0) / 2.5) * flux * np.maximum(err, 0.001)
    
    # Bounds
    window_span = max(float(jd.max() - jd.min()), 1.0)
    tE_max = PACZYNSKI_TE_MAX_FACTOR * window_span
    
    lower_opt = np.array([np.log(PACZYNSKI_U0_MIN), float(jd.min()), np.log(PACZYNSKI_TE_MIN_DAYS)], dtype=float)
    upper_opt = np.array([np.log(PACZYNSKI_U0_MAX), float(jd.max()), np.log(tE_max)], dtype=float)
    
    # Multi-start optimization
    best_result = None
    last_error = None
    
    # Starting points
    brightest_idx = np.argsort(mag)[:min(10, n)]
    t0_starts = [center_guess]
    if len(brightest_idx):
        t0_starts.append(float(np.median(jd[brightest_idx])))
    t0_starts = list(dict.fromkeys(float(np.clip(val, jd.min(), jd.max())) for val in t0_starts))
    
    tE_starts = [max(5.0, frac * window_span) for frac in FIT_MULTISTART_TE_FRACTIONS]
    tE_starts = [float(np.clip(val, PACZYNSKI_TE_MIN_DAYS, tE_max)) for val in tE_starts]
    
    def residuals_flux(opt_params: np.ndarray) -> np.ndarray:
        log_u0, t0, log_tE = opt_params
        u0 = np.exp(log_u0)
        tE = np.exp(log_tE)
        A_t = _pspl_magnification(jd, u0, t0, tE)
        _, _, model_flux = _solve_linear_flux_params(A_t, flux, flux_err)
        return (flux - model_flux) / flux_err
    
    for t0_start in t0_starts:
        for u0_start in FIT_MULTISTART_U0:
            for tE_start in tE_starts:
                x0 = np.array([np.log(u0_start), t0_start, np.log(tE_start)], dtype=float)
                x0 = np.clip(x0, lower_opt + 1e-8, upper_opt - 1e-8)
                
                try:
                    result = least_squares(
                        residuals_flux,
                        x0=x0,
                        bounds=(lower_opt, upper_opt),
                        loss='soft_l1',
                        f_scale=1.5,
                        max_nfev=3000,
                    )
                except Exception as exc:
                    last_error = repr(exc)
                    continue
                
                if not result.success or not np.all(np.isfinite(result.x)):
                    last_error = result.message
                    continue
                
                trial_resid = residuals_flux(result.x)
                trial_chi2 = float(np.nansum(trial_resid**2))
                if best_result is None or trial_chi2 < best_result['chi2']:
                    best_result = {'result': result, 'chi2': trial_chi2}
    
    if best_result is None:
        return {'success': False, 'error': last_error or 'flux_multistart_failed', 'model_name': 'paczynski_flux'}
    
    result = best_result['result']
    log_u0, t0, log_tE = result.x.astype(float)
    u0 = float(np.exp(log_u0))
    tE = float(np.exp(log_tE))
    
    # Final linear solve for Fs, Fb
    A_t = _pspl_magnification(jd, u0, t0, tE)
    Fs, Fb, model_flux = _solve_linear_flux_params(A_t, flux, flux_err)
    
    # Convert model flux back to mag
    model_flux_safe = np.maximum(model_flux, FLUX_MIN_RELATIVE)
    model_mag = ref_mag - 2.5 * np.log10(model_flux_safe)
    
    # Compute chi2, BIC
    resid = (flux - model_flux) / flux_err
    chi2 = float(np.nansum(resid**2))
    k = 5  # u0, t0, tE, Fs, Fb
    dof = max(n - k, 1)
    reduced_chi2 = chi2 / dof
    bic = chi2 + k * np.log(max(n, 2))
    
    # Blend fraction
    total_flux = Fs + Fb
    blend_fraction = Fb / total_flux if total_flux > 0 else 0.0
    
    return {
        'model_name': 'paczynski_flux',
        'success': True,
        'u0': u0,
        't0': t0,
        'tE': tE,
        'Fs': Fs,
        'Fb': Fb,
        'ref_mag': ref_mag,
        'blend_fraction': blend_fraction,
        'model_flux': model_flux,
        'model_mag': model_mag,
        'flux': flux,
        'flux_err': flux_err,
        'residuals': resid,
        'chi2': chi2,
        'reduced_chi2': reduced_chi2,
        'bic': bic,
        'n_points': n,
        'params': np.array([u0, t0, tE, Fs, Fb], dtype=float),
    }


def _fit_model_suite(df: pd.DataFrame, *, center_guess: float, width_seed: float) -> dict[str, object]:
    jd = df['JD'].to_numpy(dtype=float)
    mag = df['mag'].to_numpy(dtype=float)
    err = np.clip(df['error'].to_numpy(dtype=float), 0.01, None)

    half_window = _estimate_return_half_window(jd, mag, err, center_guess, width_seed)
    fit_mask = np.abs(jd - center_guess) <= half_window
    if int(fit_mask.sum()) < 30:
        half_window = max(half_window, 360.0)
        fit_mask = np.abs(jd - center_guess) <= half_window
    if int(fit_mask.sum()) < 20:
        fit_mask = np.ones_like(jd, dtype=bool)
        half_window = 0.5 * float(jd.max() - jd.min())

    def run_all(current_mask: np.ndarray, current_center: float, current_width_seed: float) -> dict[str, object]:
        jd_fit = jd[current_mask]
        mag_fit = mag[current_mask]
        err_fit = err[current_mask]
        fits = {}
        for model_name in ('flat', 'gaussian', 'fred', 'paczynski'):
            fits[model_name] = _fit_model(
                model_name,
                jd_fit,
                mag_fit,
                err_fit,
                center_guess=current_center,
                width_seed=current_width_seed,
            )
        return {
            'fit_mask': current_mask.copy(),
            'half_window': float(half_window),
            'jd_fit': jd_fit,
            'mag_fit': mag_fit,
            'err_fit': err_fit,
            'fits': fits,
        }

    suite = run_all(fit_mask, center_guess, width_seed)
    pac = suite['fits']['paczynski']
    if pac.get('success'):
        pac_params = pac['params']
        pac_t0 = float(pac_params[1])
        pac_tE = float(abs(pac_params[2]))
        refined_half = max(half_window, _estimate_return_half_window(jd, mag, err, pac_t0, max(width_seed, 8.0 * pac_tE)), 10.0 * pac_tE)
        refined_half = float(np.clip(refined_half, 240.0, 2200.0))
        refined_mask = np.abs(jd - pac_t0) <= refined_half
        if int(refined_mask.sum()) >= int(np.sum(fit_mask)) + 10:
            half_window = refined_half
            suite = run_all(refined_mask, pac_t0, max(width_seed, pac_tE))

    fits = suite['fits']
    comparison_model_names = ('flat', 'gaussian', 'fred', 'paczynski')
    curve_alt_model_names = ('gaussian', 'fred')
    successful = {name: fit for name, fit in fits.items() if name in comparison_model_names and fit.get('success')}
    best_model = min(successful, key=lambda name: successful[name]['bic']) if successful else None

    pac = successful.get('paczynski')
    gaussian_fit = fits.get('gaussian', {})
    curve_alts = {name: fit for name, fit in successful.items() if name in curve_alt_model_names}
    best_alt_model = min(curve_alts, key=lambda name: curve_alts[name]['bic']) if curve_alts else None
    best_alt_bic = float(curve_alts[best_alt_model]['bic']) if best_alt_model is not None else np.nan
    delta_bic_vs_gaussian = np.nan
    delta_bic_vs_fred = np.nan
    if pac is not None and gaussian_fit.get('success'):
        delta_bic_vs_gaussian = float(gaussian_fit['bic'] - pac['bic'])
    if pac is not None and 'fred' in successful:
        delta_bic_vs_fred = float(successful['fred']['bic'] - pac['bic'])
    delta_bic_vs_flat = np.nan
    delta_bic_vs_best_alt = np.nan
    if pac is not None and 'flat' in successful:
        delta_bic_vs_flat = float(successful['flat']['bic'] - pac['bic'])
    if pac is not None and np.isfinite(best_alt_bic):
        delta_bic_vs_best_alt = float(best_alt_bic - pac['bic'])

    suite['best_model'] = best_model
    suite['best_alt_model'] = best_alt_model
    suite['delta_bic_vs_flat'] = delta_bic_vs_flat
    suite['delta_bic_vs_gaussian'] = delta_bic_vs_gaussian
    suite['delta_bic_vs_fred'] = delta_bic_vs_fred
    suite['delta_bic_vs_best_alt'] = delta_bic_vs_best_alt
    suite['best_alt_bic'] = best_alt_bic
    return suite


def _pacz_quality_metrics(seed_result: dict[str, object]) -> dict[str, object]:
    pac = seed_result['fits'].get('paczynski', {})
    if not pac.get('success'):
        return {
            'fit_ok': False,
            'warnings': ['paczynski_fit_failed'],
            'shoulder_left': 0,
            'shoulder_right': 0,
            'n_strong_points': 0,
            'paczynski_tau_coverage_score': np.nan,
            'paczynski_coverage_n_bins_hit': 0,
            'paczynski_coverage_n_bins': int(PAC_COVERAGE_N_BINS),
            'paczynski_coverage_max_weighted_gap': np.nan,
            'paczynski_coverage_frac_points_in_tau_window': np.nan,
        }

    jd_fit = seed_result['jd_fit']
    mag_fit = seed_result['mag_fit']
    pac_model = pac['model']
    A0, t0, tE, baseline, slope = pac['params']
    lower, upper = pac['bounds']
    t_ref = pac['t_ref']
    baseline_model = flat_trend(jd_fit, baseline, slope, t_ref)
    event_depth_model = baseline_model - pac_model
    max_depth_model = float(np.nanmax(event_depth_model)) if len(event_depth_model) else 0.0
    data_depth = baseline_model - mag_fit

    shoulder_mask = event_depth_model >= max(0.15 * max_depth_model, 0.03)
    left_mask = (jd_fit < t0) & shoulder_mask
    right_mask = (jd_fit > t0) & shoulder_mask
    shoulder_left = int(np.sum(left_mask))
    shoulder_right = int(np.sum(right_mask))
    strong_threshold = max(0.5 * max_depth_model, 0.04)
    n_strong_points = int(np.sum(data_depth >= strong_threshold))

    warnings = []
    fit_ok = True
    delta_bic_vs_fred = _raw_delta_bic(seed_result.get('delta_bic_vs_fred'))

    if not np.isfinite(seed_result.get('delta_bic_vs_flat', np.nan)) or float(seed_result['delta_bic_vs_flat']) < PAC_WEAK_VS_FLAT_MIN_DELTA_BIC:
        warnings.append('weak_vs_flat')
        fit_ok = False
    if np.isfinite(delta_bic_vs_fred) and delta_bic_vs_fred <= -NON_PACZYNSKI_SELECTION_DELTA_BIC_THRESHOLD:
        warnings.append('significant_fred_preferred')
        fit_ok = False
    if float(pac['reduced_chi2']) > 5.0:
        warnings.append('high_reduced_chi2')
        fit_ok = False
    if _finite_float(tE) is None or tE <= 0.0:
        warnings.append('invalid_tE')
        fit_ok = False
    t0_lower_gap = float(t0) - float(lower[1])
    t0_upper_gap = float(upper[1]) - float(t0)
    t0_margin = max(10.0, 0.02 * max(float(upper[1] - lower[1]), 1.0))
    if t0_lower_gap <= t0_margin or t0_upper_gap <= t0_margin:
        warnings.append('t0_near_bound')
        fit_ok = False

    tE_lower_limit = max(float(lower[2]) * 5.0, 5.0)
    tE_upper_margin = max(20.0, 0.10 * float(upper[2]))
    if float(tE) <= tE_lower_limit or float(upper[2]) - float(tE) <= tE_upper_margin:
        warnings.append('tE_near_bound')
        fit_ok = False

    A0_lower_limit = max(float(lower[0]) + 0.02, 1.02)
    A0_upper_margin = max(2.0, 0.05 * float(upper[0]))
    if float(A0) <= A0_lower_limit or float(upper[0]) - float(A0) <= A0_upper_margin:
        warnings.append('A0_near_bound')
        fit_ok = False
    if shoulder_left < 3 or shoulder_right < 3:
        warnings.append('insufficient_shoulders')
        fit_ok = False
    if n_strong_points < 2:
        warnings.append('single_point_peak')
        fit_ok = False

    cov = _paczynski_weighted_coverage(jd_fit, pac)
    cov_score = cov.get('paczynski_tau_coverage_score')
    if np.isfinite(cov_score):
        if float(cov_score) < PAC_COVERAGE_FAIL_THRESHOLD:
            warnings.append('low_event_coverage')
            fit_ok = False
        elif float(cov_score) < PAC_COVERAGE_WARN_THRESHOLD:
            warnings.append('event_coverage_below_recommended')

    return {
        'fit_ok': fit_ok,
        'warnings': warnings,
        'shoulder_left': shoulder_left,
        'shoulder_right': shoulder_right,
        'n_strong_points': n_strong_points,
        'max_model_depth': max_depth_model,
        **cov,
    }


def _best_selected_fit(seed_result: dict[str, object]) -> dict[str, object]:
    best_model = seed_result.get('selected_model') or seed_result.get('best_model')
    if not best_model:
        return {}
    return seed_result['fits'].get(best_model, {})


# =============================================================================
# MORPHOLOGY METRICS
# =============================================================================

def _compute_morphology_metrics(
    jd: np.ndarray,
    mag: np.ndarray,
    err: np.ndarray,
    pac_fit: dict[str, object],
) -> dict[str, object]:
    """Compute morphology metrics: rise/decay time, skewness, autocorr, symmetry, excursions.
    
    Args:
        jd: Time array (JD - 2450000)
        mag: Magnitude array
        err: Error array
        pac_fit: Paczynski fit result dict
    
    Returns:
        Dict with rise_time_days, decay_time_days, rise_decay_ratio, event_skewness,
        residual_autocorr, symmetry_score, excursion_fraction, vonneumann_ratio.
    """
    from scipy.stats import skew as scipy_skew
    from malca.config import (
        MORPH_EVENT_WINDOW_TE_FACTOR, MORPH_OUTSIDE_WINDOW_TE_FACTOR,
        MORPH_EXCURSION_SIGMA_THRESHOLD, MORPH_SYMMETRY_MIN_POINTS,
    )
    
    results = {
        'rise_time_days': np.nan,
        'decay_time_days': np.nan,
        'rise_decay_ratio': np.nan,
        'event_skewness': np.nan,
        'residual_autocorr': np.nan,
        'symmetry_score': np.nan,
        'excursion_fraction': np.nan,
        'vonneumann_ratio': np.nan,
    }
    
    if not pac_fit.get('success'):
        return results
    
    # Extract parameters
    params = pac_fit.get('params')
    if params is None or len(params) < 3:
        return results
    
    t0 = float(params[1])
    tE = float(abs(params[2]))
    
    if not np.isfinite(t0) or not np.isfinite(tE) or tE <= 0:
        return results
    
    # Get model if available
    model_mag = pac_fit.get('model')
    if model_mag is None:
        return results
    
    # Rise and decay time from model
    # Find times when model reaches 10% and 90% of peak depth
    baseline = float(params[3]) if len(params) > 3 else np.median(mag)
    peak_mag = np.min(model_mag)
    depth = baseline - peak_mag
    
    if depth > 0.01:
        thresh_10 = baseline - 0.1 * depth
        thresh_90 = baseline - 0.9 * depth
        
        # Before peak
        before_peak = jd < t0
        after_peak = jd >= t0
        
        if np.sum(before_peak) > 2 and np.sum(after_peak) > 2:
            model_before = model_mag[before_peak]
            model_after = model_mag[after_peak]
            jd_before = jd[before_peak]
            jd_after = jd[after_peak]
            
            # Rise: time from 10% to 90% depth (before peak)
            try:
                idx_10_rise = np.where(model_before <= thresh_10)[0]
                idx_90_rise = np.where(model_before <= thresh_90)[0]
                if len(idx_10_rise) > 0 and len(idx_90_rise) > 0:
                    t_10_rise = jd_before[idx_10_rise[0]]
                    t_90_rise = jd_before[idx_90_rise[-1]]
                    results['rise_time_days'] = abs(t_90_rise - t_10_rise)
            except Exception:
                pass
            
            # Decay: time from 90% to 10% depth (after peak)
            try:
                idx_90_decay = np.where(model_after <= thresh_90)[0]
                idx_10_decay = np.where(model_after <= thresh_10)[0]
                if len(idx_90_decay) > 0 and len(idx_10_decay) > 0:
                    t_90_decay = jd_after[idx_90_decay[0]]
                    t_10_decay = jd_after[idx_10_decay[-1]]
                    results['decay_time_days'] = abs(t_10_decay - t_90_decay)
            except Exception:
                pass
    
    # Rise/decay ratio
    if np.isfinite(results['rise_time_days']) and np.isfinite(results['decay_time_days']):
        if results['decay_time_days'] > 0:
            results['rise_decay_ratio'] = results['rise_time_days'] / results['decay_time_days']
    
    # Event window mask
    event_mask = np.abs(jd - t0) <= MORPH_EVENT_WINDOW_TE_FACTOR * tE
    outside_mask = np.abs(jd - t0) > MORPH_OUTSIDE_WINDOW_TE_FACTOR * tE
    
    # Event skewness
    if np.sum(event_mask) >= 5:
        try:
            results['event_skewness'] = float(scipy_skew(mag[event_mask], nan_policy='omit'))
        except Exception:
            pass
    
    # Residual autocorrelation (lag-1)
    residuals = pac_fit.get('residuals')
    if residuals is not None and len(residuals) > 5:
        try:
            resid_sorted_idx = np.argsort(jd)
            resid_sorted = np.asarray(residuals)[resid_sorted_idx]
            valid = np.isfinite(resid_sorted)
            if np.sum(valid) > 5:
                r = resid_sorted[valid]
                r_mean = np.mean(r)
                r_centered = r - r_mean
                var = np.sum(r_centered**2)
                if var > 0:
                    autocorr = np.sum(r_centered[:-1] * r_centered[1:]) / var
                    results['residual_autocorr'] = float(autocorr)
        except Exception:
            pass
    
    # Symmetry score: compare left vs right of peak
    left_mask = (jd < t0) & event_mask
    right_mask = (jd >= t0) & event_mask
    
    if np.sum(left_mask) >= MORPH_SYMMETRY_MIN_POINTS and np.sum(right_mask) >= MORPH_SYMMETRY_MIN_POINTS:
        try:
            # Reflect right side around t0 and compare
            jd_left = jd[left_mask]
            mag_left = mag[left_mask]
            jd_right = jd[right_mask]
            mag_right = mag[right_mask]
            
            # Interpolate right side at mirrored left times
            jd_right_mirrored = 2 * t0 - jd_right
            
            # Compare areas
            left_area = np.trapz(baseline - mag_left, jd_left) if len(jd_left) > 1 else 0
            right_area = np.trapz(baseline - mag_right, jd_right) if len(jd_right) > 1 else 0
            
            total_area = abs(left_area) + abs(right_area)
            if total_area > 0:
                results['symmetry_score'] = abs(left_area - right_area) / total_area
        except Exception:
            pass
    
    # Excursion fraction outside event window
    if np.sum(outside_mask) >= 3:
        try:
            outside_resid = (mag[outside_mask] - baseline) / err[outside_mask]
            n_excursions = np.sum(np.abs(outside_resid) > MORPH_EXCURSION_SIGMA_THRESHOLD)
            results['excursion_fraction'] = float(n_excursions) / float(np.sum(outside_mask))
        except Exception:
            pass
    
    # Von Neumann ratio of residuals
    if residuals is not None and len(residuals) > 5:
        try:
            resid_sorted_idx = np.argsort(jd)
            r = np.asarray(residuals)[resid_sorted_idx]
            valid = np.isfinite(r)
            if np.sum(valid) > 5:
                r_valid = r[valid]
                delta_sq = np.sum(np.diff(r_valid)**2)
                var = np.sum((r_valid - np.mean(r_valid))**2)
                if var > 0:
                    results['vonneumann_ratio'] = float(delta_sq / var)
        except Exception:
            pass
    
    return results


# =============================================================================
# CV/NOVA SCORING
# =============================================================================

def _compute_cv_nova_score(
    jd: np.ndarray,
    mag: np.ndarray,
    err: np.ndarray,
    pac_fit: dict[str, object],
    fred_fit: dict[str, object],
) -> dict[str, object]:
    """Compute CV/nova contamination score.
    
    Indicators that suggest CV/nova rather than microlensing:
    - FRED model preferred over Paczynski (ΔBIC)
    - Asymmetric rise/decay (fast rise, slow decay)
    - Secondary peaks/outbursts
    - High amplitude with poor Paczynski fit
    
    Returns dict with cv_nova_score (0-1), fred_preferred, rise_decay_asymmetry, etc.
    """
    from malca.config import (
        SECONDARY_PEAK_MIN_SEPARATION_DAYS,
        SECONDARY_PEAK_MIN_AMPLITUDE_FRAC,
    )
    
    results = {
        'cv_nova_score': 0.0,
        'fred_preferred': False,
        'delta_bic_fred_vs_pac': np.nan,
        'rise_decay_asymmetry': np.nan,
        'secondary_peak_detected': False,
        'amplitude_mag': np.nan,
    }
    
    pac_success = pac_fit.get('success', False)
    fred_success = fred_fit.get('success', False)
    
    # Amplitude
    amplitude = float(np.nanmax(mag) - np.nanmin(mag)) if len(mag) > 0 else 0.0
    results['amplitude_mag'] = amplitude
    
    # ΔBIC FRED vs Paczynski
    if pac_success and fred_success:
        pac_bic = float(pac_fit.get('bic', np.inf))
        fred_bic = float(fred_fit.get('bic', np.inf))
        delta = pac_bic - fred_bic  # Positive = FRED preferred
        results['delta_bic_fred_vs_pac'] = delta
        results['fred_preferred'] = delta > 2.0
    
    # Rise/decay asymmetry from FRED params
    if fred_success:
        fred_params = fred_fit.get('params')
        if fred_params is not None and len(fred_params) >= 4:
            tau_rise = float(fred_params[2])
            tau_decay = float(fred_params[3])
            if tau_rise > 0:
                results['rise_decay_asymmetry'] = tau_decay / tau_rise
    
    # Secondary peak detection via residuals
    if pac_success and pac_fit.get('residuals') is not None:
        residuals = np.asarray(pac_fit['residuals'])
        err_safe = np.maximum(err, 0.01)
        normalized_resid = residuals
        
        # Look for consecutive negative residuals (secondary brightening)
        significant_neg = normalized_resid < -3.0
        
        if np.sum(significant_neg) >= 3:
            # Check for clustering (consecutive negative residuals)
            sort_idx = np.argsort(jd)
            sig_sorted = significant_neg[sort_idx]
            
            # Count consecutive runs
            max_run = 0
            current_run = 0
            for val in sig_sorted:
                if val:
                    current_run += 1
                    max_run = max(max_run, current_run)
                else:
                    current_run = 0
            
            if max_run >= 3:
                results['secondary_peak_detected'] = True
    
    # Compute composite score (0-1)
    score = 0.0
    
    # FRED preference (0-30 points)
    if results['fred_preferred']:
        delta = results['delta_bic_fred_vs_pac']
        if np.isfinite(delta):
            score += min(30.0, 5.0 * np.log10(1.0 + max(delta, 0.0)))
    
    # Asymmetry (0-25 points) - CV/novae have fast rise, slow decay
    asym = results['rise_decay_asymmetry']
    if np.isfinite(asym) and asym > 1.0:
        score += min(25.0, 10.0 * np.log10(asym))
    
    # Secondary peak (0-20 points)
    if results['secondary_peak_detected']:
        score += 20.0
    
    # High amplitude (0-15 points)
    if amplitude > 0.5:
        score += min(15.0, 10.0 * np.log10(1.0 + amplitude))
    
    # Poor Paczynski fit (0-10 points)
    if pac_success:
        pac_chi2 = float(pac_fit.get('reduced_chi2', 1.0))
        if pac_chi2 > 3.0:
            score += min(10.0, 2.0 * np.log10(pac_chi2))
    
    results['cv_nova_score'] = min(1.0, score / 100.0)
    
    return results


# =============================================================================
# PERIODICITY SCANNING
# =============================================================================

def _scan_periodicity(
    jd: np.ndarray,
    mag: np.ndarray,
    err: np.ndarray,
    *,
    scan_residuals: bool = True,
    model_mag: np.ndarray | None = None,
) -> dict[str, object]:
    """Scan for periodicity using LSP and PDM.
    
    Runs period search on raw lightcurve and optionally on model residuals.
    Uses malca.periodogram functions.
    
    Returns dict with lsp_best_period, lsp_best_power, pdm_best_period, pdm_best_theta,
    resid_* versions, and periodicity_detected flag.
    """
    from malca.config import (
        PERIOD_MIN_DAYS, PERIOD_MAX_DAYS,
        RESIDUAL_PERIOD_POWER_THRESHOLD,
    )
    
    results = {
        'lsp_best_period': np.nan,
        'lsp_best_power': np.nan,
        'pdm_best_period': np.nan,
        'pdm_best_theta': np.nan,
        'resid_lsp_best_period': np.nan,
        'resid_lsp_best_power': np.nan,
        'resid_pdm_best_period': np.nan,
        'resid_pdm_best_theta': np.nan,
        'periodicity_detected': False,
    }
    
    # Import periodogram functions
    try:
        from malca.periodogram import lsp_find_period, pdm_find_period
    except ImportError:
        return results
    
    valid = np.isfinite(jd) & np.isfinite(mag) & np.isfinite(err)
    if np.sum(valid) < 20:
        return results
    
    jd_v = jd[valid]
    mag_v = mag[valid]
    err_v = err[valid]
    
    # LSP on raw LC
    try:
        lsp_result = lsp_find_period(
            jd_v, mag_v, err_v,
            min_period=PERIOD_MIN_DAYS,
            max_period=min(PERIOD_MAX_DAYS, 0.5 * (jd_v.max() - jd_v.min())),
        )
        if lsp_result is not None:
            results['lsp_best_period'] = float(lsp_result.get('best_period', np.nan))
            results['lsp_best_power'] = float(lsp_result.get('best_power', np.nan))
    except Exception:
        pass
    
    # PDM on raw LC
    try:
        pdm_result = pdm_find_period(
            jd_v, mag_v, err_v,
            min_period=PERIOD_MIN_DAYS,
            max_period=min(PERIOD_MAX_DAYS, 0.5 * (jd_v.max() - jd_v.min())),
        )
        if pdm_result is not None:
            results['pdm_best_period'] = float(pdm_result.get('best_period', np.nan))
            results['pdm_best_theta'] = float(pdm_result.get('best_theta', np.nan))
    except Exception:
        pass
    
    # Scan residuals if model provided
    if scan_residuals and model_mag is not None:
        model_v = model_mag[valid] if len(model_mag) == len(jd) else None
        if model_v is not None and len(model_v) == len(jd_v):
            resid = mag_v - model_v
            
            try:
                lsp_resid = lsp_find_period(
                    jd_v, resid, err_v,
                    min_period=PERIOD_MIN_DAYS,
                    max_period=min(PERIOD_MAX_DAYS, 0.5 * (jd_v.max() - jd_v.min())),
                )
                if lsp_resid is not None:
                    results['resid_lsp_best_period'] = float(lsp_resid.get('best_period', np.nan))
                    results['resid_lsp_best_power'] = float(lsp_resid.get('best_power', np.nan))
            except Exception:
                pass
            
            try:
                pdm_resid = pdm_find_period(
                    jd_v, resid, err_v,
                    min_period=PERIOD_MIN_DAYS,
                    max_period=min(PERIOD_MAX_DAYS, 0.5 * (jd_v.max() - jd_v.min())),
                )
                if pdm_resid is not None:
                    results['resid_pdm_best_period'] = float(pdm_resid.get('best_period', np.nan))
                    results['resid_pdm_best_theta'] = float(pdm_resid.get('best_theta', np.nan))
            except Exception:
                pass
    
    # Periodicity detection flag
    lsp_power = results['lsp_best_power']
    pdm_theta = results['pdm_best_theta']
    resid_lsp_power = results['resid_lsp_best_power']
    
    if (np.isfinite(lsp_power) and lsp_power > 0.3) or \
       (np.isfinite(pdm_theta) and pdm_theta < 0.5) or \
       (np.isfinite(resid_lsp_power) and resid_lsp_power > RESIDUAL_PERIOD_POWER_THRESHOLD):
        results['periodicity_detected'] = True
    
    return results


# =============================================================================
# PER-CANDIDATE QUALITY SCORE
# =============================================================================

def _compute_candidate_quality_score(
    summary: dict[str, object],
    morphology: dict[str, object],
    periodicity: dict[str, object],
    cv_nova: dict[str, object],
    quality_metrics: dict[str, object],
) -> dict[str, object]:
    """Compute composite quality score for a single candidate.
    
    Components (weights):
      - Fit quality (30%): reduced χ², shoulders, BIC vs flat
      - Morphology (20%): symmetry, rise/decay ratio, autocorr
      - Contamination (25%): CV score, periodicity, excursions
      - Coverage (15%): LC span, n_points, tau coverage
      - Parallax (10%): convergence status
    
    Returns dict with quality_score (0-1), quality_tier, quality_flags.
    """
    results = {
        'quality_score': 0.0,
        'quality_tier': 'Suspect',
        'quality_flags': [],
    }
    
    flags = []
    
    # -------------------------------------------------------------------------
    # FIT QUALITY (30%)
    # -------------------------------------------------------------------------
    fit_score = 0.0
    
    # Reduced chi-squared (0-40 pts)
    chi2 = float(summary.get('paczynski_reduced_chi2', np.nan))
    if np.isfinite(chi2):
        if chi2 <= 2.0:
            fit_score += 40.0
        elif chi2 >= 10.0:
            fit_score += 0.0
            flags.append('high_chi2')
        else:
            fit_score += 40.0 * (10.0 - chi2) / 8.0
    
    # Shoulders (0-30 pts)
    shoulder_left = int(quality_metrics.get('shoulder_left', 0))
    shoulder_right = int(quality_metrics.get('shoulder_right', 0))
    shoulder_points = min(30.0, 5.0 * (shoulder_left + shoulder_right))
    fit_score += shoulder_points
    if shoulder_left < 3:
        flags.append('weak_left_shoulder')
    if shoulder_right < 3:
        flags.append('weak_right_shoulder')
    
    # BIC vs flat (0-20 pts)
    delta_bic = float(summary.get('delta_bic_vs_flat', np.nan))
    if np.isfinite(delta_bic):
        if delta_bic >= 10.0:
            fit_score += 20.0
        elif delta_bic >= 2.0:
            fit_score += 10.0 + 10.0 * (delta_bic - 2.0) / 8.0
        elif delta_bic < 0:
            flags.append('flat_preferred')
    
    # Strong points (0-10 pts)
    n_strong = int(quality_metrics.get('n_strong_points', 0))
    fit_score += min(10.0, 5.0 * n_strong)
    if n_strong < 2:
        flags.append('few_strong_points')
    
    fit_component = min(1.0, fit_score / 100.0)
    
    # -------------------------------------------------------------------------
    # MORPHOLOGY (20%)
    # -------------------------------------------------------------------------
    morph_score = 0.0
    morph_count = 0
    
    # Rise/decay ratio (should be ~1 for microlensing)
    rd_ratio = float(morphology.get('rise_decay_ratio', np.nan))
    if np.isfinite(rd_ratio) and rd_ratio > 0:
        ratio_dev = max(rd_ratio, 1.0 / rd_ratio)
        if ratio_dev <= 3.0:
            morph_score += 25.0 * (3.0 - ratio_dev) / 2.0
        else:
            flags.append('asymmetric_event')
        morph_count += 1
    
    # Symmetry score (low = good)
    symmetry = float(morphology.get('symmetry_score', np.nan))
    if np.isfinite(symmetry):
        if symmetry <= 0.5:
            morph_score += 25.0 * (0.5 - symmetry) / 0.5
        else:
            flags.append('low_symmetry')
        morph_count += 1
    
    # Residual autocorrelation (low = good)
    autocorr = float(morphology.get('residual_autocorr', np.nan))
    if np.isfinite(autocorr):
        autocorr_abs = abs(autocorr)
        if autocorr_abs <= 0.5:
            morph_score += 25.0 * (0.5 - autocorr_abs) / 0.5
        else:
            flags.append('correlated_residuals')
        morph_count += 1
    
    # Excursion fraction (low = good)
    excursion = float(morphology.get('excursion_fraction', np.nan))
    if np.isfinite(excursion):
        if excursion <= 0.1:
            morph_score += 25.0
        elif excursion <= 0.3:
            morph_score += 25.0 * (0.3 - excursion) / 0.2
        else:
            flags.append('baseline_excursions')
        morph_count += 1
    
    morph_component = min(1.0, morph_score / 100.0) if morph_count > 0 else 0.5
    
    # -------------------------------------------------------------------------
    # CONTAMINATION (25%)
    # -------------------------------------------------------------------------
    contam_score = 100.0  # Start at max, subtract for contamination
    
    # CV/Nova score (high = bad)
    cv_score = float(cv_nova.get('cv_nova_score', 0.0))
    if np.isfinite(cv_score):
        contam_score -= cv_score * 40.0
        if cv_score > 0.3:
            flags.append('cv_nova_like')
    
    # Periodicity (detected = bad)
    if periodicity.get('periodicity_detected'):
        contam_score -= 30.0
        flags.append('periodic_signal')
    
    # Secondary peak (bad)
    if cv_nova.get('secondary_peak_detected'):
        contam_score -= 20.0
        flags.append('secondary_peak')
    
    # FRED strongly preferred
    if cv_nova.get('fred_preferred'):
        delta_fred = float(cv_nova.get('delta_bic_fred_vs_pac', 0.0))
        if np.isfinite(delta_fred) and delta_fred > 6.0:
            contam_score -= 10.0
            flags.append('fred_preferred')
    
    contam_component = max(0.0, min(1.0, contam_score / 100.0))
    
    # -------------------------------------------------------------------------
    # COVERAGE (15%)
    # -------------------------------------------------------------------------
    coverage_score = 0.0
    
    n_points = int(summary.get('n_points_fit', 0))
    if n_points >= 50:
        coverage_score += 50.0
    else:
        coverage_score += 50.0 * n_points / 50.0
        if n_points < 30:
            flags.append('few_points')
    
    # Tau coverage score
    tau_cov = float(quality_metrics.get('paczynski_tau_coverage_score', np.nan))
    if np.isfinite(tau_cov):
        coverage_score += 50.0 * min(1.0, tau_cov)
    else:
        coverage_score += 25.0
    
    coverage_component = min(1.0, coverage_score / 100.0)
    
    # -------------------------------------------------------------------------
    # PARALLAX (10%)
    # -------------------------------------------------------------------------
    parallax_score = 50.0  # Neutral default
    
    parallax_attempted = bool(summary.get('parallax_attempted', False))
    parallax_preferred = bool(summary.get('parallax_preferred', False))
    
    if parallax_attempted:
        if parallax_preferred:
            parallax_score = 90.0
        else:
            parallax_score = 60.0
    
    parallax_component = parallax_score / 100.0
    
    # -------------------------------------------------------------------------
    # COMBINE
    # -------------------------------------------------------------------------
    total = (
        0.30 * fit_component +
        0.20 * morph_component +
        0.25 * contam_component +
        0.15 * coverage_component +
        0.10 * parallax_component
    )
    
    # Determine tier
    if total >= 0.8:
        tier = 'Gold'
    elif total >= 0.6:
        tier = 'Silver'
    elif total >= 0.5:
        tier = 'Bronze'
    else:
        tier = 'Suspect'
    
    results['quality_score'] = float(total)
    results['quality_tier'] = tier
    results['quality_flags'] = flags
    
    return results


def _accepted_seed_rank(seed_result: dict[str, object]) -> tuple:
    pac = seed_result['fits'].get('paczynski', {})
    quality = seed_result.get('quality', {})
    return (
        int(bool(pac.get('success'))),
        int(seed_result.get('best_model') == 'paczynski'),
        int(bool(quality.get('fit_ok'))),
        _raw_delta_bic(seed_result.get('delta_bic_vs_fred', -1e9)) if np.isfinite(seed_result.get('delta_bic_vs_fred', np.nan)) else -1e9,
        _raw_delta_bic(seed_result.get('delta_bic_vs_flat', -1e9)) if np.isfinite(seed_result.get('delta_bic_vs_flat', np.nan)) else -1e9,
        -float(pac.get('reduced_chi2', np.inf)),
    )


def _significant_alt_model_name(seed_result: dict[str, object]) -> str | None:
    fred_fit = seed_result['fits'].get('fred', {})
    delta_bic_vs_fred = _raw_delta_bic(seed_result.get('delta_bic_vs_fred'))
    if fred_fit.get('success') and np.isfinite(delta_bic_vs_fred) and delta_bic_vs_fred <= -NON_PACZYNSKI_SELECTION_DELTA_BIC_THRESHOLD:
        return 'fred'
    return None


def _significant_alt_seed_rank(seed_result: dict[str, object]) -> tuple:
    alt_model = _significant_alt_model_name(seed_result)
    alt_fit = seed_result['fits'].get(alt_model, {}) if alt_model else {}
    delta_bic_vs_fred = _raw_delta_bic(seed_result.get('delta_bic_vs_fred'))
    reduced_chi2 = float(alt_fit.get('reduced_chi2', np.inf))
    n_points = int(alt_fit.get('n_points', 0))
    return (
        int(bool(alt_fit.get('success'))),
        -float(delta_bic_vs_fred) if np.isfinite(delta_bic_vs_fred) else -1e9,
        -reduced_chi2 if np.isfinite(reduced_chi2) else -1e9,
        n_points,
    )


def _fallback_pacz_seed_rank(seed_result: dict[str, object]) -> tuple:
    pac = seed_result['fits'].get('paczynski', {})
    quality = seed_result.get('quality', {})
    reduced_chi2 = float(pac.get('reduced_chi2', np.inf))
    return (
        int(bool(pac.get('success'))),
        int(quality.get('shoulder_left', 0) + quality.get('shoulder_right', 0)),
        int(quality.get('n_strong_points', 0)),
        _raw_delta_bic(seed_result.get('delta_bic_vs_flat', -1e9)) if np.isfinite(seed_result.get('delta_bic_vs_flat', np.nan)) else -1e9,
        -reduced_chi2 if np.isfinite(reduced_chi2) else -1e9,
    )


def _fallback_seed_rank(seed_result: dict[str, object]) -> tuple:
    best_fit = _best_selected_fit(seed_result)
    best_model = seed_result.get('best_model')
    reduced_chi2 = float(best_fit.get('reduced_chi2', np.inf))
    n_points = int(best_fit.get('n_points', 0))
    support_score = (n_points / max(reduced_chi2, 0.5)) if np.isfinite(reduced_chi2) else -1.0
    return (
        int(bool(best_fit.get('success'))),
        float(support_score),
        -reduced_chi2 if np.isfinite(reduced_chi2) else -1e9,
        n_points,
        int(best_model not in (None, 'flat')),
        int(seed_result.get('quality', {}).get('shoulder_left', 0) + seed_result.get('quality', {}).get('shoulder_right', 0)),
    )


def _select_best_seed_result(seed_results: list[dict[str, object]]) -> tuple[dict[str, object], str]:
    accepted = [seed_result for seed_result in seed_results if seed_result.get('quality', {}).get('fit_ok')]
    if accepted:
        return max(accepted, key=_accepted_seed_rank), 'paczynski_qc'
    significant_alt = [seed_result for seed_result in seed_results if _significant_alt_model_name(seed_result)]
    if significant_alt:
        return max(significant_alt, key=_significant_alt_seed_rank), 'significant_alt_model'
    pac_fallback = [seed_result for seed_result in seed_results if seed_result['fits'].get('paczynski', {}).get('success')]
    if pac_fallback:
        return max(pac_fallback, key=_fallback_pacz_seed_rank), 'fallback_paczynski'
    return max(seed_results, key=_fallback_seed_rank), 'fallback_best_model'


def fit_candidate_context(context: dict[str, object]) -> dict[str, object]:
    df = context['df']
    row = context['row']
    width_seed = _pick_width_seed(row)
    seeds = _candidate_seeds(df, row)
    seed_results = []
    for seed in seeds:
        suite = _fit_model_suite(df, center_guess=float(seed['t0_guess']), width_seed=width_seed)
        suite['seed_method'] = str(seed['seed_method'])
        suite['seed_t0_guess'] = float(seed['t0_guess'])
        suite['quality'] = _pacz_quality_metrics(suite)
        seed_results.append(suite)

    best_seed_result, selection_mode = _select_best_seed_result(seed_results)
    quality = best_seed_result['quality']
    if selection_mode in ('paczynski_qc', 'fallback_paczynski') and best_seed_result['fits'].get('paczynski', {}).get('success'):
        selected_model_name = 'paczynski'
    elif selection_mode == 'significant_alt_model':
        selected_model_name = _significant_alt_model_name(best_seed_result)
    else:
        selected_model_name = str(best_seed_result.get('best_model')) if best_seed_result.get('best_model') is not None else None
    best_seed_result['selected_model'] = selected_model_name
    selected_fit = _best_selected_fit(best_seed_result)
    pac = best_seed_result['fits'].get('paczynski', {})
    raw_pacz_tE = float(abs(pac['params'][2])) if pac.get('success') else np.nan
    raw_pacz_t0 = float(pac['params'][1]) if pac.get('success') else np.nan
    pac_is_displayable = bool(pac.get('success')) and selected_model_name == 'paczynski'
    display_raw_tE = raw_pacz_tE if pac_is_displayable else np.nan
    display_raw_t0 = raw_pacz_t0 if pac_is_displayable else np.nan
    reported_tE = display_raw_tE if quality.get('fit_ok') else np.nan

    payload = context['payload']
    ra_deg = _finite_float(payload.get('ra_deg'))
    if ra_deg is None:
        ra_deg = _finite_float(payload.get('ra'))
    dec_deg = _finite_float(payload.get('dec_deg'))
    if dec_deg is None:
        dec_deg = _finite_float(payload.get('dec'))
    gaia_dr3_source_id = _text_value(payload.get('gaia_id'))
    vsx_name = _text_value(payload.get('vsx_name'))
    vsx_class = _text_value(payload.get('vsx_class'))
    vsx_sep_arcsec = _finite_float(payload.get('vsx_sep_arcsec'))
    asassn_var_name = _text_value(payload.get('asassn_var_name'))
    asassn_var_type = _text_value(payload.get('asassn_var_type'))
    simbad_sep_arcsec = _finite_float(payload.get('simbad_sep_arcsec'))

    fit_t0_jd_minus_2450000 = raw_pacz_t0 if np.isfinite(raw_pacz_t0) else np.nan
    fit_t0_jd = raw_pacz_t0 + 2450000.0 if np.isfinite(raw_pacz_t0) else np.nan
    peak_window_start_jd_minus_2450000 = raw_pacz_t0 - 2.0 * raw_pacz_tE if np.isfinite(raw_pacz_t0) and np.isfinite(raw_pacz_tE) else np.nan
    peak_window_end_jd_minus_2450000 = raw_pacz_t0 + 2.0 * raw_pacz_tE if np.isfinite(raw_pacz_t0) and np.isfinite(raw_pacz_tE) else np.nan
    peak_window_start_jd = peak_window_start_jd_minus_2450000 + 2450000.0 if np.isfinite(peak_window_start_jd_minus_2450000) else np.nan
    peak_window_end_jd = peak_window_end_jd_minus_2450000 + 2450000.0 if np.isfinite(peak_window_end_jd_minus_2450000) else np.nan

    if selected_model_name == 'paczynski':
        if pac.get('success') and np.isfinite(raw_pacz_tE) and float(raw_pacz_tE) >= PARALLAX_MIN_TE_DAYS:
            parallax_result = _fit_parallax_diagnostics(context, best_seed_result, ra_deg=ra_deg, dec_deg=dec_deg)
        elif pac.get('success'):
            parallax_result = _empty_parallax_result('not_attempted:tE_below_threshold')
        else:
            parallax_result = _empty_parallax_result('not_attempted:no_paczynski_fit')
    else:
        parallax_result = _empty_parallax_result('not_attempted:selected_model_not_paczynski')

    # Compute flux-space fit for additional metrics
    flux_fit = _fit_model_flux_space(
        best_seed_result['jd_fit'],
        best_seed_result['mag_fit'],
        best_seed_result['err_fit'],
        center_guess=raw_pacz_t0 if np.isfinite(raw_pacz_t0) else float(best_seed_result['seed_t0_guess']),
        width_seed=_pick_width_seed(row),
    )

    # Compute morphology metrics
    morphology = _compute_morphology_metrics(
        best_seed_result['jd_fit'],
        best_seed_result['mag_fit'],
        best_seed_result['err_fit'],
        pac,
    )

    # CV/Nova scoring
    fred_fit = best_seed_result['fits'].get('fred', {})
    cv_nova = _compute_cv_nova_score(
        best_seed_result['jd_fit'],
        best_seed_result['mag_fit'],
        best_seed_result['err_fit'],
        pac,
        fred_fit,
    )

    # Periodicity scanning
    model_mag = pac.get('model') if pac.get('success') else None
    periodicity = _scan_periodicity(
        df['JD'].to_numpy(),
        df['mag'].to_numpy(),
        df['error'].to_numpy() if 'error' in df.columns else df.get('mag_err', np.full(len(df), 0.01)).to_numpy(),
        scan_residuals=True,
        model_mag=np.interp(df['JD'].to_numpy(), best_seed_result['jd_fit'], model_mag) if model_mag is not None else None,
    )

    brightest = df.nsmallest(min(10, len(df)), 'mag')
    summary = {
        'candidate_id': context['candidate_id'],
        'asas_sn_id': context['asas_sn_id'],
        'lc_path': str(context['lc_path']),
        'band_used': context['band_label'],
        'n_points_total': int(len(df)),
        'min_mag_t0_guess': float(df.loc[int(df['mag'].idxmin()), 'JD']),
        'brightest10_median_t0_guess': float(brightest['JD'].median()) if not brightest.empty else np.nan,
        'pipeline_jump_t0': _finite_float(row.get('jump_best_t0')),
        'seed_method': best_seed_result['seed_method'],
        'seed_t0_guess': best_seed_result['seed_t0_guess'],
        'best_model': selected_model_name,
        'best_model_by_bic': best_seed_result.get('best_model'),
        'best_alt_model': best_seed_result.get('best_alt_model'),
        'selection_mode': selection_mode,
        'fit_ok': bool(quality.get('fit_ok')),
        'reported_tE_days': reported_tE,
        'raw_paczynski_tE_days': raw_pacz_tE,
        'display_raw_paczynski_tE_days': display_raw_tE,
        'fit_t0': display_raw_t0,
        'fit_t0_time_system': 'JD-2450000',
        'fit_t0_jd_minus_2450000': fit_t0_jd_minus_2450000,
        'fit_t0_jd': fit_t0_jd,
        'peak_window_time_system': 'JD-2450000',
        'peak_window_start_jd_minus_2450000': peak_window_start_jd_minus_2450000,
        'peak_window_end_jd_minus_2450000': peak_window_end_jd_minus_2450000,
        'peak_window_start_jd': peak_window_start_jd,
        'peak_window_end_jd': peak_window_end_jd,
        'fit_reduced_chi2': float(selected_fit.get('reduced_chi2', np.nan)),
        'paczynski_reduced_chi2': float(pac.get('reduced_chi2', np.nan)) if pac.get('success') else np.nan,
        'delta_bic_vs_flat': best_seed_result.get('delta_bic_vs_flat'),
        'delta_bic_vs_gaussian': best_seed_result.get('delta_bic_vs_gaussian'),
        'delta_bic_vs_fred': best_seed_result.get('delta_bic_vs_fred'),
        'delta_bic_vs_best_alt': best_seed_result.get('delta_bic_vs_best_alt'),
        'n_points_fit': int(np.sum(best_seed_result['fit_mask'])),
        'half_window_days': float(best_seed_result['half_window']),
        'shoulder_left': int(quality.get('shoulder_left', 0)),
        'shoulder_right': int(quality.get('shoulder_right', 0)),
        'n_strong_points': int(quality.get('n_strong_points', 0)),
        'paczynski_tau_coverage_score': _finite_float(quality.get('paczynski_tau_coverage_score')),
        'paczynski_coverage_n_bins_hit': int(quality.get('paczynski_coverage_n_bins_hit', 0)),
        'paczynski_coverage_n_bins': int(quality.get('paczynski_coverage_n_bins', PAC_COVERAGE_N_BINS)),
        'paczynski_coverage_max_weighted_gap': _finite_float(quality.get('paczynski_coverage_max_weighted_gap')),
        'paczynski_coverage_frac_points_in_tau_window': _finite_float(quality.get('paczynski_coverage_frac_points_in_tau_window')),
        'fit_warning': ','.join(quality.get('warnings', [])),
        'ra_deg': ra_deg,
        'dec_deg': dec_deg,
        'gaia_dr3_source_id': gaia_dr3_source_id,
        'asassn_source_id': _text_value(context['asas_sn_id']),
        'asassn_var_name': asassn_var_name,
        'asassn_var_type': asassn_var_type,
        'nearest_simbad_object': _text_value(row.get('simbad_main_id')) or _text_value(payload.get('simbad_main_id')),
        'simbad_otype': _text_value(row.get('simbad_otype')) or _text_value(payload.get('simbad_otype')),
        'simbad_sep_arcsec': simbad_sep_arcsec,
        'nearest_vsx_object': vsx_name,
        'vsx_class': vsx_class,
        'vsx_sep_arcsec': vsx_sep_arcsec,
        'microlens_match': _bool_flag(row.get('microlens_match')),
        'microlens_catalog': _text_value(row.get('microlens_catalog')),
        'microlens_name': _text_value(row.get('microlens_name')),
        'microlens_alt_name': _text_value(row.get('microlens_alt_name')),
        'microlens_te_days': _finite_float(row.get('microlens_te_days')),
        'microlens_sep_arcsec': _finite_float(row.get('microlens_sep_arcsec')),
        'vetting_likely_known': _bool_flag(row.get('vetting_likely_known')),
        'catalog_source': _text_value(row.get('catalog_source')),
        'gaia_var_class': _text_value(payload.get('gaia_var_class')) or _text_value(row.get('gaia_var_class')),
        'ztf_var_type': _text_value(payload.get('ztf_var_type')) or _text_value(row.get('ztf_var_type')),
        # Flux-space fit parameters
        'flux_u0': float(flux_fit.get('u0', np.nan)) if flux_fit.get('success') else np.nan,
        'flux_Fs': float(flux_fit.get('Fs', np.nan)) if flux_fit.get('success') else np.nan,
        'flux_Fb': float(flux_fit.get('Fb', np.nan)) if flux_fit.get('success') else np.nan,
        'flux_blend_fraction': float(flux_fit.get('blend_fraction', np.nan)) if flux_fit.get('success') else np.nan,
        # Morphology metrics
        'morph_rise_time_days': morphology.get('rise_time_days', np.nan),
        'morph_decay_time_days': morphology.get('decay_time_days', np.nan),
        'morph_rise_decay_ratio': morphology.get('rise_decay_ratio', np.nan),
        'morph_event_skewness': morphology.get('event_skewness', np.nan),
        'morph_residual_autocorr': morphology.get('residual_autocorr', np.nan),
        'morph_symmetry_score': morphology.get('symmetry_score', np.nan),
        'morph_excursion_fraction': morphology.get('excursion_fraction', np.nan),
        'morph_vonneumann_ratio': morphology.get('vonneumann_ratio', np.nan),
        # CV/Nova metrics
        'cv_nova_score': cv_nova.get('cv_nova_score', np.nan),
        'cv_nova_fred_preferred': cv_nova.get('fred_preferred', False),
        'cv_nova_rise_decay_asymmetry': cv_nova.get('rise_decay_asymmetry', np.nan),
        'cv_nova_secondary_peak': cv_nova.get('secondary_peak_detected', False),
        'cv_nova_amplitude_mag': cv_nova.get('amplitude_mag', np.nan),
        # Periodicity metrics
        'period_lsp_best': periodicity.get('lsp_best_period', np.nan),
        'period_lsp_power': periodicity.get('lsp_best_power', np.nan),
        'period_pdm_best': periodicity.get('pdm_best_period', np.nan),
        'period_pdm_theta': periodicity.get('pdm_best_theta', np.nan),
        'period_resid_lsp_best': periodicity.get('resid_lsp_best_period', np.nan),
        'period_resid_lsp_power': periodicity.get('resid_lsp_best_power', np.nan),
        'periodicity_detected': periodicity.get('periodicity_detected', False),
    }
    summary.update(_flatten_parallax_summary(parallax_result))

    # Compute per-candidate quality score
    candidate_quality = _compute_candidate_quality_score(
        summary=summary,
        morphology=morphology,
        periodicity=periodicity,
        cv_nova=cv_nova,
        quality_metrics=quality,
    )
    summary['quality_score'] = candidate_quality['quality_score']
    summary['quality_tier'] = candidate_quality['quality_tier']
    summary['quality_flags'] = ','.join(candidate_quality['quality_flags'])

    return {
        'context': context,
        'seed_results': seed_results,
        'best_seed_result': best_seed_result,
        'parallax': parallax_result,
        'flux_fit': flux_fit,
        'morphology': morphology,
        'cv_nova': cv_nova,
        'periodicity': periodicity,
        'candidate_quality': candidate_quality,
        'summary': summary,
    }


def _load_candidate_metadata_from_candidates_parquet(
    candidate_ids: list[str] | set[str] | tuple[str, ...],
) -> dict[str, dict[str, object]]:
    """
    Load minimal metadata keyed by candidate/asassn ID from ``output/candidates.parquet``.

    Returns a mapping ``candidate_id -> {'ra_deg': float, 'dec_deg': float}`` for IDs present
    in the local candidates catalog.
    """
    path = REPO_ROOT / 'output' / 'candidates.parquet'
    if not path.exists():
        return {}
    try:
        df = pd.read_parquet(path, columns=['asas_sn_id', 'ra_deg', 'dec_deg'])
    except Exception:
        return {}
    if df.empty:
        return {}

    df = df.copy()
    df['asas_sn_id'] = df['asas_sn_id'].map(_candidate_id_match_str)
    df['ra_deg'] = pd.to_numeric(df['ra_deg'], errors='coerce')
    df['dec_deg'] = pd.to_numeric(df['dec_deg'], errors='coerce')
    df = df.dropna(subset=['asas_sn_id', 'ra_deg', 'dec_deg'])
    if df.empty:
        return {}

    wanted = {_candidate_id_match_str(x) for x in candidate_ids}
    if not wanted:
        return {}
    df = df.loc[df['asas_sn_id'].isin(wanted), ['asas_sn_id', 'ra_deg', 'dec_deg']]
    if df.empty:
        return {}

    out: dict[str, dict[str, object]] = {}
    for _, row in df.drop_duplicates('asas_sn_id', keep='first').iterrows():
        cid = str(row['asas_sn_id'])
        out[cid] = {
            'ra_deg': float(row['ra_deg']),
            'dec_deg': float(row['dec_deg']),
        }
    return out


def _fetch_candidate_metadata_from_skypatrol2(
    candidate_ids: list[str] | set[str] | tuple[str, ...],
) -> dict[str, dict[str, object]]:
    """
    Best-effort SkyPatrol2 fallback (pyasassn) for ``ra_deg``/``dec_deg`` by ``asas_sn_id``.

    This is intentionally defensive because pyasassn query APIs may differ across versions.
    """
    try:
        from pyasassn.client import SkyPatrolClient  # type: ignore
    except Exception:
        return {}

    candidate_ids_norm = [_candidate_id_match_str(x) for x in candidate_ids]
    candidate_ids_norm = [x for x in candidate_ids_norm if x]
    if not candidate_ids_norm:
        return {}

    # Query the *client* for rows; `client.catalogs.master_list` is just a schema table.
    # Use `query_list` which actually returns catalog rows including ra/dec.
    out: dict[str, dict[str, object]] = {}
    try:
        client = SkyPatrolClient()
    except Exception:
        return {}

    # Chunk to be defensive against server limits.
    chunk_size = 200
    for i in range(0, len(candidate_ids_norm), chunk_size):
        chunk = candidate_ids_norm[i : i + chunk_size]
        try:
            ids_int = [int(x) for x in chunk]
        except Exception:
            ids_int = []
        if not ids_int:
            continue

        try:
            df = client.query_list(
                ids_int,
                id_col='asas_sn_id',
                catalog='master_list',
                cols=['asas_sn_id', 'ra_deg', 'dec_deg'],
                download=False,
                threads=1,
            )
        except Exception:
            continue

        if df is None or getattr(df, 'empty', True):
            continue

        df = df.copy()
        if 'asas_sn_id' in df.columns:
            df['asas_sn_id'] = df['asas_sn_id'].map(_candidate_id_match_str)
        df['ra_deg'] = pd.to_numeric(df.get('ra_deg'), errors='coerce')
        df['dec_deg'] = pd.to_numeric(df.get('dec_deg'), errors='coerce')
        df = df.dropna(subset=['asas_sn_id', 'ra_deg', 'dec_deg'])
        for _, row in df.iterrows():
            cid = str(row['asas_sn_id'])
            ra = float(row['ra_deg'])
            dec = float(row['dec_deg'])
            if np.isfinite(ra) and np.isfinite(dec):
                out[cid] = {'ra_deg': ra, 'dec_deg': dec}

    return out


def _enrich_candidate_metadata_with_gaia_ids(
    candidate_metadata_by_id: dict[str, dict[str, object]],
    *,
    max_sep_arcsec: float = 1.0,
) -> dict[str, dict[str, object]]:
    """
    Add Gaia DR3 ``gaia_id`` (``source_id``) via conservative coordinate cone-match.
    """
    if not candidate_metadata_by_id:
        return candidate_metadata_by_id

    rows: list[dict[str, object]] = []
    cids: list[str] = []
    for cid, meta in candidate_metadata_by_id.items():
        if meta.get('gaia_id'):
            continue
        ra = _finite_float(meta.get('ra_deg'))
        dec = _finite_float(meta.get('dec_deg'))
        if ra is None or dec is None:
            continue
        cids.append(str(cid))
        rows.append({'_idx': len(cids) - 1, 'ra': float(ra), 'dec': float(dec)})

    if not rows:
        return candidate_metadata_by_id

    coords_df = pd.DataFrame(rows)
    try:
        gaia_hits = batch_gaia_cone_query(
            coords_df,
            select_cols="g.ra, g.dec",
            extra_where="",
            match_radius_arcsec=float(max_sep_arcsec),
            chunk_size=500,
            n_workers=2,
            verbose=False,
        )
    except Exception:
        return candidate_metadata_by_id

    if gaia_hits.empty:
        return candidate_metadata_by_id

    hits = gaia_hits.copy()
    hits['_idx'] = pd.to_numeric(hits.get('_idx'), errors='coerce')
    hits['sep_arcsec'] = pd.to_numeric(hits.get('sep_arcsec'), errors='coerce')
    hits = hits.dropna(subset=['_idx', 'source_id', 'sep_arcsec'])
    if hits.empty:
        return candidate_metadata_by_id
    hits = hits.sort_values(['_idx', 'sep_arcsec'], kind='mergesort').drop_duplicates('_idx', keep='first')

    for _, row in hits.iterrows():
        idx = int(row['_idx'])
        if idx < 0 or idx >= len(cids):
            continue
        cid = cids[idx]
        sid = _candidate_id_match_str(row.get('source_id'))
        if not sid:
            continue
        candidate_metadata_by_id.setdefault(cid, {})
        candidate_metadata_by_id[cid]['gaia_id'] = sid

    return candidate_metadata_by_id


def _fit_candidate_context_from_db_task(
    db_path: str,
    candidate_id: str,
    *,
    prefer_g_band: bool = True,
) -> dict[str, object]:
    """Worker-safe single-candidate fit from review DB."""
    db = Path(db_path).expanduser().resolve()
    plot_dir = infer_plot_dir_from_source(db)
    with sqlite3.connect(db) as conn:
        init_db(conn)
        context = _load_candidate_context(
            conn,
            str(candidate_id),
            plot_dir=plot_dir,
            prefer_g_band=prefer_g_band,
        )
    return fit_candidate_context(context)


def _fit_candidate_context_from_lightcurve_task(
    candidate_id: str,
    lc_paths: list[str],
    metadata: dict[str, object] | None = None,
    *,
    prefer_g_band: bool = True,
) -> dict[str, object] | None:
    """Worker-safe single-candidate fit from direct light-curve paths."""
    df: pd.DataFrame | None = None
    band_label: str = 'all'
    used_lc_path: Path | None = None
    for lc_path_s in lc_paths:
        lc_path = Path(lc_path_s)
        try:
            _df, _band_label = _prepare_lightcurve_df(lc_path, prefer_g_band=prefer_g_band)
        except Exception:
            continue
        if _df.empty:
            continue
        df = _df
        band_label = _band_label
        used_lc_path = lc_path
        break

    if df is None or used_lc_path is None:
        return None

    payload = {'candidate_id': str(candidate_id)}
    meta = metadata or {}
    if _finite_float(meta.get('ra_deg')) is not None:
        payload['ra_deg'] = float(meta['ra_deg'])
    if _finite_float(meta.get('dec_deg')) is not None:
        payload['dec_deg'] = float(meta['dec_deg'])
    gid = _candidate_id_match_str(meta.get('gaia_id'))
    if gid:
        payload['gaia_id'] = gid

    context = {
        'candidate_id': str(candidate_id),
        'asas_sn_id': str(candidate_id),
        'row': {},
        'payload': payload,
        'lc_path': used_lc_path,
        'df': df,
        'band_label': band_label,
    }
    return fit_candidate_context(context)


def _fit_lightcurve_only_candidates(
    candidate_id_to_lc_paths: dict[str, list[Path]],
    *,
    candidate_metadata_by_id: dict[str, dict[str, object]] | None = None,
    fit_workers: int = 1,
    prefer_g_band: bool = True,
    show_progress: bool = False,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """
    Fit candidates when we have a native light-curve path but no `review.db` row.

    We construct a minimal `context` (empty row/payload) and let the fitter fall back to
    data-driven seeds. Parallax diagnostics will be skipped because coordinates are missing.
    """
    items = sorted(candidate_id_to_lc_paths.items(), key=lambda kv: kv[0])
    if show_progress and sys.stderr.isatty():
        items = tqdm(items, desc="Fitting lightcurves (no review.db)", unit="cid")

    results: list[dict[str, object]] = []
    if int(fit_workers) <= 1:
        for candidate_id, lc_paths in items:
            result = _fit_candidate_context_from_lightcurve_task(
                str(candidate_id),
                [str(p) for p in lc_paths],
                (candidate_metadata_by_id or {}).get(str(candidate_id), {}),
                prefer_g_band=prefer_g_band,
            )
            if result is not None:
                results.append(result)
    else:
        with cf.ProcessPoolExecutor(max_workers=int(fit_workers)) as ex:
            futures = [
                ex.submit(
                    _fit_candidate_context_from_lightcurve_task,
                    str(candidate_id),
                    [str(p) for p in lc_paths],
                    (candidate_metadata_by_id or {}).get(str(candidate_id), {}),
                    prefer_g_band=prefer_g_band,
                )
                for candidate_id, lc_paths in items
            ]
            iterator = cf.as_completed(futures)
            if show_progress and sys.stderr.isatty():
                iterator = tqdm(iterator, total=len(futures), desc="LC-only fits", unit="cid")
            for fut in iterator:
                result = fut.result()
                if result is not None:
                    results.append(result)

    results_df = pd.DataFrame([result['summary'] for result in results]).sort_values('candidate_id').reset_index(drop=True)
    return results_df, results


def fit_microlensing_candidates(
    db_path: str | Path,
    *,
    candidate_ids: list[str],
    fit_workers: int = 1,
    prefer_g_band: bool = True,
    show_progress: bool = True,
    progress_desc: str = "Fitting candidates",
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    db_path = Path(db_path).expanduser().resolve()
    plot_dir = infer_plot_dir_from_source(db_path)
    raw_n = len(candidate_ids)
    candidate_ids = exclude_visual_inspection_bad_ids(list(candidate_ids))
    dropped = raw_n - len(candidate_ids)
    if dropped and show_progress:
        print(
            f"  Excluding {dropped} subjective visual-review BAD LC(s); see VISUAL_INSPECTION_BAD_IDS in microlensing.py.",
            flush=True,
        )
    if not candidate_ids:
        return pd.DataFrame(), []
    results: list[dict[str, object]] = []
    ids_loop = list(candidate_ids)
    if show_progress and sys.stderr.isatty() and int(fit_workers) <= 1:
        ids_loop = tqdm(ids_loop, desc=progress_desc, unit="cid")
    if int(fit_workers) <= 1:
        with sqlite3.connect(db_path) as conn:
            init_db(conn)
            for candidate_id in ids_loop:
                context = _load_candidate_context(
                    conn,
                    str(candidate_id),
                    plot_dir=plot_dir,
                    prefer_g_band=prefer_g_band,
                )
                results.append(fit_candidate_context(context))
    else:
        with cf.ProcessPoolExecutor(max_workers=int(fit_workers)) as ex:
            futures = [
                ex.submit(
                    _fit_candidate_context_from_db_task,
                    str(db_path),
                    str(candidate_id),
                    prefer_g_band=prefer_g_band,
                )
                for candidate_id in ids_loop
            ]
            iterator = cf.as_completed(futures)
            if show_progress and sys.stderr.isatty():
                iterator = tqdm(iterator, total=len(futures), desc=progress_desc, unit="cid")
            for fut in iterator:
                results.append(fut.result())

    results_df = pd.DataFrame([result['summary'] for result in results]).sort_values('candidate_id').reset_index(drop=True)
    return results_df, results


# Back-compat name used in older notebooks / snippets
fit_march18_candidates = fit_microlensing_candidates


def plot_candidate_fit(result: dict[str, object], *, figsize: tuple[float, float] = (13.0, 10.0)):
    context = result['context']
    summary = result['summary']
    best_seed = result['best_seed_result']
    pac = best_seed['fits'].get('paczynski', {})
    best_model_name = best_seed.get('selected_model') or best_seed.get('best_model')
    best_model_fit = best_seed['fits'].get(best_model_name, {}) if best_model_name else {}
    pac_success = bool(pac.get('success'))
    df = context['df']
    err_col = 'error' if 'error' in df.columns else 'mag_err'
    parallax = result.get('parallax', {}) or {}
    parallax_best = parallax.get('branches', {}).get(parallax.get('best_branch', ''), {}) if parallax.get('best_branch') else {}
    show_parallax = False  # bool(parallax_best.get('success')) and summary.get('parallax_attempted', False)
    
    # Flux-space fit data
    flux_fit = result.get('flux_fit', {}) or {}
    flux_fit_success = bool(flux_fit.get('success'))

    plot_jd_offset = 8000.0
    jd_axis_label = 'JD - 2458000 [d]'
    mag_label = r'$g$ [mag]'
    flux_label = r'Relative Flux'

    def _plot_jd(values):
        return np.asarray(values, dtype=float) - plot_jd_offset

    def _plot_jd_scalar(value: float) -> float:
        return float(value) - plot_jd_offset

    fig = plt.figure(figsize=figsize, dpi=MICROLENSING_FIT_PDF_DPI)
    # 3 panels: full LC (mag), zoomed LC (mag), residuals
    gs = fig.add_gridspec(3, 1, height_ratios=[2.8, 1.8, 0.9], hspace=0.30)
    ax = fig.add_subplot(gs[0])
    ax_zoom = fig.add_subplot(gs[1])
    ax_res = fig.add_subplot(gs[2], sharex=ax_zoom)
    _ctx_caption = _plot_crossmatch_context_caption(summary)
    _top_margin = 0.88 if _ctx_caption else 0.93
    fig.subplots_adjust(left=0.08, right=0.84, top=_top_margin, bottom=0.10)

    fit_mask = np.asarray(best_seed['fit_mask'], dtype=bool)
    fit_df = df.loc[fit_mask]
    jd_dense = np.linspace(float(best_seed['jd_fit'].min()), float(best_seed['jd_fit'].max()), 700)

    # Curves: Paczynski always when it fit. Gaussian/FRED only when:
    # - best is Paczynski: do not plot alts unless Pac struggles vs flat
    #   (delta_bic_vs_flat = BIC_flat - BIC_pac <= -PLOT_ALT_WHEN_PAC_VS_FLAT_DELTA_BIC); then each alt with BIC < BIC_pac.
    # - best is Gaussian or FRED: always plot Pac + that model; also plot the other alt if it beats Pac on BIC.
    # - best is something else (e.g. flat): plot any gaussian/fred that beats Pac on BIC (diagnostic).
    dense_models: dict[str, np.ndarray] = {}
    if pac_success:
        dense_models['paczynski'] = _evaluate_model('paczynski', pac['params'], jd_dense, pac['t_ref'])
    delta_flat_pac = best_seed.get('delta_bic_vs_flat')
    pac_struggles_vs_flat = (
        np.isfinite(delta_flat_pac) and float(delta_flat_pac) <= -PLOT_ALT_WHEN_PAC_VS_FLAT_DELTA_BIC
    )
    pac_bic = pac.get('bic') if pac_success else None

    def _alt_beats_pac_bic(m_fit: dict[str, object]) -> bool:
        if not pac_success or pac_bic is None:
            return False
        m_bic = m_fit.get('bic')
        return (
            m_bic is not None
            and np.isfinite(m_bic)
            and np.isfinite(pac_bic)
            and float(m_bic) < float(pac_bic)
        )

    for m_name in ('gaussian', 'fred'):
        m_fit = best_seed['fits'].get(m_name, {})
        if not m_fit.get('success'):
            continue
        include = False
        if best_model_name == 'paczynski':
            include = bool(pac_struggles_vs_flat and _alt_beats_pac_bic(m_fit))
        elif best_model_name == m_name:
            include = True
        elif best_model_name in ('gaussian', 'fred'):
            include = _alt_beats_pac_bic(m_fit)
        else:
            include = _alt_beats_pac_bic(m_fit)
        if include:
            dense_models[m_name] = _evaluate_model(m_name, m_fit['params'], jd_dense, m_fit['t_ref'])

    parallax_dense = None
    if show_parallax and np.isfinite(summary.get('ra_deg', np.nan)) and np.isfinite(summary.get('dec_deg', np.nan)):
        parallax_dense = _evaluate_parallax_branch_mag(parallax_best, jd_dense, float(summary['ra_deg']), float(summary['dec_deg']))

    def _zoom_center_and_scale() -> tuple[float, float]:
        if show_parallax:
            return float(summary['parallax_best_t0_jd_minus_2450000']), max(float(summary['parallax_best_tE_days']), 5.0)
        if best_model_fit.get('success') and best_model_name in {'paczynski', 'gaussian', 'fred'}:
            params = np.asarray(best_model_fit['params'], dtype=float)
            center = float(params[1])
            if best_model_name in {'paczynski', 'gaussian'}:
                scale = float(abs(params[2]))
            else:
                scale = float(max(abs(params[2]), abs(params[3])))
            return center, max(scale, 5.0)
        if pac_success:
            return float(pac['params'][1]), max(float(abs(pac['params'][2])), 5.0)
        return float(summary['seed_t0_guess']), max(float(0.12 * best_seed['half_window']), 10.0)

    zoom_center, zoom_scale = _zoom_center_and_scale()
    zoom_half_window = float(np.clip(max(35.0, 3.5 * zoom_scale), 35.0, max(80.0, float(best_seed['half_window']))))
    zoom_mask = np.abs(df['JD'] - zoom_center) <= zoom_half_window
    if int(np.sum(zoom_mask)) < 8:
        zoom_half_window = float(np.clip(max(60.0, 0.25 * float(best_seed['half_window'])), 60.0, max(120.0, float(best_seed['half_window']))))
        zoom_mask = np.abs(df['JD'] - zoom_center) <= zoom_half_window
    zoom_df = df.loc[zoom_mask]
    zoom_fit_df = fit_df.loc[np.abs(fit_df['JD'] - zoom_center) <= zoom_half_window] if not fit_df.empty else fit_df

    _model_legend_tex = {
        'paczynski': r'$\mathrm{Paczynski}$',
        'fred': r'$\mathrm{FRED}$',
        'gaussian': r'$\mathrm{Gaussian}$',
    }
    colors = {'paczynski': 'red', 'fred': 'blue', 'gaussian': 'orange'}
    # Figure-level info panel uses a white bbox; keep model/data artists above it where they overlap.
    _z_lc_points = 4
    _z_vline = 4.5
    _z_model_curves = 6

    for axis in (ax, ax_zoom):
        axis.errorbar(
            _plot_jd(df['JD']), df['mag'], yerr=df[err_col],
            fmt='k.', alpha=0.25 if axis is ax_zoom else 0.7, markersize=3, elinewidth=0.8, capsize=0,
            zorder=_z_lc_points,
        )
        axis.errorbar(
            _plot_jd(fit_df['JD']), fit_df['mag'], yerr=fit_df[err_col],
            fmt='k.', alpha=0.9, markersize=4, elinewidth=1.0, capsize=0,
            zorder=_z_lc_points + 0.1,
        )
        for m_name, m_dense in dense_models.items():
            kw = {
                'color': colors.get(m_name, 'k'),
                'linewidth': 2.0 if m_name == 'paczynski' else 1.6,
                'zorder': _z_model_curves,
            }
            if m_name != 'paczynski':
                kw['linestyle'] = '--'
            if axis is ax:
                kw['label'] = _model_legend_tex.get(m_name, m_name)
            axis.plot(_plot_jd(jd_dense), m_dense, **kw)
        if pac_success:
            axis.axvline(
                _plot_jd_scalar(float(pac['params'][1])),
                color='tab:orange', linestyle=':', linewidth=1.2,
                zorder=_z_vline,
                label=r'$t_0\,\mathrm{(Paczynski)}$' if axis is ax else None,
            )
        if parallax_dense is not None:
            kw_par = {'color': 'tab:cyan', 'linewidth': 2.0, 'linestyle': '-.', 'zorder': _z_model_curves}
            if axis is ax:
                kw_par['label'] = r'$\mathrm{Parallax}$'
            axis.plot(_plot_jd(jd_dense), parallax_dense, **kw_par)
        axis.axvline(
            _plot_jd_scalar(float(summary['seed_t0_guess'])),
            color='tab:brown', linestyle='--', linewidth=1.0, alpha=0.8, zorder=_z_vline,
        )
        axis.grid(alpha=0.2)
        axis.invert_yaxis()

    for artist in [*list(ax_zoom.collections), *list(ax_zoom.lines)]:
        artist.remove()
    ax_zoom.errorbar(
        _plot_jd(zoom_df['JD']), zoom_df['mag'], yerr=zoom_df[err_col],
        fmt='k.', alpha=0.8, markersize=4, elinewidth=1.0, capsize=0, zorder=_z_lc_points,
    )
    ax_zoom.errorbar(
        _plot_jd(zoom_fit_df['JD']), zoom_fit_df['mag'], yerr=zoom_fit_df[err_col],
        fmt='k.', alpha=0.95, markersize=5, elinewidth=1.2, capsize=0, zorder=_z_lc_points + 0.1,
    )
    zoom_dense_mask = np.abs(jd_dense - zoom_center) <= zoom_half_window
    for m_name, m_dense in dense_models.items():
        kw = {
            'color': colors.get(m_name, 'k'),
            'linewidth': 2.0 if m_name == 'paczynski' else 1.6,
            'zorder': _z_model_curves,
        }
        if m_name != 'paczynski':
            kw['linestyle'] = '--'
        ax_zoom.plot(_plot_jd(jd_dense[zoom_dense_mask]), m_dense[zoom_dense_mask], **kw)
    if parallax_dense is not None:
        ax_zoom.plot(
            _plot_jd(jd_dense[zoom_dense_mask]), parallax_dense[zoom_dense_mask],
            color='tab:cyan', linewidth=2.0, linestyle='-.', zorder=_z_model_curves,
        )
    ax_zoom.axvline(
        _plot_jd_scalar(float(summary['seed_t0_guess'])),
        color='tab:brown', linestyle='--', linewidth=1.0, alpha=0.8, zorder=_z_vline,
    )
    if pac_success:
        ax_zoom.axvline(
            _plot_jd_scalar(float(pac['params'][1])), color='tab:orange', linestyle=':', linewidth=1.2, zorder=_z_vline,
        )
    if show_parallax and np.isfinite(summary.get('parallax_best_t0_jd_minus_2450000', np.nan)):
        ax_zoom.axvline(
            _plot_jd_scalar(float(summary['parallax_best_t0_jd_minus_2450000'])),
            color='tab:cyan', linestyle=':', linewidth=1.2, zorder=_z_vline,
        )
    ax_zoom.set_xlim(_plot_jd_scalar(zoom_center - zoom_half_window), _plot_jd_scalar(zoom_center + zoom_half_window))
    ax_zoom.set_ylabel(mag_label)
    ax_zoom.tick_params(axis='x', which='both', labelbottom=False, labeltop=False, top=True)

    # Bottom panel: always residuals vs Paczynski model (when available).
    if pac_success:
        residual_model = np.asarray(pac['model'], dtype=float)
    else:
        residual_model = np.full_like(best_seed['mag_fit'], np.nan)

    residuals = best_seed['mag_fit'] - residual_model
    zoom_res_mask = np.abs(best_seed['jd_fit'] - zoom_center) <= zoom_half_window
    if int(np.sum(zoom_res_mask)) < 4:
        zoom_res_mask = np.ones_like(best_seed['jd_fit'], dtype=bool)
    ax_res.axhline(0.0, color='0.4', linewidth=1.0)
    ax_res.errorbar(
        _plot_jd(best_seed['jd_fit'][zoom_res_mask]),
        residuals[zoom_res_mask],
        yerr=np.asarray(best_seed['err_fit'], dtype=float)[zoom_res_mask],
        fmt='k.', alpha=0.9, markersize=4, elinewidth=1.0, capsize=0,
    )
    ax_res.set_ylabel(r'$\mathrm{Residual\ [mag]}$')
    ax_res.set_xlabel(jd_axis_label)
    ax_res.grid(alpha=0.2)
    ax_res.invert_yaxis()

    title_tE_days = summary['reported_tE_days']
    if not np.isfinite(title_tE_days):
        title_tE_days = summary['raw_paczynski_tE_days']
    _tE_title = title_tE_days if np.isfinite(title_tE_days) else float('nan')
    pac_rchi = float(pac.get('reduced_chi2', np.nan)) if pac_success else np.nan
    if np.isfinite(pac_rchi):
        _chi2nu_title = rf'$\chi^2_\nu={pac_rchi:.2f}$'
    else:
        _chi2nu_title = r'$\chi^2_\nu=\mathrm{nan}$'
    _title_line1 = (
        f'{_chi2nu_title} | {summary["candidate_id"]} | ' + rf"$t_\mathrm{{E}}={_tE_title:.3f}\,\mathrm{{d}}$"
    )
    if _ctx_caption:
        ax.set_title(f'{_title_line1}\n{_ctx_caption}', fontsize=9)
    else:
        ax.set_title(_title_line1, fontsize=10)
    ax.set_ylabel(mag_label)
    ax.set_xlabel(jd_axis_label)
    ax.tick_params(axis='x', labelbottom=True, labeltop=False)

    def _mtxt(val: object) -> str:
        return str(val).replace('_', r'\_').replace('%', r'\%')

    best_bic = best_model_fit.get('bic', np.nan)
    dbic_lines = []
    for m_name in ['flat', 'gaussian', 'fred', 'paczynski']:
        if m_name == best_model_name:
            continue
        m_bic = best_seed['fits'].get(m_name, {}).get('bic')
        if m_bic is not None and np.isfinite(m_bic) and np.isfinite(best_bic):
            dbic_val = float(m_bic) - float(best_bic)
            dbic_lines.append(rf'$\Delta\mathrm{{BIC}}(\mathrm{{{m_name}}})={dbic_val:.2f}$')
    _chi2nu_panel_names = {'paczynski': 'Paczynski', 'fred': 'FRED', 'gaussian': 'Gaussian'}
    chi2nu_lines: list[str] = []
    for m_name in ('paczynski', 'gaussian', 'fred'):
        if m_name not in dense_models:
            continue
        if m_name == 'paczynski':
            continue
        fit = best_seed['fits'].get(m_name, {})
        rchi = float(fit.get('reduced_chi2', np.nan))
        lab = _chi2nu_panel_names.get(m_name, m_name)
        if np.isfinite(rchi):
            chi2nu_lines.append(rf'$\chi^2_\nu(\mathrm{{{lab}}})={rchi:.2f}$')
        else:
            chi2nu_lines.append(rf'$\chi^2_\nu(\mathrm{{{lab}}})=\mathrm{{nan}}$')
    if not chi2nu_lines:
        _chi2_nu = float(summary.get('fit_reduced_chi2', np.nan))
        _chi2_nu_s = f'{_chi2_nu:.2f}' if np.isfinite(_chi2_nu) else r'\mathrm{nan}'
        chi2nu_lines.append(rf'$\chi^2_\nu(\mathrm{{best}})={_chi2_nu_s}$')

    # Flux-space blending info
    blend_lines = []
    if flux_fit_success:
        Fs_val = float(flux_fit.get('Fs', np.nan))
        Fb_val = float(flux_fit.get('Fb', np.nan))
        blend_frac = float(flux_fit.get('blend_fraction', np.nan))
        if np.isfinite(blend_frac):
            blend_lines.append(rf'$f_\mathrm{{blend}}={blend_frac:.2f}$')
        if np.isfinite(Fs_val) and np.isfinite(Fb_val):
            blend_lines.append(rf'$F_s/F_b={Fs_val:.2f}/{Fb_val:.2f}$')

    info_lines = [
        rf'$\mathrm{{best}}=\mathrm{{{_mtxt(summary.get("best_model"))}}}$',
        rf'$\mathrm{{ok}}=\mathrm{{{_mtxt(summary.get("fit_ok"))}}}$',
        rf'$\mathrm{{mode}}=\mathrm{{{_mtxt(summary.get("selection_mode"))}}}$',
        *chi2nu_lines,
        *dbic_lines,
        *blend_lines,
    ]
    # Stats / diagnostics box: figure-level panel outside the axes.
    _info_fs = 7.0
    fig.text(
        0.855,
        0.54,
        '\n'.join(info_lines),
        transform=fig.transFigure,
        fontsize=_info_fs,
        ha='left',
        va='center',
        zorder=1,
        bbox={'facecolor': 'white', 'alpha': 0.92, 'edgecolor': '0.8', 'boxstyle': 'round,pad=0.25'},
    )
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        leg = ax.legend(
            handles,
            labels,
            loc='lower right',
            fontsize=_info_fs,
            framealpha=0.92,
            fancybox=True,
            borderpad=0.35,
            labelspacing=0.35,
            handlelength=1.6,
            handletextpad=0.5,
            borderaxespad=0.35,
        )
        leg.set_zorder(_z_model_curves + 1)
    return fig, (ax, ax_zoom, ax_res)



from malca.vetting import (
    MICROLENS_CACHE_DIR,
    fetch_microlensing_event_catalog,
    _safe_text,
)

EXTERNAL_MICROLENS_RADIUS_ARCSEC = 2.0
GAIA_ALERT_RADIUS_ARCSEC = 2.0
REFRESH_EXTERNAL_MICROLENS_TABLES = False
RUN_GAIA_ALERT_LOOKUP = True
MANUAL_OGLE_EWS_CSV_CANDIDATES = [
    REPO_ROOT / 'input' / 'ogle-ews-220326.csv',
    REPO_ROOT / 'input' / 'ogle_ews_220326.csv',
]
MANUAL_OGLE_EWS_CSV_PATH = next((path for path in MANUAL_OGLE_EWS_CSV_CANDIDATES if path.exists()), MANUAL_OGLE_EWS_CSV_CANDIDATES[-1])
MICROLENS_SOURCE_PREFIX = {
    'OGLE-EWS-220326': 'ogle_ews_220326',
    'OGLE-EWS': 'ogle_ews',
    'KMTNet': 'kmtnet',
    'MOA': 'moa',
}
MICROLENS_SOURCE_PRIORITY = {
    'OGLE-EWS-220326': 0,
    'OGLE-EWS': 1,
    'KMTNet': 2,
    'MOA': 3,
}
EXTERNAL_MATCH_DETAIL_COLS = [
    'candidate_id',
    'source',
    'event_id',
    'alias',
    'sep_arcsec',
    'catalog_t0_time_system',
    'catalog_t0_jd',
    'delta_t0_days',
    'catalog_tE_days',
    'catalog_tE_kind',
    'delta_tE_days',
    'delta_log_tE',
    'status',
    'event_year',
    'source_url',
    'match_rank',
    'source_match_rank',
]


def signed_log10_series(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors='coerce')
    return np.sign(values) * np.log10(1.0 + np.abs(values))


def _external_cache_path(name: str) -> Path:
    root = Path(MICROLENS_CACHE_DIR).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    return root / name


def _empty_external_match_summary(candidate_ids: pd.Series | list[str]) -> pd.DataFrame:
    candidate_ids = [str(cid) for cid in candidate_ids]
    df = pd.DataFrame({'candidate_id': candidate_ids})
    defaults: dict[str, object] = {
        'external_known_microlens': False,
        'external_best_microlens_source': '',
        'external_best_microlens_event_id': '',
        'external_best_microlens_alias': '',
        'external_best_microlens_sep_arcsec': np.nan,
        'external_best_microlens_catalog_t0_jd': np.nan,
        'external_best_microlens_catalog_t0_time_system': '',
        'external_best_microlens_delta_t0_days': np.nan,
        'external_best_microlens_catalog_tE_days': np.nan,
        'external_best_microlens_catalog_tE_kind': '',
        'external_best_microlens_delta_tE_days': np.nan,
        'external_best_microlens_delta_log_tE': np.nan,
        'external_best_microlens_status': '',
        'external_best_microlens_event_year': np.nan,
        'external_best_microlens_source_url': '',
    }
    per_source_defaults: dict[str, object] = {}
    for prefix in MICROLENS_SOURCE_PREFIX.values():
        per_source_defaults.update({
            f'{prefix}_event_id': '',
            f'{prefix}_alias': '',
            f'{prefix}_sep_arcsec': np.nan,
            f'{prefix}_catalog_t0_jd': np.nan,
            f'{prefix}_catalog_t0_time_system': '',
            f'{prefix}_delta_t0_days': np.nan,
            f'{prefix}_catalog_tE_days': np.nan,
            f'{prefix}_catalog_tE_kind': '',
            f'{prefix}_delta_tE_days': np.nan,
            f'{prefix}_delta_log_tE': np.nan,
            f'{prefix}_status': '',
            f'{prefix}_source_url': '',
        })
    for col, default in {**defaults, **per_source_defaults}.items():
        df[col] = default
    return df


def _empty_gaia_alert_summary(candidate_ids: pd.Series | list[str]) -> pd.DataFrame:
    candidate_ids = [str(cid) for cid in candidate_ids]
    df = pd.DataFrame({'candidate_id': candidate_ids})
    df['gaia_alert_name'] = ''
    df['gaia_alert_class'] = ''
    df['gaia_alert_sep_arcsec'] = np.nan
    df['gaia_alert_microlens_like'] = False
    df['gaia_alert_lookup_status'] = 'not_run'
    return df


def _normalize_event_id(event_raw: object, prefix: str) -> str:
    text = _safe_text(event_raw)
    if not text:
        return ''
    if text.upper().startswith(prefix.upper() + '-'):
        return text
    return f'{prefix}-{text}'










def build_external_microlens_catalog(*, force_refresh: bool = False) -> pd.DataFrame:
    return fetch_microlensing_event_catalog(show_tqdm=True)


def _abs_diff(left: pd.Series, right: pd.Series) -> pd.Series:
    left = pd.to_numeric(left, errors='coerce')
    right = pd.to_numeric(right, errors='coerce')
    return (left - right).abs()


def _delta_log_te(left: pd.Series, right: pd.Series) -> pd.Series:
    left = pd.to_numeric(left, errors='coerce')
    right = pd.to_numeric(right, errors='coerce')
    valid = (left > 0.0) & (right > 0.0)
    out = pd.Series(np.nan, index=left.index, dtype=float)
    out.loc[valid] = np.abs(np.log10(left.loc[valid] / right.loc[valid]))
    return out


def crossmatch_external_microlensing(master_table: pd.DataFrame, *, radius_arcsec: float, force_refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_df = _empty_external_match_summary(master_table['candidate_id'])
    catalog = build_external_microlens_catalog(force_refresh=force_refresh)
    if catalog.empty:
        return summary_df, pd.DataFrame(columns=EXTERNAL_MATCH_DETAIL_COLS)

    candidate_cols = ['candidate_id', 'ra_deg', 'dec_deg', 'fit_t0_jd', 'raw_paczynski_tE_days']
    candidate_table = master_table[candidate_cols].copy()
    valid_candidates = candidate_table['ra_deg'].notna() & candidate_table['dec_deg'].notna()
    valid_catalog = catalog['ra'].notna() & catalog['dec'].notna()
    if not valid_candidates.any() or not valid_catalog.any():
        return summary_df, pd.DataFrame(columns=EXTERNAL_MATCH_DETAIL_COLS)

    cand_valid = candidate_table.loc[valid_candidates].reset_index(drop=True)
    cat_valid = catalog.loc[valid_catalog].reset_index(drop=True)

    cand_coords = SkyCoord(ra=cand_valid['ra_deg'].to_numpy(dtype=float), dec=cand_valid['dec_deg'].to_numpy(dtype=float), unit='deg')
    cat_coords = SkyCoord(ra=cat_valid['ra'].to_numpy(dtype=float), dec=cat_valid['dec'].to_numpy(dtype=float), unit='deg')
    idx_cat, idx_cand, sep2d, _ = cand_coords.search_around_sky(cat_coords, radius_arcsec * u.arcsec)
    if len(idx_cand) == 0:
        return summary_df, pd.DataFrame(columns=EXTERNAL_MATCH_DETAIL_COLS)

    matches = cat_valid.iloc[np.asarray(idx_cat, dtype=int)].copy().reset_index(drop=True)
    matched_candidates = cand_valid.iloc[np.asarray(idx_cand, dtype=int)].reset_index(drop=True)
    matches['candidate_id'] = matched_candidates['candidate_id'].astype(str)
    matches['candidate_fit_t0_jd'] = matched_candidates['fit_t0_jd'].to_numpy(dtype=float)
    matches['candidate_fit_tE_days'] = matched_candidates['raw_paczynski_tE_days'].to_numpy(dtype=float)
    matches['sep_arcsec'] = np.asarray(sep2d.arcsec, dtype=float)
    matches['delta_t0_days'] = _abs_diff(matches['candidate_fit_t0_jd'], matches['catalog_t0_jd'])
    matches['delta_tE_days'] = _abs_diff(matches['candidate_fit_tE_days'], matches['catalog_tE_days'])
    matches['delta_log_tE'] = _delta_log_te(matches['candidate_fit_tE_days'], matches['catalog_tE_days'])
    matches['source_rank'] = matches['source'].map(MICROLENS_SOURCE_PRIORITY).fillna(999).astype(int)
    matches['t0_missing_rank'] = matches['delta_t0_days'].isna().astype(int)
    matches['tE_missing_rank'] = matches['delta_log_tE'].isna().astype(int)
    matches = matches.sort_values(
        ['candidate_id', 'sep_arcsec', 't0_missing_rank', 'delta_t0_days', 'tE_missing_rank', 'delta_log_tE', 'source_rank', 'event_id'],
        na_position='last',
    ).reset_index(drop=True)
    matches['match_rank'] = matches.groupby('candidate_id').cumcount() + 1
    matches['source_match_rank'] = matches.groupby(['candidate_id', 'source']).cumcount() + 1

    summary_rows: list[dict[str, object]] = []
    for candidate_id, group in matches.groupby('candidate_id', sort=False):
        row = _empty_external_match_summary([candidate_id]).iloc[0].to_dict()
        row['candidate_id'] = str(candidate_id)
        row['external_known_microlens'] = True
        best = group.iloc[0]
        row.update({
            'external_best_microlens_source': _safe_text(best.get('source')),
            'external_best_microlens_event_id': _safe_text(best.get('event_id')),
            'external_best_microlens_alias': _safe_text(best.get('alias')),
            'external_best_microlens_sep_arcsec': float(best.get('sep_arcsec')) if pd.notna(best.get('sep_arcsec')) else np.nan,
            'external_best_microlens_catalog_t0_jd': float(best.get('catalog_t0_jd')) if pd.notna(best.get('catalog_t0_jd')) else np.nan,
            'external_best_microlens_catalog_t0_time_system': _safe_text(best.get('catalog_t0_time_system')),
            'external_best_microlens_delta_t0_days': float(best.get('delta_t0_days')) if pd.notna(best.get('delta_t0_days')) else np.nan,
            'external_best_microlens_catalog_tE_days': float(best.get('catalog_tE_days')) if pd.notna(best.get('catalog_tE_days')) else np.nan,
            'external_best_microlens_catalog_tE_kind': _safe_text(best.get('catalog_tE_kind')),
            'external_best_microlens_delta_tE_days': float(best.get('delta_tE_days')) if pd.notna(best.get('delta_tE_days')) else np.nan,
            'external_best_microlens_delta_log_tE': float(best.get('delta_log_tE')) if pd.notna(best.get('delta_log_tE')) else np.nan,
            'external_best_microlens_status': _safe_text(best.get('status')),
            'external_best_microlens_event_year': float(best.get('event_year')) if pd.notna(best.get('event_year')) else np.nan,
            'external_best_microlens_source_url': _safe_text(best.get('source_url')),
        })
        for source_name, prefix in MICROLENS_SOURCE_PREFIX.items():
            source_group = group.loc[group['source'] == source_name]
            if source_group.empty:
                continue
            match = source_group.iloc[0]
            row.update({
                f'{prefix}_event_id': _safe_text(match.get('event_id')),
                f'{prefix}_alias': _safe_text(match.get('alias')),
                f'{prefix}_sep_arcsec': float(match.get('sep_arcsec')) if pd.notna(match.get('sep_arcsec')) else np.nan,
                f'{prefix}_catalog_t0_jd': float(match.get('catalog_t0_jd')) if pd.notna(match.get('catalog_t0_jd')) else np.nan,
                f'{prefix}_catalog_t0_time_system': _safe_text(match.get('catalog_t0_time_system')),
                f'{prefix}_delta_t0_days': float(match.get('delta_t0_days')) if pd.notna(match.get('delta_t0_days')) else np.nan,
                f'{prefix}_catalog_tE_days': float(match.get('catalog_tE_days')) if pd.notna(match.get('catalog_tE_days')) else np.nan,
                f'{prefix}_catalog_tE_kind': _safe_text(match.get('catalog_tE_kind')),
                f'{prefix}_delta_tE_days': float(match.get('delta_tE_days')) if pd.notna(match.get('delta_tE_days')) else np.nan,
                f'{prefix}_delta_log_tE': float(match.get('delta_log_tE')) if pd.notna(match.get('delta_log_tE')) else np.nan,
                f'{prefix}_status': _safe_text(match.get('status')),
                f'{prefix}_source_url': _safe_text(match.get('source_url')),
            })
        summary_rows.append(row)

    summary_df = _empty_external_match_summary(master_table['candidate_id']).merge(
        pd.DataFrame(summary_rows),
        on='candidate_id',
        how='left',
        suffixes=('', '_match'),
    )
    for col in summary_df.columns:
        if col.endswith('_match'):
            base_col = col[:-6]
            summary_df[base_col] = summary_df[col].combine_first(summary_df[base_col])
            summary_df = summary_df.drop(columns=[col])

    detail_df = matches[EXTERNAL_MATCH_DETAIL_COLS].copy()
    return summary_df, detail_df


def crossmatch_external_gaia_alerts(master_table: pd.DataFrame, *, radius_arcsec: float, run_lookup: bool) -> pd.DataFrame:
    summary_df = _empty_gaia_alert_summary(master_table['candidate_id'])
    if not run_lookup:
        summary_df['gaia_alert_lookup_status'] = 'skipped'
        return summary_df
    try:
        from malca.ltv.crossmatch import crossmatch_gaia_alerts
    except Exception as exc:
        summary_df['gaia_alert_lookup_status'] = f'import_failed:{type(exc).__name__}'
        return summary_df

    try:
        gaia_df = crossmatch_gaia_alerts(
            master_table[['candidate_id', 'ra_deg', 'dec_deg']].copy(),
            ra_column='ra_deg',
            dec_column='dec_deg',
            match_radius_arcsec=radius_arcsec,
            verbose=False,
        )
    except Exception as exc:
        summary_df['gaia_alert_lookup_status'] = f'query_failed:{type(exc).__name__}'
        return summary_df

    out = summary_df.merge(
        gaia_df[['candidate_id', 'gaia_alert_name', 'gaia_alert_class', 'gaia_alert_sep_arcsec']].copy(),
        on='candidate_id',
        how='left',
        suffixes=('', '_new'),
    )
    for col in ('gaia_alert_name', 'gaia_alert_class', 'gaia_alert_sep_arcsec'):
        out[col] = out[f'{col}_new'].combine_first(out[col])
        out = out.drop(columns=[f'{col}_new'])
    out['gaia_alert_name'] = out['gaia_alert_name'].fillna('')
    out['gaia_alert_class'] = out['gaia_alert_class'].fillna('')
    out['gaia_alert_microlens_like'] = out['gaia_alert_class'].astype(str).str.contains(r'ULENS|MICROLENS|LENS', case=False, regex=True, na=False)
    out['gaia_alert_lookup_status'] = 'queried'
    return out


def run_microlensing_pipeline(
    *,
    repo_root: Path | None = None,
    db_path: Path | None = None,
    plot_lc: bool = False,
    crossmatch: bool = False,
    crossmatch_dust: bool = True,
    crossmatch_unwise: bool = True,
    crossmatch_neowise: bool = True,
    crossmatch_starhorse: str | None = "tap",
    crossmatch_gaia_catalog: Path | None = None,
    crossmatch_vsx_csv: Path | None = None,
    crossmatch_method: str = 'tap',
    crossmatch_cache_dir: Path | None = None,
    crossmatch_no_cache: bool = False,
    crossmatch_refresh_cache: bool = False,
    fit_workers: int = 1,
) -> None:
    """Run March 18 fits, CSV + sky map, jumps-14 cohort; optional LC PDFs and characterize/vet crossmatch.

    Subjective **bad** LCs (:data:`VISUAL_INSPECTION_BAD_IDS`) are never fitted or written; **probably_bad**
    rows are kept and flagged via ``visual_inspection_subjective_*`` columns.
    """
    show_progress = True
    global REPO_ROOT, DB_PATH, MICROLENSING_OUTPUT_ROOT, MICROLENSING_FIT_PDF_DIR

    rr = (repo_root or find_repo_root(_repo_search_start())).resolve()
    REPO_ROOT = rr
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    DB_PATH = (
        Path(db_path).expanduser().resolve()
        if db_path is not None
        else (REPO_ROOT / "output" / "runs" / "runs_march18_bundle_all" / "review" / "review.db")
    )
    MICROLENSING_OUTPUT_ROOT = (REPO_ROOT / "output" / "microlensing").resolve()
    MICROLENSING_FIT_PDF_DIR = (MICROLENSING_OUTPUT_ROOT / "fit_pdfs").resolve()
    MICROLENSING_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    cache_root = (
        Path(crossmatch_cache_dir).expanduser().resolve()
        if crossmatch_cache_dir is not None
        else (MICROLENSING_OUTPUT_ROOT / "cache").resolve()
    )

    if show_progress:
        print("[1/6] Fitting March 18 review candidates …", flush=True)
    results_df, fit_results = fit_microlensing_candidates(
        DB_PATH,
        candidate_ids=MARCH18_CANDIDATE_IDS,
        fit_workers=fit_workers,
        prefer_g_band=True,
        show_progress=show_progress,
        progress_desc="March 18 cohort",
    )

    if show_progress:
        print("[2/6] Cross-matching external microlensing catalogs …", flush=True)
    external_summary_df, external_match_details_df = crossmatch_external_microlensing(
        results_df,
        radius_arcsec=EXTERNAL_MICROLENS_RADIUS_ARCSEC,
        force_refresh=REFRESH_EXTERNAL_MICROLENS_TABLES,
    )
    if show_progress:
        print("[3/6] Gaia Alerts cross-match …", flush=True)
    gaia_alert_summary_df = crossmatch_external_gaia_alerts(
        results_df,
        radius_arcsec=GAIA_ALERT_RADIUS_ARCSEC,
        run_lookup=RUN_GAIA_ALERT_LOOKUP,
    )

    if show_progress:
        print("[4/6] Building master table & summaries …", flush=True)
    master_df = results_df.copy()
    master_df = master_df.merge(external_summary_df, on='candidate_id', how='left')
    master_df = master_df.merge(gaia_alert_summary_df, on='candidate_id', how='left')


    asassn_ml_path = REPO_ROOT / 'input' / 'asas_sn_microlens.csv'
    master_df['asassn_ml_match'] = False
    master_df['asassn_ml_name'] = ''
    master_df['asassn_ml_sep_arcsec'] = np.nan
    if asassn_ml_path.exists():
        asml = pd.read_csv(asassn_ml_path, header=None)
        asml_ra = pd.to_numeric(asml.iloc[:, 12], errors='coerce')
        asml_dec = pd.to_numeric(asml.iloc[:, 13], errors='coerce')
        asml_name = asml.iloc[:, 0].replace('', pd.NA).combine_first(asml.iloc[:, 1]).fillna('unknown')
        valid = asml_ra.notna() & asml_dec.notna()
        if valid.any():
            cat_coords = SkyCoord(ra=asml_ra[valid].values, dec=asml_dec[valid].values, unit='deg')
            asml_names = asml_name[valid].values
        
            master_coords = SkyCoord(ra=master_df['ra_deg'].values, dec=master_df['dec_deg'].values, unit='deg')
            idx_cat, sep2d, _ = master_coords.match_to_catalog_sky(cat_coords)
            match_mask = sep2d <= 5.0 * u.arcsec
        
            if len(idx_cat) > 0:
                master_df.loc[match_mask, 'asassn_ml_match'] = True
                master_df.loc[match_mask, 'asassn_ml_name'] = asml_names[idx_cat][match_mask]
                master_df.loc[match_mask, 'asassn_ml_sep_arcsec'] = sep2d.arcsec[match_mask]
    master_df['external_known_microlens'] = master_df['external_known_microlens'].fillna(False).astype(bool)
    master_df['gaia_alert_microlens_like'] = master_df['gaia_alert_microlens_like'].fillna(False).astype(bool)
    master_df['external_known_microlens'] = master_df['external_known_microlens'] | master_df['gaia_alert_microlens_like'] | master_df['asassn_ml_match']
    master_df['log10_delta_bic_vs_flat'] = signed_log10_series(master_df['delta_bic_vs_flat'])
    master_df['log10_delta_bic_vs_gaussian'] = signed_log10_series(master_df['delta_bic_vs_gaussian'])
    master_df['log10_delta_bic_vs_fred'] = signed_log10_series(master_df['delta_bic_vs_fred'])
    master_df['log10_delta_bic_vs_best_alt'] = signed_log10_series(master_df['delta_bic_vs_best_alt'])
    master_df['vizier_url'] = [
        _vizier_cone_search_url_deg(r, d) for r, d in zip(master_df['ra_deg'], master_df['dec_deg'])
    ]
    master_df = _add_milky_way_line_of_sight_columns(master_df)

    summary_lookup_cols = [col for col in master_df.columns if col not in results_df.columns]
    summary_lookup = master_df.set_index('candidate_id')[summary_lookup_cols].to_dict('index')
    for result in fit_results:
        result['summary'].update(summary_lookup.get(result['summary']['candidate_id'], {}))

    master_cols = [
        'candidate_id',
        'asassn_source_id',
        'asassn_var_name',
        'asassn_var_type',
        'gaia_dr3_source_id',
        'ra_deg',
        'dec_deg',
        'gal_l_deg',
        'gal_b_deg',
        'mw_line_of_sight_region',
        'vizier_url',
        'fit_t0_time_system',
        'fit_t0_jd_minus_2450000',
        'fit_t0_jd',
        'raw_paczynski_tE_days',
        'parallax_attempted',
        'parallax_fit_ok',
        'parallax_preferred',
        'parallax_status',
        'parallax_warning',
        'parallax_best_branch',
        'parallax_delta_bic',
        'parallax_branch_delta_bic',
        'parallax_best_t0_jd_minus_2450000',
        'parallax_best_tE_days',
        'parallax_best_u0',
        'parallax_best_piE_N',
        'parallax_best_piE_E',
        'parallax_best_piE',
        'parallax_best_chi2',
        'parallax_best_reduced_chi2',
        'parallax_best_bic',
        'parallax_best_acceptance_rate',
        'parallax_best_n_samples',
        'parallax_pos_t0_jd_minus_2450000',
        'parallax_pos_tE_days',
        'parallax_pos_u0',
        'parallax_pos_piE_N',
        'parallax_pos_piE_E',
        'parallax_pos_piE',
        'parallax_pos_chi2',
        'parallax_neg_t0_jd_minus_2450000',
        'parallax_neg_tE_days',
        'parallax_neg_u0',
        'parallax_neg_piE_N',
        'parallax_neg_piE_E',
        'parallax_neg_piE',
        'parallax_neg_chi2',
        'peak_window_time_system',
        'peak_window_start_jd_minus_2450000',
        'peak_window_end_jd_minus_2450000',
        'peak_window_start_jd',
        'peak_window_end_jd',
        'fit_ok',
        'selection_mode',
        'seed_method',
        'best_model',
        'fit_reduced_chi2',
        'paczynski_reduced_chi2',
        'log10_delta_bic_vs_flat',
        'log10_delta_bic_vs_gaussian',
        'log10_delta_bic_vs_fred',
        'log10_delta_bic_vs_best_alt',
        'external_known_microlens',
        'external_best_microlens_source',
        'external_best_microlens_event_id',
        'external_best_microlens_sep_arcsec',
        'external_best_microlens_catalog_t0_time_system',
        'external_best_microlens_catalog_t0_jd',
        'external_best_microlens_delta_t0_days',
        'external_best_microlens_catalog_tE_days',
        'external_best_microlens_catalog_tE_kind',
        'external_best_microlens_delta_tE_days',
        'external_best_microlens_delta_log_tE',
        'gaia_alert_name',
        'gaia_alert_class',
        'gaia_alert_sep_arcsec',
        'gaia_alert_microlens_like',
        'gaia_alert_lookup_status',
        'asassn_ml_match',
        'asassn_ml_name',
        'asassn_ml_sep_arcsec',
        'ogle_ews_event_id',
        'ogle_ews_sep_arcsec',
        'ogle_ews_catalog_t0_time_system',
        'ogle_ews_catalog_t0_jd',
        'ogle_ews_delta_t0_days',
        'ogle_ews_catalog_tE_days',
        'ogle_ews_catalog_tE_kind',
        'ogle_ews_delta_tE_days',
        'ogle_ews_delta_log_tE',
        'kmtnet_event_id',
        'kmtnet_sep_arcsec',
        'kmtnet_catalog_t0_time_system',
        'kmtnet_catalog_t0_jd',
        'kmtnet_delta_t0_days',
        'kmtnet_catalog_tE_days',
        'kmtnet_catalog_tE_kind',
        'kmtnet_delta_tE_days',
        'kmtnet_delta_log_tE',
        'moa_event_id',
        'moa_sep_arcsec',
        'moa_catalog_t0_time_system',
        'moa_catalog_t0_jd',
        'moa_delta_t0_days',
        'moa_catalog_tE_days',
        'moa_catalog_tE_kind',
        'moa_delta_tE_days',
        'moa_delta_log_tE',
        'microlens_match',
        'microlens_catalog',
        'microlens_name',
        'microlens_alt_name',
        'microlens_te_days',
        'microlens_sep_arcsec',
        'vetting_likely_known',
        'catalog_source',
        'nearest_simbad_object',
        'simbad_otype',
        'simbad_sep_arcsec',
        'nearest_vsx_object',
        'vsx_class',
        'vsx_sep_arcsec',
        'paczynski_tau_coverage_score',
        'paczynski_coverage_n_bins_hit',
        'paczynski_coverage_n_bins',
        'paczynski_coverage_max_weighted_gap',
        'paczynski_coverage_frac_points_in_tau_window',
        'fit_warning',
    ]
    raw_delta_bic_cols = {'delta_bic_vs_flat', 'delta_bic_vs_gaussian', 'delta_bic_vs_fred', 'delta_bic_vs_best_alt'}
    ordered_master_cols = [
        *master_cols,
        *[col for col in master_df.columns if col not in master_cols and col not in raw_delta_bic_cols],
    ]

    _preview_dataframe('master_df (ordered columns)', master_df[ordered_master_cols])

    external_match_details_df = external_match_details_df.sort_values(['candidate_id', 'match_rank', 'source_match_rank']).reset_index(drop=True)
    _preview_dataframe('external_match_details', external_match_details_df[EXTERNAL_MATCH_DETAIL_COLS] if not external_match_details_df.empty else pd.DataFrame(columns=EXTERNAL_MATCH_DETAIL_COLS))

    gaia_alert_matches_df = master_df.loc[
        master_df['gaia_alert_name'].fillna('').astype(str) != '',
        ['candidate_id', 'gaia_alert_name', 'gaia_alert_class', 'gaia_alert_sep_arcsec', 'gaia_alert_microlens_like', 'gaia_alert_lookup_status'],
    ].copy()
    _preview_dataframe('gaia_alert_matches', gaia_alert_matches_df if not gaia_alert_matches_df.empty else pd.DataFrame(columns=['candidate_id', 'gaia_alert_name', 'gaia_alert_class', 'gaia_alert_sep_arcsec', 'gaia_alert_microlens_like', 'gaia_alert_lookup_status']))




    if show_progress:
        if plot_lc:
            print(
                "[5/6] LC fit PDFs (March 18 + jumps-14) written at end after crossmatch table (use --plot-lc) …",
                flush=True,
            )
        else:
            print("[5/6] Skipping LC fit PDFs (pass --plot-lc to write) …", flush=True)




    if show_progress:
        print("[5b] Jumps-14 single-bucket cohort (plots → DB resolve → fits) …", flush=True)
    # Jumps 14–14.5 uncategorized single-bucket cohort.
    JUMPS14_PLOTS_BASE = REPO_ROOT / 'output' / 'runs' / 'plots' / 'jumps_14_14.5_uncategorized'
    SINGLE_BUCKETS = ('single-fred', 'single-paczysnki', 'single-unclear')

    collected: set[str] = set()
    for sub in SINGLE_BUCKETS:
        bucket = JUMPS14_PLOTS_BASE / sub
        if not bucket.is_dir():
            print(f'Missing plot dir (skip): {bucket}')
            continue
        for f in bucket.rglob('*.png'):
            if f.name.endswith('_candidate.png'):
                collected.add(f.name[: -len('_candidate.png')])

    march18_id_set = {str(x) for x in MARCH18_CANDIDATE_IDS}
    target_ids = sorted(collected - march18_id_set)
    tb = len(target_ids)
    target_ids = exclude_visual_inspection_bad_ids(target_ids)
    if tb != len(target_ids) and show_progress:
        print(
            f"  Excluded {tb - len(target_ids)} plot-bucket ID(s) in VISUAL_INSPECTION_BAD_IDS (subjective bad LC).",
            flush=True,
        )
    print(
        f'Plot bucket IDs: {len(collected)} unique; after excluding March 18: {tb} to resolve; '
        f'after excluding subjective BAD: {len(target_ids)}.',
        flush=True,
    )

    runs_root = (REPO_ROOT / 'output' / 'runs').resolve()
    all_dbs = [
        db_path
        for db_path in runs_root.rglob('review.db')
        if 'plots' not in db_path.relative_to(runs_root).parts
    ]

    id_to_db: dict[str, Path] = {}
    for db_path in all_dbs:
        try:
            with sqlite3.connect(db_path) as conn:
                db_cids = set(
                    pd.read_sql_query('SELECT candidate_id FROM candidates', conn)['candidate_id'].astype(str),
                )
            for cid in target_ids:
                if cid in db_cids and cid not in id_to_db:
                    id_to_db[cid] = db_path
        except Exception as exc:
            print(f'Skipping {db_path}: {exc}')

    missing = set(target_ids) - set(id_to_db)
    if missing:
        sample = sorted(missing)[:20]
        print(f'Warning: {len(missing)} target IDs not in any review.db (first 20): {sample}')

    # Fallback: if a candidate has a light-curve in some run bundle but is absent from all `review.db`,
    # fit using the light-curve directly.
    lc_only_single_results_df_list: list[pd.DataFrame] = []
    lc_only_fit_results: list[dict[str, object]] = []
    if missing:
        if show_progress:
            print(f"Attempting light-curve-only fits for {len(missing)} missing candidate(s) …", flush=True)

        lightcurve_bundle_dirs = [
            p for p in runs_root.rglob('bundle_assets/lightcurves')
            if p.is_dir()
        ]

        # Prefer representations that are known to be loadable by `load_lightcurve_df`.
        ext_order = ('dat3', 'dat2', 'dat', 'raw2')
        candidate_id_to_lc_paths: dict[str, list[Path]] = {}

        for cid in sorted(missing):
            lc_paths: list[Path] = []
            # First, check for expected filenames directly in each lightcurve bundle directory
            # (mirrors `resolve_lightcurve_path` which also doesn't recurse).
            for lc_dir in lightcurve_bundle_dirs:
                dir_paths: list[Path] = []
                for ext in ext_order:
                    p = lc_dir / f"{cid}.{ext}"
                    if p.exists():
                        dir_paths.append(p)
                if dir_paths:
                    lc_paths = dir_paths
                    break

            # If direct lookup failed, do a last-resort recursive lookup for this candidate+extension.
            if not lc_paths:
                for lc_dir in lightcurve_bundle_dirs:
                    for ext in ext_order:
                        p = next(lc_dir.rglob(f"{cid}.{ext}"), None)
                        if p is not None:
                            lc_paths = [p]
                            break
                    if lc_paths:
                        break

            if lc_paths:
                candidate_id_to_lc_paths[cid] = lc_paths

        if candidate_id_to_lc_paths:
            # Step 1: recover coordinates from local output/candidates.parquet keyed by asas_sn_id.
            candidate_metadata_by_id = _load_candidate_metadata_from_candidates_parquet(
                list(candidate_id_to_lc_paths.keys()),
            )
            # Step 2 (fallback): for IDs still missing coordinates, try SkyPatrol2 (pyasassn).
            missing_meta_ids = sorted(
                cid for cid in candidate_id_to_lc_paths
                if cid not in candidate_metadata_by_id
            )
            if missing_meta_ids:
                sp_meta = _fetch_candidate_metadata_from_skypatrol2(missing_meta_ids)
                if sp_meta:
                    candidate_metadata_by_id.update(sp_meta)

            # Step 3: conservative Gaia cone-match (1 arcsec) to recover Gaia source_id.
            candidate_metadata_by_id = _enrich_candidate_metadata_with_gaia_ids(
                candidate_metadata_by_id,
                max_sep_arcsec=1.0,
            )

            if show_progress:
                n_with_gaia = sum(
                    1
                    for m in candidate_metadata_by_id.values()
                    if _candidate_id_match_str(m.get('gaia_id'))
                )
                print(
                    f"  Metadata recovered for {len(candidate_metadata_by_id)} / "
                    f"{len(candidate_id_to_lc_paths)} lightcurve-only candidate(s) "
                    f"(local candidates.parquet first, SkyPatrol2 fallback); "
                    f"Gaia IDs recovered for {n_with_gaia}.",
                    flush=True,
                )

            res_df, res_fits = _fit_lightcurve_only_candidates(
                candidate_id_to_lc_paths,
                candidate_metadata_by_id=candidate_metadata_by_id,
                fit_workers=fit_workers,
                prefer_g_band=True,
                show_progress=False,
            )
            if not res_df.empty:
                lc_only_single_results_df_list.append(res_df)
                lc_only_fit_results.extend(res_fits)

        if show_progress:
            still_missing = sorted(set(missing) - set(candidate_id_to_lc_paths))
            if still_missing:
                print(
                    f"  Lightcurve fallback found {len(candidate_id_to_lc_paths)} / {len(missing)} missing; "
                    f"still missing {len(still_missing)} (no light-curve file found).",
                    flush=True,
                )
            else:
                print(
                    f"  Lightcurve fallback found {len(candidate_id_to_lc_paths)} / {len(missing)} missing.",
                    flush=True,
                )

    db_to_ids: dict[Path, list[str]] = {}
    for cid, db_path in id_to_db.items():
        db_to_ids.setdefault(db_path, []).append(cid)
    db_to_ids = {db: sorted(cids) for db, cids in db_to_ids.items()}

    print(f'Resolved {len(id_to_db)} candidates across {len(db_to_ids)} review.db files.')

    jumps14_single_results_df_list: list[pd.DataFrame] = []
    jumps14_fit_results: list[dict[str, object]] = []

    db_loop = sorted(db_to_ids.items(), key=lambda kv: str(kv[0]))
    if show_progress and sys.stderr.isatty():
        db_loop = tqdm(db_loop, desc="Jumps-14 review DBs", unit="db")
    for db_path, cids in db_loop:
        print(f'Fitting {len(cids)} candidates from {db_path} …', flush=True)
        res_df, res_fits = fit_microlensing_candidates(
            db_path,
            candidate_ids=cids,
            fit_workers=fit_workers,
            prefer_g_band=True,
            show_progress=False,
        )
        jumps14_single_results_df_list.append(res_df)
        jumps14_fit_results.extend(res_fits)

    if lc_only_single_results_df_list:
        jumps14_single_results_df_list.extend(lc_only_single_results_df_list)
        jumps14_fit_results.extend(lc_only_fit_results)

    if jumps14_single_results_df_list:
        jumps14_single_results_df = pd.concat(
            jumps14_single_results_df_list, ignore_index=True,
        )
    else:
        jumps14_single_results_df = pd.DataFrame()

    print(f'jumps14_single_results_df rows: {len(jumps14_single_results_df)}')

    if len(jumps14_single_results_df):
        jumps14_single_results_df = jumps14_single_results_df.copy()
        jumps14_single_results_df['vizier_url'] = [
            _vizier_cone_search_url_deg(r, d)
            for r, d in zip(jumps14_single_results_df['ra_deg'], jumps14_single_results_df['dec_deg'])
        ]
        jumps14_single_results_df = _add_milky_way_line_of_sight_columns(jumps14_single_results_df)

    if len(jumps14_single_results_df):
        jumps_aligned = jumps14_single_results_df.reindex(columns=master_df.columns)
        microlensing_table_df = pd.concat([master_df, jumps_aligned], ignore_index=True).drop_duplicates(
            'candidate_id', keep='first'
        )
    else:
        microlensing_table_df = master_df

    if not microlensing_table_df.empty and 'candidate_id' in microlensing_table_df.columns:
        _bn = _visual_inspection_bad_id_norms()
        _cidn = microlensing_table_df['candidate_id'].map(_candidate_id_match_str)
        _drop_mask = _cidn.isin(_bn)
        if _drop_mask.any() and show_progress:
            print(
                f"  Dropping {_drop_mask.sum()} row(s) in VISUAL_INSPECTION_BAD_IDS before crossmatch/export.",
                flush=True,
            )
        microlensing_table_df = microlensing_table_df.loc[~_drop_mask].reset_index(drop=True)

    if crossmatch:
        if show_progress:
            print(
                "[4c] Crossmatch suite: malca.characterize + malca.vetting (not reimplemented here) …",
                flush=True,
            )
        characterize_checkpoint_path: Path | None = None
        vetting_checkpoint_path: Path | None = None
        if not crossmatch_no_cache:
            _cache_ids = sorted({str(cid) for cid in microlensing_table_df['candidate_id'].dropna().astype(str).tolist()})
            _cache_payload = {
                "candidate_ids": _cache_ids,
                "crossmatch_dust": bool(crossmatch_dust),
                "crossmatch_unwise": bool(crossmatch_unwise),
                "crossmatch_neowise": bool(crossmatch_neowise),
                "crossmatch_starhorse": str(crossmatch_starhorse or ""),
                "crossmatch_method": str(crossmatch_method),
                "crossmatch_gaia_catalog": (
                    str(Path(crossmatch_gaia_catalog).expanduser().resolve())
                    if crossmatch_gaia_catalog is not None
                    else ""
                ),
                "crossmatch_vsx_csv": (
                    str(Path(crossmatch_vsx_csv).expanduser().resolve())
                    if crossmatch_vsx_csv is not None
                    else ""
                ),
            }
            _cache_key = hashlib.sha256(
                json.dumps(_cache_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()[:16]
            cache_root.mkdir(parents=True, exist_ok=True)
            characterize_checkpoint_path = cache_root / f"crossmatch_char_{_cache_key}.parquet"
            vetting_checkpoint_path = cache_root / f"crossmatch_vet_{_cache_key}.parquet"
            if crossmatch_refresh_cache:
                for _ckpt in (characterize_checkpoint_path, vetting_checkpoint_path):
                    if _ckpt.exists():
                        _ckpt.unlink()
                if show_progress:
                    print(f"  Refreshed crossmatch cache key={_cache_key}", flush=True)
            elif show_progress:
                print(f"  Crossmatch cache key={_cache_key}", flush=True)
        elif show_progress:
            print("  Crossmatch cache disabled (--crossmatch-no-cache).", flush=True)

        microlensing_table_df = _run_microlensing_crossmatch_enrichment(
            microlensing_table_df,
            repo_root=REPO_ROOT,
            show_progress=show_progress,
            dust=crossmatch_dust,
            unwise=crossmatch_unwise,
            neowise_lc=crossmatch_neowise,
            starhorse=crossmatch_starhorse,
            gaia_catalog=crossmatch_gaia_catalog,
            vsx_crossmatch=crossmatch_vsx_csv,
            vet_method=crossmatch_method,
            characterize_checkpoint_path=characterize_checkpoint_path,
            vetting_checkpoint_path=vetting_checkpoint_path,
        )

    microlensing_table_df = _add_visual_inspection_columns(microlensing_table_df)
    microlensing_table_df = _ensure_microlensing_quality_columns(microlensing_table_df)

    if plot_lc and not microlensing_table_df.empty:
        _inject_microlensing_table_into_summaries(microlensing_table_df, fit_results)
        _inject_microlensing_table_into_summaries(microlensing_table_df, jumps14_fit_results)

    _ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    microlensing_results_path = MICROLENSING_OUTPUT_ROOT / f'microlensing_results_{_ts}.parquet'
    microlensing_table_df.to_parquet(microlensing_results_path, index=False)
    if show_progress:
        print(f"Results table written to {microlensing_results_path}", flush=True)

    microlensing_sky_path = MICROLENSING_OUTPUT_ROOT / f'microlensing_sky_{_ts}.pdf'
    _save_microlensing_full_sky_plot(microlensing_table_df, microlensing_sky_path, dpi=MICROLENSING_FIT_PDF_DPI)
    if show_progress:
        print(f"Full-sky map written to {microlensing_sky_path}", flush=True)

    microlensing_grid_path = MICROLENSING_OUTPUT_ROOT / f'microlensing_grid_{_ts}.pdf'
    _save_microlensing_candidate_grid_plot(
        microlensing_table_df,
        microlensing_grid_path,
        min_tier='Bronze',
        max_candidates=None,
        dpi=MICROLENSING_FIT_PDF_DPI,
        fit_results=fit_results,
        jumps14_fit_results=jumps14_fit_results,
    )
    if show_progress:
        print(f"Candidate grid written to {microlensing_grid_path}", flush=True)

    microlensing_cmd_path = MICROLENSING_OUTPUT_ROOT / f'microlensing_cmd_{_ts}.pdf'
    _save_microlensing_cmd_plot(microlensing_table_df, microlensing_cmd_path, dpi=MICROLENSING_FIT_PDF_DPI)
    if show_progress:
        print(f"CMD summary written to {microlensing_cmd_path}", flush=True)

    if plot_lc:
        MICROLENSING_FIT_PDF_DIR.mkdir(parents=True, exist_ok=True)
        _all_fit_pdf_results = list(fit_results) + list(jumps14_fit_results)
        jpdf_iter = _all_fit_pdf_results
        if show_progress and sys.stderr.isatty():
            jpdf_iter = tqdm(_all_fit_pdf_results, desc="LC fit PDFs", unit="pdf")
        for result in jpdf_iter:
            summary = result['summary']
            fig, _axes = plot_candidate_fit(result)
            out_pdf = MICROLENSING_FIT_PDF_DIR / f'{_microlensing_fit_pdf_stem(summary)}.pdf'
            fig.savefig(
                out_pdf,
                bbox_inches='tight',
                format='pdf',
                dpi=MICROLENSING_FIT_PDF_DPI,
                facecolor=fig.get_facecolor(),
                edgecolor='none',
            )
            plt.close(fig)
        if show_progress:
            print(
                f'Saved {len(fit_results)} March 18 + {len(jumps14_fit_results)} jumps-14 '
                f'LC fit PDFs under {MICROLENSING_FIT_PDF_DIR}',
                flush=True,
            )
    elif show_progress and (len(fit_results) or len(jumps14_fit_results)):
        print('Skipping LC fit PDFs (--plot-lc not set).', flush=True)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Microlensing analysis: March 18 cohort fits, catalog cross-matches, "
            "one CSV plus sky/grid/CMD PDFs under output/microlensing/. "
            "Per-candidate LC PDFs are written only with --plot-lc."
        ),
    )
    p.add_argument(
        "--plot-lc",
        action="store_true",
        help="Write per-candidate light-curve fit PDFs under output/microlensing/fit_pdfs/.",
    )
    p.add_argument(
        "--crossmatch",
        action="store_true",
        help=(
            "After fits, run malca.characterize_candidates_df + malca.vet_candidates (Gaia context, SIMBAD, "
            "ZTF/ASAS-SN vars, TNS, ALeRCE, eROSITA, …; ATLAS forced photometry is not used); merges into the results Parquet. "
            "Requires local Gaia parquet for full characterize (see --crossmatch-gaia-catalog)."
        ),
    )
    p.add_argument(
        "--crossmatch-off",
        action="store_true",
        help="Disable crossmatch enrichment (overrides --crossmatch and all --crossmatch-* toggles).",
    )
    g_x = p.add_argument_group("Crossmatch (only if --crossmatch)")
    g_x.add_argument(
        "--crossmatch-dust",
        action="store_true",
        default=True,
        help="Enable dustmaps3d in characterize (default: on).",
    )
    g_x.add_argument(
        "--crossmatch-unwise",
        action="store_true",
        default=True,
        help="Enable unWISE/unTimely variability in characterize (default: on).",
    )
    g_x.add_argument(
        "--crossmatch-no-unwise",
        action="store_true",
        default=False,
        help="Disable unWISE/unTimely variability even when --crossmatch is enabled.",
    )
    g_x.add_argument(
        "--crossmatch-neowise",
        action="store_true",
        default=True,
        help="Enable NEOWISE LC fetch in vetting (slow) (default: on).",
    )
    g_x.add_argument(
        "--crossmatch-no-neowise",
        action="store_true",
        default=False,
        help="Disable NEOWISE LC fetch even when --crossmatch is enabled.",
    )
    g_x.add_argument(
        "--crossmatch-starhorse",
        nargs="?",
        const="tap",
        default="tap",
        metavar="MODE",
        help="StarHorse in characterize: use flag alone or pass tap (default: tap).",
    )
    g_x.add_argument(
        "--crossmatch-gaia-catalog",
        type=Path,
        default=None,
        help="Gaia DR3 parquet for characterize (default: <repo>/output/cache/catalogs/gaia/gaia_dr3_crossmatched.parquet).",
    )
    g_x.add_argument(
        "--crossmatch-vsx",
        dest="crossmatch_vsx_csv",
        type=Path,
        default=None,
        help="ASAS-SN x VSX crossmatch Parquet for characterize (default: malca.config VSX_CROSSMATCH_PATH under repo).",
    )
    g_x.add_argument(
        "--crossmatch-method",
        choices=("tap", "xmatch"),
        default="tap",
        help="Vetting backend for TAP-based catalog steps (default: tap).",
    )
    g_x.add_argument(
        "--crossmatch-cache-dir",
        type=Path,
        default=None,
        help="Crossmatch checkpoint cache dir (default: <repo>/output/microlensing/cache).",
    )
    g_x.add_argument(
        "--crossmatch-no-cache",
        action="store_true",
        help="Disable crossmatch checkpoint cache read/write.",
    )
    g_x.add_argument(
        "--crossmatch-refresh-cache",
        action="store_true",
        help="Ignore existing crossmatch checkpoint cache for this run key and recompute.",
    )
    p.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root (default: walk from cwd for pyproject.toml).",
    )
    p.add_argument(
        "--db-path",
        type=Path,
        default=None,
        help="Override path to review.db (default: runs_march18_bundle_all/review/review.db).",
    )
    p.add_argument(
        "--fit-workers",
        type=int,
        default=1,
        help="Number of process workers for per-candidate fitting (default: 1, i.e. serial).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    if getattr(args, "crossmatch_off", False):
        # Explicit override requested by user.
        args.crossmatch = False
        args.crossmatch_dust = False
        args.crossmatch_unwise = False
        args.crossmatch_neowise = False
        args.crossmatch_starhorse = None
    else:
        # Allow stacking of individual "disable X" flags.
        if getattr(args, "crossmatch_no_unwise", False):
            args.crossmatch_unwise = False
        if getattr(args, "crossmatch_no_neowise", False):
            args.crossmatch_neowise = False
    run_microlensing_pipeline(
        repo_root=args.repo_root,
        db_path=args.db_path,
        plot_lc=args.plot_lc,
        crossmatch=args.crossmatch,
        crossmatch_dust=args.crossmatch_dust,
        crossmatch_unwise=args.crossmatch_unwise,
        crossmatch_neowise=args.crossmatch_neowise,
        crossmatch_starhorse=args.crossmatch_starhorse,
        crossmatch_gaia_catalog=args.crossmatch_gaia_catalog,
        crossmatch_vsx_csv=args.crossmatch_vsx_csv,
        crossmatch_method=args.crossmatch_method,
        crossmatch_cache_dir=args.crossmatch_cache_dir,
        crossmatch_no_cache=args.crossmatch_no_cache,
        crossmatch_refresh_cache=args.crossmatch_refresh_cache,
        fit_workers=max(1, int(args.fit_workers)),
    )


# Back-compat for older imports
run_march18_einstein_pipeline = run_microlensing_pipeline


if __name__ == "__main__":
    main()
