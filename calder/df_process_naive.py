import os
from pathlib import Path
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt

# Try to import helpers whether running as a package or from repo root
try:
    # When imported as calder.df_process_naive
    from calder.utils import read_lc_dat, plot_light_curve
except Exception:
    try:
        # When running from repo root with PYTHONPATH=. and importing df_process_naive
        from utils import read_lc_dat, plot_light_curve
    except Exception:  # pragma: no cover
        read_lc_dat = None  # type: ignore
        plot_light_curve = None  # type: ignore


def _read_lc_dat_minimal(asassn_id, lc_dir):
    """Lightweight .dat loader with no astropy/scipy dependency.
    Returns (df_g, df_v) or (empty, empty) if file not found.
    """
    try:
        path = os.path.join(lc_dir, f"{asassn_id}.dat")
        if not os.path.exists(path):
            return pd.DataFrame(), pd.DataFrame()

        columns = [
            "JD",
            "mag",
            "error",
            "good_bad",
            "camera#",
            "v_g_band",
            "saturated",
            "cam_field",
        ]
        df = pd.read_fwf(path, header=None, names=columns)

        # Split camera/field if present
        if "cam_field" in df.columns:
            try:
                df[["camera_name", "field"]] = df["cam_field"].astype(str).str.split("/", expand=True)
                df = df.drop(columns=["cam_field"])
            except Exception:
                pass

        # Basic dtype enforcement
        for c in ("JD", "mag", "error"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        for c in ("good_bad", "camera#", "v_g_band", "saturated"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        dfg = df.loc[df.get("v_g_band", 0) == 0].reset_index(drop=True)
        dfv = df.loc[df.get("v_g_band", 1) == 1].reset_index(drop=True)
        return dfg, dfv
    except Exception:
        return pd.DataFrame(), pd.DataFrame()


def _basic_plot_light_curve(df, title: str, band_label: str = "g"):
    """Minimal plotting without astropy/scipy."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    try:
        jd = pd.to_numeric(df["JD"], errors="coerce") - (2.458 * 10 ** 6)
        mag = pd.to_numeric(df["mag"], errors="coerce")
    except Exception:
        jd = df["JD"]
        mag = df["mag"]
    ax.scatter(jd, mag, s=6, alpha=0.6)
    try:
        ax.set_xlim(jd.min() - 300, jd.max() + 150)
        ax.set_ylim(mag.min() - 0.1, mag.max() + 0.1)
    except Exception:
        pass
    ax.set_xlabel("Julian Date $- 2458000$ [d]")
    ax.set_ylabel(f"{band_label.upper()} [mag]")
    ax.set_title(title)
    ax.invert_yaxis()
    return fig, ax


def read_df_csv_naive(
    csv_path,
    require_both: bool = False,
    out_csv_path=None,
    write_csv: bool = True,
    index: bool = False,
    # If True, also generate a PDF plot per selected row
    plot: bool = False,
    # Band to plot when plotting is enabled: "g" or "v"
    plot_band: str = "g",
    # If provided, override output directory for plots; otherwise derived from CSV location
    plots_root: Optional[Path] = None,
    # Whether to overlay peak markers in plots
    peak_option: bool = False,
):
    """
    Read a CSV summarizing peak-finding results and return only rows where
    either band (default) or both bands (if ``require_both=True``) have
    a non-zero number of peaks. All columns are preserved.

    Behavior:
    - Coerces ``g_n_peaks`` and ``v_n_peaks`` to numeric; NaN -> 0.
    - Filters rows by (g_n_peaks > 0) OR (v_n_peaks > 0) by default.
      If ``require_both=True``, uses AND instead.
    - Adds a ``source_file`` column with the basename of ``csv_path`` for
      provenance.
    - Leaves array-like columns (e.g., ``g_peaks_idx``, ``g_peaks_jd``) as raw
      strings; no parsing is performed.

    Parameters
    ----------
    csv_path : str or Path
        Path to the CSV file to read.
    require_both : bool, optional
        If True, require non-zero peaks in both bands (AND). If False (default),
        accept non-zero peaks in either band (OR).
    out_csv_path : str or Path, optional
        Where to write the selected rows as CSV. If None and ``write_csv`` is
        True, writes next to the input as ``<stem>_selected_dippers.csv``.
    write_csv : bool, optional
        If True (default), write the filtered rows to CSV.
    index : bool, optional
        Whether to include the index when writing CSV (default False).

    Returns
    -------
    pandas.DataFrame
        Filtered DataFrame with all original columns retained, plus
        ``source_file``.
    """

    file = Path(csv_path)
    df = pd.read_csv(file).copy()

    # Ensure required columns exist
    for col in ("g_n_peaks", "v_n_peaks"):
        if col not in df.columns:
            raise KeyError(
                f"Column '{col}' is missing; cannot select nonzero-peak rows."
            )

    # Coerce counts to numeric and treat NaNs as zero
    df["g_n_peaks"] = pd.to_numeric(df["g_n_peaks"], errors="coerce").fillna(0)
    df["v_n_peaks"] = pd.to_numeric(df["v_n_peaks"], errors="coerce").fillna(0)

    # Selection mask: either band by default; both if requested
    if require_both:
        mask = (df["g_n_peaks"] > 0) & (df["v_n_peaks"] > 0)
    else:
        mask = (df["g_n_peaks"] > 0) | (df["v_n_peaks"] > 0)

    out = df.loc[mask].reset_index(drop=True)

    # Provenance
    out["source_file"] = file.name

    # Optionally write CSV of selected rows
    if write_csv:
        dest = (
            Path(out_csv_path)
            if out_csv_path is not None
            else file.parent / f"{file.stem}_selected_dippers.csv"
        )
        out.to_csv(dest, index=index)

    # Optionally generate plots for each selected row
    if plot and len(out) > 0:
        try:
            _plot_selected_from_df(
                out,
                csv_file=file,
                band=plot_band,
                plots_root=plots_root,
                peak_option=peak_option,
            )
        except Exception as e:  # pragma: no cover
            print(f"[warn] Plotting failed for {file.name}: {e}")

    return out


def _find_ra_dec(index_csv: str, asas_sn_id: str):
    """Try to read RA/DEC from the provided index CSV.

    Returns (ra_deg, dec_deg) as floats or (None, None) on failure.
    """
    try:
        # Read only necessary columns if possible
        usecols = None
        # Probe columns first
        hdr = pd.read_csv(index_csv, nrows=0)
        cols = set(hdr.columns)
        id_col = None
        for c in ("asas_sn_id", "asassn_id", "asassnid", "id"):
            if c in cols:
                id_col = c
                break
        ra_col = None
        for c in ("ra_deg", "ra", "ra_j2000"):
            if c in cols:
                ra_col = c
                break
        dec_col = None
        for c in ("dec_deg", "dec", "dec_j2000"):
            if c in cols:
                dec_col = c
                break

        if id_col is None or ra_col is None or dec_col is None:
            return None, None

        usecols = [id_col, ra_col, dec_col]
        df = pd.read_csv(index_csv, usecols=usecols, dtype={id_col: str})
        row = df.loc[df[id_col].astype(str) == str(asas_sn_id)]
        if row.empty:
            return None, None
        ra = float(row.iloc[0][ra_col])
        dec = float(row.iloc[0][dec_col])
        return ra, dec
    except Exception:
        return None, None


def _plot_selected_from_df(
    df_sel: pd.DataFrame,
    csv_file: Path,
    band: str = "g",
    plots_root: Optional[Path] = None,
    peak_option: bool = False,
):
    """Generate light-curve PDF plots for each selected row.

    - Loads light curves via ``read_lc_dat`` (per row's ``lc_dir`` and ``asas_sn_id``)
    - Calls ``plot_light_curve`` for the requested band
    - Saves PDFs under ``<csv_dir>/plots/<csv_stem>/<band>/<asas_sn_id>.pdf``
    - If RA/DEC cannot be read from the ``index_csv``, falls back to a generic
      title based on the ASAS-SN id.
    """
    # Use utils helpers when available; otherwise fall back
    have_utils_reader = read_lc_dat is not None
    have_utils_plotter = plot_light_curve is not None

    # Resolve output directory for this CSV's plots
    root = Path(plots_root) if plots_root is not None else csv_file.parent
    subdir = root / "plots" / csv_file.stem / band.lower()
    os.makedirs(subdir, exist_ok=True)

    for i, row in df_sel.iterrows():
        asn = str(row.get("asas_sn_id", "")).strip()
        lc_dir = row.get("lc_dir")
        if not asn or not isinstance(lc_dir, str) or not lc_dir:
            print(f"[skip] Row {i}: missing asas_sn_id or lc_dir")
            continue

        try:
            # Optional path rewrite via environment: LC_DIR_REWRITE="old_prefix::new_prefix"
            try:
                r = os.environ.get("LC_DIR_REWRITE")
                if r and "::" in r and isinstance(lc_dir, str):
                    old, new = r.split("::", 1)
                    if lc_dir.startswith(old):
                        lc_dir = lc_dir.replace(old, new, 1)
            except Exception:
                pass

            # Skip if this band shows no dips for this object (if column exists)
            ncol = "v_n_peaks" if band.lower() == "v" else "g_n_peaks"
            nval = row.get(ncol)
            try:
                if nval is not None and float(nval) <= 0:
                    # explicitly requested: only plot bands that have dips
                    continue
            except Exception:
                pass

            # Load light curves
            if have_utils_reader:
                dfg, dfv = read_lc_dat(asn, lc_dir)
            else:
                dfg, dfv = _read_lc_dat_minimal(asn, lc_dir)

            # Choose band
            if band.lower() == "v":
                df = dfv
            else:
                df = dfg

            if df is None or df.empty:
                print(f"[skip] {asn}: no data in band '{band}'")
                continue

            # Attempt RA/DEC from index CSV
            index_csv = row.get("index_csv")
            ra, dec = (None, None)
            if isinstance(index_csv, str) and index_csv:
                ra, dec = _find_ra_dec(index_csv, asn)

            # Plot
            try:
                if have_utils_plotter:
                    if ra is None or dec is None:
                        # Use placeholder RA/DEC then override title
                        plot_light_curve(df, 0.0, 0.0, peak_option=peak_option)
                        plt.gca().set_title(f"ASASSN {asn}")
                    else:
                        plot_light_curve(df, ra, dec, peak_option=peak_option)
                    if band.lower() == "v":
                        try:
                            plt.gca().set_ylabel("V [mag]")
                        except Exception:
                            pass
                else:
                    # Minimal fallback plotter
                    _basic_plot_light_curve(df, title=f"ASASSN {asn}", band_label=band)
            except Exception as e:
                print(f"[warn] {asn}: plot failed: {e}")
                continue

            # Save
            out_path = subdir / f"{asn}.pdf"
            try:
                plt.savefig(out_path, bbox_inches="tight")
            finally:
                plt.close()
            print(f"[plot] {out_path}")
        except Exception as e:
            print(f"[warn] {asn}: error {e}")
