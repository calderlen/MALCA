import os
import sqlite3
import json
from pathlib import Path
import pandas as pd
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import LogNorm
from malca.plotting.lightcurve_publication import apply_publication_rcparams

# Apply publication LaTeX font and style globally
apply_publication_rcparams(plt)
# Override specific publication defaults to force 10pt across the board
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10
})
from astropy.coordinates import SkyCoord
from dustmaps.sfd import SFDQuery
import astropy.units as u

REPO_ROOT = Path(__file__).resolve().parents[1]
DUSTMAPS_DIR = REPO_ROOT / "data" / "dustmaps"
SFD_DIR = DUSTMAPS_DIR / "sfd"
DUST_GRID_N_L = 2880
DUST_GRID_N_B = 1440


def _perceptual_gray_cmap(name="dust_perceptual_gray", lstar_min=0.0, lstar_max=100.0):
    """Neutral grayscale with roughly uniform steps in CIE L* lightness."""
    lstar = np.linspace(lstar_max, lstar_min, 256)
    y = np.where(lstar > 8.0, ((lstar + 16.0) / 116.0) ** 3, lstar / 903.3)
    srgb = np.where(y <= 0.0031308, 12.92 * y, 1.055 * np.power(y, 1.0 / 2.4) - 0.055)
    srgb = np.clip(srgb, 0.0, 1.0)
    colors = np.column_stack([srgb, srgb, srgb, np.ones_like(srgb)])
    return LinearSegmentedColormap.from_list(name, colors)


DUST_CMAP = _perceptual_gray_cmap()


def _project_cache_has_sfd():
    return all((SFD_DIR / filename).exists() for filename in ("SFD_dust_4096_ngp.fits", "SFD_dust_4096_sgp.fits"))


def _sfd_query():
    if _project_cache_has_sfd():
        return SFDQuery(map_dir=str(SFD_DIR))
    return SFDQuery()


def _dust_norm_from_ebv(ebv):
    positive = np.asarray(ebv, dtype=float)
    positive = positive[np.isfinite(positive) & (positive > 0.0)]
    if positive.size == 0:
        raise ValueError("Dust grid has no positive finite E(B-V) values to plot.")
    vmin = max(float(np.nanpercentile(positive, 4.0)), 1.0e-3)
    vmax = float(np.nanpercentile(positive, 99.5))
    if vmax <= vmin:
        vmax = vmin * 10.0
    return LogNorm(vmin=vmin, vmax=vmax)


def _make_matched_dust_grid(extinction_map):
    x_edges = np.linspace(-180.0, 180.0, DUST_GRID_N_L + 1)
    y_edges = np.linspace(-90.0, 90.0, DUST_GRID_N_B + 1)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    xx, yy = np.meshgrid(x_centers, y_centers)
    l_deg = xx % 360.0
    b_deg = yy

    if extinction_map == "sfd":
        coords = SkyCoord(l=l_deg.ravel() * u.deg, b=b_deg.ravel() * u.deg, frame="galactic")
        ebv = _sfd_query().query(coords).reshape(DUST_GRID_N_B, DUST_GRID_N_L)
    else:
        from dustmaps3d import dustmaps3d
        distance_kpc = np.full(l_deg.size, 1000.0, dtype=float)
        ebv_1d, *_ = dustmaps3d(l_deg.ravel(), b_deg.ravel(), distance_kpc)
        ebv = np.asarray(ebv_1d, dtype=float).reshape(DUST_GRID_N_B, DUST_GRID_N_L)

    return x_edges, y_edges, ebv


def _draw_matched_dust_background(ax, extinction_map):
    x_edges, y_edges, ebv = _make_matched_dust_grid(extinction_map)
    masked_ebv = np.ma.masked_invalid(np.ma.masked_less_equal(np.asarray(ebv, dtype=float), 0.0))
    return ax.pcolormesh(
        -np.deg2rad(np.asarray(x_edges)[::-1]),
        np.deg2rad(y_edges),
        masked_ebv[:, ::-1],
        shading="auto",
        cmap=DUST_CMAP,
        norm=_dust_norm_from_ebv(ebv),
        alpha=1.0,
        zorder=0,
        rasterized=True,
    )


def _galactic_to_matched_mollweide(l_deg, b_deg):
    l_wrapped = ((np.asarray(l_deg, dtype=float) + 180.0) % 360.0) - 180.0
    return -np.deg2rad(l_wrapped), np.deg2rad(np.asarray(b_deg, dtype=float))

def main():
    import argparse

    default_run = REPO_ROOT / "output/runs/dat3-full-extended_2026-07-01-v4"
    default_microlensing = REPO_ROOT / "output/microlensing/microlensing_results_20260619_001924.parquet"

    parser = argparse.ArgumentParser()
    parser.add_argument("--extinction-map", choices=["sfd", "3d"], default="sfd")
    parser.add_argument("--run-root", type=Path, default=default_run)
    parser.add_argument(
        "--review-db",
        type=Path,
        default=None,
        help="Review SQLite DB; defaults to RUN_ROOT/review/review.db.",
    )
    parser.add_argument(
        "--microlensing-parquet",
        type=Path,
        default=default_microlensing,
        help="Optional microlensing fit summary used to tag FRED candidates.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for figure outputs; defaults to RUN_ROOT/results.",
    )
    args = parser.parse_args()

    run_root = args.run_root.expanduser().resolve()
    review_db = (
        args.review_db.expanduser().resolve()
        if args.review_db is not None
        else (run_root / "review" / "review.db")
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else (run_root / "results")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = "_3d" if args.extinction_map == "3d" else ""
    out_pdf = output_dir / f"all_classes_mollweide{suffix}.pdf"
    out_png = output_dir / f"all_classes_mollweide{suffix}.png"
    print("Gathering data...")
    print(f"Review DB: {review_db}")
    # 1. Get coords from DB
    conn = sqlite3.connect(review_db)
    candidate_columns = {
        str(row[1]) for row in conn.execute("PRAGMA table_info(candidates)")
    }
    coord_columns = [col for col in ("candidate_id", "ra", "dec", "payload_json") if col in candidate_columns]
    df_coords = pd.read_sql_query(
        f"SELECT {', '.join(coord_columns)} FROM candidates",
        conn,
    )
    df_coords["candidate_id"] = df_coords["candidate_id"].astype(str)

    def _resolve_coord(row: pd.Series, column: str, payload_key: str) -> object:
        value = row.get(column, np.nan)
        if pd.notna(value):
            return value
        payload = row.get("payload_json")
        if isinstance(payload, str) and payload.strip():
            try:
                payload_data = json.loads(payload)
            except json.JSONDecodeError:
                payload_data = {}
            external = payload_data.get("external_stats", {}) if isinstance(payload_data, dict) else {}
            for source in (external, payload_data if isinstance(payload_data, dict) else {}):
                if isinstance(source, dict) and pd.notna(source.get(payload_key)):
                    return source.get(payload_key)
        return np.nan

    if "ra" not in df_coords.columns:
        df_coords["ra"] = np.nan
    if "dec" not in df_coords.columns:
        df_coords["dec"] = np.nan
    df_coords["ra"] = df_coords.apply(lambda row: _resolve_coord(row, "ra", "ra_deg"), axis=1)
    df_coords["dec"] = df_coords.apply(lambda row: _resolve_coord(row, "dec", "dec_deg"), axis=1)
    df_coords = df_coords[["candidate_id", "ra", "dec"]]

    # 2. Get event_class from reviews
    df_labels = pd.read_sql_query(
        """
        SELECT candidate_id, event_class
        FROM reviews
        WHERE event_class IN ('dipper', 'ltv', 'microlensing')
        """,
        conn,
    )
    conn.close()
    df_labels["candidate_id"] = df_labels["candidate_id"].astype(str)
    print(f"Loaded {len(df_labels)} reviewed dipper/LTV/microlensing candidates")

    # 3. Optional microlensing best_model overlay for FRED tagging
    df_pq = pd.DataFrame(columns=["candidate_id", "best_model"])
    microlensing_parquet = args.microlensing_parquet.expanduser().resolve()
    if microlensing_parquet.exists():
        df_pq = pd.read_parquet(microlensing_parquet, columns=["candidate_id", "best_model"])
        df_pq["candidate_id"] = df_pq["candidate_id"].astype(str)
        print(f"Loaded microlensing best_model from {microlensing_parquet}")
    else:
        print(f"Microlensing parquet not found; skipping FRED overlay: {microlensing_parquet}")

    # Merge
    df_merged = df_labels.merge(df_pq, on="candidate_id", how="left")
    df_merged = df_merged.merge(df_coords, on="candidate_id", how="left")

    df_merged['final_plot_class'] = 'other'
    df_merged.loc[df_merged['event_class'] == 'ltv', 'final_plot_class'] = 'LTV'
    df_merged.loc[df_merged['event_class'] == 'dipper', 'final_plot_class'] = 'Dipper'
    df_merged.loc[df_merged['event_class'] == 'microlensing', 'final_plot_class'] = 'Microlensing'

    # Override with FRED if best_model is fred
    df_merged.loc[df_merged['best_model'] == 'fred', 'final_plot_class'] = 'FRED'
    
    df_merged = df_merged.dropna(subset=['ra', 'dec'])

    print(f"Evaluating {args.extinction_map.upper()} Dust Map...")
    
    # Generate matplotlib mollweide grid
    fig, ax = plt.subplots(figsize=(7.0, 4.0), subplot_kw={'projection': 'mollweide'})
    _draw_matched_dust_background(ax, args.extinction_map)
    
    import matplotlib.patheffects as pe
    
    ax.grid(True, alpha=0.35, linestyle='--', linewidth=0.6)
    tick_labels = np.arange(180, -181, -30)
    ax.set_xticks(-np.deg2rad(tick_labels))
    ax.set_xticklabels([f"{int(value)}°" for value in tick_labels])
    ax.set_xlabel(r'$\ell$ [$^\circ$]')
    ax.set_ylabel(r'$b$ [$^\circ$]')
    ax.set_title("")  # Removed plot title
    
    # Make tick labels stand out against dark dust map without obscuring markers
    for label in ax.get_xticklabels():
        label.set_path_effects([pe.withStroke(linewidth=1.5, foreground='white')])
        label.set_color('black')
        
    # Fix misaligned b labels by drawing them manually (reduced spaces so they hug the edge)
    ax.set_yticklabels([])
    for lat_deg in range(-75, 90, 15):
        if lat_deg == 0: continue
        lat = np.radians(lat_deg)
        txt = ax.text(-np.pi, lat, f"{lat_deg}° ", ha='right', va='center', fontsize=10)
        txt.set_path_effects([pe.withStroke(linewidth=1.5, foreground='white')])

    print("Plotting candidates...")
    from malca.plotting.lightcurve_publication import CMD_BUCKET_STYLE
    
    styles = {
        'Microlensing': CMD_BUCKET_STYLE['Microlensing'],
        'Dipper': CMD_BUCKET_STYLE['Dipper'],
        'LTV': CMD_BUCKET_STYLE['LTV'],
        'FRED': {'color': '#1f77b4', 'marker': 'o', 'size': 14, 'zorder': 9},  # Blue circles, smaller size
        'other': CMD_BUCKET_STYLE['Unknown']
    }
    
    for cls in ['other', 'Dipper', 'LTV', 'FRED', 'Microlensing']: # Plot in z-order
        subset = df_merged[df_merged['final_plot_class'] == cls]
        if len(subset) == 0:
            continue
            
        style = styles[cls]
        marker_size = float(style["size"]) * 1.5  # Scaled down to prevent extreme overlapping
        
        # Convert ICRS RA/Dec to Galactic l/b
        c = SkyCoord(ra=subset['ra'].values, dec=subset['dec'].values, unit='deg', frame='icrs')
        l_rad, b_rad = _galactic_to_matched_mollweide(c.galactic.l.deg, c.galactic.b.deg)
            
        ax.scatter(
            l_rad, 
            b_rad, 
            c=style['color'], 
            s=marker_size, 
            marker=style['marker'],
            alpha=0.95,
            edgecolors='black',
            linewidths=0.5,
            label=f"{cls} ({len(subset)})",
            zorder=style.get('zorder', 5)
        )

    # Adjust legend to nestle into the empty bottom-right space of the Mollweide axes box
    ax.legend(
        loc='lower right', 
        bbox_to_anchor=(1.01, -0.02),
        fontsize=8,
        markerscale=1.0,
        framealpha=1.0,
        edgecolor='black',
        borderaxespad=0.0,
        borderpad=0.4,
        labelspacing=0.4,
        handletextpad=0.4
    )
    
    plt.savefig(out_pdf, bbox_inches='tight', dpi=300)
    plt.savefig(out_png, bbox_inches='tight', dpi=300)
    print(f"Saved to {out_pdf} and {out_png}")

if __name__ == "__main__":
    main()
