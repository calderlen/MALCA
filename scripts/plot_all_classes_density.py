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
    parser = argparse.ArgumentParser()
    parser.add_argument("--extinction-map", choices=["sfd", "3d"], default="sfd")
    args = parser.parse_args()

    suffix = "_3d" if args.extinction_map == "3d" else ""
    out_pdf = f"output/runs/runs_march18_bundle_all/results/all_classes_mollweide{suffix}.pdf"
    out_png = f"output/runs/runs_march18_bundle_all/results/all_classes_mollweide{suffix}.png"
    print("Gathering data...")
    # 1. Get coords from DB
    conn = sqlite3.connect('output/runs/runs_march18_bundle_all/review/review.taxonomy_filled.db')
    c = conn.cursor()
    c.execute('SELECT candidate_id, payload_json FROM candidates')
    rows = []
    for cid, p_json in c.fetchall():
        p = json.loads(p_json)
        ext = p.get('external_stats', {})
        ra = ext.get('ra_deg', p.get('ra_deg'))
        dec = ext.get('dec_deg', p.get('dec_deg'))
        rows.append({'candidate_id': cid, 'ra': ra, 'dec': dec})
    df_coords = pd.DataFrame(rows)
    df_coords['candidate_id'] = df_coords['candidate_id'].astype(str)

    # 2. Get event_class from CSV
    csv_path = 'output/runs/runs_march18_bundle_all/results/march18_review_cmd_dustmaps_full.csv'
    df_csv = pd.read_csv(csv_path)
    df_csv['candidate_id'] = df_csv['candidate_id'].astype(str)

    # 3. Get best_model from parquet
    pq_path = 'output/microlensing/microlensing_results_20260619_001924.parquet'
    df_pq = pd.read_parquet(pq_path, columns=['candidate_id', 'best_model'])
    df_pq['candidate_id'] = df_pq['candidate_id'].astype(str)

    # Merge
    df_merged = df_csv[['candidate_id', 'event_class']].merge(df_pq, on='candidate_id', how='left')
    df_merged = df_merged.merge(df_coords, on='candidate_id', how='left')

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
