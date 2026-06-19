import os
import sqlite3
import json
import pandas as pd
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from malca.lightcurve_publication import apply_publication_rcparams

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

# Output path
out_pdf = "output_migrated_camera_field_20260606/runs/runs_march18_bundle_all/results/all_classes_mollweide.pdf"
out_png = "output_migrated_camera_field_20260606/runs/runs_march18_bundle_all/results/all_classes_mollweide.png"

def main():
    print("Gathering data...")
    # 1. Get coords from DB
    conn = sqlite3.connect('output_migrated_camera_field_20260606/runs/runs_march18_bundle_all/review/review.taxonomy_filled.db')
    c = conn.cursor()
    c.execute('SELECT candidate_id, payload_json FROM candidates')
    rows = []
    for cid, p_json in c.fetchall():
        p = json.loads(p_json)
        ext = p.get('external_stats', {})
        ra = ext.get('ra_deg')
        dec = ext.get('dec_deg')
        rows.append({'candidate_id': cid, 'ra': ra, 'dec': dec})
    df_coords = pd.DataFrame(rows)
    df_coords['candidate_id'] = df_coords['candidate_id'].astype(str)

    # 2. Get event_class from CSV
    csv_path = 'output_migrated_camera_field_20260606/runs/runs_march18_bundle_all/results/march18_review_cmd_dustmaps_full.csv'
    df_csv = pd.read_csv(csv_path)
    df_csv['candidate_id'] = df_csv['candidate_id'].astype(str)

    # 3. Get best_model from parquet
    pq_path = 'output_migrated_camera_field_20260606/microlensing/microlensing_results_20260619_001924.parquet'
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

    print("Evaluating SFD Dust Map...")
    import astropy.units as u
    
    # Generate matplotlib mollweide grid
    fig, ax = plt.subplots(figsize=(7.0, 4.0), subplot_kw={'projection': 'mollweide'})
    
    lon_grid = np.linspace(-np.pi, np.pi, 800)
    lat_grid = np.linspace(-np.pi/2, np.pi/2, 400)
    Lon, Lat = np.meshgrid(lon_grid, lat_grid)
    
    # SkyCoord expects longitude to be positive, so we pass it as is and let Astropy wrap it
    coords = SkyCoord(l=Lon*u.rad, b=Lat*u.rad, frame='galactic')
    
    sfd = SFDQuery()
    ebv = sfd(coords)
    ebv[np.isnan(ebv)] = 0
    ebv_log = np.log10(ebv + 0.1)
    
    ax.pcolormesh(
        Lon, Lat, ebv_log,
        cmap='Greys',
        vmin=np.percentile(ebv_log, 5),
        vmax=np.percentile(ebv_log, 99),
        rasterized=True,
        shading='auto',
        zorder=0
    )
    
    import matplotlib.patheffects as pe
    
    ax.grid(True, alpha=0.35, linestyle='--', linewidth=0.6)
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
    from malca.lightcurve_publication import CMD_BUCKET_STYLE
    
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
        l_rad = np.radians(((c.galactic.l.deg + 180.0) % 360.0) - 180.0)
        b_rad = np.radians(c.galactic.b.deg)
            
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
