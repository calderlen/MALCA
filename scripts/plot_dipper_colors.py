import os
import sqlite3
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.ticker import AutoMinorLocator
from malca.lightcurve_publication import apply_publication_rcparams

apply_publication_rcparams(plt)
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12
})

def main():
    # 1. Get event_class from CSV
    csv_path = 'output_migrated_camera_field_20260606/runs/runs_march18_bundle_all/results/march18_review_cmd_dustmaps_full.csv'
    df_csv = pd.read_csv(csv_path)
    dipper_ids = df_csv.loc[df_csv['event_class'] == 'dipper', 'candidate_id'].astype(str).tolist()
    
    # 2. Get features from DB
    conn = sqlite3.connect('output_migrated_camera_field_20260606/runs/runs_march18_bundle_all/review/review.taxonomy_filled.db')
    c = conn.cursor()
    c.execute('SELECT candidate_id, payload_json FROM candidates')
    
    rows = []
    for cid, p_json in c.fetchall():
        if str(cid) in dipper_ids:
            p = json.loads(p_json)
            ext = p.get('external_stats', {})
            rows.append({
                'candidate_id': cid,
                'w1_w2': ext.get('w1_w2'),
                'H_K': ext.get('H_K'),
                'vphas_r_ha': ext.get('vphas_r_ha'),
                'vphas_r_i': ext.get('vphas_r_i'),
                'yso_class': ext.get('yso_class')
            })
            
    df = pd.DataFrame(rows)
    
    # Cast to float
    for col in ['w1_w2', 'H_K', 'vphas_r_ha', 'vphas_r_i']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    out_dir = Path('output_migrated_camera_field_20260606/runs/runs_march18_bundle_all/results')
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: VPHAS+ r-i vs r-Ha
    df_vphas = df.dropna(subset=['vphas_r_i', 'vphas_r_ha'])
    if not df_vphas.empty:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(df_vphas['vphas_r_i'], df_vphas['vphas_r_ha'], color='k', edgecolor='k', s=40, zorder=5, label=f'Dippers ({len(df_vphas)})')
        ax.axhline(0.0, color='k', linestyle='--', linewidth=0.5, zorder=1) # Ha excess threshold is usually > 0 or a function of r-i
        ax.set_xlabel(r'$r - i$ [mag]')
        ax.set_ylabel(r'$r - H\alpha$ [mag]')
        ax.legend(loc='best')
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        fig.tight_layout()
        out_vphas = out_dir / 'dipper_vphas_color_color.pdf'
        fig.savefig(out_vphas)
        print(f"Saved {out_vphas}")
        plt.close(fig)
    else:
        print("No VPHAS+ data available to plot.")

    # Plot 2: 2MASS vs WISE H-K vs W1-W2
    df_ir = df.dropna(subset=['H_K', 'w1_w2'])
    if not df_ir.empty:
        fig, ax = plt.subplots(figsize=(6, 5))
        
        # Plot YSO classification regions if desired, but for now just scatter
        # Main sequence is generally w1-w2 < 0.3, Class II is w1-w2 > 0.3
        
        ax.scatter(df_ir['w1_w2'], df_ir['H_K'], color='k', edgecolor='k', s=40, zorder=5, label=f'Dippers ({len(df_ir)})')
        
        # Simple guidelines for YSO classes (Koenig & Leisawitz 2014 approximate)
        ax.axhline(0.3, color='k', linestyle='--', linewidth=0.5, zorder=1)
        ax.axvline(0.3, color='k', linestyle='--', linewidth=0.5, zorder=1)
        
        ax.set_xlabel(r'$W_1 - W_2$ [mag]')
        ax.set_ylabel(r'$H - K$ [mag]')
        ax.legend(loc='best')
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        fig.tight_layout()
        out_ir = out_dir / 'dipper_wise_color_color.pdf'
        fig.savefig(out_ir)
        print(f"Saved {out_ir}")
        plt.close(fig)
    else:
        print("No WISE/2MASS data available to plot.")

if __name__ == '__main__':
    main()
