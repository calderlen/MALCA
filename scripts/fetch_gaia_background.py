import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astroquery.gaia import Gaia

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
OUTPUT_CSV = "gaia_allsky_15mag.csv"
OUTPUT_PARQUET = "gaia_allsky_15mag.parquet"

def fetch_gaia_background():
    """
    Downloads ~35 million stars from Gaia DR3 down to G=15.
    Requires ESA Archive credentials due to the 3M row anonymous limit.
    """
    print("Please log in to your ESA Gaia Archive account...")
    Gaia.login()

    # The ADQL Query
    # We include 'parallax' and 'ruwe' just in case you need them later
    query = """
    SELECT 
        source_id, phot_g_mean_mag, bp_rp, parallax, ruwe
    FROM 
        gaiadr3.gaia_source
    WHERE 
        phot_g_mean_mag <= 15
        AND bp_rp IS NOT NULL
        AND ruwe < 1.4
    """

    print("Submitting query to ESA servers...")
    print("This will download a large CSV. Do not interrupt your connection.")
    
    # We use dump_to_file to completely bypass Astropy's VOTable XML parser.
    # This writes the raw compressed CSV directly to disk.
    job = Gaia.launch_job_async(
        query, 
        dump_to_file=True, 
        output_format='csv', 
        output_file=OUTPUT_CSV
    )
    
    print(f"Download complete: {OUTPUT_CSV}")
    
    print("Loading CSV into Pandas and converting to Parquet...")
    # Load the CSV. We use pyarrow engine for speed if available.
    df = pd.read_csv(OUTPUT_CSV, engine='pyarrow')
    
    # Save to Parquet format
    df.to_parquet(OUTPUT_PARQUET, engine='pyarrow')
    print(f"Successfully saved {len(df)} rows to {OUTPUT_PARQUET}")
    
    # Clean up the large CSV to save disk space
    os.remove(OUTPUT_CSV)
    print(f"Cleaned up temporary CSV. You are ready to plot!")

def plot_local_heatmap(parquet_path):
    """
    Loads the Parquet file and generates a 2D Histogram (Heatmap).
    """
    if not os.path.exists(parquet_path):
        print(f"Error: {parquet_path} not found. Run the fetch function first.")
        return

    print("Loading Parquet data...")
    df = pd.read_parquet(parquet_path, engine='pyarrow')

    # Mock ASAS-SN candidates (replace with your real data)
    candidate_bp_rp = np.array([0.5, 1.2, 2.1, 0.3, 1.7])
    candidate_g_mag = np.array([12.1, 14.3, 11.5, 14.9, 13.0])

    print("Generating plot...")
    fig, ax = plt.subplots(figsize=(9, 11))

    # hist2d is incredibly fast, even for 35M points
    h, xedges, yedges, img = ax.hist2d(
        df['bp_rp'], 
        df['phot_g_mean_mag'], 
        bins=[250, 250],         # 250x250 grid for high resolution
        range=[[-0.5, 4.0], [3.0, 16.0]], 
        cmap='Blues', 
        norm=LogNorm()           # Log scale makes rare stars visible
    )
    cbar = fig.colorbar(img, ax=ax, label='Stars per bin')

    # Overlay candidates
    ax.scatter(
        candidate_bp_rp, 
        candidate_g_mag, 
        color='crimson', 
        edgecolor='black',
        s=80, 
        marker='*',
        label='ASAS-SN Candidates'
    )

    ax.invert_yaxis()
    ax.set_xlabel(r'Color Index ($G_{BP} - G_{RP}$)', fontsize=12)
    ax.set_ylabel(r'Apparent Magnitude ($G$)', fontsize=12)
    ax.set_title('Gaia All-Sky DR3 ($G \le 15$) vs. Candidates', fontsize=14)
    ax.grid(True, linestyle=':', alpha=0.4)
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # If the parquet file doesn't exist, fetch the data
    if not os.path.exists(OUTPUT_PARQUET):
        fetch_gaia_background()
    
    # Plot it
    plot_local_heatmap(OUTPUT_PARQUET)
