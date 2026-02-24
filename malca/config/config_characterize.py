"""Characterization settings: Gaia/StarHorse/dust/BANYAN/IPHAS/unWISE/cluster/SFR query params."""

# Gaia query
GAIA_CHUNK_SIZE = 1000
STARHORSE_TAP_CHUNK_SIZE = 1000

# Crossmatch radii
IPHAS_MAX_SEP_ARCSEC = 1.0
CLUSTER_MAX_SEP_ARCSEC = 1.0
UNWISE_MAX_SEP_ARCSEC = 3.0
GALEX_MAX_SEP_ARCSEC = 3.0
APASS_MAX_SEP_ARCSEC = 1.0
ALLWISE_MAX_SEP_ARCSEC = 2.0
NEIGHBOR_RADIUS_ARCSEC = 15.0
NEIGHBOR_CHUNK_SIZE = 1000
SPECTRA_RADIUS_ARCSEC = 3.0
SPECTRA_CHUNK_SIZE = 1000

# unWISE quality thresholds
UNWISE_TIMEOUT_SECONDS = 30
UNWISE_FRACFLUX_MIN = 0.5
UNWISE_QF_MIN = 0.9
UNWISE_VARIABILITY_ZSCORE = 3.0
UNWISE_EXPECTED_SCATTER_BASE = 0.02
UNWISE_EXPECTED_SCATTER_SLOPE = 0.01
UNWISE_EXPECTED_SCATTER_MAG_REF = 14
UNWISE_WORKERS = 8
UNWISE_CHECKPOINT_EVERY = 200
UNWISE_MAX_RETRIES = 3

# SFR proximity
SFR_MAX_DIST_KPC = 1.5
SFR_DIST_TOLERANCE_FRACTION = 0.5

# SFR catalog (Prisinzano+2022)
SFR_CATALOG = [
    {"name": "Orion Nebula Cluster", "ra": 83.82, "dec": -5.39, "dist_pc": 400, "radius_deg": 1.0},
    {"name": "Cygnus X", "ra": 307.0, "dec": 40.5, "dist_pc": 1400, "radius_deg": 3.0},
    {"name": "Taurus", "ra": 68.0, "dec": 26.0, "dist_pc": 140, "radius_deg": 5.0},
    {"name": "Ophiuchus", "ra": 246.8, "dec": -24.5, "dist_pc": 140, "radius_deg": 3.0},
    {"name": "Scorpius-Centaurus", "ra": 240.0, "dec": -25.0, "dist_pc": 145, "radius_deg": 10.0},
    {"name": "Perseus", "ra": 55.0, "dec": 32.0, "dist_pc": 300, "radius_deg": 3.0},
    {"name": "Serpens", "ra": 277.5, "dec": 1.2, "dist_pc": 415, "radius_deg": 1.0},
    {"name": "Lupus", "ra": 240.0, "dec": -38.0, "dist_pc": 160, "radius_deg": 3.0},
]

# BANYAN Sigma
BANYAN_MIN_ASSOC_PROB = 0.1

# IPHAS H-alpha
IPHAS_HA_EXCESS_THRESHOLD = 0.25

# Post-review vetting
VETTING_SIMBAD_RADIUS_ARCSEC = 5.0
VETTING_ASASSN_RADIUS_ARCSEC = 5.0
