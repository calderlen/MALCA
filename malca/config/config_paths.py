"""File paths, API endpoints, and cache directories."""

from pathlib import Path

VSX_CROSSMATCH_PATH = Path("input/vsx/asassn_x_vsx_matches_20250919_2252.csv")
VSX_RAW_CATALOG_PATH = Path("input/vsx/vsxcat.090525.csv")
ASASSN_INDEX_PATH = Path("input/asassn_index_masked_concat_cleaned_20250919_154524_brotli.parquet")
STARHORSE_DEFAULT_PATH = "input/starhorse/starhorse2021.parquet"
MIST_GRID_PATH = Path("input/mist/mist_cmd_minimal.csv")
DEFAULT_CACHE_DIR = Path("~/.cache/malca/catalogs")
GAIA_CACHE_FILE = Path("output/gaia_cache.parquet")
LTV_OUTPUT_DIR = Path("output/ltv")
LTV_INJECTION_OUTPUT_DIR = LTV_OUTPUT_DIR / "injection"
SYDNEY_LTV_CSV_PATH = Path("input/SydneyLTVs.csv")
LCV2_ROOT = Path("/data/poohbah/1/assassin/rowan.90/lcsv2")
UNTIMELY_API_URL = "https://irsa.ipac.caltech.edu/cgi-bin/Gator/nph-query"
STARHORSE_TAP_URL = "https://gaia.aip.de/tap"
GAIA_AIP_TAP_URL = "https://gaia.aip.de/tap"
GAIA_LOCAL_CATALOG = Path("input/gaia/gaia_dr3_crossmatched.parquet")
DEFAULT_OUTPUT_DIR = Path("output")
LCV2_MASKED_ROOT = Path("/data/poohbah/1/assassin/lenhart/malca-older/calder/lcsv2_masked")
SKYPATROL_CACHE_DIR = Path("output/cache/skypatrol")
LTV_CACHE_DIR = Path("/tmp/ltv_cache")
