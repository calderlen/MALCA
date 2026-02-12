"""File paths, API endpoints, and cache directories."""

from pathlib import Path

VSX_CROSSMATCH_PATH = Path("input/vsx/asassn_x_vsx_matches_20250919_2252_compat.csv")
VSX_RAW_CATALOG_PATH = Path("input/vsx/vsxcat.090525.csv")
ASASSN_INDEX_PATH = Path("input/asassn_index_masked_concat_cleaned_20250919_154524_brotli.parquet")
STARHORSE_DEFAULT_PATH = "input/starhorse/starhorse2021.parquet"
MIST_GRID_PATH = Path("input/mist/mist_cmd_minimal.csv")
DEFAULT_CACHE_DIR = Path("~/.cache/malca/catalogs")
GAIA_CACHE_FILE = Path("output/gaia_cache.parquet")
LCV2_ROOT = Path("/data/poohbah/1/assassin/rowan.90/lcsv2")
UNTIMELY_API_URL = "https://irsa.ipac.caltech.edu/cgi-bin/Gator/nph-query"
STARHORSE_TAP_URL = "https://gaia.aip.de/tap"
GAIA_AIP_TAP_URL = "https://gea.esac.esa.int/tap-server/tap"
GAIA_LOCAL_CATALOG = Path("input/gaia/gaia_dr3_crossmatched.parquet")
