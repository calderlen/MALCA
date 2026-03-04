"""
Post-review vetting: check whether candidates are already known objects.

Queries:
 1. SIMBAD — object type, identifiers, bibliography count
 2. Gaia DR3 variability tables — variability flag + classification
 3. ASAS-SN Variable Stars Database (VizieR II/366) — known ASAS-SN variables
 4. ZTF periodic variables (Chen+ 2020, VizieR J/ApJS/249/18) — recent ZTF discoveries
 5. TNS (Transient Name Server) — supernovae, novae, CVs, transients
 6. Gaia DR3 eclipsing binary parameters — periods for dominant contaminant class
 7. ALeRCE ZTF broker — ZTF ML classification
 8. ATLAS forced photometry — independent cyan/orange confirmation
 9. Gaia DR3 epoch photometry — space-based variability confirmation
10. eROSITA X-ray catalog — youth indicator
11. Proper motion consistency — cluster membership validation
12. NEOWISE light curves — IR time-series for dipper confirmation

Usage:
    from malca.vetting import vet_candidates
    df_vetted = vet_candidates(df)
"""
from __future__ import annotations

import io
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import pandas as pd
import pyvo
import requests
from astropy.coordinates import SkyCoord
from astropy.table import Table
import astropy.units as u
from tqdm import tqdm

try:
    from astroquery.xmatch import XMatch
except Exception:
    class XMatch:  # type: ignore[no-redef]
        @staticmethod
        def query(*_args, **_kwargs):
            raise ImportError("astroquery is required for XMatch queries")

from malca.config.config_paths import GAIA_AIP_TAP_URL
from malca.config.config_characterize import GAIA_CHUNK_SIZE
from malca.config.config_ltv import VIZIER_TAP_URL, SIMBAD_TAP_URL
from malca.utils import batch_tap_crossmatch

# Vetting configuration
SIMBAD_RADIUS_ARCSEC = 5.0
SIMBAD_BATCH_SIZE = 500
SIMBAD_RETRY_DELAY = 5
SIMBAD_MAX_RETRIES = 3

GAIA_VAR_CHUNK_SIZE = GAIA_CHUNK_SIZE
GAIA_ESA_TAP_URL = "https://gea.esac.esa.int/tap-server/tap"
GAIA_TAP_URLS = [GAIA_ESA_TAP_URL, GAIA_AIP_TAP_URL]
ASASSN_VAR_CATALOG = "II/366/catv2021"
ASASSN_VAR_LOCAL_CSV = Path(__file__).resolve().parent.parent / "input" / "asassn_variables_x.csv"
ASASSN_VAR_RADIUS_ARCSEC = 5.0

# Module-level cache for the local ASAS-SN catalog
_asassn_cache: dict = {}

ALERCE_API_BASE = "https://api.alerce.online"
ALERCE_RADIUS_ARCSEC = 3.0
ALERCE_BATCH_SIZE = 50

ATLAS_API_BASE = "https://fallingstar-data.com/forcedphot"
ATLAS_POLL_INTERVAL = 10
ATLAS_MAX_POLL = 120
ATLAS_MJD_MIN = 57000  # ~2015

ZTF_VAR_CATALOG = "J/ApJS/249/18/table2"
ZTF_VAR_RADIUS_ARCSEC = 3.0

TNS_API_BASE = "https://www.wis-tns.org/api"
TNS_RADIUS_ARCSEC = 5.0
TNS_BATCH_SIZE = 50
TNS_LOCAL_INPUT_DIR = Path(__file__).resolve().parent.parent / "input"
TNS_LOCAL_CSVS = [
    TNS_LOCAL_INPUT_DIR / "tns_public_objects.csv",
    TNS_LOCAL_INPUT_DIR / "tns_sne.csv",
]

# Module-level cache for the local TNS catalog
_tns_cache: dict = {}

EROSITA_CATALOG = "J/A+A/682/A34/erass1-m"
EROSITA_LOCAL_FITS = Path(__file__).resolve().parent.parent / "input" / "eRASS1_Main.v1.2.fits"
EROSITA_RADIUS_ARCSEC = 10.0

# Module-level cache for the local eROSITA catalog
_erosita_cache: dict = {}

NEOWISE_MAX_SEP_ARCSEC = 3.0


# =============================================================================
# SIMBAD BATCH QUERY
# =============================================================================


def query_simbad_batch(
    df: pd.DataFrame,
    radius_arcsec: float = SIMBAD_RADIUS_ARCSEC,
    chunk_size: int = SIMBAD_BATCH_SIZE,
    method: Literal["tap", "xmatch"] = "tap",
) -> pd.DataFrame:
    """
    Query SIMBAD by coordinates for all candidates.

    method='tap'    — batch TAP upload (best for large batches).
    method='xmatch' — CDS XMatch service (reliable for small batches).

    Adds columns: simbad_main_id, simbad_otype, simbad_nbref, simbad_sep_arcsec.
    """
    df = df.copy()
    for col in ("simbad_main_id", "simbad_otype", "simbad_nbref", "simbad_sep_arcsec"):
        df[col] = np.nan if col == "simbad_sep_arcsec" or col == "simbad_nbref" else ""

    valid = df["ra"].notna() & df["dec"].notna()
    if not valid.any():
        return df

    n = int(valid.sum())

    if method == "xmatch":
        return _simbad_via_xmatch(df, valid, n, radius_arcsec)
    else:
        return _simbad_via_tap(df, valid, n, radius_arcsec, chunk_size)


def _simbad_via_xmatch(
    df: pd.DataFrame, valid, n: int, radius_arcsec: float,
) -> pd.DataFrame:
    """SIMBAD lookup via CDS XMatch (reliable for small batches)."""
    print(f"SIMBAD: querying {n} candidates via CDS XMatch (radius={radius_arcsec}\")")

    source_table = Table()
    source_table["_idx"] = np.array(df.index[valid])
    source_table["ra"] = df.loc[valid, "ra"].values
    source_table["dec"] = df.loc[valid, "dec"].values

    matched = 0
    try:
        result = XMatch.query(
            cat1=source_table,
            cat2="simbad",
            max_distance=radius_arcsec * u.arcsec,
            colRA1="ra", colDec1="dec",
        )
        if result is not None and len(result) > 0:
            result_df = result.to_pandas()
            # Normalise column names (XMatch may return main_id or main_type)
            col_map = {}
            for c in result_df.columns:
                cl = c.lower()
                if cl == "main_id":
                    col_map[c] = "main_id"
                elif cl == "main_type":
                    col_map[c] = "otype"
                elif cl == "nbref":
                    col_map[c] = "nbref"
            result_df = result_df.rename(columns=col_map)

            if "nbref" in result_df.columns:
                result_df["nbref"] = pd.to_numeric(result_df["nbref"], errors="coerce").fillna(0).astype(int)
                result_df = result_df.sort_values("nbref", ascending=False).drop_duplicates(subset="_idx", keep="first")
            elif "angDist" in result_df.columns:
                result_df = result_df.sort_values("angDist").drop_duplicates(subset="_idx", keep="first")
            else:
                result_df = result_df.drop_duplicates(subset="_idx", keep="first")

            for _, row in result_df.iterrows():
                idx = int(row["_idx"])
                if idx in df.index:
                    df.loc[idx, "simbad_main_id"] = str(row.get("main_id", "") or "")
                    df.loc[idx, "simbad_otype"] = str(row.get("otype", "") or "")
                    df.loc[idx, "simbad_nbref"] = int(row.get("nbref", 0) or 0)
                    sep = row.get("angDist", np.nan)
                    df.loc[idx, "simbad_sep_arcsec"] = round(float(sep), 3) if pd.notna(sep) else np.nan
                    matched += 1
    except Exception as e:
        print(f"SIMBAD: XMatch query failed: {e}")

    print(f"SIMBAD: {matched}/{n} candidates matched")
    return df


def _simbad_via_tap(
    df: pd.DataFrame, valid, n: int, radius_arcsec: float, chunk_size: int,
) -> pd.DataFrame:
    """SIMBAD lookup via batch TAP upload (original bulk path)."""
    print(f"SIMBAD: querying {n} candidates via TAP (radius={radius_arcsec}\")")

    coords_df = pd.DataFrame({
        "_idx": df.index[valid],
        "ra": df.loc[valid, "ra"].values,
        "dec": df.loc[valid, "dec"].values,
    })

    result = batch_tap_crossmatch(
        coords_df,
        tap_url=SIMBAD_TAP_URL,
        catalog_table="basic",
        select_cols="c.main_id, c.otype, c.nbref",
        ra_col="ra",
        dec_col="dec",
        match_radius_arcsec=radius_arcsec,
        chunk_size=chunk_size,
        n_workers=4,
        verbose=True,
        desc="SIMBAD TAP",
    )

    matched = 0
    if not result.empty:
        # Keep best match per source (most references)
        result["nbref"] = pd.to_numeric(result.get("nbref"), errors="coerce").fillna(0).astype(int)
        result = result.sort_values("nbref", ascending=False).drop_duplicates(subset="_idx", keep="first")
        for _, row in result.iterrows():
            idx = int(row["_idx"])
            if idx in df.index:
                df.loc[idx, "simbad_main_id"] = str(row.get("main_id", ""))
                df.loc[idx, "simbad_otype"] = str(row.get("otype", ""))
                df.loc[idx, "simbad_nbref"] = int(row["nbref"])
                df.loc[idx, "simbad_sep_arcsec"] = round(float(row["sep_arcsec"]), 3)
                matched += 1

    print(f"SIMBAD: {matched}/{n} candidates matched")
    return df


# =============================================================================
# GAIA DR3 VARIABILITY TABLES
# =============================================================================


def query_gaia_variability(
    df: pd.DataFrame,
    chunk_size: int = GAIA_VAR_CHUNK_SIZE,
) -> pd.DataFrame:
    """
    Query Gaia DR3 vari_summary + vari_classifier_result.

    Adds columns: gaia_var_flag (bool), gaia_var_class, gaia_var_score.
    Requires a 'gaia_id' column with Gaia DR3 source_ids.
    """
    df = df.copy()
    df["gaia_var_flag"] = False
    df["gaia_var_class"] = ""
    df["gaia_var_score"] = np.nan

    if "gaia_id" not in df.columns:
        print("Warning: Gaia variability query requires 'gaia_id' column, skipping")
        return df

    valid = df["gaia_id"].notna()
    if not valid.any():
        return df

    # Normalize IDs
    gaia_ids = []
    idx_map = {}  # gaia_id_str -> list of df indices
    for idx, val in df.loc[valid, "gaia_id"].items():
        sid = str(val).strip()
        if sid.isdigit():
            gaia_ids.append(sid)
            idx_map.setdefault(sid, []).append(idx)
    gaia_ids = list(set(gaia_ids))

    if not gaia_ids:
        return df

    print(f"Gaia variability: querying {len(gaia_ids)} source_ids")

    # Try multiple TAP servers (ESA primary, AIP mirror as fallback)
    tap = None
    for tap_url in GAIA_TAP_URLS:
        try:
            _tap = pyvo.dal.TAPService(tap_url)
            test_query = f"SELECT source_id FROM gaiadr3.vari_summary WHERE source_id = {gaia_ids[0]}"
            _tap.run_sync(test_query)
            tap = _tap
            break
        except Exception:
            print(f"  Gaia TAP {tap_url} unavailable, trying next...")
            continue

    if tap is None:
        print("  Warning: all Gaia TAP servers unreachable, skipping variability query")
        return df

    # Query vari_summary (is it flagged as variable?)
    summary_results = {}
    for i in tqdm(range(0, len(gaia_ids), chunk_size), desc="Gaia vari_summary"):
        chunk = gaia_ids[i : i + chunk_size]
        ids_str = ",".join(chunk)
        query = f"""
            SELECT source_id,
                   in_vari_classification_result
            FROM gaiadr3.vari_summary
            WHERE source_id IN ({ids_str})
        """
        for attempt in range(3):
            try:
                result = tap.run_sync(query)
                for row in result:
                    sid = str(row["source_id"])
                    summary_results[sid] = bool(row["in_vari_classification_result"])
                break
            except Exception as e:
                if attempt < 2:
                    time.sleep(5 * (attempt + 1))
                else:
                    print(f"  Gaia vari_summary chunk {i} failed: {e}")

    # Query vari_classifier_result (what class?)
    classifier_results = {}
    for i in tqdm(range(0, len(gaia_ids), chunk_size), desc="Gaia vari_classifier"):
        chunk = gaia_ids[i : i + chunk_size]
        ids_str = ",".join(chunk)
        query = f"""
            SELECT source_id, best_class_name, best_class_score
            FROM gaiadr3.vari_classifier_result
            WHERE source_id IN ({ids_str})
        """
        for attempt in range(3):
            try:
                result = tap.run_sync(query)
                for row in result:
                    sid = str(row["source_id"])
                    classifier_results[sid] = (
                        str(row["best_class_name"]),
                        float(row["best_class_score"]) if row["best_class_score"] is not None else np.nan,
                    )
                break
            except Exception as e:
                if attempt < 2:
                    time.sleep(5 * (attempt + 1))
                else:
                    print(f"  Gaia vari_classifier chunk {i} failed: {e}")

    # Apply results
    matched = 0
    for sid, indices in idx_map.items():
        is_var = summary_results.get(sid, False)
        cls_info = classifier_results.get(sid)
        for idx in indices:
            df.loc[idx, "gaia_var_flag"] = is_var
            if cls_info is not None:
                df.loc[idx, "gaia_var_class"] = cls_info[0]
                df.loc[idx, "gaia_var_score"] = cls_info[1]
                matched += 1

    flagged = sum(1 for v in summary_results.values() if v)
    print(f"Gaia variability: {flagged} flagged as variable, {matched} with classification")
    return df


# =============================================================================
# ASAS-SN VARIABLE STAR CATALOG (VizieR II/366)
# =============================================================================


def crossmatch_asassn_variables(
    df: pd.DataFrame,
    radius_arcsec: float = ASASSN_VAR_RADIUS_ARCSEC,
    chunk_size: int = 1000,
    method: Literal["tap", "local"] = "tap",
    local_csv: Path | str | None = None,
) -> pd.DataFrame:
    """
    Crossmatch against the ASAS-SN Variable Stars Database.

    method='tap'   — batch VizieR TAP (best for large batches).
    method='local' — local CSV crossmatch via SkyCoord (instant, no network).

    Note: II/366 is not available on the CDS XMatch server.

    Adds columns: asassn_var_name, asassn_var_type, asassn_var_period.
    """
    df = df.copy()
    df["asassn_var_name"] = ""
    df["asassn_var_type"] = ""
    df["asassn_var_period"] = np.nan

    valid = df["ra"].notna() & df["dec"].notna()
    if not valid.any():
        return df

    n_valid = int(valid.sum())

    if method == "local":
        return _asassn_via_local(df, valid, n_valid, radius_arcsec, local_csv)
    else:
        return _asassn_via_tap(df, valid, n_valid, radius_arcsec, chunk_size)


def _asassn_via_local(
    df: pd.DataFrame, valid, n_valid: int, radius_arcsec: float,
    local_csv: Path | str | None = None,
) -> pd.DataFrame:
    """ASAS-SN crossmatch via local CSV + SkyCoord (instant)."""
    csv_path = Path(local_csv) if local_csv else ASASSN_VAR_LOCAL_CSV
    if not csv_path.exists():
        print(f"ASAS-SN variables: local CSV not found at {csv_path}, skipping")
        return df

    # Load and cache the catalog
    cache_key = str(csv_path)
    if cache_key not in _asassn_cache:
        print(f"ASAS-SN variables: loading local catalog from {csv_path.name}...")
        cat = pd.read_csv(csv_path, usecols=["ID", "RAJ2000", "DEJ2000", "ML_classification", "Period"])
        cat = cat.dropna(subset=["RAJ2000", "DEJ2000"])
        cat_coord = SkyCoord(ra=cat["RAJ2000"].values, dec=cat["DEJ2000"].values, unit="deg")
        _asassn_cache[cache_key] = (cat, cat_coord)
        print(f"ASAS-SN variables: cached {len(cat)} entries")

    cat, cat_coord = _asassn_cache[cache_key]

    print(f"ASAS-SN variables: crossmatching {n_valid} candidates via local catalog (radius={radius_arcsec}\")")

    src_coord = SkyCoord(
        ra=df.loc[valid, "ra"].values, dec=df.loc[valid, "dec"].values, unit="deg",
    )
    idx_cat, sep2d, _ = src_coord.match_to_catalog_sky(cat_coord)
    max_sep = radius_arcsec * u.arcsec

    matched = 0
    for i, df_idx in enumerate(df.index[valid]):
        if sep2d[i] <= max_sep:
            row = cat.iloc[idx_cat[i]]
            df.loc[df_idx, "asassn_var_name"] = str(row.get("ID", "") or "")
            df.loc[df_idx, "asassn_var_type"] = str(row.get("ML_classification", "") or "")
            try:
                df.loc[df_idx, "asassn_var_period"] = float(row["Period"]) if pd.notna(row.get("Period")) else np.nan
            except (ValueError, TypeError):
                pass
            matched += 1

    print(f"ASAS-SN variables: {matched} matches")
    return df


def _asassn_via_tap(
    df: pd.DataFrame, valid, n_valid: int, radius_arcsec: float, chunk_size: int,
) -> pd.DataFrame:
    """ASAS-SN crossmatch via batch VizieR TAP (original bulk path)."""
    print(f"ASAS-SN variables: crossmatching {n_valid} candidates via TAP (radius={radius_arcsec}\")")

    coords_df = pd.DataFrame({
        "_idx": df.index[valid],
        "ra": df.loc[valid, "ra"].values,
        "dec": df.loc[valid, "dec"].values,
    })

    result = batch_tap_crossmatch(
        coords_df,
        tap_url=VIZIER_TAP_URL,
        catalog_table=f'"{ASASSN_VAR_CATALOG}"',
        select_cols='c."ASASSN-V", c."Type", c."Per"',
        ra_col="RAJ2000",
        dec_col="DEJ2000",
        match_radius_arcsec=radius_arcsec,
        chunk_size=chunk_size,
        n_workers=4,
        verbose=True,
        desc="ASAS-SN TAP",
    )

    matched = 0
    if not result.empty:
        result = result.sort_values("sep_arcsec").drop_duplicates(subset="_idx", keep="first")
        for _, row in result.iterrows():
            idx = int(row["_idx"])
            if idx in df.index:
                df.loc[idx, "asassn_var_name"] = str(row.get("ASASSN-V", "") or "")
                df.loc[idx, "asassn_var_type"] = str(row.get("Type", "") or "")
                try:
                    df.loc[idx, "asassn_var_period"] = float(row["Per"]) if pd.notna(row.get("Per")) else np.nan
                except (ValueError, TypeError):
                    pass
                matched += 1

    print(f"ASAS-SN variables: {matched} matches")
    return df


# =============================================================================
# ZTF PERIODIC VARIABLES (Chen+ 2020, VizieR J/ApJS/249/18)
# =============================================================================


def crossmatch_ztf_variables(
    df: pd.DataFrame,
    radius_arcsec: float = ZTF_VAR_RADIUS_ARCSEC,
    chunk_size: int = 1000,
    method: Literal["tap", "xmatch"] = "tap",
) -> pd.DataFrame:
    """
    Crossmatch against ZTF periodic variable catalog (Chen+ 2020).

    method='tap'    — batch VizieR TAP upload (best for large batches).
    method='xmatch' — CDS XMatch service (reliable for small batches).

    ~781k periodic variables from ZTF DR2.  Adds columns: ztf_var_type, ztf_var_period, ztf_var_amp.
    """
    df = df.copy()
    df["ztf_var_type"] = ""
    df["ztf_var_period"] = np.nan
    df["ztf_var_amp"] = np.nan

    valid = df["ra"].notna() & df["dec"].notna()
    if not valid.any():
        return df

    n_valid = int(valid.sum())

    if method == "xmatch":
        print(f"ZTF variables: crossmatching {n_valid} candidates via CDS XMatch (radius={radius_arcsec}\")")
        source_table = Table()
        source_table["_idx"] = np.array(df.index[valid])
        source_table["ra"] = df.loc[valid, "ra"].values
        source_table["dec"] = df.loc[valid, "dec"].values

        try:
            result_tab = XMatch.query(
                cat1=source_table,
                cat2="vizier:J/ApJS/249/18/table2",
                max_distance=radius_arcsec * u.arcsec,
                colRA1="ra", colDec1="dec",
            )
            result = result_tab.to_pandas() if result_tab is not None and len(result_tab) > 0 else pd.DataFrame()
        except Exception as e:
            print(f"ZTF variables: XMatch query failed: {e}")
            result = pd.DataFrame()

        if not result.empty and "angDist" in result.columns:
            result = result.sort_values("angDist").drop_duplicates(subset="_idx", keep="first")
    else:
        print(f"ZTF variables: crossmatching {n_valid} candidates via TAP (radius={radius_arcsec}\")")
        coords_df = pd.DataFrame({
            "_idx": df.index[valid],
            "ra": df.loc[valid, "ra"].values,
            "dec": df.loc[valid, "dec"].values,
        })
        result = batch_tap_crossmatch(
            coords_df,
            tap_url=VIZIER_TAP_URL,
            catalog_table='"J/ApJS/249/18/table2"',
            select_cols='c."Type", c."Per", c."gAmp", c."rAmp"',
            ra_col="RAJ2000",
            dec_col="DEJ2000",
            match_radius_arcsec=radius_arcsec,
            chunk_size=chunk_size,
            n_workers=4,
            verbose=True,
            desc="ZTF vars TAP",
        )
        if not result.empty:
            result = result.sort_values("sep_arcsec").drop_duplicates(subset="_idx", keep="first")

    matched = 0
    if not result.empty:
        for _, row in result.iterrows():
            idx = int(row["_idx"])
            if idx in df.index:
                df.loc[idx, "ztf_var_type"] = str(row.get("Type", "") or "")
                try:
                    df.loc[idx, "ztf_var_period"] = float(row["Per"]) if pd.notna(row.get("Per")) else np.nan
                except (ValueError, TypeError):
                    pass
                # Use g-band amplitude, fall back to r-band
                amp = np.nan
                for amp_col in ("gAmp", "rAmp"):
                    try:
                        v = row.get(amp_col)
                        if pd.notna(v):
                            amp = float(v)
                            break
                    except (ValueError, TypeError):
                        pass
                df.loc[idx, "ztf_var_amp"] = amp
                matched += 1

    print(f"ZTF variables: {matched} matches")
    return df


# =============================================================================
# TNS (TRANSIENT NAME SERVER)
# =============================================================================


def _load_tns_catalog(csv_paths: list[Path]) -> tuple[pd.DataFrame, SkyCoord] | None:
    """Load and merge TNS catalogs from CSVs. Returns (cat, cat_coord) or None if no data."""
    cache_key = tuple(str(p) for p in csv_paths)
    if cache_key in _tns_cache:
        return _tns_cache[cache_key]

    rows: list[dict] = []

    for csv_path in csv_paths:
        if not Path(csv_path).exists():
            continue
        try:
            if csv_path.name == "tns_public_objects.csv":
                cat = pd.read_csv(csv_path, skiprows=1, low_memory=False)
                if cat.empty:
                    continue
                # Columns: name_prefix, name, ra, declination, redshift, type, discoverydate
                for _, r in cat.iterrows():
                    ra = r.get("ra")
                    dec = r.get("declination")
                    if pd.isna(ra) or pd.isna(dec):
                        continue
                    try:
                        ra_f = float(ra)
                        dec_f = float(dec)
                    except (ValueError, TypeError):
                        continue
                    prefix = str(r.get("name_prefix", "") or "").strip()
                    name = str(r.get("name", "") or "").strip()
                    tns_name = f"{prefix}{name}" if prefix and name else (name or prefix)
                    rows.append({
                        "ra": ra_f,
                        "dec": dec_f,
                        "name": tns_name,
                        "type": str(r.get("type", "") or "").strip(),
                        "redshift": r.get("redshift"),
                        "discovery_date": str(r.get("discoverydate", "") or "")[:10] if pd.notna(r.get("discoverydate")) else "",
                    })
            elif csv_path.name == "tns_sne.csv":
                cat = pd.read_csv(csv_path, low_memory=False)
                if cat.empty:
                    continue
                for _, r in cat.iterrows():
                    ra_str = r.get("RA")
                    dec_str = r.get("DEC")
                    if pd.isna(ra_str) or pd.isna(dec_str):
                        continue
                    try:
                        c = SkyCoord(ra=str(ra_str), dec=str(dec_str), unit=(u.hourangle, u.deg))
                        ra_f = c.ra.deg
                        dec_f = c.dec.deg
                    except Exception:
                        continue
                    name = str(r.get("Name", "") or "").strip()
                    obj_type = str(r.get("Obj. Type", "") or "").strip()
                    disc_col = "Discovery Date (UT)"
                    disc_val = r.get(disc_col, "") if disc_col in r else ""
                    disc_date = str(disc_val)[:10] if pd.notna(disc_val) else ""
                    rows.append({
                        "ra": ra_f,
                        "dec": dec_f,
                        "name": name,
                        "type": obj_type,
                        "redshift": r.get("Redshift"),
                        "discovery_date": disc_date,
                    })
        except Exception as e:
            print(f"TNS: warning loading {csv_path.name}: {e}")
            continue

    if not rows:
        _tns_cache[cache_key] = None
        return None

    cat = pd.DataFrame(rows)
    cat = cat.dropna(subset=["ra", "dec"])
    if cat.empty:
        _tns_cache[cache_key] = None
        return None
    cat_coord = SkyCoord(ra=cat["ra"].values, dec=cat["dec"].values, unit="deg")
    _tns_cache[cache_key] = (cat, cat_coord)
    print(f"TNS: loaded local catalog: {len(cat)} entries from {len(csv_paths)} file(s)")
    return cat, cat_coord


def crossmatch_tns(
    df: pd.DataFrame,
    radius_arcsec: float = TNS_RADIUS_ARCSEC,
    tns_api_key: str | None = None,
    local_csvs: list[Path] | None = None,
) -> pd.DataFrame:
    """
    Crossmatch against the Transient Name Server via local catalogs.

    Uses tns_public_objects.csv and tns_sne.csv in ~/code/malca/input (or local_csvs override).

    Adds columns: tns_name, tns_type, tns_redshift, tns_disc_date.
    """
    df = df.copy()
    df["tns_name"] = ""
    df["tns_type"] = ""
    df["tns_redshift"] = np.nan
    df["tns_disc_date"] = ""

    valid = df["ra"].notna() & df["dec"].notna()
    if not valid.any():
        return df

    n_valid = int(valid.sum())
    paths = list(local_csvs) if local_csvs else TNS_LOCAL_CSVS
    loaded = _load_tns_catalog(paths)

    if loaded is None:
        print("TNS: no catalog data loaded (check input/tns_public_objects.csv, input/tns_sne.csv), skipping")
        return df

    cat, cat_coord = loaded
    print(f"TNS: crossmatching {n_valid} candidates via local catalog (radius={radius_arcsec}\")")

    src_coord = SkyCoord(
        ra=df.loc[valid, "ra"].values,
        dec=df.loc[valid, "dec"].values,
        unit="deg",
    )
    idx_cat, sep2d, _ = src_coord.match_to_catalog_sky(cat_coord)
    max_sep = radius_arcsec * u.arcsec

    matched = 0
    for i, df_idx in enumerate(df.index[valid]):
        if sep2d[i] <= max_sep:
            row = cat.iloc[idx_cat[i]]
            df.loc[df_idx, "tns_name"] = str(row.get("name", "") or "")
            df.loc[df_idx, "tns_type"] = str(row.get("type", "") or "")
            try:
                z = row.get("redshift")
                df.loc[df_idx, "tns_redshift"] = float(z) if pd.notna(z) and str(z).strip() else np.nan
            except (ValueError, TypeError):
                pass
            df.loc[df_idx, "tns_disc_date"] = str(row.get("discovery_date", "") or "")
            matched += 1

    print(f"TNS: {matched} transient matches")
    return df


# =============================================================================
# GAIA DR3 ECLIPSING BINARY PARAMETERS
# =============================================================================


def query_gaia_eb_params(
    df: pd.DataFrame,
    chunk_size: int = GAIA_VAR_CHUNK_SIZE,
) -> pd.DataFrame:
    """
    Query Gaia DR3 vari_eclipsing_binary for detailed EB parameters.

    Only queries sources already classified as ECL by the Gaia classifier.
    Adds columns: gaia_eb_period, gaia_eb_morph, gaia_eb_global_ranking.
    """
    df = df.copy()
    df["gaia_eb_period"] = np.nan
    df["gaia_eb_morph"] = ""
    df["gaia_eb_global_ranking"] = np.nan

    if "gaia_id" not in df.columns:
        return df

    # Only look up sources classified as ECL
    ecl_mask = df.get("gaia_var_class", pd.Series("", index=df.index)).str.upper() == "ECL"
    if not ecl_mask.any():
        print("Gaia EB params: no ECL-classified sources, skipping")
        return df

    gaia_ids = []
    idx_map = {}
    for idx, val in df.loc[ecl_mask, "gaia_id"].items():
        sid = str(val).strip()
        if sid.isdigit():
            gaia_ids.append(sid)
            idx_map.setdefault(sid, []).append(idx)
    gaia_ids = list(set(gaia_ids))

    if not gaia_ids:
        return df

    n_ecl = len(gaia_ids)
    print(f"Gaia EB params: querying {n_ecl} ECL-classified sources")
    tap = None
    for tap_url in GAIA_TAP_URLS:
        try:
            tap = pyvo.dal.TAPService(tap_url)
            break
        except Exception:
            continue

    if tap is None:
        print("  Warning: all Gaia TAP servers unreachable, skipping EB params")
        return df

    eb_results = {}
    for i in tqdm(range(0, len(gaia_ids), chunk_size), desc="Gaia EB params"):
        chunk = gaia_ids[i : i + chunk_size]
        ids_str = ",".join(chunk)
        query = f"""
            SELECT source_id, frequency, model_type, global_ranking
            FROM gaiadr3.vari_eclipsing_binary
            WHERE source_id IN ({ids_str})
        """
        for attempt in range(3):
            try:
                result = tap.run_sync(query)
                for row in result:
                    sid = str(row["source_id"])
                    freq = row["frequency"]
                    period = 1.0 / float(freq) if freq and float(freq) > 0 else np.nan
                    morph = str(row["model_type"]) if row["model_type"] else ""
                    ranking = float(row["global_ranking"]) if row["global_ranking"] is not None else np.nan
                    eb_results[sid] = (period, morph, ranking)
                break
            except Exception as e:
                if attempt < 2:
                    time.sleep(5 * (attempt + 1))
                else:
                    print(f"  Gaia EB chunk {i} failed: {e}")

    matched = 0
    for sid, indices in idx_map.items():
        info = eb_results.get(sid)
        if info is None:
            continue
        period, morph, ranking = info
        for idx in indices:
            df.loc[idx, "gaia_eb_period"] = period
            df.loc[idx, "gaia_eb_morph"] = morph
            df.loc[idx, "gaia_eb_global_ranking"] = ranking
            matched += 1

    print(f"Gaia EB params: {matched} sources with orbital parameters")
    return df


# =============================================================================
# ALeRCE ZTF BROKER
# =============================================================================


def _alerce_request_with_retry(method, url, max_retries=3, **kwargs):
    """HTTP request with retry on 429 rate-limit responses."""
    kwargs.setdefault("timeout", 60)
    for attempt in range(max_retries):
        try:
            resp = method(url, **kwargs)
            if resp.status_code == 429:
                time.sleep(min(2 ** attempt, 8))
                continue
            return resp
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(1)
    return None


def _alerce_query_single(ra: float, dec: float, radius_arcsec: float) -> dict | None:
    """Cone search + probability lookup for one candidate. Returns result dict or None."""
    defaults = {
        "alerce_oid": "", "alerce_ndet": 0,
        "alerce_lc_class": "", "alerce_lc_prob": np.nan,
        "alerce_stamp_class": "", "alerce_stamp_prob": np.nan,
    }

    # Cone search
    resp = _alerce_request_with_retry(
        requests.get,
        f"{ALERCE_API_BASE}/ztf/v1/objects/",
        params={
            "ra": ra,
            "dec": dec,
            "radius": radius_arcsec,
            "page_size": 5,
            "order_by": "ndet",
            "order_mode": "DESC",
        },
    )
    if resp is None or resp.status_code != 200:
        return None
    items = resp.json().get("items", [])
    if not items:
        return None

    obj = items[0]
    oid = obj.get("oid", "")
    result = dict(defaults)
    result["alerce_oid"] = oid
    result["alerce_ndet"] = int(obj.get("ndet", 0))

    # Probability lookup
    if oid:
        resp = _alerce_request_with_retry(
            requests.get,
            f"{ALERCE_API_BASE}/ztf/v1/objects/{oid}/probabilities",
        )
        if resp is not None and resp.status_code == 200:
            probs = resp.json()
            lc_probs = [p for p in probs if p.get("classifier_name", "").startswith("lc_classifier")]
            if lc_probs:
                best_lc = max(lc_probs, key=lambda p: p.get("probability", 0))
                result["alerce_lc_class"] = best_lc.get("class_name", "")
                result["alerce_lc_prob"] = best_lc.get("probability", np.nan)
            stamp_probs = [p for p in probs if p.get("classifier_name", "").startswith("stamp_classifier")]
            if stamp_probs:
                best_stamp = max(stamp_probs, key=lambda p: p.get("probability", 0))
                result["alerce_stamp_class"] = best_stamp.get("class_name", "")
                result["alerce_stamp_prob"] = best_stamp.get("probability", np.nan)

    return result


def query_alerce(
    df: pd.DataFrame,
    radius_arcsec: float = ALERCE_RADIUS_ARCSEC,
    workers: int = 8,
) -> pd.DataFrame:
    """
    Query ALeRCE ZTF broker for classification.

    Adds columns: alerce_oid, alerce_ndet, alerce_lc_class, alerce_lc_prob,
                  alerce_stamp_class, alerce_stamp_prob.
    """
    df = df.copy()
    df["alerce_oid"] = ""
    df["alerce_ndet"] = 0
    df["alerce_lc_class"] = ""
    df["alerce_lc_prob"] = np.nan
    df["alerce_stamp_class"] = ""
    df["alerce_stamp_prob"] = np.nan

    valid = df["ra"].notna() & df["dec"].notna()
    if not valid.any():
        return df

    n_valid = int(valid.sum())
    print(f"ALeRCE: querying {n_valid} candidates (radius={radius_arcsec}\", workers={workers})")
    matched = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_alerce_query_single, float(df.loc[idx, "ra"]),
                            float(df.loc[idx, "dec"]), radius_arcsec): idx
            for idx in df.index[valid]
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="ALeRCE"):
            idx = futures[fut]
            try:
                result = fut.result()
            except Exception:
                continue
            if result is None:
                continue
            for k, v in result.items():
                df.loc[idx, k] = v
            matched += 1

    print(f"ALeRCE: {matched}/{n_valid} candidates matched")
    return df


# =============================================================================
# ATLAS FORCED PHOTOMETRY
# =============================================================================


def _atlas_submit_job(
    ra: float, dec: float, token: str, mjd_min: float = ATLAS_MJD_MIN,
) -> str | None:
    """Submit an ATLAS forced photometry job. Returns task URL or None."""
    try:
        resp = requests.post(
            f"{ATLAS_API_BASE}/queue/",
            headers={"Authorization": f"Token {token}"},
            data={"ra": ra, "dec": dec, "mjd_min": mjd_min},
            timeout=30,
        )
        if resp.status_code == 429:
            return None
        resp.raise_for_status()
        return resp.url
    except Exception:
        return None


def _atlas_poll_result(task_url: str, token: str) -> pd.DataFrame | None:
    """Poll an ATLAS task until complete, return photometry DataFrame or None."""
    for _ in range(ATLAS_MAX_POLL):
        try:
            resp = requests.get(
                task_url,
                headers={"Authorization": f"Token {token}"},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("finishtimestamp"):
                result_url = data.get("result_url")
                if result_url:
                    phot_resp = requests.get(
                        result_url,
                        headers={"Authorization": f"Token {token}"},
                        timeout=60,
                    )
                    phot_resp.raise_for_status()
                    text = phot_resp.text
                    # Strip comment lines
                    lines = [l for l in text.split("\n") if not l.startswith("###")]
                    if lines:
                        return pd.read_csv(io.StringIO("\n".join(lines)), delim_whitespace=True)
                return None
        except Exception:
            pass
        time.sleep(ATLAS_POLL_INTERVAL)
    return None


def query_atlas_forced_phot(
    df: pd.DataFrame,
    token: str | None = None,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Query ATLAS forced photometry for independent variability confirmation.

    Requires an ATLAS API token (register at https://fallingstar-data.com/forcedphot/).

    Adds columns: atlas_has_phot, atlas_n_det_cyan, atlas_n_det_orange,
                  atlas_cyan_range, atlas_orange_range.
    If *output_dir* is set, saves the full photometry DataFrame per candidate
    as ``atlas_lc_<candidate_id>.parquet``.
    """
    df = df.copy()
    df["atlas_has_phot"] = False
    df["atlas_n_det_cyan"] = 0
    df["atlas_n_det_orange"] = 0
    df["atlas_cyan_range"] = np.nan
    df["atlas_orange_range"] = np.nan

    token = token or os.environ.get("MALCA_ATLAS_TOKEN") or os.environ.get("ATLAS_API_TOKEN")
    if not token:
        print("ATLAS: no API token provided, skipping (register at https://fallingstar-data.com/forcedphot/)")
        return df

    valid = df["ra"].notna() & df["dec"].notna()
    if not valid.any():
        return df

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    n_valid = int(valid.sum())
    print(f"ATLAS: submitting {n_valid} forced photometry jobs")
    matched = 0

    for idx in tqdm(df.index[valid], desc="ATLAS forced phot"):
        ra, dec = float(df.loc[idx, "ra"]), float(df.loc[idx, "dec"])

        task_url = _atlas_submit_job(ra, dec, token)
        if task_url is None:
            continue

        phot = _atlas_poll_result(task_url, token)
        if phot is None or phot.empty:
            continue

        df.loc[idx, "atlas_has_phot"] = True

        # Separate cyan (c) and orange (o) bands
        if "F" in phot.columns:
            cyan = phot[phot["F"] == "c"]
            orange = phot[phot["F"] == "o"]
        elif "filter" in phot.columns:
            cyan = phot[phot["filter"] == "c"]
            orange = phot[phot["filter"] == "o"]
        else:
            matched += 1
            continue

        mag_col = "m" if "m" in phot.columns else "mag" if "mag" in phot.columns else None
        if mag_col is None:
            matched += 1
            continue

        if len(cyan) > 0:
            c_mags = pd.to_numeric(cyan[mag_col], errors="coerce").dropna()
            df.loc[idx, "atlas_n_det_cyan"] = len(c_mags)
            if len(c_mags) >= 2:
                df.loc[idx, "atlas_cyan_range"] = round(float(c_mags.max() - c_mags.min()), 4)

        if len(orange) > 0:
            o_mags = pd.to_numeric(orange[mag_col], errors="coerce").dropna()
            df.loc[idx, "atlas_n_det_orange"] = len(o_mags)
            if len(o_mags) >= 2:
                df.loc[idx, "atlas_orange_range"] = round(float(o_mags.max() - o_mags.min()), 4)

        if output_dir and not phot.empty:
            cand_id = str(df.loc[idx, "candidate_id"]) if "candidate_id" in df.columns else str(idx)
            phot.to_parquet(Path(output_dir) / f"atlas_lc_{cand_id}.parquet", index=False)

        matched += 1

    print(f"ATLAS: {matched}/{n_valid} candidates with photometry")
    return df


# =============================================================================
# ZTF LIGHT CURVE FETCHING (IRSA TAP)
# =============================================================================

IRSA_TAP_URL = "https://irsa.ipac.caltech.edu/TAP"


def fetch_ztf_lightcurves(
    df: pd.DataFrame,
    radius_arcsec: float = 2.0,
    output_dir: Path | None = None,
    workers: int = 4,
) -> pd.DataFrame:
    """
    Fetch ZTF light curves from IRSA ZTF DR22.

    Queries the ``ztf_objects_dr22`` table for matching objects, then downloads
    their light curves from ``ztf_objects_dr22.lightcurve``.

    Adds columns: ztf_lc_n_det, ztf_lc_g_range, ztf_lc_r_range.
    If *output_dir* is set, saves per-candidate parquet files as
    ``ztf_lc_<candidate_id>.parquet``.
    """
    df = df.copy()
    df["ztf_lc_n_det"] = 0
    df["ztf_lc_g_range"] = np.nan
    df["ztf_lc_r_range"] = np.nan

    valid = df["ra"].notna() & df["dec"].notna()
    if not valid.any():
        return df

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    n_valid = int(valid.sum())
    print(f"ZTF LCs: fetching {n_valid} light curves")

    tap = pyvo.dal.TAPService(IRSA_TAP_URL)

    def _fetch_one(idx: int) -> tuple:
        ra, dec = float(df.loc[idx, "ra"]), float(df.loc[idx, "dec"])
        if not (np.isfinite(ra) and np.isfinite(dec)):
            return (idx, 0, np.nan, np.nan)

        try:
            # Find matching ZTF objects
            obj_query = f"""
            SELECT oid
            FROM ztf_objects_dr22
            WHERE CONTAINS(
                POINT('ICRS', ra, dec),
                CIRCLE('ICRS', {ra:.7f}, {dec:.7f}, {radius_arcsec / 3600.0})
            ) = 1
            """
            obj_result = tap.run_sync(obj_query)
            obj_table = obj_result.to_table()
            if obj_table is None or len(obj_table) == 0:
                return (idx, 0, np.nan, np.nan)

            oids = [str(row["oid"]) for row in obj_table]
            oid_list = ",".join(oids)

            # Fetch light curve data
            lc_query = f"""
            SELECT oid, hjd, mag, magerr, filtercode, catflags
            FROM ztf_objects_dr22.lightcurve
            WHERE oid IN ({oid_list})
            AND catflags = 0
            ORDER BY hjd ASC
            """
            lc_result = tap.run_sync(lc_query)
            lc_table = lc_result.to_table()
            if lc_table is None or len(lc_table) == 0:
                return (idx, 0, np.nan, np.nan)

            lc = lc_table.to_pandas()

            # Normalize column names
            col_map = {}
            for c in lc.columns:
                cl = c.lower()
                if cl == "hjd":
                    col_map[c] = "mjd"
                elif cl == "filtercode":
                    col_map[c] = "band"
                else:
                    col_map[c] = cl
            lc = lc.rename(columns=col_map)

            # Convert HJD to MJD (approximate: MJD = HJD - 2400000.5)
            if "mjd" in lc.columns:
                lc["mjd"] = pd.to_numeric(lc["mjd"], errors="coerce")
                mask = lc["mjd"] > 2400000
                lc.loc[mask, "mjd"] = lc.loc[mask, "mjd"] - 2400000.5

            # Map filter codes to band names
            if "band" in lc.columns:
                band_map = {1: "zg", 2: "zr", 3: "zi"}
                lc["band"] = pd.to_numeric(lc["band"], errors="coerce").map(band_map).fillna(lc["band"].astype(str))

            n_det = len(lc)
            mag = pd.to_numeric(lc.get("mag"), errors="coerce")
            g_mask = lc["band"] == "zg" if "band" in lc.columns else pd.Series(False, index=lc.index)
            r_mask = lc["band"] == "zr" if "band" in lc.columns else pd.Series(False, index=lc.index)
            g_mags = mag[g_mask].dropna()
            r_mags = mag[r_mask].dropna()
            g_range = float(g_mags.max() - g_mags.min()) if len(g_mags) >= 2 else np.nan
            r_range = float(r_mags.max() - r_mags.min()) if len(r_mags) >= 2 else np.nan

            if output_dir and not lc.empty:
                cand_id = str(df.loc[idx, "candidate_id"]) if "candidate_id" in df.columns else str(idx)
                lc.to_parquet(Path(output_dir) / f"ztf_lc_{cand_id}.parquet", index=False)

            return (idx, n_det, g_range, r_range)
        except Exception:
            return (idx, 0, np.nan, np.nan)

    matched = 0
    valid_idx = df.index[valid].tolist()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_fetch_one, idx): idx for idx in valid_idx}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="ZTF LCs"):
            idx, n_det, g_range, r_range = fut.result()
            df.loc[idx, "ztf_lc_n_det"] = n_det
            df.loc[idx, "ztf_lc_g_range"] = g_range
            df.loc[idx, "ztf_lc_r_range"] = r_range
            if n_det > 0:
                matched += 1

    print(f"ZTF LCs: {matched}/{n_valid} with data")
    return df


# =============================================================================
# GAIA DR3 EPOCH PHOTOMETRY
# =============================================================================


def query_gaia_epoch_photometry(
    df: pd.DataFrame,
    chunk_size: int = GAIA_VAR_CHUNK_SIZE,
) -> pd.DataFrame:
    """
    Check Gaia DR3 epoch photometry availability and basic stats.

    Adds columns: gaia_epoch_available, gaia_epoch_n_obs, gaia_epoch_g_range.
    Requires 'gaia_id' column.
    """
    df = df.copy()
    df["gaia_epoch_available"] = False
    df["gaia_epoch_n_obs"] = 0
    df["gaia_epoch_g_range"] = np.nan

    if "gaia_id" not in df.columns:
        print("Warning: Gaia epoch photometry requires 'gaia_id' column, skipping")
        return df

    valid = df["gaia_id"].notna()
    if not valid.any():
        return df

    # Normalize IDs
    gaia_ids = []
    idx_map = {}
    for idx, val in df.loc[valid, "gaia_id"].items():
        sid = str(val).strip()
        if sid.isdigit():
            gaia_ids.append(sid)
            idx_map.setdefault(sid, []).append(idx)
    gaia_ids = list(set(gaia_ids))

    if not gaia_ids:
        return df

    print(f"Gaia epoch photometry: checking {len(gaia_ids)} source_ids")
    tap = None
    for tap_url in GAIA_TAP_URLS:
        try:
            _tap = pyvo.dal.TAPService(tap_url)
            test_query = f"SELECT source_id FROM gaiadr3.vari_summary WHERE source_id = {gaia_ids[0]}"
            _tap.run_sync(test_query)
            tap = _tap
            break
        except Exception:
            continue

    if tap is None:
        print("  Warning: all Gaia TAP servers unreachable, skipping epoch photometry")
        return df

    # Query vari_summary for observation counts and magnitude ranges
    # (epoch photometry itself is huge — we use vari_summary stats instead)
    epoch_results = {}
    for i in tqdm(range(0, len(gaia_ids), chunk_size), desc="Gaia epoch stats"):
        chunk = gaia_ids[i : i + chunk_size]
        ids_str = ",".join(chunk)
        query = f"""
            SELECT source_id,
                   num_selected_g_fov,
                   range_mag_g_fov
            FROM gaiadr3.vari_summary
            WHERE source_id IN ({ids_str})
        """
        for attempt in range(3):
            try:
                result = tap.run_sync(query)
                for row in result:
                    sid = str(row["source_id"])
                    n_obs = int(row["num_selected_g_fov"]) if row["num_selected_g_fov"] is not None else 0
                    g_range = float(row["range_mag_g_fov"]) if row["range_mag_g_fov"] is not None else np.nan
                    epoch_results[sid] = (n_obs, g_range)
                break
            except Exception as e:
                if attempt < 2:
                    time.sleep(5 * (attempt + 1))
                else:
                    print(f"  Gaia epoch stats chunk {i} failed: {e}")

    # Apply
    matched = 0
    for sid, indices in idx_map.items():
        info = epoch_results.get(sid)
        if info is None:
            continue
        n_obs, g_range = info
        for idx in indices:
            df.loc[idx, "gaia_epoch_available"] = n_obs > 0
            df.loc[idx, "gaia_epoch_n_obs"] = n_obs
            df.loc[idx, "gaia_epoch_g_range"] = g_range
            matched += 1

    print(f"Gaia epoch photometry: {matched} sources with time-series data")
    return df


def fetch_gaia_epoch_lcs(
    df: pd.DataFrame,
    chunk_size: int = 50,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Download full Gaia DR3 epoch photometry time series.

    Only fetches for candidates with ``gaia_epoch_available == True`` (or that
    have a valid ``gaia_id``).  Stores per-candidate parquet files as
    ``gaia_epoch_lc_<candidate_id>.parquet``.

    Adds columns: gaia_epoch_lc_n_g, gaia_epoch_lc_g_range.
    """
    df = df.copy()
    df["gaia_epoch_lc_n_g"] = 0
    df["gaia_epoch_lc_g_range"] = np.nan

    if "gaia_id" not in df.columns:
        print("Gaia epoch LCs: requires 'gaia_id' column, skipping")
        return df

    # Only fetch for candidates with epoch photometry available
    if "gaia_epoch_available" in df.columns:
        valid = df["gaia_id"].notna() & df["gaia_epoch_available"].astype(bool)
    else:
        valid = df["gaia_id"].notna()
    if not valid.any():
        print("Gaia epoch LCs: no candidates with epoch photometry available")
        return df

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Build gaia_id -> index mapping
    gaia_ids = []
    idx_map: dict[str, list] = {}
    for idx, val in df.loc[valid, "gaia_id"].items():
        sid = str(val).strip()
        if sid.isdigit():
            gaia_ids.append(sid)
            idx_map.setdefault(sid, []).append(idx)
    gaia_ids = list(set(gaia_ids))

    if not gaia_ids:
        return df

    n_total = len(gaia_ids)
    print(f"Gaia epoch LCs: downloading time series for {n_total} sources")

    # Find working TAP server
    tap = None
    for tap_url in GAIA_TAP_URLS:
        try:
            _tap = pyvo.dal.TAPService(tap_url)
            test = f"SELECT source_id FROM gaiadr3.epoch_photometry WHERE source_id = {gaia_ids[0]} AND transit_id IS NOT NULL"
            _tap.run_sync(test, maxrec=1)
            tap = _tap
            break
        except Exception:
            continue

    if tap is None:
        print("  Warning: all Gaia TAP servers unreachable, skipping epoch LC download")
        return df

    matched = 0
    for i in tqdm(range(0, len(gaia_ids), chunk_size), desc="Gaia epoch LCs"):
        chunk = gaia_ids[i : i + chunk_size]
        ids_str = ",".join(chunk)
        query = f"""
            SELECT source_id, transit_id,
                   g_transit_time AS "time",
                   g_transit_mag AS mag,
                   g_transit_mag_error AS mag_error,
                   'G' AS band,
                   rejected_by_variability
            FROM gaiadr3.epoch_photometry
            WHERE source_id IN ({ids_str})
            ORDER BY source_id, g_transit_time
        """
        for attempt in range(3):
            try:
                result = tap.run_sync(query)
                table = result.to_table()
                if table is None or len(table) == 0:
                    break
                lc_all = table.to_pandas()

                # Process per source
                for sid in chunk:
                    sid_int = int(sid)
                    src_lc = lc_all[lc_all["source_id"] == sid_int].copy()
                    if src_lc.empty:
                        continue

                    src_lc["time"] = pd.to_numeric(src_lc["time"], errors="coerce")
                    src_lc["mag"] = pd.to_numeric(src_lc["mag"], errors="coerce")
                    src_lc["mag_error"] = pd.to_numeric(src_lc["mag_error"], errors="coerce")
                    src_lc = src_lc.dropna(subset=["time", "mag"])

                    n_g = len(src_lc)
                    g_mags = src_lc["mag"].dropna()
                    g_range = float(g_mags.max() - g_mags.min()) if len(g_mags) >= 2 else np.nan

                    for df_idx in idx_map.get(sid, []):
                        df.loc[df_idx, "gaia_epoch_lc_n_g"] = n_g
                        df.loc[df_idx, "gaia_epoch_lc_g_range"] = g_range

                    if output_dir and not src_lc.empty:
                        for df_idx in idx_map.get(sid, []):
                            cand_id = str(df.loc[df_idx, "candidate_id"]) if "candidate_id" in df.columns else str(df_idx)
                            src_lc.to_parquet(Path(output_dir) / f"gaia_epoch_lc_{cand_id}.parquet", index=False)
                            break  # one file per gaia source

                    matched += 1
                break
            except Exception as e:
                if attempt < 2:
                    time.sleep(5 * (attempt + 1))
                else:
                    print(f"  Gaia epoch LC chunk {i} failed: {e}")

    print(f"Gaia epoch LCs: {matched}/{n_total} with time-series data")
    return df


# =============================================================================
# eROSITA X-RAY CATALOG
# =============================================================================


def crossmatch_erosita(
    df: pd.DataFrame,
    radius_arcsec: float = EROSITA_RADIUS_ARCSEC,
    chunk_size: int = 1000,
    method: Literal["tap", "xmatch", "local"] = "tap",
    local_fits: Path | str | None = None,
) -> pd.DataFrame:
    """
    Crossmatch against eROSITA-DE DR1 (Merloni+2024).

    method='tap'    — batch VizieR TAP upload (best for large batches).
    method='xmatch' — CDS XMatch service (reliable for small batches).
    method='local'  — local FITS crossmatch via SkyCoord (instant, no network).

    X-ray detection is a strong youth indicator for YSO candidates.
    Adds columns: xray_det, xray_flux, xray_sep_arcsec.
    """
    df = df.copy()
    df["xray_det"] = False
    df["xray_flux"] = np.nan
    df["xray_sep_arcsec"] = np.nan

    valid = df["ra"].notna() & df["dec"].notna()
    if not valid.any():
        return df

    n_valid = int(valid.sum())

    if method == "local":
        return _erosita_via_local(df, valid, n_valid, radius_arcsec, local_fits)
    elif method == "xmatch":
        print(f"eROSITA: crossmatching {n_valid} candidates via CDS XMatch (radius={radius_arcsec}\")")
        source_table = Table()
        source_table["_idx"] = np.array(df.index[valid])
        source_table["ra"] = df.loc[valid, "ra"].values
        source_table["dec"] = df.loc[valid, "dec"].values

        try:
            result_tab = XMatch.query(
                cat1=source_table,
                cat2="vizier:J/A+A/682/A34/erass1-m",
                max_distance=radius_arcsec * u.arcsec,
                colRA1="ra", colDec1="dec",
            )
            result = result_tab.to_pandas() if result_tab is not None and len(result_tab) > 0 else pd.DataFrame()
        except Exception as e:
            print(f"eROSITA: XMatch query failed: {e}")
            result = pd.DataFrame()

        sep_col = "angDist" if "angDist" in result.columns else "sep_arcsec"
        if not result.empty and sep_col in result.columns:
            result = result.sort_values(sep_col).drop_duplicates(subset="_idx", keep="first")
    else:
        print(f"eROSITA: crossmatching {n_valid} candidates via TAP (radius={radius_arcsec}\")")
        coords_df = pd.DataFrame({
            "_idx": df.index[valid],
            "ra": df.loc[valid, "ra"].values,
            "dec": df.loc[valid, "dec"].values,
        })
        result = batch_tap_crossmatch(
            coords_df,
            tap_url=VIZIER_TAP_URL,
            catalog_table='"J/A+A/682/A34/erass1-m"',
            select_cols='c."MLFlux1"',
            ra_col="RA_ICRS",
            dec_col="DE_ICRS",
            match_radius_arcsec=radius_arcsec,
            chunk_size=chunk_size,
            n_workers=4,
            verbose=True,
            desc="eROSITA TAP",
        )
        sep_col = "sep_arcsec"
        if not result.empty:
            result = result.sort_values(sep_col).drop_duplicates(subset="_idx", keep="first")

    matched = 0
    if not result.empty:
        for _, row in result.iterrows():
            idx = int(row["_idx"])
            if idx in df.index:
                df.loc[idx, "xray_det"] = True
                sep = row.get("angDist", row.get("sep_arcsec", np.nan))
                df.loc[idx, "xray_sep_arcsec"] = round(float(sep), 3) if pd.notna(sep) else np.nan
                try:
                    df.loc[idx, "xray_flux"] = float(row["MLFlux1"])
                except (ValueError, TypeError, KeyError):
                    pass
                matched += 1

    print(f"eROSITA: {matched} X-ray matches")
    return df


def _erosita_via_local(
    df: pd.DataFrame, valid, n_valid: int, radius_arcsec: float,
    local_fits: Path | str | None = None,
) -> pd.DataFrame:
    """eROSITA crossmatch via local FITS + SkyCoord (instant)."""
    from astropy.io import fits as pyfits

    fits_path = Path(local_fits) if local_fits else EROSITA_LOCAL_FITS
    if not fits_path.exists():
        print(f"eROSITA: local FITS not found at {fits_path}, skipping")
        return df

    # Load and cache the catalog (only RA, DEC, ML_FLUX_1)
    cache_key = str(fits_path)
    if cache_key not in _erosita_cache:
        print(f"eROSITA: loading local catalog from {fits_path.name}...")
        with pyfits.open(fits_path, memmap=True) as hdul:
            tbl = hdul[1].data
            ra_arr = tbl["RA"].astype(np.float64)
            dec_arr = tbl["DEC"].astype(np.float64)
            flux_arr = tbl["ML_FLUX_1"].astype(np.float32)
        cat_coord = SkyCoord(ra=ra_arr, dec=dec_arr, unit="deg")
        _erosita_cache[cache_key] = (flux_arr, cat_coord)
        print(f"eROSITA: cached {len(ra_arr)} sources")

    flux_arr, cat_coord = _erosita_cache[cache_key]

    print(f"eROSITA: crossmatching {n_valid} candidates via local catalog (radius={radius_arcsec}\")")

    src_coord = SkyCoord(
        ra=df.loc[valid, "ra"].values, dec=df.loc[valid, "dec"].values, unit="deg",
    )
    idx_cat, sep2d, _ = src_coord.match_to_catalog_sky(cat_coord)
    max_sep = radius_arcsec * u.arcsec

    matched = 0
    for i, df_idx in enumerate(df.index[valid]):
        if sep2d[i] <= max_sep:
            df.loc[df_idx, "xray_det"] = True
            df.loc[df_idx, "xray_sep_arcsec"] = round(sep2d[i].arcsec, 3)
            try:
                df.loc[df_idx, "xray_flux"] = float(flux_arr[idx_cat[i]])
            except (ValueError, TypeError):
                pass
            matched += 1

    print(f"eROSITA: {matched} X-ray matches")
    return df


# =============================================================================
# PROPER MOTION CONSISTENCY
# =============================================================================


def check_pm_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Check proper motion consistency with cluster membership.

    For candidates that have cluster_name and proper motions (pmra, pmdec),
    compute the offset from the cluster mean PM in sigma units.

    Adds column: pm_cluster_offset_sigma.
    """
    df = df.copy()
    df["pm_cluster_offset_sigma"] = np.nan

    required = {"cluster_name", "pmra", "pmdec", "pmra_error" if "pmra_error" in df.columns else "pmra"}
    has_cluster = "cluster_name" in df.columns
    has_pm = "pmra" in df.columns and "pmdec" in df.columns
    if not has_cluster or not has_pm:
        print("PM consistency: requires cluster_name, pmra, pmdec columns, skipping")
        return df

    # Find candidates with cluster membership
    in_cluster = df["cluster_name"].notna() & (df["cluster_name"] != "")
    if not in_cluster.any():
        print("PM consistency: no candidates with cluster membership")
        return df

    # Compute cluster mean PM from the candidates themselves (grouped by cluster)
    cluster_groups = df.loc[in_cluster].groupby("cluster_name")
    cluster_stats = {}
    for name, group in cluster_groups:
        pm_ra = group["pmra"].dropna()
        pm_dec = group["pmdec"].dropna()
        if len(pm_ra) >= 2 and len(pm_dec) >= 2:
            cluster_stats[name] = {
                "pmra_mean": pm_ra.mean(),
                "pmdec_mean": pm_dec.mean(),
                "pmra_std": max(pm_ra.std(), 0.5),  # floor at 0.5 mas/yr
                "pmdec_std": max(pm_dec.std(), 0.5),
            }

    if not cluster_stats:
        # If only single members per cluster, use PM errors if available
        pmra_err_col = "pmra_error" if "pmra_error" in df.columns else None
        pmdec_err_col = "pmdec_error" if "pmdec_error" in df.columns else None
        if pmra_err_col and pmdec_err_col:
            for idx in df.index[in_cluster]:
                # No cluster mean available — flag as nan
                pass
        print("PM consistency: insufficient cluster members for PM comparison")
        return df

    # Compute offset
    matched = 0
    for idx in df.index[in_cluster]:
        cluster = df.loc[idx, "cluster_name"]
        stats = cluster_stats.get(cluster)
        if stats is None:
            continue
        pmra = df.loc[idx, "pmra"]
        pmdec = df.loc[idx, "pmdec"]
        if pd.isna(pmra) or pd.isna(pmdec):
            continue

        d_ra = (pmra - stats["pmra_mean"]) / stats["pmra_std"]
        d_dec = (pmdec - stats["pmdec_mean"]) / stats["pmdec_std"]
        offset_sigma = np.sqrt(d_ra**2 + d_dec**2)
        df.loc[idx, "pm_cluster_offset_sigma"] = round(float(offset_sigma), 2)
        matched += 1

    print(f"PM consistency: computed for {matched} cluster members")
    return df


# =============================================================================
# NEOWISE LIGHT CURVES
# =============================================================================


def query_neowise_lightcurves(
    df: pd.DataFrame,
    max_sep_arcsec: float = NEOWISE_MAX_SEP_ARCSEC,
    output_dir: Path | None = None,
    workers: int = 4,
) -> pd.DataFrame:
    """
    Fetch full NEOWISE light curves for candidates.

    Stores per-epoch W1/W2 photometry (if output_dir set, saves individual LC parquets).
    Adds columns: neowise_n_epochs, neowise_w1_range, neowise_w2_range.
    """
    from astroquery.ipac.irsa import Irsa

    df = df.copy()
    df["neowise_n_epochs"] = 0
    df["neowise_w1_range"] = np.nan
    df["neowise_w2_range"] = np.nan

    valid = df["ra"].notna() & df["dec"].notna()
    if not valid.any():
        return df

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    n_valid = int(valid.sum())
    print(f"NEOWISE LCs: fetching {n_valid} light curves")

    def _fetch_one(idx: int) -> tuple:
        ra, dec = float(df.loc[idx, "ra"]), float(df.loc[idx, "dec"])
        if not (np.isfinite(ra) and np.isfinite(dec)):
            return (idx, 0, np.nan, np.nan)

        query = f"""
        SELECT mjd, w1mpro, w1sigmpro, w2mpro, w2sigmpro, w1snr, w2snr,
               qual_frame, qi_fact, cc_flags
        FROM neowiser_p1bs_psd
        WHERE CONTAINS(
            POINT('ICRS', ra, dec),
            CIRCLE('ICRS', {ra:.7f}, {dec:.7f}, {max_sep_arcsec / 3600.0})
        ) = 1
        ORDER BY mjd ASC
        """
        try:
            result = Irsa.query_tap(query)
            table = result.to_table()
            if table is None or len(table) == 0:
                return (idx, 0, np.nan, np.nan)

            lc = table.to_pandas()

            # Quality filters (same as characterize.py)
            if "qual_frame" in lc.columns:
                qual = pd.to_numeric(lc["qual_frame"], errors="coerce")
                lc = lc[qual.isin([0, 1])]
            if "cc_flags" in lc.columns:
                cc = lc["cc_flags"].astype(str)
                lc = lc[~cc.str.contains("[^0]", regex=True, na=False)]
            if "qi_fact" in lc.columns:
                qf = pd.to_numeric(lc["qi_fact"], errors="coerce")
                lc = lc[qf >= 0.9]
            if "w1snr" in lc.columns:
                lc = lc[pd.to_numeric(lc["w1snr"], errors="coerce") >= 3.0]

            if lc.empty:
                return (idx, 0, np.nan, np.nan)

            w1 = pd.to_numeric(lc.get("w1mpro"), errors="coerce").dropna()
            w2 = pd.to_numeric(lc.get("w2mpro"), errors="coerce").dropna()
            n_epochs = len(lc)
            w1_range = float(w1.max() - w1.min()) if len(w1) >= 2 else np.nan
            w2_range = float(w2.max() - w2.min()) if len(w2) >= 2 else np.nan

            # Save individual LC if output_dir set
            if output_dir and not lc.empty:
                cand_id = str(df.loc[idx, "candidate_id"]) if "candidate_id" in df.columns else str(idx)
                lc.to_parquet(Path(output_dir) / f"neowise_lc_{cand_id}.parquet", index=False)

            return (idx, n_epochs, w1_range, w2_range)
        except Exception:
            return (idx, 0, np.nan, np.nan)

    matched = 0
    valid_idx = df.index[valid].tolist()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_fetch_one, idx): idx for idx in valid_idx}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="NEOWISE LCs"):
            idx, n_epochs, w1_range, w2_range = fut.result()
            df.loc[idx, "neowise_n_epochs"] = n_epochs
            df.loc[idx, "neowise_w1_range"] = w1_range
            df.loc[idx, "neowise_w2_range"] = w2_range
            if n_epochs > 0:
                matched += 1

    print(f"NEOWISE LCs: {matched}/{n_valid} with data")
    return df


# =============================================================================
# TESS LIGHT CURVES
# =============================================================================


def fetch_tess_lightcurves(
    df: pd.DataFrame,
    output_dir: Path | None = None,
    workers: int = 2,
) -> pd.DataFrame:
    """
    Fetch TESS light curves via ``lightkurve``.

    Prefers 2-min cadence SPOC, falls back to QLP/FFI products.
    If review-mode TESS overlays are re-enabled later, this fetcher should be
    tightened to choose one best search match/product per sector instead of
    concatenating every light curve returned by the search cone.

    Adds columns: tess_n_sectors, tess_total_points, tess_flux_range.
    If *output_dir* is set, saves per-candidate parquet files as
    ``tess_lc_<candidate_id>.parquet``.
    """
    try:
        import lightkurve as lk
    except ImportError:
        print("TESS LCs: lightkurve not installed, skipping (pip install lightkurve)")
        df = df.copy()
        df["tess_n_sectors"] = 0
        df["tess_total_points"] = 0
        df["tess_flux_range"] = np.nan
        return df

    df = df.copy()
    df["tess_n_sectors"] = 0
    df["tess_total_points"] = 0
    df["tess_flux_range"] = np.nan

    valid = df["ra"].notna() & df["dec"].notna()
    if not valid.any():
        return df

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    n_valid = int(valid.sum())
    print(f"TESS LCs: fetching {n_valid} light curves")

    def _fetch_one(idx: int) -> tuple:
        ra, dec = float(df.loc[idx, "ra"]), float(df.loc[idx, "dec"])
        if not (np.isfinite(ra) and np.isfinite(dec)):
            return (idx, 0, 0, np.nan)

        try:
            coord = SkyCoord(ra=ra, dec=dec, unit="deg")
            search = lk.search_lightcurve(coord, radius=21, mission="TESS")
            if search is None or len(search) == 0:
                return (idx, 0, 0, np.nan)

            # Prefer SPOC 2-min, then QLP, then any
            spoc = search[search.author == "SPOC"]
            if len(spoc) > 0:
                lc_collection = spoc.download_all(quality_bitmask="default")
            else:
                qlp = search[search.author == "QLP"]
                if len(qlp) > 0:
                    lc_collection = qlp.download_all(quality_bitmask="default")
                else:
                    lc_collection = search.download_all(quality_bitmask="default")

            if lc_collection is None or len(lc_collection) == 0:
                return (idx, 0, 0, np.nan)

            rows = []
            sectors = set()
            for lc_obj in lc_collection:
                t = lc_obj.time.value
                f = lc_obj.flux.value
                fe = lc_obj.flux_err.value if lc_obj.flux_err is not None else np.full_like(f, np.nan)
                q = lc_obj.quality.value if hasattr(lc_obj, "quality") and lc_obj.quality is not None else np.zeros(len(t), dtype=int)
                sector = getattr(lc_obj.meta, "SECTOR", None) if hasattr(lc_obj, "meta") else None
                if sector is None:
                    sector = getattr(lc_obj, "SECTOR", 0)
                sectors.add(sector)
                for j in range(len(t)):
                    rows.append({
                        "time": float(t[j]),
                        "flux": float(f[j]),
                        "flux_err": float(fe[j]),
                        "quality": int(q[j]),
                        "sector": int(sector) if sector is not None else 0,
                    })

            if not rows:
                return (idx, 0, 0, np.nan)

            lc_df = pd.DataFrame(rows)
            lc_df = lc_df[np.isfinite(lc_df["flux"])].copy()

            n_sectors = len(sectors)
            total_points = len(lc_df)
            flux_vals = lc_df["flux"].dropna()
            flux_range = float(flux_vals.max() - flux_vals.min()) if len(flux_vals) >= 2 else np.nan

            if output_dir and not lc_df.empty:
                cand_id = str(df.loc[idx, "candidate_id"]) if "candidate_id" in df.columns else str(idx)
                lc_df.to_parquet(Path(output_dir) / f"tess_lc_{cand_id}.parquet", index=False)

            return (idx, n_sectors, total_points, flux_range)
        except Exception:
            return (idx, 0, 0, np.nan)

    matched = 0
    valid_idx = df.index[valid].tolist()

    # lightkurve queries MAST — use low parallelism
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_fetch_one, idx): idx for idx in valid_idx}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="TESS LCs"):
            idx, n_sectors, total_points, flux_range = fut.result()
            df.loc[idx, "tess_n_sectors"] = n_sectors
            df.loc[idx, "tess_total_points"] = total_points
            df.loc[idx, "tess_flux_range"] = flux_range
            if n_sectors > 0:
                matched += 1

    print(f"TESS LCs: {matched}/{n_valid} with data")
    return df


# =============================================================================
# KEPLER/K2 LIGHT CURVES
# =============================================================================


def fetch_kepler_k2_lightcurves(
    df: pd.DataFrame,
    output_dir: Path | None = None,
    workers: int = 2,
) -> pd.DataFrame:
    """
    Fetch Kepler/K2 light curves via ``lightkurve``.

    Adds columns: kepler_n_quarters, kepler_total_points, kepler_flux_range.
    If *output_dir* is set, saves per-candidate parquet files as
    ``kepler_lc_<candidate_id>.parquet``.
    """
    try:
        import lightkurve as lk
    except ImportError:
        print("Kepler LCs: lightkurve not installed, skipping")
        df = df.copy()
        df["kepler_n_quarters"] = 0
        df["kepler_total_points"] = 0
        df["kepler_flux_range"] = np.nan
        return df

    df = df.copy()
    df["kepler_n_quarters"] = 0
    df["kepler_total_points"] = 0
    df["kepler_flux_range"] = np.nan

    valid = df["ra"].notna() & df["dec"].notna()
    if not valid.any():
        return df

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    n_valid = int(valid.sum())
    print(f"Kepler/K2 LCs: fetching {n_valid} light curves")

    def _fetch_one(idx: int) -> tuple:
        ra, dec = float(df.loc[idx, "ra"]), float(df.loc[idx, "dec"])
        if not (np.isfinite(ra) and np.isfinite(dec)):
            return (idx, 0, 0, np.nan)

        try:
            coord = SkyCoord(ra=ra, dec=dec, unit="deg")
            search = lk.search_lightcurve(coord, radius=21, mission=("Kepler", "K2"))
            if search is None or len(search) == 0:
                return (idx, 0, 0, np.nan)

            lc_collection = search.download_all()
            if lc_collection is None or len(lc_collection) == 0:
                return (idx, 0, 0, np.nan)

            rows = []
            quarters = set()
            for lc_obj in lc_collection:
                t = lc_obj.time.value
                f = lc_obj.flux.value
                fe = lc_obj.flux_err.value if lc_obj.flux_err is not None else np.full_like(f, np.nan)
                q = lc_obj.quality.value if hasattr(lc_obj, "quality") and lc_obj.quality is not None else np.zeros(len(t), dtype=int)
                quarter = getattr(lc_obj.meta, "QUARTER", getattr(lc_obj.meta, "CAMPAIGN", None)) if hasattr(lc_obj, "meta") else None
                if quarter is None:
                    quarter = getattr(lc_obj, "QUARTER", getattr(lc_obj, "CAMPAIGN", 0))
                quarters.add(quarter)
                for j in range(len(t)):
                    rows.append({
                        "time": float(t[j]),
                        "flux": float(f[j]),
                        "flux_err": float(fe[j]),
                        "quality": int(q[j]),
                        "quarter": int(quarter) if quarter is not None else 0,
                    })

            if not rows:
                return (idx, 0, 0, np.nan)

            lc_df = pd.DataFrame(rows)
            lc_df = lc_df[np.isfinite(lc_df["flux"])].copy()

            n_quarters = len(quarters)
            total_points = len(lc_df)
            flux_vals = lc_df["flux"].dropna()
            flux_range = float(flux_vals.max() - flux_vals.min()) if len(flux_vals) >= 2 else np.nan

            if output_dir and not lc_df.empty:
                cand_id = str(df.loc[idx, "candidate_id"]) if "candidate_id" in df.columns else str(idx)
                lc_df.to_parquet(Path(output_dir) / f"kepler_lc_{cand_id}.parquet", index=False)

            return (idx, n_quarters, total_points, flux_range)
        except Exception:
            return (idx, 0, 0, np.nan)

    matched = 0
    valid_idx = df.index[valid].tolist()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_fetch_one, idx): idx for idx in valid_idx}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Kepler LCs"):
            idx, n_quarters, total_points, flux_range = fut.result()
            df.loc[idx, "kepler_n_quarters"] = n_quarters
            df.loc[idx, "kepler_total_points"] = total_points
            df.loc[idx, "kepler_flux_range"] = flux_range
            if n_quarters > 0:
                matched += 1

    print(f"Kepler/K2 LCs: {matched}/{n_valid} with data")
    return df


# =============================================================================
# AAVSO LIGHT CURVES
# =============================================================================


def fetch_aavso_lightcurves(
    df: pd.DataFrame,
    output_dir: Path | None = None,
    workers: int = 4,
) -> pd.DataFrame:
    """
    Fetch AAVSO light curves via WebObs scraping.

    Adds columns: aavso_lc_n_points.
    If *output_dir* is set, saves per-candidate parquet files as
    ``aavso_lc_<candidate_id>.parquet``.
    """
    df = df.copy()
    df["aavso_lc_n_points"] = 0
    num_pages = 50 # Limit max pages (10,000 pts) per source

    # Needs to match by simbad_main_id, tns_name, asassn_var_name or vsx_name if available.
    # AAVSO searches rely on star name. If no name, skipped.
    name_cols = [c for c in ["simbad_main_id", "asassn_var_name", "vsx_name", "tns_name", "ztf_var_name", "candidate_id"] if c in df.columns]
    
    valid = pd.Series(False, index=df.index)
    best_names = pd.Series("", index=df.index)
    
    for idx in df.index:
        for col in name_cols:
            val = df.loc[idx, col]
            if pd.notna(val) and str(val).strip() != "" and "J" not in str(val) and "TIC" not in str(val) and "Gaia" not in str(val) and len(str(val)) < 20: 
                valid[idx] = True
                best_names[idx] = str(val).strip()
                break
                
    # Fall back to vsx_name if simbad is not available, try to format correctly.
    # In vetting, it's mostly "V* XX YYY" or similar which works well.
    if not valid.any():
        return df

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    n_valid = int(valid.sum())
    print(f"AAVSO LCs: fetching {n_valid} light curves by name")

    def _fetch_one(idx: int) -> tuple:
        star_name = best_names[idx]
        obj_url = star_name.replace("V*", "").strip().replace(" ", "+")
        
        try:
            dfs = []
            import io
            for page in range(1, num_pages + 1):
                url = f"https://app.aavso.org/webobs/results/?star={obj_url}&num_results=200&obs_types=ccd&page={page}"
                res = requests.get(url, timeout=10)
                if "No observations found" in res.text or "Error" in res.text:
                    break
                
                try:
                    tables = pd.read_html(io.StringIO(res.text))
                except ValueError:
                    break
                
                if not tables:
                    break
                
                page_df = tables[0]
                # Columns are ['Star', 'JD', 'Calendar Date', 'Mag', 'Err', 'Filter', 'Observer', 'Cmp1', 'Cmp2', 'Chart', 'Comments', ...]
                if "JD" not in page_df.columns or "Mag" not in page_df.columns:
                    break
                    
                dfs.append(page_df[["JD", "Mag", "Err", "Filter", "Observer"]])
                if len(page_df) < 200:
                    break
            
            if not dfs:
                return (idx, 0)
                
            lc_df = pd.concat(dfs, ignore_index=True)
            # Clean up types and limit rows with values
            lc_df["Mag"] = pd.to_numeric(lc_df["Mag"].astype(str).str.replace("<", ""), errors="coerce")
            lc_df["Err"] = pd.to_numeric(lc_df["Err"], errors="coerce")
            lc_df["JD"] = pd.to_numeric(lc_df["JD"], errors="coerce")
            lc_df = lc_df.dropna(subset=["JD", "Mag"])
            
            # Map columns to lowercase standard
            lc_df = lc_df.rename(columns={"JD": "mjd", "Mag": "mag", "Err": "mag_err", "Filter": "filter", "Observer": "observer"})
            lc_df["mjd"] = lc_df["mjd"] - 2400000.5 # Convert JD to MJD
            
            n_points = len(lc_df)
            
            if output_dir and n_points > 0:
                cand_id = str(df.loc[idx, "candidate_id"]) if "candidate_id" in df.columns else str(idx)
                lc_df.to_parquet(Path(output_dir) / f"aavso_lc_{cand_id}.parquet", index=False)

            return (idx, n_points)
        except Exception:
            return (idx, 0)

    matched = 0
    valid_idx = df.index[valid].tolist()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_fetch_one, idx): idx for idx in valid_idx}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="AAVSO LCs"):
            idx, n_points = fut.result()
            df.loc[idx, "aavso_lc_n_points"] = n_points
            if n_points > 0:
                matched += 1

    print(f"AAVSO LCs: {matched}/{n_valid} with data")
    return df


# =============================================================================
# PAN-STARRS LIGHT CURVES
# =============================================================================


def fetch_panstarrs_lightcurves(
    df: pd.DataFrame,
    output_dir: Path | None = None,
    workers: int = 4,
) -> pd.DataFrame:
    """
    Fetch Pan-STARRS (PS1 DR2) epoch photometry.

    Adds columns: ps1_lc_n_points.
    If *output_dir* is set, saves per-candidate parquet files as
    ``ps1_lc_<candidate_id>.parquet``.
    """
    df = df.copy()
    df["ps1_lc_n_points"] = 0

    valid = df["ra"].notna() & df["dec"].notna()
    if not valid.any():
        return df

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    n_valid = int(valid.sum())
    print(f"Pan-STARRS LCs: fetching {n_valid} light curves")

    def _fetch_one(idx: int) -> tuple:
        ra, dec = float(df.loc[idx, "ra"]), float(df.loc[idx, "dec"])
        if not (np.isfinite(ra) and np.isfinite(dec)):
            return (idx, 0)

        # Skip southern hemisphere queries (-30 limit for PS1)
        if dec < -30.5:
            return (idx, 0)

        try:
            url = f"https://catalogs.mast.stsci.edu/api/v0.1/panstarrs/dr2/detection.csv?ra={ra}&dec={dec}&radius=0.0015&pagesize=10000&format=csv"
            res = requests.get(url, timeout=20)
            if res.status_code != 200 or "obsTime" not in res.text:
                return (idx, 0)

            import io
            lc_df = pd.read_csv(io.StringIO(res.text))
            
            if lc_df.empty or "obsTime" not in lc_df.columns:
                return (idx, 0)

            # Filter by infoFlag if present
            if "infoFlag" in lc_df.columns:
                # Keep only detections without DEFECT(2048), SATURATED(4096), FIT_FAIL(8)
                bad_mask = (
                    ((lc_df["infoFlag"] & 2048) != 0) | 
                    ((lc_df["infoFlag"] & 4096) != 0) | 
                    ((lc_df["infoFlag"] & 8) != 0)
                )
                lc_df = lc_df[~bad_mask].copy()

            if lc_df.empty:
                return (idx, 0)
                
            # Rename for consistency mapping
            lc_df = lc_df.rename(columns={
                "filterID": "filter",
                "obsTime": "mjd",
                "psfFlux": "flux_psf",
                "psfFluxErr": "flux_psf_err"
            })
            
            # Map filters from ID to string (1=g, 2=r, 3=i, 4=z, 5=y)
            filter_map = {1: "g_ps", 2: "r_ps", 3: "i_ps", 4: "z_ps", 5: "y_ps"}
            lc_df["filter"] = lc_df["filter"].map(filter_map)
            
            # Convert AB fluxes to AB magnitudes properly (-2.5*log10(flux) + 8.90) 
            # PS1 fluxes are in Jansky * 10^36... actually MAST API returns 
            # Jy according to MAST schema? No, it's microJanskys or similar.
            # Lightcurvy uses `mag_psf = -2.5*log10(flux_psf) + 8.90` (mJy -> AB_mag)
            
            valid_flux = lc_df["flux_psf"] > 0
            lc_df = lc_df[valid_flux].copy()
            
            lc_df["mag"] = -2.5 * np.log10(lc_df["flux_psf"]) + 8.90
            lc_df["mag_err"] = 1.08 * (lc_df["flux_psf_err"] / lc_df["flux_psf"])
            
            # Cleanup
            lc_df = lc_df.dropna(subset=["mjd", "mag"])
            # MAST's obsTime can arrive as JD; normalize to actual MJD to match our schema.
            if not lc_df.empty and float(pd.to_numeric(lc_df["mjd"], errors="coerce").median()) > 1_000_000.0:
                lc_df["mjd"] = pd.to_numeric(lc_df["mjd"], errors="coerce") - 2400000.5
            n_points = len(lc_df)

            if output_dir and n_points > 0:
                cand_id = str(df.loc[idx, "candidate_id"]) if "candidate_id" in df.columns else str(idx)
                lc_df.to_parquet(Path(output_dir) / f"ps1_lc_{cand_id}.parquet", index=False)

            return (idx, n_points)
        except Exception:
            return (idx, 0)

    matched = 0
    valid_idx = df.index[valid].tolist()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_fetch_one, idx): idx for idx in valid_idx}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Pan-STARRS LCs"):
            idx, n_points = fut.result()
            df.loc[idx, "ps1_lc_n_points"] = n_points
            if n_points > 0:
                matched += 1

    print(f"Pan-STARRS LCs: {matched}/{n_valid} with data")
    return df


# =============================================================================
# CRTS LIGHT CURVES
# =============================================================================


def fetch_crts_lightcurves(
    df: pd.DataFrame,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Fetch CRTS light curves using VizieR TAP (II/341/data table).

    Adds columns: crts_lc_n_points.
    If *output_dir* is set, saves per-candidate parquet files as
    ``crts_lc_<candidate_id>.parquet``.
    """
    df = df.copy()
    df["crts_lc_n_points"] = 0

    valid = df["ra"].notna() & df["dec"].notna()
    if not valid.any():
        return df

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    n_valid = int(valid.sum())
    print(f"CRTS LCs: fetching {n_valid} light curves via TAP")

    try:
        import pyvo
    except ImportError:
        print("CRTS LCs: pyvo not installed, skipping")
        return df

    # We do a batch query against II/341/data which has epoch photometry.
    # Because II/341/data is massive, doing a batch crossmatch is best.
    coords_df = pd.DataFrame({
        "_idx": df.index[valid],
        "ra": df.loc[valid, "ra"].values,
        "dec": df.loc[valid, "dec"].values,
    })
    
    # Actually, II/341/data does not have RA/DEC, only "ID" which links to II/341/ptss
    # It has "RAJ2000" and "DEJ2000" but usually crossmatching directly on data tables is disallowed.
    # Let's crossmatch against the main catalog II/341/crts_prss first.
    t0 = time.perf_counter()
    prss_result = batch_tap_crossmatch(
        coords_df,
        tap_url=VIZIER_TAP_URL,
        catalog_table='"II/341/crts_prss"',
        select_cols='c."ID"',
        ra_col="RAJ2000",
        dec_col="DEJ2000",
        match_radius_arcsec=3.0,
        chunk_size=1000,
        n_workers=2,
        desc="CRTS crossmatch",
    )
    
    if prss_result.empty:
        print("CRTS LCs: no counterparts found")
        return df
        
    # Keep closest match per input index
    prss_result = prss_result.sort_values("sep_arcsec").drop_duplicates(subset="_idx", keep="first")
    crts_ids = prss_result["ID"].dropna().astype(str).tolist()
    id_to_idx = dict(zip(prss_result["ID"].astype(str), prss_result["_idx"]))

    if not crts_ids:
        return df

    print(f"CRTS LCs: fetching full LCs for {len(crts_ids)} matches...")
    
    # Query the II/341/data table for all matched IDs
    tap_serv = pyvo.dal.TAPService(VIZIER_TAP_URL)
    
    # Doing chunks of 100 IDs
    chunk_size = 100
    all_lcs = []
    
    for i in tqdm(range(0, len(crts_ids), chunk_size), desc="CRTS Epoch"):
        chunk_ids = crts_ids[i:i+chunk_size]
        ids_str = ", ".join(f"'{cid}'" for cid in chunk_ids)
        query = f"""
        SELECT "ID", "ObsTime", "Mag", "e_Mag"
        FROM "II/341/data"
        WHERE "ID" IN ({ids_str})
        """
        try:
            res = tap_serv.search(query, maxrec=500000)
            res_df = res.to_table().to_pandas()
            if not res_df.empty:
                all_lcs.append(res_df)
        except Exception as e:
            print(f"CRTS fetch error: {e}")
            continue

    if not all_lcs:
        print("CRTS LCs: no epoch data retrieved")
        return df

    full_data = pd.concat(all_lcs, ignore_index=True)
    full_data = full_data.rename(columns={"ObsTime": "mjd", "Mag": "mag", "e_Mag": "mag_err", "ID": "CRTS_ID"})
    full_data["CRTS_ID"] = full_data["CRTS_ID"].astype(str)
    
    matched = 0
    grouped = full_data.groupby("CRTS_ID")
    for cid, group in grouped:
        if cid not in id_to_idx:
            continue
        idx = id_to_idx[cid]
        
        lc_df = group.sort_values("mjd").dropna(subset=["mjd", "mag"]).copy()
        n_points = len(lc_df)
        df.loc[idx, "crts_lc_n_points"] = n_points
        
        if output_dir and n_points > 0:
            cand_id = str(df.loc[idx, "candidate_id"]) if "candidate_id" in df.columns else str(idx)
            lc_df.to_parquet(Path(output_dir) / f"crts_lc_{cand_id}.parquet", index=False)
            matched += 1

    print(f"CRTS LCs: {matched}/{n_valid} with data")
    return df


# =============================================================================
# EXTERNAL LC ORCHESTRATOR
# =============================================================================



def fetch_external_lcs(
    df: pd.DataFrame,
    *,
    output_dir: Path | None = None,
    run_atlas: bool = True,
    run_ztf: bool = True,
    run_gaia_epoch: bool = True,
    run_tess: bool = False,
    run_kepler: bool = False,
    run_aavso: bool = False,
    run_ps1: bool = True,
    run_crts: bool = True,
    atlas_token: str | None = None,
    workers: int = 4,
    checkpoint_path: Path | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """
    Orchestrator for fetching external light curves from all sources.

    Calls each fetch function in sequence with *output_dir*.
    Supports checkpoint resume (same pattern as ``vet_candidates``).
    NEOWISE is NOT included here (already part of vetting).
    """
    def _emit(msg: str) -> None:
        if progress_callback:
            progress_callback(msg)
        else:
            print(msg)

    # Normalise coordinate column names
    if "ra" not in df.columns and "ra_deg" in df.columns:
        df = df.rename(columns={"ra_deg": "ra"})
    if "dec" not in df.columns and "dec_deg" in df.columns:
        df = df.rename(columns={"dec_deg": "dec"})

    # Resume from checkpoint if available
    _resumed = False
    if checkpoint_path and checkpoint_path.exists():
        try:
            df = pd.read_parquet(checkpoint_path)
            _resumed = True
            _emit(f"Resumed external LCs from checkpoint: {checkpoint_path}")
        except Exception:
            pass

    total_start = time.perf_counter()
    _emit(f"EXTERNAL LIGHT CURVES: {len(df)} candidates")

    _MODULE_MARKERS = {
        "ATLAS LCs": "atlas_has_phot",
        "ZTF LCs": "ztf_lc_n_det",
        "Gaia epoch LCs": "gaia_epoch_lc_n_g",
        "TESS LCs": "tess_n_sectors",
        "Kepler LCs": "kepler_n_quarters",
        "AAVSO LCs": "aavso_lc_n_points",
        "Pan-STARRS LCs": "ps1_lc_n_points",
        "CRTS LCs": "crts_lc_n_points",
    }

    def _module_done(name):
        if not _resumed:
            return False
        col = _MODULE_MARKERS.get(name)
        if col is None or col not in df.columns:
            return False
        s = df[col]
        return s.notna().any() and (s != 0).any()

    def _run_module(name, func, **kwargs):
        nonlocal df
        if _module_done(name):
            _emit(f"{name} skipped (already in checkpoint)")
            return
        t0 = time.perf_counter()
        df = func(df, **kwargs)
        _emit(f"{name} completed in {time.perf_counter() - t0:.1f}s")
        if checkpoint_path:
            df.to_parquet(checkpoint_path, index=False)

    if run_atlas:
        _run_module("ATLAS LCs", query_atlas_forced_phot, token=atlas_token, output_dir=output_dir)

    if run_ztf:
        _run_module("ZTF LCs", fetch_ztf_lightcurves, output_dir=output_dir, workers=workers)

    if run_gaia_epoch:
        _run_module("Gaia epoch LCs", fetch_gaia_epoch_lcs, output_dir=output_dir)

    if run_tess:
        _run_module("TESS LCs", fetch_tess_lightcurves, output_dir=output_dir, workers=min(workers, 2))

    if run_kepler:
        _run_module("Kepler LCs", fetch_kepler_k2_lightcurves, output_dir=output_dir, workers=min(workers, 2))

    if run_aavso:
        _run_module("AAVSO LCs", fetch_aavso_lightcurves, output_dir=output_dir, workers=workers)

    if run_ps1:
        _run_module("Pan-STARRS LCs", fetch_panstarrs_lightcurves, output_dir=output_dir, workers=workers)

    if run_crts:
        _run_module("CRTS LCs", fetch_crts_lightcurves, output_dir=output_dir)

    elapsed = time.perf_counter() - total_start
    _emit(f"External LCs completed in {elapsed:.1f}s")
    return df


# =============================================================================
# ORCHESTRATION
# =============================================================================


def vet_candidates(
    df: pd.DataFrame,
    *,
    run_simbad: bool = True,
    run_gaia_var: bool = True,
    run_asassn_var: bool = True,
    run_ztf_var: bool = True,
    run_tns: bool = True,
    run_gaia_eb: bool = True,
    run_alerce: bool = True,
    run_atlas: bool = True,
    run_gaia_epoch: bool = True,
    run_erosita: bool = True,
    run_pm_check: bool = True,
    run_neowise_lc: bool = True,
    simbad_radius_arcsec: float = SIMBAD_RADIUS_ARCSEC,
    asassn_radius_arcsec: float = ASASSN_VAR_RADIUS_ARCSEC,
    ztf_var_radius_arcsec: float = ZTF_VAR_RADIUS_ARCSEC,
    tns_radius_arcsec: float = TNS_RADIUS_ARCSEC,
    alerce_radius_arcsec: float = ALERCE_RADIUS_ARCSEC,
    erosita_radius_arcsec: float = EROSITA_RADIUS_ARCSEC,
    gaia_var_chunk_size: int = GAIA_VAR_CHUNK_SIZE,
    atlas_token: str | None = None,
    atlas_output_dir: Path | None = None,
    tns_api_key: str | None = None,
    alerce_workers: int = 8,
    neowise_output_dir: Path | None = None,
    neowise_workers: int = 4,
    checkpoint_path: Path | None = None,
    method: Literal["tap", "xmatch"] = "tap",
) -> pd.DataFrame:
    """
    Run all vetting queries on a candidate DataFrame.

    Parameters
    ----------
    df : DataFrame with at minimum 'ra', 'dec' columns.
         'gaia_id' column needed for Gaia variability queries.
    run_simbad : query SIMBAD for object type, bibliography
    run_gaia_var : query Gaia DR3 variability tables
    run_asassn_var : crossmatch ASAS-SN variable star catalog
    run_ztf_var : crossmatch ZTF periodic variables (Chen+ 2020)
    run_tns : crossmatch Transient Name Server
    run_gaia_eb : query Gaia DR3 eclipsing binary parameters (ECL sources only)
    run_alerce : query ALeRCE ZTF broker
    run_atlas : query ATLAS forced photometry (requires token)
    run_gaia_epoch : check Gaia epoch photometry availability
    run_erosita : crossmatch eROSITA X-ray catalog
    run_pm_check : proper motion consistency with clusters
    run_neowise_lc : fetch full NEOWISE light curves
    checkpoint_path : if set, save intermediate results after each module
    method : 'tap' (batch TAP upload, default) or 'xmatch' (CDS XMatch,
        better for small batches / review GUI).  Propagated to SIMBAD,
        ZTF vars, TNS, and eROSITA crossmatch functions.

    Returns
    -------
    DataFrame with vetting columns added.
    """
    # Normalise coordinate column names (pipeline uses ra_deg/dec_deg).
    if "ra" not in df.columns and "ra_deg" in df.columns:
        df = df.rename(columns={"ra_deg": "ra"})
    if "dec" not in df.columns and "dec_deg" in df.columns:
        df = df.rename(columns={"dec_deg": "dec"})

    # Resume from checkpoint if available.
    _resumed = False
    if checkpoint_path and checkpoint_path.exists():
        try:
            df = pd.read_parquet(checkpoint_path)
            _resumed = True
            print(f"Resumed from checkpoint: {checkpoint_path}")
        except Exception:
            pass

    total_start = time.perf_counter()
    print(f"\n{'='*60}")
    print(f"POST-REVIEW VETTING: {len(df)} candidates")
    print(f"{'='*60}\n")

    # Map each module to a marker column — if that column has data, skip.
    _MODULE_MARKERS = {
        "SIMBAD": "simbad_main_id",
        "Gaia variability": "gaia_var_flag",
        "Gaia epoch photometry": "gaia_epoch_available",
        "ASAS-SN variables": "asassn_var_name",
        "ZTF variables": "ztf_var_type",
        "TNS": "tns_name",
        "Gaia EB params": "gaia_eb_period",
        "ALeRCE": "alerce_oid",
        "eROSITA": "xray_det",
        "ATLAS forced phot": "atlas_has_phot",
        "PM consistency": "pm_cluster_offset_sigma",
        "NEOWISE LCs": "neowise_n_epochs",
    }

    def _module_done(name):
        """Check if a module's marker column already has data (from checkpoint)."""
        if not _resumed:
            return False
        col = _MODULE_MARKERS.get(name)
        if col is None or col not in df.columns:
            return False
        s = df[col]
        if s.dtype == object:
            return (s.fillna("").astype(str).str.strip() != "").any()
        return s.notna().any()

    def _run_module(name, func, **kwargs):
        nonlocal df
        if _module_done(name):
            print(f"  {name} — skipped (already in checkpoint)\n")
            return
        t0 = time.perf_counter()
        df = func(df, **kwargs)
        print(f"  {name} completed in {time.perf_counter() - t0:.1f}s\n")
        if checkpoint_path:
            df.to_parquet(checkpoint_path, index=False)

    if run_simbad:
        _run_module("SIMBAD", query_simbad_batch, radius_arcsec=simbad_radius_arcsec, method=method)

    if run_gaia_var:
        _run_module("Gaia variability", query_gaia_variability, chunk_size=gaia_var_chunk_size)

    if run_gaia_epoch:
        _run_module("Gaia epoch photometry", query_gaia_epoch_photometry, chunk_size=gaia_var_chunk_size)

    if run_asassn_var:
        # ASAS-SN II/366 is not on CDS XMatch; use local CSV when method='xmatch'
        _asassn_method = "local" if method == "xmatch" else "tap"
        _run_module("ASAS-SN variables", crossmatch_asassn_variables,
                    radius_arcsec=asassn_radius_arcsec, method=_asassn_method)

    if run_ztf_var:
        _run_module("ZTF variables", crossmatch_ztf_variables, radius_arcsec=ztf_var_radius_arcsec, method=method)

    if run_tns:
        _run_module("TNS", crossmatch_tns, radius_arcsec=tns_radius_arcsec, tns_api_key=tns_api_key)

    if run_gaia_eb:
        _run_module("Gaia EB params", query_gaia_eb_params, chunk_size=gaia_var_chunk_size)

    if run_alerce:
        _run_module("ALeRCE", query_alerce, radius_arcsec=alerce_radius_arcsec, workers=alerce_workers)

    if run_erosita:
        # eROSITA: prefer local FITS when method='xmatch' (if file exists)
        _erosita_method = "local" if method == "xmatch" and EROSITA_LOCAL_FITS.exists() else method
        _run_module("eROSITA", crossmatch_erosita, radius_arcsec=erosita_radius_arcsec, method=_erosita_method)

    if run_atlas:
        _run_module("ATLAS forced phot", query_atlas_forced_phot, token=atlas_token, output_dir=atlas_output_dir)

    if run_pm_check:
        _run_module("PM consistency", check_pm_consistency)

    if run_neowise_lc:
        _run_module("NEOWISE LCs", query_neowise_lightcurves,
                    output_dir=neowise_output_dir, workers=neowise_workers)

    # Summary
    _print_vetting_summary(df, total_start)
    return df


def _print_vetting_summary(df: pd.DataFrame, total_start: float) -> None:
    """Print comprehensive vetting summary."""
    print(f"\n{'='*60}")
    print("VETTING SUMMARY")
    print(f"{'='*60}")

    if "simbad_main_id" in df.columns:
        n = (df["simbad_main_id"] != "").sum()
        print(f"  SIMBAD matches:         {n}/{len(df)}")
        if n > 0:
            print(f"  Median SIMBAD refs:     {df.loc[df['simbad_main_id'] != '', 'simbad_nbref'].median():.0f}")

    if "gaia_var_flag" in df.columns:
        print(f"  Gaia variable flag:     {df['gaia_var_flag'].sum()}/{len(df)}")
    if "gaia_var_class" in df.columns:
        n = (df["gaia_var_class"] != "").sum()
        print(f"  Gaia classified:        {n}/{len(df)}")
        if n > 0:
            for cls, cnt in df.loc[df["gaia_var_class"] != "", "gaia_var_class"].value_counts().head(5).items():
                print(f"    {cls}: {cnt}")

    if "gaia_epoch_available" in df.columns:
        print(f"  Gaia epoch available:   {df['gaia_epoch_available'].sum()}/{len(df)}")

    if "asassn_var_type" in df.columns:
        n = (df["asassn_var_type"] != "").sum()
        print(f"  ASAS-SN var matches:    {n}/{len(df)}")

    if "ztf_var_type" in df.columns:
        n = (df["ztf_var_type"] != "").sum()
        print(f"  ZTF var matches:        {n}/{len(df)}")
        if n > 0:
            for cls, cnt in df.loc[df["ztf_var_type"] != "", "ztf_var_type"].value_counts().head(5).items():
                print(f"    {cls}: {cnt}")

    if "tns_name" in df.columns:
        n = (df["tns_name"] != "").sum()
        print(f"  TNS transients:         {n}/{len(df)}")
        if n > 0:
            for cls, cnt in df.loc[df["tns_type"] != "", "tns_type"].value_counts().head(5).items():
                print(f"    {cls}: {cnt}")

    if "gaia_eb_period" in df.columns:
        n = df["gaia_eb_period"].notna().sum()
        print(f"  Gaia EB params:         {n}/{len(df)}")

    if "alerce_oid" in df.columns:
        n = (df["alerce_oid"] != "").sum()
        print(f"  ALeRCE matches:         {n}/{len(df)}")
        if n > 0:
            lc_cls = df.loc[df["alerce_lc_class"] != "", "alerce_lc_class"].value_counts().head(5)
            if len(lc_cls) > 0:
                print(f"  ALeRCE LC classes:")
                for cls, cnt in lc_cls.items():
                    print(f"    {cls}: {cnt}")

    if "xray_det" in df.columns:
        n = df["xray_det"].sum()
        print(f"  eROSITA X-ray det:      {n}/{len(df)}")

    if "atlas_has_phot" in df.columns:
        n = df["atlas_has_phot"].sum()
        print(f"  ATLAS photometry:       {n}/{len(df)}")

    if "pm_cluster_offset_sigma" in df.columns:
        n = df["pm_cluster_offset_sigma"].notna().sum()
        if n > 0:
            outliers = (df["pm_cluster_offset_sigma"] > 3).sum()
            print(f"  PM consistency:         {n} checked, {outliers} outliers (>3σ)")

    if "neowise_n_epochs" in df.columns:
        n = (df["neowise_n_epochs"] > 0).sum()
        print(f"  NEOWISE LCs:            {n}/{len(df)}")

    # Flag "likely known" vs "potentially new"
    known_mask = pd.Series(False, index=df.index)
    
    # We only want to flag true for variables, not just generic objects.
    if "gaia_var_class" in df.columns:
        known_mask |= df["gaia_var_class"] != ""
    if "asassn_var_type" in df.columns:
        known_mask |= df["asassn_var_type"] != ""
    if "ztf_var_type" in df.columns:
        known_mask |= df["ztf_var_type"] != ""
    if "tns_name" in df.columns:
        known_mask |= df["tns_name"] != ""
    if "alerce_lc_class" in df.columns:
        known_mask |= df["alerce_lc_class"] != ""
    if "vsx_class" in df.columns:
        known_mask |= df["vsx_class"].fillna("").astype(str).str.strip() != ""

    if "simbad_otype" in df.columns:
        def is_var_otype(x):
            s = str(x).strip()
            if not s: return False
            if 'V*' in s: return True
            matches = {'EB*', 'YSO', 'SN', 'Nova', 'Catac', 'RR*', 'Cepheid', 'Mira', 'BYDra', 'RSCVn', 'Symbiotic', 'ELL', 'Blazar', 'QSO', 'AGN'}
            s_low = s.lower()
            return any(m.lower() in s_low for m in matches)
        known_mask |= df["simbad_otype"].apply(is_var_otype)
    df["vetting_likely_known"] = known_mask

    n_known = known_mask.sum()
    n_new = len(df) - n_known
    print(f"\n  Likely known:           {n_known}")
    print(f"  Potentially new:        {n_new}")
    print(f"\n  Total time: {time.perf_counter() - total_start:.1f}s")


# =============================================================================
# CLI ENTRY POINT
# =============================================================================


def main():
    """CLI for standalone vetting."""
    import argparse

    parser = argparse.ArgumentParser(description="Post-review vetting of MALCA candidates")
    parser.add_argument("input", type=Path, help="Input parquet/CSV with candidates (needs ra, dec columns)")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output parquet path (default: <input>_vetted.parquet)")
    parser.add_argument("--min-score", type=float, default=None, help="Only vet candidates with interest_score >= this value")
    parser.add_argument("--simbad-radius", type=float, default=SIMBAD_RADIUS_ARCSEC, help=f"SIMBAD search radius in arcsec (default: {SIMBAD_RADIUS_ARCSEC})")
    parser.add_argument("--asassn-radius", type=float, default=ASASSN_VAR_RADIUS_ARCSEC, help=f"ASAS-SN crossmatch radius in arcsec (default: {ASASSN_VAR_RADIUS_ARCSEC})")
    parser.add_argument("--alerce-radius", type=float, default=ALERCE_RADIUS_ARCSEC, help=f"ALeRCE search radius in arcsec (default: {ALERCE_RADIUS_ARCSEC})")
    parser.add_argument("--erosita-radius", type=float, default=EROSITA_RADIUS_ARCSEC, help=f"eROSITA search radius in arcsec (default: {EROSITA_RADIUS_ARCSEC})")
    parser.add_argument("--no-simbad", action="store_true", help="Skip SIMBAD query")
    parser.add_argument("--no-gaia-var", action="store_true", help="Skip Gaia DR3 variability query")
    parser.add_argument("--no-gaia-epoch", action="store_true", help="Skip Gaia DR3 epoch photometry check")
    parser.add_argument("--no-asassn-var", action="store_true", help="Skip ASAS-SN variable catalog crossmatch")
    parser.add_argument("--no-ztf-var", action="store_true", help="Skip ZTF periodic variables crossmatch")
    parser.add_argument("--ztf-var-radius", type=float, default=ZTF_VAR_RADIUS_ARCSEC, help=f"ZTF variable crossmatch radius in arcsec (default: {ZTF_VAR_RADIUS_ARCSEC})")
    parser.add_argument("--no-tns", action="store_true", help="Skip TNS transient crossmatch")
    parser.add_argument("--tns-radius", type=float, default=TNS_RADIUS_ARCSEC, help=f"TNS crossmatch radius in arcsec (default: {TNS_RADIUS_ARCSEC})")
    parser.add_argument("--tns-api-key", type=str, default=None, help="TNS API key (ignored; TNS uses local catalog)")
    parser.add_argument("--no-gaia-eb", action="store_true", help="Skip Gaia DR3 eclipsing binary parameters")
    parser.add_argument("--no-alerce", action="store_true", help="Skip ALeRCE ZTF query")
    parser.add_argument("--alerce-workers", type=int, default=8, help="Parallel workers for ALeRCE queries (default: 8)")
    parser.add_argument("--no-erosita", action="store_true", help="Skip eROSITA X-ray crossmatch")
    parser.add_argument("--no-pm-check", action="store_true", help="Skip proper motion consistency check")
    parser.add_argument("--no-atlas", action="store_true", help="Skip ATLAS forced photometry (default: enabled)")
    parser.add_argument("--atlas-token", type=str, default=None, help="ATLAS forced photometry API token (or set MALCA_ATLAS_TOKEN env var)")
    parser.add_argument("--neowise-lc", dest="neowise_lc", action="store_true", help="Fetch full NEOWISE light curves (default: enabled)")
    parser.add_argument("--no-neowise-lc", dest="neowise_lc", action="store_false", help="Skip full NEOWISE light curves")
    parser.add_argument("--neowise-output-dir", type=Path, default=None, help="Directory to save individual NEOWISE LCs")
    parser.add_argument("--neowise-workers", type=int, default=4, help="Parallel workers for NEOWISE queries")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Checkpoint path (default: <input>_vetting_CHECKPOINT.parquet)")
    parser.add_argument("--no-checkpoint", action="store_true", help="Disable checkpoint saving/resume")

    parser.set_defaults(neowise_lc=True)

    args = parser.parse_args()

    # Load input
    path = args.input.expanduser()
    if path.suffix.lower() in (".parquet", ".pq"):
        df = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    print(f"Loaded {len(df)} candidates from {path}")

    # Default checkpoint: <input>_vetting_CHECKPOINT.parquet
    if args.no_checkpoint:
        _ckpt_path = None
    elif args.checkpoint:
        _ckpt_path = args.checkpoint
    else:
        _ckpt_path = path.with_name(path.stem + "_vetting_CHECKPOINT.parquet")

    # Filter by score if requested
    if args.min_score is not None and "interest_score" in df.columns:
        before = len(df)
        df = df[df["interest_score"] >= args.min_score].copy()
        print(f"Filtered to {len(df)} candidates with score >= {args.min_score} (from {before})")

    # Run vetting
    df = vet_candidates(
        df,
        run_simbad=not args.no_simbad,
        run_gaia_var=not args.no_gaia_var,
        run_gaia_epoch=not args.no_gaia_epoch,
        run_asassn_var=not args.no_asassn_var,
        run_ztf_var=not args.no_ztf_var,
        run_tns=not args.no_tns,
        run_gaia_eb=not args.no_gaia_eb,
        run_alerce=not args.no_alerce,
        run_erosita=not args.no_erosita,
        run_atlas=not args.no_atlas,
        run_pm_check=not args.no_pm_check,
        run_neowise_lc=args.neowise_lc,
        simbad_radius_arcsec=args.simbad_radius,
        asassn_radius_arcsec=args.asassn_radius,
        ztf_var_radius_arcsec=args.ztf_var_radius,
        tns_radius_arcsec=args.tns_radius,
        alerce_radius_arcsec=args.alerce_radius,
        alerce_workers=args.alerce_workers,
        erosita_radius_arcsec=args.erosita_radius,
        atlas_token=args.atlas_token or os.environ.get("MALCA_ATLAS_TOKEN"),
        tns_api_key=args.tns_api_key or os.environ.get("MALCA_TNS_API_KEY"),
        neowise_output_dir=args.neowise_output_dir,
        neowise_workers=args.neowise_workers,
        checkpoint_path=_ckpt_path,
    )

    # Save output
    out_path = args.output or path.with_name(path.stem + "_vetted.parquet")
    df.to_parquet(out_path, index=False)
    print(f"\nSaved vetted results to {out_path}")

    # Clean up checkpoint on successful completion.
    if _ckpt_path and _ckpt_path.exists():
        _ckpt_path.unlink()
        print(f"Checkpoint removed: {_ckpt_path}")


if __name__ == "__main__":
    main()
