"""
Post-review vetting: check whether candidates are already known objects.

Queries:
1. SIMBAD — object type, identifiers, bibliography count
2. Gaia DR3 variability tables — variability flag + classification
3. ASAS-SN Variable Stars Database (VizieR II/366) — known ASAS-SN variables
4. ALeRCE ZTF broker — ZTF ML classification
5. ATLAS forced photometry — independent cyan/orange confirmation
6. Gaia DR3 epoch photometry — space-based variability confirmation
7. eROSITA X-ray catalog — youth indicator
8. Proper motion consistency — cluster membership validation
9. NEOWISE light curves — IR time-series for dipper confirmation

Usage:
    from malca.vetting import vet_candidates
    df_vetted = vet_candidates(df)
"""
from __future__ import annotations

import io
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pyvo
import requests
from astropy.coordinates import SkyCoord
from astropy.table import Table
import astropy.units as u
from astroquery.simbad import Simbad
from tqdm import tqdm

from malca.config.config_paths import GAIA_AIP_TAP_URL
from malca.config.config_characterize import GAIA_CHUNK_SIZE

# Vetting configuration
SIMBAD_RADIUS_ARCSEC = 5.0
SIMBAD_BATCH_SIZE = 500
SIMBAD_RETRY_DELAY = 5
SIMBAD_MAX_RETRIES = 3

GAIA_VAR_CHUNK_SIZE = GAIA_CHUNK_SIZE
ASASSN_VAR_CATALOG = "II/366/catv2021"
ASASSN_VAR_RADIUS_ARCSEC = 5.0

ALERCE_API_BASE = "https://api.alerce.online"
ALERCE_RADIUS_ARCSEC = 3.0
ALERCE_BATCH_SIZE = 50

ATLAS_API_BASE = "https://fallingstar-data.com/forcedphot"
ATLAS_POLL_INTERVAL = 10
ATLAS_MAX_POLL = 120
ATLAS_MJD_MIN = 57000  # ~2015

EROSITA_CATALOG = "J/A+A/682/A34/erass1-m"
EROSITA_RADIUS_ARCSEC = 10.0

NEOWISE_MAX_SEP_ARCSEC = 3.0


# =============================================================================
# SIMBAD BATCH QUERY
# =============================================================================


def query_simbad_batch(
    df: pd.DataFrame,
    radius_arcsec: float = SIMBAD_RADIUS_ARCSEC,
) -> pd.DataFrame:
    """
    Query SIMBAD by coordinates for all candidates.

    Adds columns: simbad_main_id, simbad_otype, simbad_nbref, simbad_sep_arcsec.
    Uses the SIMBAD TAP service for efficient batch queries.
    """
    df = df.copy()
    for col in ("simbad_main_id", "simbad_otype", "simbad_nbref", "simbad_sep_arcsec"):
        df[col] = np.nan if col == "simbad_sep_arcsec" or col == "simbad_nbref" else ""

    valid = df["ra"].notna() & df["dec"].notna()
    if not valid.any():
        return df

    df_valid = df.loc[valid].copy()
    n = len(df_valid)
    print(f"SIMBAD: querying {n} candidates (radius={radius_arcsec}\")")

    simbad = Simbad()
    simbad.add_votable_fields("otype", "nbref")
    simbad.ROW_LIMIT = -1  # no row limit
    simbad.timeout = 120

    radius = radius_arcsec * u.arcsec
    matched = 0

    for i in tqdm(range(0, n, SIMBAD_BATCH_SIZE), desc="SIMBAD batch"):
        batch = df_valid.iloc[i : i + SIMBAD_BATCH_SIZE]
        coords = SkyCoord(
            ra=batch["ra"].values, dec=batch["dec"].values, unit="deg", frame="icrs"
        )

        for attempt in range(SIMBAD_MAX_RETRIES):
            try:
                result = simbad.query_region(coords, radius=radius)
                break
            except Exception as e:
                if attempt < SIMBAD_MAX_RETRIES - 1:
                    print(f"  SIMBAD retry {attempt + 1}: {e}")
                    time.sleep(SIMBAD_RETRY_DELAY * (attempt + 1))
                else:
                    print(f"  SIMBAD batch {i}-{i+len(batch)} failed: {e}")
                    result = None

        if result is None or len(result) == 0:
            continue

        # Match results back to input by nearest coord
        result_coords = SkyCoord(
            ra=result["ra"], dec=result["dec"], unit="deg", frame="icrs"
        )

        for j, (idx, row) in enumerate(batch.iterrows()):
            src = SkyCoord(ra=row["ra"], dec=row["dec"], unit="deg", frame="icrs")
            seps = src.separation(result_coords).arcsec
            within = seps <= radius_arcsec
            if not within.any():
                continue

            # Take the match with most references (most studied object)
            candidates_within = np.where(within)[0]
            nbrefs = np.array([
                int(result["nbref"][k]) if result["nbref"][k] is not None else 0
                for k in candidates_within
            ])
            best = candidates_within[np.argmax(nbrefs)]

            df.loc[idx, "simbad_main_id"] = str(result["main_id"][best])
            df.loc[idx, "simbad_otype"] = str(result["otype"][best])
            df.loc[idx, "simbad_nbref"] = int(nbrefs[np.argmax(nbrefs)])
            df.loc[idx, "simbad_sep_arcsec"] = round(seps[best], 3)
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
    tap = pyvo.dal.TAPService("https://gea.esac.esa.int/tap-server/tap")

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
    batch_size: int = 50,
) -> pd.DataFrame:
    """
    Crossmatch against the ASAS-SN Variable Stars Database (Jayasinghe+ 2018-2021).

    Uses VizieR query_region in batches. Adds columns: asassn_var_name, asassn_var_type, asassn_var_period.
    """
    from astroquery.vizier import Vizier

    df = df.copy()
    df["asassn_var_name"] = ""
    df["asassn_var_type"] = ""
    df["asassn_var_period"] = np.nan

    valid = df["ra"].notna() & df["dec"].notna()
    if not valid.any():
        return df

    n_valid = int(valid.sum())
    print(f"ASAS-SN variables: crossmatching {n_valid} candidates (radius={radius_arcsec}\")")

    viz = Vizier(columns=["ASASSN-V", "Type", "Per", "RAJ2000", "DEJ2000"], row_limit=-1)
    radius = radius_arcsec * u.arcsec
    matched = 0
    valid_indices = df.index[valid]

    for i in tqdm(range(0, n_valid, batch_size), desc="ASAS-SN vars"):
        batch_idx = valid_indices[i : i + batch_size]
        batch = df.loc[batch_idx]
        coords = SkyCoord(
            ra=batch["ra"].values, dec=batch["dec"].values, unit="deg", frame="icrs"
        )

        for attempt in range(3):
            try:
                results = viz.query_region(coords, radius=radius, catalog=ASASSN_VAR_CATALOG)
                break
            except Exception as e:
                if attempt < 2:
                    time.sleep(3 * (attempt + 1))
                else:
                    print(f"  ASAS-SN batch {i} failed: {e}")
                    results = None

        if results is None or len(results) == 0:
            continue

        result_table = results[0]
        result_coords = SkyCoord(
            ra=result_table["RAJ2000"], dec=result_table["DEJ2000"], unit="deg", frame="icrs"
        )

        for j, idx in enumerate(batch_idx):
            src = SkyCoord(ra=df.loc[idx, "ra"], dec=df.loc[idx, "dec"], unit="deg", frame="icrs")
            seps = src.separation(result_coords).arcsec
            within = seps <= radius_arcsec
            if not within.any():
                continue

            best = np.argmin(seps)
            row = result_table[best]
            name = str(row["ASASSN-V"]) if row["ASASSN-V"] else ""
            vtype = str(row["Type"]) if row["Type"] else ""
            try:
                p = row["Per"]
                period = float(p) if p is not None and p is not np.ma.masked else np.nan
            except (ValueError, TypeError):
                period = np.nan

            df.loc[idx, "asassn_var_name"] = name
            df.loc[idx, "asassn_var_type"] = vtype
            df.loc[idx, "asassn_var_period"] = period
            matched += 1

    print(f"ASAS-SN variables: {matched} matches")
    return df


# =============================================================================
# ALeRCE ZTF BROKER
# =============================================================================


def _alerce_conesearch_single(ra: float, dec: float, radius_arcsec: float) -> dict | None:
    """Query ALeRCE for a single coordinate. Returns best match or None."""
    try:
        resp = requests.get(
            f"{ALERCE_API_BASE}/ztf/v1/objects/",
            params={
                "conesearch_input[ra]": ra,
                "conesearch_input[dec]": dec,
                "conesearch_input[radius]": radius_arcsec,
                "page_size": 5,
                "order_by": "ndet",
                "order_mode": "DESC",
            },
            timeout=60,
        )
        if resp.status_code == 429:
            time.sleep(5)
            return None
        if resp.status_code != 200:
            return None
        data = resp.json()
        items = data.get("items", [])
        if not items:
            return None
        return items[0]  # most detections = most relevant
    except Exception:
        return None


def _alerce_get_probabilities(oid: str) -> dict:
    """Get classification probabilities for an ALeRCE object."""
    out = {"lc_class": "", "lc_prob": np.nan, "stamp_class": "", "stamp_prob": np.nan}
    try:
        resp = requests.get(
            f"{ALERCE_API_BASE}/ztf/v1/objects/{oid}/probabilities",
            timeout=60,
        )
        if resp.status_code != 200:
            return out
        probs = resp.json()
        # Find best LC classifier result
        lc_probs = [p for p in probs if p.get("classifier_name", "").startswith("lc_classifier")]
        if lc_probs:
            best_lc = max(lc_probs, key=lambda p: p.get("probability", 0))
            out["lc_class"] = best_lc.get("class_name", "")
            out["lc_prob"] = best_lc.get("probability", np.nan)
        # Find best stamp classifier result
        stamp_probs = [p for p in probs if p.get("classifier_name", "").startswith("stamp_classifier")]
        if stamp_probs:
            best_stamp = max(stamp_probs, key=lambda p: p.get("probability", 0))
            out["stamp_class"] = best_stamp.get("class_name", "")
            out["stamp_prob"] = best_stamp.get("probability", np.nan)
    except Exception:
        pass
    return out


def query_alerce(
    df: pd.DataFrame,
    radius_arcsec: float = ALERCE_RADIUS_ARCSEC,
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
    print(f"ALeRCE: querying {n_valid} candidates (radius={radius_arcsec}\")")
    matched = 0

    for idx in tqdm(df.index[valid], desc="ALeRCE"):
        ra, dec = float(df.loc[idx, "ra"]), float(df.loc[idx, "dec"])
        obj = _alerce_conesearch_single(ra, dec, radius_arcsec)
        if obj is None:
            continue

        oid = obj.get("oid", "")
        df.loc[idx, "alerce_oid"] = oid
        df.loc[idx, "alerce_ndet"] = int(obj.get("ndet", 0))

        # Get classification probabilities
        if oid:
            probs = _alerce_get_probabilities(oid)
            df.loc[idx, "alerce_lc_class"] = probs["lc_class"]
            df.loc[idx, "alerce_lc_prob"] = probs["lc_prob"]
            df.loc[idx, "alerce_stamp_class"] = probs["stamp_class"]
            df.loc[idx, "alerce_stamp_prob"] = probs["stamp_prob"]
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
) -> pd.DataFrame:
    """
    Query ATLAS forced photometry for independent variability confirmation.

    Requires an ATLAS API token (register at https://fallingstar-data.com/forcedphot/).

    Adds columns: atlas_has_phot, atlas_n_det_cyan, atlas_n_det_orange,
                  atlas_cyan_range, atlas_orange_range.
    """
    df = df.copy()
    df["atlas_has_phot"] = False
    df["atlas_n_det_cyan"] = 0
    df["atlas_n_det_orange"] = 0
    df["atlas_cyan_range"] = np.nan
    df["atlas_orange_range"] = np.nan

    if not token:
        print("ATLAS: no API token provided, skipping (register at https://fallingstar-data.com/forcedphot/)")
        return df

    valid = df["ra"].notna() & df["dec"].notna()
    if not valid.any():
        return df

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

        matched += 1

    print(f"ATLAS: {matched}/{n_valid} candidates with photometry")
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
    tap = pyvo.dal.TAPService("https://gea.esac.esa.int/tap-server/tap")

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


# =============================================================================
# eROSITA X-RAY CATALOG
# =============================================================================


def crossmatch_erosita(
    df: pd.DataFrame,
    radius_arcsec: float = EROSITA_RADIUS_ARCSEC,
    batch_size: int = 50,
) -> pd.DataFrame:
    """
    Crossmatch against eROSITA-DE DR1 (Merloni+2024).

    X-ray detection is a strong youth indicator for YSO candidates.
    Adds columns: xray_det, xray_flux, xray_sep_arcsec.
    """
    from astroquery.vizier import Vizier

    df = df.copy()
    df["xray_det"] = False
    df["xray_flux"] = np.nan
    df["xray_sep_arcsec"] = np.nan

    valid = df["ra"].notna() & df["dec"].notna()
    if not valid.any():
        return df

    n_valid = int(valid.sum())
    print(f"eROSITA: crossmatching {n_valid} candidates (radius={radius_arcsec}\")")

    viz = Vizier(columns=["RA_ICRS", "DE_ICRS", "MLFlux1", "DetLike0"], row_limit=-1)
    radius = radius_arcsec * u.arcsec
    matched = 0
    valid_indices = df.index[valid]

    for i in tqdm(range(0, n_valid, batch_size), desc="eROSITA"):
        batch_idx = valid_indices[i : i + batch_size]
        batch = df.loc[batch_idx]
        coords = SkyCoord(
            ra=batch["ra"].values, dec=batch["dec"].values, unit="deg", frame="icrs"
        )

        for attempt in range(3):
            try:
                results = viz.query_region(coords, radius=radius, catalog=EROSITA_CATALOG)
                break
            except Exception as e:
                if attempt < 2:
                    time.sleep(3 * (attempt + 1))
                else:
                    print(f"  eROSITA batch {i} failed: {e}")
                    results = None

        if results is None or len(results) == 0:
            continue

        result_table = results[0]
        result_coords = SkyCoord(
            ra=result_table["RA_ICRS"], dec=result_table["DE_ICRS"], unit="deg", frame="icrs"
        )

        for j, idx in enumerate(batch_idx):
            src = SkyCoord(ra=df.loc[idx, "ra"], dec=df.loc[idx, "dec"], unit="deg", frame="icrs")
            seps = src.separation(result_coords).arcsec
            within = seps <= radius_arcsec
            if not within.any():
                continue

            best = np.argmin(seps)
            row = result_table[best]

            df.loc[idx, "xray_det"] = True
            df.loc[idx, "xray_sep_arcsec"] = round(float(seps[best]), 3)
            try:
                df.loc[idx, "xray_flux"] = float(row["MLFlux1"])
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
# ORCHESTRATION
# =============================================================================


def vet_candidates(
    df: pd.DataFrame,
    *,
    run_simbad: bool = True,
    run_gaia_var: bool = True,
    run_asassn_var: bool = True,
    run_alerce: bool = True,
    run_atlas: bool = False,
    run_gaia_epoch: bool = True,
    run_erosita: bool = True,
    run_pm_check: bool = True,
    run_neowise_lc: bool = False,
    simbad_radius_arcsec: float = SIMBAD_RADIUS_ARCSEC,
    asassn_radius_arcsec: float = ASASSN_VAR_RADIUS_ARCSEC,
    alerce_radius_arcsec: float = ALERCE_RADIUS_ARCSEC,
    erosita_radius_arcsec: float = EROSITA_RADIUS_ARCSEC,
    gaia_var_chunk_size: int = GAIA_VAR_CHUNK_SIZE,
    atlas_token: str | None = None,
    neowise_output_dir: Path | None = None,
    neowise_workers: int = 4,
    checkpoint_path: Path | None = None,
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
    run_alerce : query ALeRCE ZTF broker
    run_atlas : query ATLAS forced photometry (requires token)
    run_gaia_epoch : check Gaia epoch photometry availability
    run_erosita : crossmatch eROSITA X-ray catalog
    run_pm_check : proper motion consistency with clusters
    run_neowise_lc : fetch full NEOWISE light curves
    checkpoint_path : if set, save intermediate results after each module

    Returns
    -------
    DataFrame with vetting columns added.
    """
    total_start = time.perf_counter()
    print(f"\n{'='*60}")
    print(f"POST-REVIEW VETTING: {len(df)} candidates")
    print(f"{'='*60}\n")

    def _run_module(name, func, **kwargs):
        nonlocal df
        t0 = time.perf_counter()
        df = func(df, **kwargs)
        print(f"  {name} completed in {time.perf_counter() - t0:.1f}s\n")
        if checkpoint_path:
            df.to_parquet(checkpoint_path, index=False)

    if run_simbad:
        _run_module("SIMBAD", query_simbad_batch, radius_arcsec=simbad_radius_arcsec)

    if run_gaia_var:
        _run_module("Gaia variability", query_gaia_variability, chunk_size=gaia_var_chunk_size)

    if run_gaia_epoch:
        _run_module("Gaia epoch photometry", query_gaia_epoch_photometry, chunk_size=gaia_var_chunk_size)

    if run_asassn_var:
        _run_module("ASAS-SN variables", crossmatch_asassn_variables, radius_arcsec=asassn_radius_arcsec)

    if run_alerce:
        _run_module("ALeRCE", query_alerce, radius_arcsec=alerce_radius_arcsec)

    if run_erosita:
        _run_module("eROSITA", crossmatch_erosita, radius_arcsec=erosita_radius_arcsec)

    if run_atlas:
        _run_module("ATLAS forced phot", query_atlas_forced_phot, token=atlas_token)

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
    if "simbad_nbref" in df.columns:
        known_mask |= df["simbad_nbref"].fillna(0) >= 5
    if "gaia_var_class" in df.columns:
        known_mask |= df["gaia_var_class"] != ""
    if "asassn_var_type" in df.columns:
        known_mask |= df["asassn_var_type"] != ""
    if "alerce_lc_class" in df.columns:
        known_mask |= df["alerce_lc_class"] != ""
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
    parser.add_argument("--no-alerce", action="store_true", help="Skip ALeRCE ZTF query")
    parser.add_argument("--no-erosita", action="store_true", help="Skip eROSITA X-ray crossmatch")
    parser.add_argument("--no-pm-check", action="store_true", help="Skip proper motion consistency check")
    parser.add_argument("--atlas-token", type=str, default=None, help="ATLAS forced photometry API token")
    parser.add_argument("--neowise-lc", action="store_true", help="Fetch full NEOWISE light curves")
    parser.add_argument("--neowise-output-dir", type=Path, default=None, help="Directory to save individual NEOWISE LCs")
    parser.add_argument("--neowise-workers", type=int, default=4, help="Parallel workers for NEOWISE queries")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Checkpoint path for intermediate saves")

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
        run_alerce=not args.no_alerce,
        run_erosita=not args.no_erosita,
        run_atlas=args.atlas_token is not None,
        run_pm_check=not args.no_pm_check,
        run_neowise_lc=args.neowise_lc,
        simbad_radius_arcsec=args.simbad_radius,
        asassn_radius_arcsec=args.asassn_radius,
        alerce_radius_arcsec=args.alerce_radius,
        erosita_radius_arcsec=args.erosita_radius,
        atlas_token=args.atlas_token,
        neowise_output_dir=args.neowise_output_dir,
        neowise_workers=args.neowise_workers,
        checkpoint_path=args.checkpoint,
    )

    # Save output
    out_path = args.output or path.with_name(path.stem + "_vetted.parquet")
    df.to_parquet(out_path, index=False)
    print(f"\nSaved vetted results to {out_path}")


if __name__ == "__main__":
    main()
