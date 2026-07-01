import json
import sqlite3
import warnings
from pathlib import Path

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astroquery.vizier import Vizier

from malca.enrichment.characterize import (
    IPHAS_CACHE_COLUMNS,
    VPHAS_CACHE_COLUMNS,
    crossmatch_iphas,
    crossmatch_vphas,
)
from malca.review.store import init_db


def _jsonable(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    return value


def _copy_present_values(target: dict, row: pd.Series, columns: list[str]) -> None:
    for col in columns:
        if col not in row.index:
            continue
        value = _jsonable(row.get(col))
        if value in (None, ""):
            continue
        target[col] = value


def _sql_update(conn: sqlite3.Connection, candidate_id: str, values: dict[str, object]) -> None:
    if not values:
        return
    actual_cols = {
        str(row[1])
        for row in conn.execute("PRAGMA table_info(candidates)").fetchall()
    }
    cols = [col for col in values if col in actual_cols]
    if not cols:
        return
    assignments = ", ".join(f"{col}=?" for col in cols)
    params = [values[col] for col in cols]
    params.append(candidate_id)
    conn.execute(
        f"UPDATE candidates SET {assignments} WHERE candidate_id = ?",
        params,
    )


def update_dipper_photometry(db_path: str):
    db_path = Path(db_path).expanduser()
    print(f"Connecting to {db_path}...")
    conn = sqlite3.connect(db_path)
    init_db(conn)

    # 1. Get dipper IDs from CSV
    csv_path = "output/runs/runs_march18_bundle_all/results/march18_review_cmd_dustmaps_full.csv"
    df_csv = pd.read_csv(csv_path)
    dipper_ids = df_csv.loc[df_csv["event_class"] == "dipper", "candidate_id"].astype(str).tolist()

    # 2. Get RA and Dec from characterized.parquet
    char_df = pd.read_parquet("output/runs/runs_march18_bundle_all/results/characterized.parquet")
    if "candidate_id" in char_df.columns:
        id_col = "candidate_id"
    elif "asas_sn_id" in char_df.columns:
        id_col = "asas_sn_id"
    else:
        raise ValueError("characterized.parquet must include candidate_id or asas_sn_id")

    char_df[id_col] = char_df[id_col].astype(str)
    coords_df = char_df[char_df[id_col].isin(dipper_ids)].set_index(id_col)

    # 3. Get payload_json from database
    query = f"""
    SELECT candidate_id, payload_json
    FROM candidates
    WHERE candidate_id IN ({','.join(['?'] * len(dipper_ids))})
    """
    df_db = pd.read_sql(query, conn, params=dipper_ids)
    print(f"Found {len(df_db)} Dippers in DB.")

    halpha_df = coords_df[["ra", "dec"]].copy() if {"ra", "dec"}.issubset(coords_df.columns) else pd.DataFrame()
    if not halpha_df.empty:
        halpha_df = crossmatch_iphas(halpha_df)
        halpha_df = crossmatch_vphas(halpha_df)
    else:
        print("No RA/Dec coordinates found for H-alpha survey crossmatches.")

    allwise_vizier = Vizier(columns=["W1mag", "W2mag"], timeout=60, row_limit=1)
    tmass_vizier = Vizier(columns=["Hmag", "Kmag"], timeout=60, row_limit=1)

    updates = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _, row in df_db.iterrows():
            cid = str(row["candidate_id"])
            p_json = row["payload_json"]
            p = json.loads(p_json) if p_json else {}
            ext = p.get("external_stats", {})
            if not isinstance(ext, dict):
                ext = {}

            if cid not in coords_df.index:
                continue

            if cid in halpha_df.index:
                hrow = halpha_df.loc[cid]
                if _jsonable(hrow.get("iphas_source_catalog")):
                    _copy_present_values(ext, hrow, IPHAS_CACHE_COLUMNS)
                if _jsonable(hrow.get("vphas_source_catalog")):
                    _copy_present_values(ext, hrow, VPHAS_CACHE_COLUMNS)

            ra = coords_df.loc[cid, "ra"]
            dec = coords_df.loc[cid, "dec"]

            if pd.isna(ra) or pd.isna(dec):
                continue

            coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)

            # AllWISE
            try:
                a_res = allwise_vizier.query_region(coord, radius=3 * u.arcsec, catalog="II/328/allwise")
                if len(a_res) > 0 and len(a_res[0]) > 0:
                    match = a_res[0].to_pandas().iloc[0]
                    w1 = match.get("W1mag", None)
                    w2 = match.get("W2mag", None)
                    if pd.notna(w1):
                        ext["w1"] = float(w1)
                    if pd.notna(w2):
                        ext["w2"] = float(w2)
                    if pd.notna(w1) and pd.notna(w2):
                        ext["w1_w2"] = float(w1 - w2)
            except Exception as e:
                print(f"AllWISE query failed for {cid}: {e}")

            # 2MASS
            try:
                t_res = tmass_vizier.query_region(coord, radius=3 * u.arcsec, catalog="II/246/out")
                if len(t_res) > 0 and len(t_res[0]) > 0:
                    match = t_res[0].to_pandas().iloc[0]
                    hmag = match.get("Hmag", None)
                    kmag = match.get("Kmag", None)
                    if pd.notna(hmag) and pd.notna(kmag):
                        ext["H_K"] = float(hmag - kmag)
            except Exception as e:
                print(f"2MASS query failed for {cid}: {e}")

            p["external_stats"] = ext
            sql_values = {
                key: value
                for key, value in ext.items()
                if key in set(IPHAS_CACHE_COLUMNS + VPHAS_CACHE_COLUMNS + ["w1", "w2", "w1_w2", "H_K"])
            }
            updates.append((json.dumps(p), sql_values, cid))

    cursor = conn.cursor()
    updated_count = 0
    for new_json, sql_values, cid in updates:
        cursor.execute("UPDATE candidates SET payload_json = ? WHERE candidate_id = ?", (new_json, cid))
        _sql_update(conn, cid, sql_values)
        updated_count += 1

    conn.commit()
    conn.close()
    print(f"Updated payload_json for {updated_count} rows in candidates.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    args = parser.parse_args()
    update_dipper_photometry(args.db)
