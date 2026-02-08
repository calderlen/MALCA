"""
Gaia DR3 epoch photometry utilities.

Fetch Gaia epoch photometry via DataLink (astroquery Gaia.load_data)
and compute Delta(M_G) and Delta(BP-RP) between first and last epochs.

Optionally, a TAP table can be used if provided.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import re
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
from astropy.table import Table
from tqdm.auto import tqdm

from malca.config.config_ltv import (
    LTV_GAIA_CHUNK_SIZE,
    LTV_WORKERS,
    GAIA_EPOCH_DATA_RELEASE,
    GAIA_EPOCH_DATA_STRUCTURE,
    GAIA_EPOCH_RETRIEVAL_TYPE,
    GAIA_EPOCH_DELTA_MAG_THRESH,
    GAIA_EPOCH_DELTA_COLOR_THRESH,
)


def _chunked(iterable: Iterable, size: int):
    it = list(iterable)
    for i in range(0, len(it), size):
        yield it[i:i + size]


def _batch_gaia_epoch_tap_query(
    ids_df: pd.DataFrame,
    *,
    tap_table: str,
    time_col: str,
    g_col: str,
    bp_col: str,
    rp_col: str,
    chunk_size: int = LTV_GAIA_CHUNK_SIZE,
    n_workers: int = LTV_WORKERS,
    verbose: bool = False,
) -> pd.DataFrame:
    from astroquery.gaia import Gaia

    if ids_df.empty:
        return pd.DataFrame()

    results = []
    chunks = [ids_df.iloc[i:i + chunk_size] for i in range(0, len(ids_df), chunk_size)]

    def process_chunk(chunk_df):
        try:
            upload_table = Table.from_pandas(chunk_df[["_idx", "source_id"]])
            query = f"""
            SELECT
                u._idx AS _idx,
                e.source_id AS source_id,
                e.{time_col} AS t,
                e.{g_col} AS g_mag,
                e.{bp_col} AS bp_mag,
                e.{rp_col} AS rp_mag
            FROM TAP_UPLOAD.upload_table AS u
            JOIN {tap_table} AS e
            ON e.source_id = u.source_id
            """
            job = Gaia.launch_job_async(
                query,
                upload_resource=upload_table,
                upload_table_name="upload_table",
                verbose=False,
            )
            result = job.get_results()
            return result.to_pandas() if result is not None else pd.DataFrame()
        except Exception as e:
            if verbose:
                print(f"Gaia epoch query error: {e}")
            return pd.DataFrame()

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(process_chunk, chunk): i for i, chunk in enumerate(chunks)}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Gaia epoch photometry", disable=not verbose):
            res = future.result()
            if not res.empty:
                results.append(res)

    if not results:
        return pd.DataFrame()

    return pd.concat(results, ignore_index=True)


def _as_float_array(values) -> np.ndarray:
    if values is None:
        return np.array([], dtype=float)
    arr = np.ma.array(values).filled(np.nan)
    try:
        out = np.array(arr, dtype=float)
    except Exception:
        out = pd.to_numeric(np.array(arr).astype(str), errors="coerce").to_numpy()
    return np.ravel(out)


def _first_last(times: np.ndarray, mags: np.ndarray):
    if times.size == 0 or mags.size == 0:
        return np.nan, np.nan, np.nan, np.nan, 0
    mask = np.isfinite(times) & np.isfinite(mags)
    if not mask.any():
        return np.nan, np.nan, np.nan, np.nan, 0
    t = times[mask]
    m = mags[mask]
    order = np.argsort(t)
    return t[order[0]], t[order[-1]], m[order[0]], m[order[-1]], int(mask.sum())


def _nanmin(values):
    arr = np.array(values, dtype=float)
    return np.nanmin(arr) if np.isfinite(arr).any() else np.nan


def _nanmax(values):
    arr = np.array(values, dtype=float)
    return np.nanmax(arr) if np.isfinite(arr).any() else np.nan


def _parse_source_id_from_key(key: str | None) -> int | None:
    if not key:
        return None
    matches = list(re.findall(r"\d{15,21}", str(key)))
    if not matches:
        return None
    try:
        return int(matches[0])
    except Exception:
        return None


def _match_col(cols_map: dict[str, str], candidates: list[str]) -> str | None:
    for cand in candidates:
        if cand in cols_map:
            return cols_map[cand]
    return None


def _summarize_epoch_table(
    table: Table,
    *,
    source_id_hint: int | None = None,
    time_col: str | None = None,
    mag_col: str | None = None,
    band_col: str | None = None,
) -> list[dict]:
    cols_map = {c.lower(): c for c in table.colnames}

    # Array-style (RAW) tables with per-band arrays
    g_time_col = _match_col(
        cols_map,
        ["g_transit_time", "g_obs_time", "g_time", "time_g"],
    )
    g_mag_col = _match_col(
        cols_map,
        ["g_transit_mag", "g_mag", "mag_g"],
    )
    bp_time_col = _match_col(
        cols_map,
        ["bp_obs_time", "bp_transit_time", "bp_time", "time_bp"],
    )
    bp_mag_col = _match_col(
        cols_map,
        ["bp_mag", "bp_transit_mag", "mag_bp"],
    )
    rp_time_col = _match_col(
        cols_map,
        ["rp_obs_time", "rp_transit_time", "rp_time", "time_rp"],
    )
    rp_mag_col = _match_col(
        cols_map,
        ["rp_mag", "rp_transit_mag", "mag_rp"],
    )

    if g_time_col and g_mag_col and (bp_time_col or rp_time_col):
        rows = []
        for row in table:
            source_id = None
            if "source_id" in cols_map:
                try:
                    source_id = int(row[cols_map["source_id"]])
                except Exception:
                    source_id = None
            if source_id is None:
                source_id = source_id_hint
            if source_id is None:
                continue

            g_t, g_mag = _as_float_array(row[g_time_col]), _as_float_array(row[g_mag_col])
            bp_t, bp_mag = (_as_float_array(row[bp_time_col]) if bp_time_col else np.array([], float),
                            _as_float_array(row[bp_mag_col]) if bp_mag_col else np.array([], float))
            rp_t, rp_mag = (_as_float_array(row[rp_time_col]) if rp_time_col else np.array([], float),
                            _as_float_array(row[rp_mag_col]) if rp_mag_col else np.array([], float))

            g_t_first, g_t_last, g_mag_first, g_mag_last, g_n = _first_last(g_t, g_mag)
            bp_t_first, bp_t_last, bp_mag_first, bp_mag_last, bp_n = _first_last(bp_t, bp_mag)
            rp_t_first, rp_t_last, rp_mag_first, rp_mag_last, rp_n = _first_last(rp_t, rp_mag)

            rows.append({
                "source_id": source_id,
                "gaia_epoch_t_first": _nanmin([g_t_first, bp_t_first, rp_t_first]),
                "gaia_epoch_t_last": _nanmax([g_t_last, bp_t_last, rp_t_last]),
                "gaia_epoch_g_first": g_mag_first,
                "gaia_epoch_g_last": g_mag_last,
                "gaia_epoch_bp_first": bp_mag_first,
                "gaia_epoch_bp_last": bp_mag_last,
                "gaia_epoch_rp_first": rp_mag_first,
                "gaia_epoch_rp_last": rp_mag_last,
                "gaia_epoch_n_obs": g_n + bp_n + rp_n,
            })
        return rows

    # Row-wise tables with band column (INDIVIDUAL)
    if band_col is None:
        band_col = _match_col(cols_map, ["band", "phot_band", "passband"])
    if time_col is None:
        time_col = _match_col(cols_map, ["time", "t", "epoch", "obs_time", "transit_time"])
    if mag_col is None:
        mag_col = _match_col(cols_map, ["mag", "magnitude"])

    if band_col and time_col and mag_col:
        df = table.to_pandas()
        if "source_id" not in df.columns:
            if source_id_hint is None:
                return []
            df["source_id"] = source_id_hint
        df["band"] = df[band_col].astype(str).str.upper()
        df["t"] = pd.to_numeric(df[time_col], errors="coerce")
        df["mag"] = pd.to_numeric(df[mag_col], errors="coerce")

        rows = []
        for source_id, sdf in df.groupby("source_id", sort=False):
            g = sdf[sdf["band"] == "G"]
            bp = sdf[sdf["band"] == "BP"]
            rp = sdf[sdf["band"] == "RP"]

            g_t_first, g_t_last, g_mag_first, g_mag_last, g_n = _first_last(
                g["t"].to_numpy(), g["mag"].to_numpy()
            )
            bp_t_first, bp_t_last, bp_mag_first, bp_mag_last, bp_n = _first_last(
                bp["t"].to_numpy(), bp["mag"].to_numpy()
            )
            rp_t_first, rp_t_last, rp_mag_first, rp_mag_last, rp_n = _first_last(
                rp["t"].to_numpy(), rp["mag"].to_numpy()
            )

            rows.append({
                "source_id": int(source_id),
                "gaia_epoch_t_first": _nanmin([g_t_first, bp_t_first, rp_t_first]),
                "gaia_epoch_t_last": _nanmax([g_t_last, bp_t_last, rp_t_last]),
                "gaia_epoch_g_first": g_mag_first,
                "gaia_epoch_g_last": g_mag_last,
                "gaia_epoch_bp_first": bp_mag_first,
                "gaia_epoch_bp_last": bp_mag_last,
                "gaia_epoch_rp_first": rp_mag_first,
                "gaia_epoch_rp_last": rp_mag_last,
                "gaia_epoch_n_obs": int(len(sdf)),
            })
        return rows

    return []


def _batch_gaia_epoch_datalink(
    source_ids: list[int],
    *,
    data_release: str = GAIA_EPOCH_DATA_RELEASE,
    data_structure: str = GAIA_EPOCH_DATA_STRUCTURE,
    retrieval_type: str = GAIA_EPOCH_RETRIEVAL_TYPE,
    valid_data: bool = True,
    band: str | None = None,
    fmt: str = "votable",
    verbose: bool = False,
) -> list[dict]:
    from astroquery.gaia import Gaia

    if not source_ids:
        return []

    datalink = Gaia.load_data(
        ids=source_ids,
        data_release=data_release,
        data_structure=data_structure,
        retrieval_type=retrieval_type,
        valid_data=valid_data,
        band=band,
        format=fmt,
        verbose=verbose,
    )

    rows: list[dict] = []

    if isinstance(datalink, Table):
        rows.extend(_summarize_epoch_table(datalink))
        return rows

    if isinstance(datalink, Mapping):
        for key, tables in datalink.items():
            source_id_hint = _parse_source_id_from_key(key)
            if not isinstance(tables, list):
                tables = [tables]
            for table in tables:
                if isinstance(table, Table):
                    rows.extend(_summarize_epoch_table(table, source_id_hint=source_id_hint))
        return rows

    if isinstance(datalink, list):
        for item in datalink:
            if isinstance(item, Table):
                rows.extend(_summarize_epoch_table(item))
    return rows


def query_gaia_epoch_photometry_batch(
    df: pd.DataFrame,
    *,
    source_id_col: str = "gaia_source_id",
    tap_table: str | None = None,
    time_col: str = "time",
    g_col: str = "g_mag",
    bp_col: str = "bp_mag",
    rp_col: str = "rp_mag",
    data_release: str = GAIA_EPOCH_DATA_RELEASE,
    data_structure: str = GAIA_EPOCH_DATA_STRUCTURE,
    retrieval_type: str = GAIA_EPOCH_RETRIEVAL_TYPE,
    valid_data: bool = True,
    band: str | None = None,
    chunk_size: int = LTV_GAIA_CHUNK_SIZE,
    n_workers: int = LTV_WORKERS,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Query Gaia epoch photometry and compute first/last deltas.

    If tap_table is provided, a TAP upload is used.
    Otherwise, Gaia DataLink (Gaia.load_data) is used.

    Adds:
      - gaia_epoch_t_first, gaia_epoch_t_last
      - gaia_epoch_g_first, gaia_epoch_g_last
      - gaia_epoch_bp_first, gaia_epoch_bp_last
      - gaia_epoch_rp_first, gaia_epoch_rp_last
      - gaia_epoch_delta_MG
      - gaia_epoch_delta_bp_rp
      - gaia_epoch_n_obs
    """
    if source_id_col not in df.columns:
        if verbose:
            print(f"Warning: '{source_id_col}' not found; skipping Gaia epoch photometry.")
        return df

    df_out = df.copy()

    # Build source_id list
    ids = pd.to_numeric(df_out[source_id_col], errors="coerce")
    valid = ids.notna()
    ids_df = pd.DataFrame({
        "_idx": df_out.index[valid],
        "source_id": ids[valid].astype(np.int64),
    })

    if ids_df.empty:
        return df_out

    if tap_table is not None:
        epoch_df = _batch_gaia_epoch_tap_query(
            ids_df,
            tap_table=tap_table,
            time_col=time_col,
            g_col=g_col,
            bp_col=bp_col,
            rp_col=rp_col,
            chunk_size=chunk_size,
            n_workers=n_workers,
            verbose=verbose,
        )

        if epoch_df.empty:
            return df_out

        epoch_df = epoch_df.sort_values("t")
        grouped = epoch_df.groupby("_idx", sort=False)
        first = grouped.first()
        last = grouped.last()
        count = grouped.size().rename("gaia_epoch_n_obs")

        summary = pd.DataFrame({
            "_idx": first.index,
            "gaia_epoch_t_first": first["t"].values,
            "gaia_epoch_t_last": last["t"].values,
            "gaia_epoch_g_first": first["g_mag"].values,
            "gaia_epoch_g_last": last["g_mag"].values,
            "gaia_epoch_bp_first": first["bp_mag"].values,
            "gaia_epoch_bp_last": last["bp_mag"].values,
            "gaia_epoch_rp_first": first["rp_mag"].values,
            "gaia_epoch_rp_last": last["rp_mag"].values,
            "gaia_epoch_n_obs": count.values,
        })
    else:
        chunk_size = min(int(chunk_size), 5000)
        summaries = []
        if verbose:
            print(f"Gaia epoch DataLink: {len(ids_df):,} sources in chunks of {chunk_size}")
        for chunk in tqdm(list(_chunked(ids_df["source_id"].tolist(), chunk_size)), disable=not verbose):
            rows = _batch_gaia_epoch_datalink(
                [int(x) for x in chunk],
                data_release=data_release,
                data_structure=data_structure,
                retrieval_type=retrieval_type,
                valid_data=valid_data,
                band=band,
                verbose=verbose,
            )
            summaries.extend(rows)

        if not summaries:
            return df_out

        summary = pd.DataFrame(summaries)
        summary = summary.drop_duplicates(subset=["source_id"])
        # Map back to original rows
        summary = summary.merge(ids_df, on="source_id", how="left")

    # Delta computations
    summary["gaia_epoch_delta_MG"] = summary["gaia_epoch_g_last"] - summary["gaia_epoch_g_first"]
    summary["gaia_epoch_delta_bp_rp"] = (
        (summary["gaia_epoch_bp_last"] - summary["gaia_epoch_rp_last"])
        - (summary["gaia_epoch_bp_first"] - summary["gaia_epoch_rp_first"])
    )

    df_out["_idx"] = df_out.index
    df_out = df_out.merge(summary, on="_idx", how="left").drop(columns=["_idx"])
    return df_out


def apply_gaia_epoch_flags(
    df: pd.DataFrame,
    *,
    delta_mag_thresh: float = GAIA_EPOCH_DELTA_MAG_THRESH,
    delta_color_thresh: float = GAIA_EPOCH_DELTA_COLOR_THRESH,
) -> pd.DataFrame:
    """
    Add flags for large Gaia epoch changes.
    """
    if df.empty:
        return df

    if "gaia_epoch_delta_MG" not in df.columns or "gaia_epoch_delta_bp_rp" not in df.columns:
        return df

    df = df.copy()
    df["gaia_epoch_large_delta_MG"] = df["gaia_epoch_delta_MG"].abs() > delta_mag_thresh
    df["gaia_epoch_large_delta_bp_rp"] = df["gaia_epoch_delta_bp_rp"].abs() > delta_color_thresh
    df["gaia_epoch_large_change"] = df["gaia_epoch_large_delta_MG"] | df["gaia_epoch_large_delta_bp_rp"]
    return df
