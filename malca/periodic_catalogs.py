from __future__ import annotations

from pathlib import Path
import math
import time

from astropy import units as u
from astropy.coordinates import SkyCoord
from astroquery.vizier import Vizier
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import pyvo

from malca.config import (
    DEFAULT_CACHE_DIR,
    GAIA_AIP_TAP_URL,
    PARQUET_CACHE_COMPRESSION,
    POST_FILTER_COORD_CHUNK_SIZE,
    POST_FILTER_REL_TOL,
    VSX_CROSSMATCH_PATH,
)
from malca.gaia_ids import parse_gaia_source_id
from malca.table_io import read_parquet_table
from malca.vsx.metadata import normalize_vsx_match_columns, select_best_vsx_matches


DEFAULT_CACHE_DIR = DEFAULT_CACHE_DIR.expanduser()
COMPACT_TQDM_BAR_FORMAT = (
    "{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} "
    "[{elapsed}<{remaining}, {rate_fmt}]"
)
GAIA_EB_OUTPUT_COLUMNS = ["source_id", "period", "var_type", "global_ranking"]
GAIA_EB_CACHE_COLUMNS = [*GAIA_EB_OUTPUT_COLUMNS, "matched"]
GAIA_EB_CACHE_FLUSH_CHUNKS = 50

PERIOD_SOURCE_PRIORITY = (
    "gaia_eb",
    "vsx",
    "asassn_var",
    "ztf_periodic",
    "ogle",
)

PERIOD_HARMONIC_FACTORS = (1.0, 2.0, 0.5, 3.0, 1.0 / 3.0)

PERIODIC_CATALOG_MERGE_COLS = (
    "catalog_match",
    "catalog_period",
    "catalog_class",
    "catalog_source",
    "period_sources",
    "period_n_sources",
    "period_consensus_days",
    "period_consensus_agree",
    "period_conflict_flag",
    "period_consensus_support",
    "period_primary_source",
    "period_source_periods",
    "period_gaia_eb_match",
    "period_gaia_eb_days",
    "period_gaia_eb_class",
    "period_gaia_eb_sep_arcsec",
    "period_vsx_match",
    "period_vsx_days",
    "period_vsx_class",
    "period_vsx_sep_arcsec",
    "period_asassn_var_match",
    "period_asassn_var_days",
    "period_asassn_var_class",
    "period_asassn_var_sep_arcsec",
    "period_ztf_periodic_match",
    "period_ztf_periodic_days",
    "period_ztf_periodic_class",
    "period_ztf_periodic_sep_arcsec",
    "period_ogle_match",
    "period_ogle_days",
    "period_ogle_class",
    "period_ogle_sep_arcsec",
)


def extract_asassn_ids(df: pd.DataFrame) -> pd.Series:
    """Extract ASAS-SN IDs from canonical candidate identifier columns."""
    if "asas_sn_id" in df.columns:
        raw = df["asas_sn_id"]
    elif "source_id" in df.columns:
        raw = df["source_id"]
    elif "lc_path" in df.columns:
        raw = df["lc_path"].astype(str).map(lambda p: Path(p).stem)
    elif "path" in df.columns:
        raw = df["path"].astype(str).map(lambda p: Path(p).stem)
    else:
        return pd.Series([pd.NA] * len(df), index=df.index, dtype="object")

    out = raw.astype(str).str.strip()
    out = out.mask(out.eq(""), pd.NA)
    return out


def pick_coord_columns(df: pd.DataFrame) -> tuple[str | None, str | None]:
    """Pick available candidate coordinate columns."""
    for ra_col, dec_col in (("ra", "dec"), ("ra_deg", "dec_deg")):
        if ra_col in df.columns and dec_col in df.columns:
            return ra_col, dec_col
    return None, None


def periods_agree(period_a: float, period_b: float, *, rel_tol: float = POST_FILTER_REL_TOL) -> bool:
    """Return True when periods agree directly or via common harmonics."""
    if not (np.isfinite(period_a) and np.isfinite(period_b)):
        return False
    if period_a <= 0 or period_b <= 0:
        return False
    ratio = max(period_a, period_b) / min(period_a, period_b)
    for factor in PERIOD_HARMONIC_FACTORS:
        if factor <= 0:
            continue
        if abs(ratio - factor) / factor <= rel_tol:
            return True
    return False


def normalize_period_to_reference(period: float, reference: float) -> float:
    """Map period onto the closest harmonic around reference."""
    if not (np.isfinite(period) and np.isfinite(reference)):
        return period
    if period <= 0 or reference <= 0:
        return period

    candidates = [
        period,
        period / 2.0,
        period * 2.0,
        period / 3.0,
        period * 3.0,
    ]
    return float(
        min(
            candidates,
            key=lambda p: abs(math.log10(p) - math.log10(reference)) if p > 0 else np.inf,
        )
    )


def choose_consensus_period(
    periods_by_source: dict[str, float],
    *,
    rel_tol: float = POST_FILTER_REL_TOL,
) -> tuple[float, bool, bool, float, str]:
    """Return consensus period and agreement metadata."""
    valid = {
        src: float(p)
        for src, p in periods_by_source.items()
        if np.isfinite(p) and float(p) > 0
    }
    if not valid:
        return np.nan, False, False, np.nan, ""

    ordered_sources = sorted(
        valid.keys(),
        key=lambda s: PERIOD_SOURCE_PRIORITY.index(s) if s in PERIOD_SOURCE_PRIORITY else len(PERIOD_SOURCE_PRIORITY),
    )
    if len(valid) == 1:
        src = ordered_sources[0]
        return float(valid[src]), True, False, 1.0, src

    best_source = ""
    best_support = -1
    for src in ordered_sources:
        p = valid[src]
        support = sum(periods_agree(p, q, rel_tol=rel_tol) for q in valid.values())
        if support > best_support:
            best_support = support
            best_source = src

    reference = valid[best_source]
    inlier_sources = [src for src, p in valid.items() if periods_agree(p, reference, rel_tol=rel_tol)]
    normalized = [normalize_period_to_reference(valid[src], reference) for src in inlier_sources]
    consensus = float(np.median(normalized)) if normalized else float(reference)

    n_sources = len(valid)
    support_fraction = float(len(inlier_sources) / n_sources) if n_sources else np.nan
    agree = bool(len(inlier_sources) == n_sources)
    conflict = bool((n_sources >= 2) and (not agree))
    return consensus, agree, conflict, support_fraction, best_source


def fetch_chen2020_ztf_periodic(
    cache_dir: Path | None = None,
    force_download: bool = False,
    show_tqdm: bool = True,
) -> pd.DataFrame:
    """Fetch Chen+2020 ZTF periodic variable catalog from VizieR."""
    cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "chen2020_ztf_periodic.parquet"

    if cache_file.exists() and not force_download:
        if show_tqdm:
            tqdm.write(f"[fetch_chen2020] Loading cached catalog from {cache_file}")
        return pd.read_parquet(cache_file)

    if show_tqdm:
        tqdm.write("[fetch_chen2020] Querying VizieR J/ApJS/249/18 (this may take a few minutes)...")

    try:
        v = Vizier(columns=["RAJ2000", "DEJ2000", "Per", "Type", "GaiaEDR3"], row_limit=-1)
        tables = v.get_catalogs("J/ApJS/249/18")
        if not tables:
            raise ValueError("No tables returned from VizieR query")

        cat = tables[0].to_pandas()
        df = pd.DataFrame(
            {
                "ra": cat["RAJ2000"].astype(float),
                "dec": cat["DEJ2000"].astype(float),
                "period": cat["Per"].astype(float),
                "var_type": cat["Type"].astype(str),
            }
        )

        if "GaiaEDR3" in cat.columns:
            df["gaia_id"] = pd.to_numeric(cat["GaiaEDR3"], errors="coerce").astype("Int64")
        elif "GaiaDR3" in cat.columns:
            df["gaia_id"] = pd.to_numeric(cat["GaiaDR3"], errors="coerce").astype("Int64")
        elif "GaiaDR2" in cat.columns:
            df["gaia_id"] = pd.to_numeric(cat["GaiaDR2"], errors="coerce").astype("Int64")

        df.to_parquet(cache_file, index=False, compression=PARQUET_CACHE_COMPRESSION)
        if show_tqdm:
            tqdm.write(f"[fetch_chen2020] Cached {len(df)} sources to {cache_file}")
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to fetch Chen+2020 catalog from VizieR: {e}") from e


def fetch_asassn_variable_catalog(
    cache_dir: Path | None = None,
    force_download: bool = False,
    show_tqdm: bool = True,
) -> pd.DataFrame:
    """Fetch ASAS-SN variable star catalog (VizieR II/366/catv2021)."""
    cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "asassn_var_ii366.parquet"

    if cache_file.exists() and not force_download:
        if show_tqdm:
            tqdm.write(f"[fetch_asassn_var] Loading cached catalog from {cache_file}")
        return pd.read_parquet(cache_file)

    if show_tqdm:
        tqdm.write("[fetch_asassn_var] Querying VizieR II/366/catv2021 (this may take a few minutes)...")

    try:
        v = Vizier(columns=["ASASSN-V", "RAJ2000", "DEJ2000", "Per", "Type", "GaiaDR3"], row_limit=-1)
        tables = v.get_catalogs("II/366/catv2021")
        if not tables:
            raise ValueError("No tables returned from VizieR query")

        cat = tables[0].to_pandas()
        df = pd.DataFrame(
            {
                "source_name": cat.get("ASASSN-V", pd.Series(index=cat.index)).astype(str),
                "ra": pd.to_numeric(cat.get("RAJ2000"), errors="coerce"),
                "dec": pd.to_numeric(cat.get("DEJ2000"), errors="coerce"),
                "period": pd.to_numeric(cat.get("Per"), errors="coerce"),
                "var_type": cat.get("Type", pd.Series(index=cat.index)).astype(str),
                "gaia_id": pd.to_numeric(cat.get("GaiaDR3"), errors="coerce").astype("Int64"),
            }
        )

        df.to_parquet(cache_file, index=False, compression=PARQUET_CACHE_COMPRESSION)
        if show_tqdm:
            tqdm.write(f"[fetch_asassn_var] Cached {len(df)} sources to {cache_file}")
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to fetch ASAS-SN variable catalog from VizieR: {e}") from e


def fetch_ogle_periodic_catalog(
    cache_dir: Path | None = None,
    force_download: bool = False,
    show_tqdm: bool = True,
) -> pd.DataFrame:
    """Fetch OGLE periodic variable catalog (VizieR II/213/pvar)."""
    cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "ogle_ii213_pvar.parquet"

    if cache_file.exists() and not force_download:
        if show_tqdm:
            tqdm.write(f"[fetch_ogle] Loading cached catalog from {cache_file}")
        return pd.read_parquet(cache_file)

    if show_tqdm:
        tqdm.write("[fetch_ogle] Querying VizieR II/213/pvar...")

    try:
        v = Vizier(columns=["OGLE", "RAJ2000", "DEJ2000", "Per", "Type"], row_limit=-1)
        tables = v.get_catalogs("II/213/pvar")
        if not tables:
            raise ValueError("No tables returned from VizieR query")

        cat = tables[0].to_pandas()
        df = pd.DataFrame(
            {
                "source_name": cat.get("OGLE", pd.Series(index=cat.index)).astype(str),
                "ra": pd.to_numeric(cat.get("RAJ2000"), errors="coerce"),
                "dec": pd.to_numeric(cat.get("DEJ2000"), errors="coerce"),
                "period": pd.to_numeric(cat.get("Per"), errors="coerce"),
                "var_type": cat.get("Type", pd.Series(index=cat.index)).astype(str),
            }
        )

        df.to_parquet(cache_file, index=False, compression=PARQUET_CACHE_COMPRESSION)
        if show_tqdm:
            tqdm.write(f"[fetch_ogle] Cached {len(df)} sources to {cache_file}")
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to fetch OGLE periodic catalog from VizieR: {e}") from e


def fetch_vsx_period_catalog(
    vsx_crossmatch_csv: str | Path = VSX_CROSSMATCH_PATH,
    show_tqdm: bool = True,
) -> pd.DataFrame:
    """Load VSX crossmatch table and expose ASAS-SN keyed periods/classes."""
    path = Path(vsx_crossmatch_csv).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"VSX crossmatch file not found: {path}")

    if show_tqdm:
        tqdm.write(f"[fetch_vsx_period] Loading VSX crossmatch from {path}")

    xmatch = normalize_vsx_match_columns(read_parquet_table(path))

    required_cols = {"asas_sn_id"}
    missing = [c for c in required_cols if c not in xmatch.columns]
    if missing:
        raise ValueError(f"VSX crossmatch file missing required columns: {missing}")

    keep_cols = [c for c in ["asas_sn_id", "vsx_period", "vsx_class", "vsx_sep_arcsec", "gaia_id", "ra", "dec"] if c in xmatch.columns]
    out = xmatch[keep_cols].copy()
    if "gaia_id" in out.columns:
        out["gaia_id"] = pd.to_numeric(out["gaia_id"], errors="coerce").astype("Int64")

    out = select_best_vsx_matches(out, id_column="asas_sn_id")
    return out.rename(columns={"vsx_class": "var_type", "vsx_period": "period"}).reset_index(drop=True)


def fetch_gaia_dr3_eb_periods(
    source_ids: list[int] | None,
    *,
    cache_dir: Path | None = None,
    chunk_size: int = 1000,
    show_tqdm: bool = True,
) -> pd.DataFrame:
    """Fetch Gaia DR3 eclipsing-binary periods for source IDs, with negative lookup cache."""
    if source_ids is None or len(source_ids) == 0:
        return pd.DataFrame(columns=GAIA_EB_OUTPUT_COLUMNS)

    requested_ids = sorted({int(sid) for sid in source_ids})
    cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "gaia_dr3_eb_periods.parquet"

    cached_df = pd.DataFrame(columns=GAIA_EB_CACHE_COLUMNS)
    if cache_file.exists():
        try:
            cached_df = pd.read_parquet(cache_file)
            if "source_id" in cached_df.columns:
                cached_df["source_id"] = pd.to_numeric(cached_df["source_id"], errors="coerce").astype("Int64")
        except Exception:
            cached_df = pd.DataFrame(columns=GAIA_EB_CACHE_COLUMNS)
    for col in GAIA_EB_CACHE_COLUMNS:
        if col not in cached_df.columns:
            cached_df[col] = True if col == "matched" else np.nan if col in {"source_id", "period", "global_ranking"} else ""
    cached_df = cached_df.loc[:, GAIA_EB_CACHE_COLUMNS].copy()
    cached_df["source_id"] = pd.to_numeric(cached_df["source_id"], errors="coerce").astype("Int64")

    cached_ids = {
        int(v)
        for v in pd.to_numeric(cached_df["source_id"], errors="coerce").dropna().tolist()
    }
    missing_ids = [sid for sid in requested_ids if sid not in cached_ids]
    if show_tqdm and missing_ids:
        tqdm.write(f"[fetch_gaia_eb] Querying Gaia TAP for {len(missing_ids)} uncached source IDs")

    new_rows: list[dict[str, object]] = []

    def flush_cache() -> None:
        nonlocal cached_df, new_rows
        if not new_rows:
            return
        new_df = pd.DataFrame.from_records(new_rows, columns=GAIA_EB_CACHE_COLUMNS)
        full_df = new_df.copy() if cached_df.empty else pd.concat([cached_df, new_df], ignore_index=True)
        full_df["source_id"] = pd.to_numeric(full_df["source_id"], errors="coerce").astype("Int64")
        full_df = full_df.dropna(subset=["source_id"])
        full_df = full_df.drop_duplicates(subset=["source_id"], keep="last")
        full_df.to_parquet(cache_file, index=False, compression=PARQUET_CACHE_COMPRESSION)
        cached_df = full_df
        new_rows = []

    if missing_ids:
        tap = pyvo.dal.TAPService(GAIA_AIP_TAP_URL)
        chunks = range(0, len(missing_ids), max(1, int(chunk_size)))
        iterator = tqdm(
            chunks,
            desc="fetch_gaia_eb",
            unit="chunk",
            leave=False,
            disable=not show_tqdm,
            dynamic_ncols=True,
            bar_format=COMPACT_TQDM_BAR_FORMAT,
        )
        for i in iterator:
            chunk = missing_ids[i : i + max(1, int(chunk_size))]
            ids_str = ",".join(str(sid) for sid in chunk)
            query = f"""
                SELECT source_id, frequency, model_type, global_ranking
                FROM gaiadr3.vari_eclipsing_binary
                WHERE source_id IN ({ids_str})
            """
            attempt = 0
            while True:
                attempt += 1
                try:
                    result = tap.run_sync(query)
                    break
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    delay = min(5.0 * attempt, 60.0)
                    if show_tqdm:
                        msg = str(e).splitlines()[0].strip()
                        tqdm.write(
                            f"[fetch_gaia_eb] chunk query failed on attempt {attempt}; "
                            f"retrying in {delay:.0f}s: {msg}"
                        )
                    time.sleep(delay)

            found_ids: set[int] = set()
            for row in result:
                sid_int = int(row["source_id"])
                found_ids.add(sid_int)
                freq = row["frequency"]
                period = np.nan
                if freq is not None:
                    try:
                        fv = float(freq)
                        if np.isfinite(fv) and fv > 0:
                            period = 1.0 / fv
                    except Exception:
                        period = np.nan

                new_rows.append(
                    {
                        "source_id": sid_int,
                        "period": period,
                        "var_type": str(row["model_type"]) if row["model_type"] is not None else "",
                        "global_ranking": float(row["global_ranking"]) if row["global_ranking"] is not None else np.nan,
                        "matched": True,
                    }
                )

            for sid in chunk:
                if int(sid) not in found_ids:
                    new_rows.append(
                        {
                            "source_id": int(sid),
                            "period": np.nan,
                            "var_type": "",
                            "global_ranking": np.nan,
                            "matched": False,
                        }
                    )

            if (iterator.n + 1) % GAIA_EB_CACHE_FLUSH_CHUNKS == 0:
                flush_cache()

    flush_cache()
    full_df = cached_df
    if full_df.empty:
        return full_df

    full_df["source_id"] = pd.to_numeric(full_df["source_id"], errors="coerce").astype("Int64")
    keep = full_df["source_id"].isin(requested_ids)
    matched = full_df["matched"].astype("boolean").fillna(True).to_numpy(dtype=bool)
    return full_df.loc[keep & matched, GAIA_EB_OUTPUT_COLUMNS].reset_index(drop=True)


def parse_gaia_id_int(value: object) -> int | None:
    """Parse Gaia source ID-like values to int when possible."""
    source_id = parse_gaia_source_id(value)
    if source_id is None:
        return None
    try:
        return int(source_id)
    except Exception:
        return None


def match_period_catalog(
    df: pd.DataFrame,
    catalog_df: pd.DataFrame,
    *,
    source_label: str,
    max_sep_arcsec: float,
    period_col: str = "period",
    class_col: str = "var_type",
    gaia_col: str = "gaia_id",
    ra_col: str = "ra",
    dec_col: str = "dec",
    candidate_asassn_ids: pd.Series | None = None,
    catalog_asassn_col: str | None = None,
    show_tqdm: bool = False,
) -> pd.DataFrame:
    """Match one catalog to candidates and return per-source period columns."""
    n0 = len(df)
    match = np.zeros(n0, dtype=bool)
    period = np.full(n0, np.nan, dtype=float)
    cls = np.array([""] * n0, dtype=object)
    sep = np.full(n0, np.nan, dtype=float)

    if catalog_df is None or catalog_df.empty:
        return pd.DataFrame(
            {
                f"period_{source_label}_match": match,
                f"period_{source_label}_days": period,
                f"period_{source_label}_class": cls,
                f"period_{source_label}_sep_arcsec": sep,
            },
            index=df.index,
        )

    cat = catalog_df.copy()
    cat[period_col] = pd.to_numeric(cat[period_col], errors="coerce") if period_col in cat.columns else np.nan
    cat[class_col] = cat[class_col].fillna("").astype(str) if class_col in cat.columns else ""

    if gaia_col in cat.columns and "gaia_id" in df.columns:
        cand_gaia = pd.Series([parse_gaia_id_int(v) for v in df["gaia_id"].tolist()], index=df.index, dtype="object")
        cat_gaia = pd.Series([parse_gaia_id_int(v) for v in cat[gaia_col].tolist()], index=cat.index, dtype="object")
        cat_valid = cat.loc[cat_gaia.notna()].copy()
        if not cat_valid.empty:
            cat_valid["_gaia_id"] = cat_gaia.loc[cat_valid.index].astype(int)
            if "vsx_sep_arcsec" in cat_valid.columns:
                cat_valid["vsx_sep_arcsec"] = pd.to_numeric(cat_valid["vsx_sep_arcsec"], errors="coerce")
                cat_valid = cat_valid.sort_values("vsx_sep_arcsec", na_position="last")
            cat_valid = cat_valid.drop_duplicates(subset=["_gaia_id"], keep="first").set_index("_gaia_id")

            mapped_period = cand_gaia.map(cat_valid[period_col])
            mapped_class = cand_gaia.map(cat_valid[class_col]).fillna("")
            valid_period = mapped_period.notna() & np.isfinite(mapped_period.to_numpy(dtype=float)) & (mapped_period.to_numpy(dtype=float) > 0)
            if valid_period.any():
                idx_mask = valid_period.to_numpy()
                match[idx_mask] = True
                period[idx_mask] = mapped_period.loc[valid_period].to_numpy(dtype=float)
                cls[idx_mask] = mapped_class.loc[valid_period].astype(str).to_numpy()
                sep[idx_mask] = 0.0

    if catalog_asassn_col and catalog_asassn_col in cat.columns and candidate_asassn_ids is not None:
        cat_asas = cat[catalog_asassn_col].astype(str).str.strip()
        cat_valid = cat.loc[cat_asas.notna() & cat_asas.ne("")].copy()
        if not cat_valid.empty:
            cat_valid["_asas_id"] = cat_asas.loc[cat_valid.index]
            if "vsx_sep_arcsec" in cat_valid.columns:
                cat_valid["vsx_sep_arcsec"] = pd.to_numeric(cat_valid["vsx_sep_arcsec"], errors="coerce")
                cat_valid = cat_valid.sort_values("vsx_sep_arcsec", na_position="last")
            cat_valid = cat_valid.drop_duplicates(subset=["_asas_id"], keep="first").set_index("_asas_id")

            mapped_period = candidate_asassn_ids.map(cat_valid[period_col])
            mapped_class = candidate_asassn_ids.map(cat_valid[class_col]).fillna("")
            mapped_sep = (
                candidate_asassn_ids.map(cat_valid["vsx_sep_arcsec"])
                if "vsx_sep_arcsec" in cat_valid.columns
                else pd.Series(np.nan, index=df.index)
            )

            valid_period = mapped_period.notna() & np.isfinite(mapped_period.to_numpy(dtype=float)) & (mapped_period.to_numpy(dtype=float) > 0)
            if valid_period.any():
                idx_mask = valid_period.to_numpy() & (~match)
                match[idx_mask] = True
                period[idx_mask] = mapped_period.loc[idx_mask].to_numpy(dtype=float)
                cls[idx_mask] = mapped_class.loc[idx_mask].astype(str).to_numpy()
                sep[idx_mask] = pd.to_numeric(mapped_sep.loc[idx_mask], errors="coerce").to_numpy(dtype=float)

    ra_cand_col, dec_cand_col = pick_coord_columns(df)
    if ra_cand_col is not None and dec_cand_col is not None and ra_col in cat.columns and dec_col in cat.columns:
        remaining = ~match
        cand_ra = pd.to_numeric(df[ra_cand_col], errors="coerce").to_numpy(dtype=float)
        cand_dec = pd.to_numeric(df[dec_cand_col], errors="coerce").to_numpy(dtype=float)
        valid_cand = remaining & np.isfinite(cand_ra) & np.isfinite(cand_dec)

        cat_ra = pd.to_numeric(cat[ra_col], errors="coerce").to_numpy(dtype=float)
        cat_dec = pd.to_numeric(cat[dec_col], errors="coerce").to_numpy(dtype=float)
        cat_period = pd.to_numeric(cat[period_col], errors="coerce").to_numpy(dtype=float)
        cat_class = cat[class_col].astype(str).to_numpy(dtype=object)

        valid_cat = np.isfinite(cat_ra) & np.isfinite(cat_dec) & np.isfinite(cat_period) & (cat_period > 0)
        if valid_cand.any() and valid_cat.any():
            cat_coords = SkyCoord(ra=cat_ra[valid_cat] * u.deg, dec=cat_dec[valid_cat] * u.deg)
            cat_period_valid = cat_period[valid_cat]
            cat_class_valid = cat_class[valid_cat]

            cand_indices = np.flatnonzero(valid_cand)
            iterator = range(0, len(cand_indices), POST_FILTER_COORD_CHUNK_SIZE)
            if show_tqdm and len(cand_indices) > POST_FILTER_COORD_CHUNK_SIZE:
                iterator = tqdm(iterator, desc=f"match_{source_label}_coords", leave=False)
            for start in iterator:
                sub_idx = cand_indices[start : start + POST_FILTER_COORD_CHUNK_SIZE]
                cand_coords = SkyCoord(ra=cand_ra[sub_idx] * u.deg, dec=cand_dec[sub_idx] * u.deg)
                idx_cat, sep2d, _ = cand_coords.match_to_catalog_sky(cat_coords)
                sep_arcsec = sep2d.to(u.arcsec).value
                within = sep_arcsec <= float(max_sep_arcsec)
                if not np.any(within):
                    continue

                out_idx = sub_idx[within]
                src_idx = idx_cat[within]
                match[out_idx] = True
                period[out_idx] = cat_period_valid[src_idx]
                cls[out_idx] = cat_class_valid[src_idx]
                sep[out_idx] = sep_arcsec[within]

    return pd.DataFrame(
        {
            f"period_{source_label}_match": match,
            f"period_{source_label}_days": period,
            f"period_{source_label}_class": cls,
            f"period_{source_label}_sep_arcsec": sep,
        },
        index=df.index,
    )


_extract_asassn_ids = extract_asassn_ids
_pick_coord_columns = pick_coord_columns
_periods_agree = periods_agree
_normalize_period_to_reference = normalize_period_to_reference
_choose_consensus_period = choose_consensus_period
_parse_gaia_id_int = parse_gaia_id_int
_match_period_catalog = match_period_catalog
