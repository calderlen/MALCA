"""Versioned open-cluster membership and proximity enrichment.

Membership is determined only by exact Gaia DR3 source identifiers in the
published UCC and Hunt & Reffert member tables.  Nearest-cluster quantities
are kept as separate environmental diagnostics and never imply membership.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Iterable

from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np
import pandas as pd

from malca.catalogs.gaia_ids import normalize_gaia_source_id_series
from malca.products.feature_layers import expand_feature_layers


UCC_CATALOG_DOI = "10.5281/zenodo.20705026"
UCC_DEFAULT_RELEASE = "260615"
HR24_CATALOG_ID = "J/A+A/686/A42"
OPEN_CLUSTER_MATCH_VERSION = "1"

UCC_MEMBER_THRESHOLD = 0.5
HR24_MEMBER_THRESHOLD = 0.5
HR24_HIGH_QUALITY_CST_MIN = 5.0
HR24_HIGH_QUALITY_CMD_MIN = 0.5

UCC_OUTPUT_COLUMNS = (
    "ucc_match_status",
    "ucc_catalog_release",
    "ucc_catalog_doi",
    "ucc_n_matches",
    "ucc_listed_member",
    "ucc_p50_member",
    "ucc_good_cluster",
    "ucc_good_member",
    "ucc_cluster",
    "ucc_pmem",
    "ucc_n_members",
    "ucc_r50_arcmin",
    "ucc_core_radius_pc",
    "ucc_cluster_ra_deg",
    "ucc_cluster_dec_deg",
    "ucc_cluster_parallax_mas",
    "ucc_cluster_pmra_masyr",
    "ucc_cluster_pmdec_masyr",
    "ucc_cluster_rv_kms",
    "ucc_distance_kpc",
    "ucc_distance_std",
    "ucc_av_mag",
    "ucc_av_std",
    "ucc_age_myr",
    "ucc_age_std",
    "ucc_feh_dex",
    "ucc_feh_std",
    "ucc_mass_msun",
    "ucc_mass_std",
    "ucc_c3",
    "ucc_pdup",
    "ucc_uti",
    "ucc_bad_oc",
)

UCC_PROXIMITY_COLUMNS = (
    "ucc_nearest_cluster",
    "ucc_nearest_sep_arcmin",
    "ucc_nearest_r50_arcmin",
    "ucc_nearest_sep_r50",
    "ucc_nearest_age_myr",
    "ucc_nearest_distance_kpc",
    "ucc_nearest_uti",
    "ucc_nearest_dparallax_mas",
    "ucc_nearest_dpmra_masyr",
    "ucc_nearest_dpmdec_masyr",
)

HR24_OUTPUT_COLUMNS = (
    "hr24_match_status",
    "hr24_catalog_id",
    "hr24_n_matches",
    "hr24_listed_member",
    "hr24_p50_member",
    "hr24_bound_member",
    "hr24_high_quality_member",
    "hr24_cluster",
    "hr24_pmem",
    "hr24_cluster_type",
    "hr24_cst",
    "hr24_cmd_class_median",
    "hr24_n_members",
    "hr24_r50_deg",
    "hr24_r50_pc",
    "hr24_cluster_ra_deg",
    "hr24_cluster_dec_deg",
    "hr24_cluster_parallax_mas",
    "hr24_cluster_pmra_masyr",
    "hr24_cluster_pmdec_masyr",
    "hr24_distance_pc",
    "hr24_log_age_yr",
    "hr24_prob_jacobi",
    "hr24_mass_jacobi_msun",
    "hr24_mass_total_msun",
    "hr24_inside_jacobi_radius",
    "hr24_inside_tidal_radius",
)

LEGACY_CLUSTER_COLUMNS = (
    "cluster_name",
    "cluster_membership_prob",
    "cluster_age_myr",
    "cluster_dist_pc",
    "cluster_catalog",
)

OPEN_CLUSTER_OUTPUT_COLUMNS = tuple(
    dict.fromkeys(
        (
            "open_cluster_match_version",
            "open_cluster_gaia_id",
            *UCC_OUTPUT_COLUMNS,
            *UCC_PROXIMITY_COLUMNS,
            *HR24_OUTPUT_COLUMNS,
            *LEGACY_CLUSTER_COLUMNS,
        )
    )
)


@dataclass(frozen=True)
class OpenClusterMatchResult:
    """Source-level output, all associations, and catalogue provenance."""

    sources: pd.DataFrame
    all_matches: pd.DataFrame
    manifest: dict[str, object]


def gaia_identifier_series(frame: pd.DataFrame) -> pd.Series:
    """Return the first exact Gaia identifier available for each row."""
    result = pd.Series(pd.NA, index=frame.index, dtype="string")
    for column in ("source_id", "gaia_id", "gaia_source_id", "source_id_gaia"):
        if column not in frame.columns:
            continue
        values = normalize_gaia_source_id_series(frame[column])
        result = result.combine_first(values)
    return result


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _boolean(value: object) -> bool | pd._libs.missing.NAType:
    try:
        if value is None or pd.isna(value):
            return pd.NA
    except (TypeError, ValueError):
        if value is None:
            return pd.NA
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer, float, np.floating)):
        return bool(int(value))
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "bad"}:
        return True
    if text in {"0", "false", "f", "no", "n", "good"}:
        return False
    return pd.NA


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "size_bytes": int(path.stat().st_size),
        "sha256": _sha256(path),
    }


def _find_one(directory: Path, names: Iterable[str]) -> Path:
    candidates = [directory / name for name in names]
    found = [path for path in candidates if path.exists()]
    if not found:
        raise FileNotFoundError(
            f"None of the expected catalogue files exist under {directory}: "
            + ", ".join(path.name for path in candidates)
        )
    return found[0]


def _ucc_release(directory: Path, readme_path: Path | None) -> str:
    if readme_path is not None and readme_path.exists():
        text = readme_path.read_text(encoding="utf-8", errors="replace")
        match = re.search(r"\b(\d{6})\s+version\b", text, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    if re.fullmatch(r"\d{6}", directory.name):
        return directory.name
    return UCC_DEFAULT_RELEASE


def load_ucc_tables(
    directory: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Load and validate a pinned UCC bulk release."""
    root = Path(directory).expanduser()
    catalog_path = _find_one(root, ("UCC_cat.csv", "UCC_cat.csv.gz"))
    members_path = _find_one(root, ("UCC_members.parquet",))
    readme_candidates = [root / "README.txt", root / "README.md"]
    readme_path = next((path for path in readme_candidates if path.exists()), None)

    clusters = pd.read_csv(catalog_path, low_memory=False)
    members = pd.read_parquet(members_path, columns=["name", "Source", "probs"])
    required_clusters = {
        "name",
        "N_membs",
        "r_50",
        "r_core",
        "RA_ICRS",
        "DE_ICRS",
        "Plx",
        "pmRA",
        "pmDE",
        "Dist_[kpc]",
        "Age_[Myr]",
        "C3",
        "P_dup",
        "UTI",
        "bad_oc",
    }
    missing = sorted(required_clusters - set(clusters.columns))
    if missing:
        raise ValueError(f"UCC cluster table is missing required columns: {missing}")
    if clusters["name"].duplicated().any():
        examples = clusters.loc[clusters["name"].duplicated(False), "name"].head().tolist()
        raise ValueError(f"UCC cluster names are not unique: {examples}")

    members = members.copy()
    members["open_cluster_gaia_id"] = normalize_gaia_source_id_series(members["Source"])
    members["probs"] = pd.to_numeric(members["probs"], errors="coerce")
    members = members[members["open_cluster_gaia_id"].notna()].copy()

    rename = {
        "name": "ucc_cluster",
        "probs": "ucc_pmem",
        "N_membs": "ucc_n_members",
        "r_50": "ucc_r50_arcmin",
        "r_core": "ucc_core_radius_pc",
        "RA_ICRS": "ucc_cluster_ra_deg",
        "DE_ICRS": "ucc_cluster_dec_deg",
        "Plx": "ucc_cluster_parallax_mas",
        "pmRA": "ucc_cluster_pmra_masyr",
        "pmDE": "ucc_cluster_pmdec_masyr",
        "Rv": "ucc_cluster_rv_kms",
        "Dist_[kpc]": "ucc_distance_kpc",
        "Dist_STDDEV": "ucc_distance_std",
        "Av_[mag]": "ucc_av_mag",
        "Av_STDDEV": "ucc_av_std",
        "Age_[Myr]": "ucc_age_myr",
        "Age_STDDEV": "ucc_age_std",
        "FeH_[dex]": "ucc_feh_dex",
        "FeH_STDDEV": "ucc_feh_std",
        "Mass_[Msun]": "ucc_mass_msun",
        "Mass_STDDEV": "ucc_mass_std",
        "C3": "ucc_c3",
        "P_dup": "ucc_pdup",
        "UTI": "ucc_uti",
        "bad_oc": "ucc_bad_oc",
    }
    optional = [column for column in rename if column in clusters.columns]
    cluster_properties = clusters[optional].rename(columns=rename)
    member_associations = members.rename(
        columns={"name": "ucc_cluster", "probs": "ucc_pmem"}
    )[["open_cluster_gaia_id", "ucc_cluster", "ucc_pmem"]]
    associations = member_associations.merge(
        cluster_properties,
        on="ucc_cluster",
        how="left",
        validate="many_to_one",
    )
    associations["ucc_bad_oc"] = associations["ucc_bad_oc"].map(_boolean).astype("boolean")
    release = _ucc_release(root, readme_path)
    associations["ucc_catalog_release"] = release
    associations["ucc_catalog_doi"] = UCC_CATALOG_DOI

    manifest = {
        "catalog": "UCC",
        "release": release,
        "doi": UCC_CATALOG_DOI,
        "cluster_rows": int(len(clusters)),
        "member_rows": int(len(associations)),
        "files": {
            "clusters": _file_record(catalog_path),
            "members": _file_record(members_path),
        },
    }
    if readme_path is not None:
        manifest["files"]["readme"] = _file_record(readme_path)
    return associations, clusters, manifest


_HR24_CLUSTER_COLSPECS = (
    (0, 20),
    (21, 25),
    (280, 281),
    (282, 293),
    (294, 300),
    (319, 331),
    (332, 344),
    (370, 381),
    (418, 431),
    (474, 487),
    (488, 499),
    (500, 510),
    (511, 523),
    (524, 535),
    (536, 546),
    (547, 559),
    (560, 571),
    (572, 582),
    (599, 614),
    (758, 768),
    (795, 806),
    (807, 818),
    (819, 831),
    (991, 1001),
    (1007, 1022),
    (1039, 1054),
)
_HR24_CLUSTER_NAMES = (
    "Name",
    "ID",
    "Type",
    "CST",
    "N",
    "RAdeg",
    "DEdeg",
    "r50",
    "r50pc",
    "pmRA",
    "s_pmRA",
    "e_pmRA",
    "pmDE",
    "s_pmDE",
    "e_pmDE",
    "Plx",
    "s_Plx",
    "e_Plx",
    "dist50",
    "CMDCl50",
    "logAge16",
    "logAge50",
    "logAge84",
    "probJ",
    "MassJ",
    "MassTot",
)
_HR24_MEMBER_COLSPECS = ((8, 28), (29, 33), (34, 53), (54, 55), (56, 57), (58, 78))
_HR24_MEMBER_NAMES = ("Name", "ID", "GaiaDR3", "inrj", "inrt", "Prob")


def _read_hr24_table(path: Path, *, kind: str) -> pd.DataFrame:
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".parquet"):
        return pd.read_parquet(path)
    if suffixes.endswith(".csv") or suffixes.endswith(".csv.gz"):
        return pd.read_csv(path, low_memory=False)
    if kind == "clusters":
        return pd.read_fwf(
            path,
            colspecs=list(_HR24_CLUSTER_COLSPECS),
            names=list(_HR24_CLUSTER_NAMES),
            compression="infer",
        )
    return pd.read_fwf(
        path,
        colspecs=list(_HR24_MEMBER_COLSPECS),
        names=list(_HR24_MEMBER_NAMES),
        compression="infer",
    )


def _rename_first(frame: pd.DataFrame, target: str, aliases: Iterable[str]) -> pd.DataFrame:
    if target in frame.columns:
        return frame
    source = next((alias for alias in aliases if alias in frame.columns), None)
    return frame.rename(columns={source: target}) if source is not None else frame


def load_hr24_tables(
    directory: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Load the Hunt & Reffert (2024) cluster and member tables."""
    root = Path(directory).expanduser()
    clusters_path = _find_one(
        root,
        ("clusters.parquet", "clusters.csv", "clusters.csv.gz", "clusters.dat", "clusters.dat.gz"),
    )
    members_path = _find_one(
        root,
        ("members.parquet", "members.csv", "members.csv.gz", "members.dat", "members.dat.gz"),
    )
    clusters = _read_hr24_table(clusters_path, kind="clusters")
    members = _read_hr24_table(members_path, kind="members")
    clusters = _rename_first(clusters, "Name", ("name", "cluster", "Cluster"))
    clusters = _rename_first(clusters, "ID", ("id", "cluster_id"))
    members = _rename_first(members, "Name", ("name", "cluster", "Cluster"))
    members = _rename_first(members, "ID", ("id", "cluster_id"))
    members = _rename_first(members, "GaiaDR3", ("source_id", "Source", "gaia_source_id"))
    members = _rename_first(members, "Prob", ("probs", "prob", "membership_probability"))
    required_cluster = {"Name", "ID", "Type", "CST", "CMDCl50"}
    required_member = {"Name", "ID", "GaiaDR3", "Prob"}
    missing_cluster = sorted(required_cluster - set(clusters.columns))
    missing_member = sorted(required_member - set(members.columns))
    if missing_cluster or missing_member:
        raise ValueError(
            "Hunt & Reffert tables are missing required columns: "
            f"clusters={missing_cluster}, members={missing_member}"
        )
    if clusters["ID"].duplicated().any():
        examples = clusters.loc[clusters["ID"].duplicated(False), "ID"].head().tolist()
        raise ValueError(f"Hunt & Reffert cluster IDs are not unique: {examples}")

    members = members.copy()
    members["open_cluster_gaia_id"] = normalize_gaia_source_id_series(members["GaiaDR3"])
    members["Prob"] = pd.to_numeric(members["Prob"], errors="coerce")
    members = members[members["open_cluster_gaia_id"].notna()].copy()

    rename = {
        "Name": "hr24_cluster",
        "Prob": "hr24_pmem",
        "Type": "hr24_cluster_type",
        "CST": "hr24_cst",
        "CMDCl50": "hr24_cmd_class_median",
        "N": "hr24_n_members",
        "r50": "hr24_r50_deg",
        "r50pc": "hr24_r50_pc",
        "RAdeg": "hr24_cluster_ra_deg",
        "DEdeg": "hr24_cluster_dec_deg",
        "Plx": "hr24_cluster_parallax_mas",
        "pmRA": "hr24_cluster_pmra_masyr",
        "pmDE": "hr24_cluster_pmdec_masyr",
        "dist50": "hr24_distance_pc",
        "logAge50": "hr24_log_age_yr",
        "probJ": "hr24_prob_jacobi",
        "MassJ": "hr24_mass_jacobi_msun",
        "MassTot": "hr24_mass_total_msun",
        "inrj": "hr24_inside_jacobi_radius",
        "inrt": "hr24_inside_tidal_radius",
    }
    cluster_columns = [column for column in rename if column in clusters.columns and column != "Name"]
    cluster_properties = clusters[["ID", "Name", *cluster_columns]].rename(columns=rename)
    member_columns = [column for column in ("ID", "Name", "Prob", "inrj", "inrt") if column in members]
    member_rename = {
        key: value
        for key, value in rename.items()
        if key not in {"Name"}
    }
    member_rename["Name"] = "hr24_member_cluster"
    member_associations = members[["open_cluster_gaia_id", *member_columns]].rename(
        columns=member_rename
    )
    associations = member_associations.merge(
        cluster_properties,
        on="ID",
        how="left",
        validate="many_to_one",
    )
    if "hr24_cluster" not in associations:
        associations["hr24_cluster"] = associations["hr24_member_cluster"]
    else:
        associations["hr24_cluster"] = associations["hr24_cluster"].fillna(
            associations["hr24_member_cluster"]
        )
    associations = associations.drop(columns=["ID", "hr24_member_cluster"], errors="ignore")
    for column in ("hr24_inside_jacobi_radius", "hr24_inside_tidal_radius"):
        if column in associations:
            associations[column] = associations[column].map(_boolean).astype("boolean")
    associations["hr24_catalog_id"] = HR24_CATALOG_ID

    manifest = {
        "catalog": "Hunt & Reffert 2024",
        "catalog_id": HR24_CATALOG_ID,
        "cluster_rows": int(len(clusters)),
        "member_rows": int(len(associations)),
        "files": {
            "clusters": _file_record(clusters_path),
            "members": _file_record(members_path),
        },
    }
    return associations, clusters, manifest


def _identity_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in ("candidate_id", "asas_sn_id", "timescale", "open_cluster_gaia_id")
        if column in frame.columns
    ]


def _match_ucc(
    sources: pd.DataFrame,
    associations: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    keyed = sources[["_open_cluster_input_row", *_identity_columns(sources)]].copy()
    joined = keyed.merge(
        associations,
        on="open_cluster_gaia_id",
        how="left",
        validate="many_to_many",
    )
    joined["ucc_listed_member"] = joined["ucc_cluster"].notna()
    joined["ucc_p50_member"] = joined["ucc_listed_member"] & joined["ucc_pmem"].ge(
        UCC_MEMBER_THRESHOLD
    )
    joined["ucc_good_cluster"] = joined["ucc_bad_oc"].eq(False).fillna(False)
    joined["ucc_good_member"] = joined["ucc_p50_member"] & joined["ucc_good_cluster"]
    counts = (
        joined.loc[joined["ucc_listed_member"]]
        .groupby("_open_cluster_input_row", sort=False)
        .size()
    )
    joined["ucc_n_matches"] = joined["_open_cluster_input_row"].map(counts).fillna(0).astype(int)
    joined["ucc_match_status"] = np.select(
        [
            joined["open_cluster_gaia_id"].isna(),
            joined["ucc_good_member"],
            joined["ucc_p50_member"],
            joined["ucc_listed_member"],
        ],
        ["missing_gaia_id", "good_member", "p50_bad_cluster", "listed_below_threshold"],
        default="no_match",
    )
    sorted_joined = joined.sort_values(
        [
            "_open_cluster_input_row",
            "ucc_good_member",
            "ucc_p50_member",
            "ucc_good_cluster",
            "ucc_pmem",
            "ucc_uti",
            "ucc_pdup",
            "ucc_cluster",
        ],
        ascending=[True, False, False, False, False, False, True, True],
        kind="stable",
        na_position="last",
    )
    preferred = sorted_joined.drop_duplicates("_open_cluster_input_row", keep="first")
    all_matches = joined.loc[joined["ucc_listed_member"]].copy()
    all_matches.insert(0, "cluster_catalog", "UCC")
    return preferred, all_matches


def _match_hr24(
    sources: pd.DataFrame,
    associations: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    keyed = sources[["_open_cluster_input_row", *_identity_columns(sources)]].copy()
    joined = keyed.merge(
        associations,
        on="open_cluster_gaia_id",
        how="left",
        validate="many_to_many",
    )
    joined["hr24_listed_member"] = joined["hr24_cluster"].notna()
    joined["hr24_p50_member"] = joined["hr24_listed_member"] & joined["hr24_pmem"].ge(
        HR24_MEMBER_THRESHOLD
    )
    cluster_type = joined["hr24_cluster_type"].fillna("").astype(str).str.strip().str.lower()
    joined["hr24_bound_member"] = joined["hr24_p50_member"] & cluster_type.eq("o")
    joined["hr24_high_quality_member"] = (
        joined["hr24_bound_member"]
        & joined["hr24_cst"].gt(HR24_HIGH_QUALITY_CST_MIN)
        & joined["hr24_cmd_class_median"].gt(HR24_HIGH_QUALITY_CMD_MIN)
    )
    counts = (
        joined.loc[joined["hr24_listed_member"]]
        .groupby("_open_cluster_input_row", sort=False)
        .size()
    )
    joined["hr24_n_matches"] = joined["_open_cluster_input_row"].map(counts).fillna(0).astype(int)
    joined["hr24_match_status"] = np.select(
        [
            joined["open_cluster_gaia_id"].isna(),
            joined["hr24_high_quality_member"],
            joined["hr24_bound_member"],
            joined["hr24_p50_member"],
            joined["hr24_listed_member"],
        ],
        [
            "missing_gaia_id",
            "high_quality_bound_member",
            "bound_member",
            "p50_non_open_cluster",
            "listed_below_threshold",
        ],
        default="no_match",
    )
    sorted_joined = joined.sort_values(
        [
            "_open_cluster_input_row",
            "hr24_high_quality_member",
            "hr24_bound_member",
            "hr24_p50_member",
            "hr24_pmem",
            "hr24_cst",
            "hr24_cmd_class_median",
            "hr24_cluster",
        ],
        ascending=[True, False, False, False, False, False, False, True],
        kind="stable",
        na_position="last",
    )
    preferred = sorted_joined.drop_duplicates("_open_cluster_input_row", keep="first")
    all_matches = joined.loc[joined["hr24_listed_member"]].copy()
    all_matches.insert(0, "cluster_catalog", "Hunt & Reffert 2024")
    return preferred, all_matches


def _assign_columns(
    sources: pd.DataFrame,
    preferred: pd.DataFrame,
    columns: Iterable[str],
) -> pd.DataFrame:
    output = sources.copy()
    lookup = preferred.set_index("_open_cluster_input_row")
    input_rows = output["_open_cluster_input_row"]
    for column in columns:
        if column in lookup.columns:
            output[column] = input_rows.map(lookup[column])
    return output


def add_ucc_proximity(sources: pd.DataFrame, raw_clusters: pd.DataFrame) -> pd.DataFrame:
    """Attach nearest reliable UCC-centre diagnostics without implying membership."""
    output = sources.copy()
    for column in UCC_PROXIMITY_COLUMNS:
        if column not in output:
            output[column] = pd.NA
    required = {"name", "RA_ICRS", "DE_ICRS", "bad_oc"}
    if not required.issubset(raw_clusters.columns) or not {"ra", "dec"}.issubset(output.columns):
        return output

    clusters = raw_clusters.copy()
    clusters["_bad"] = clusters["bad_oc"].map(_boolean).astype("boolean")
    cluster_ra = pd.to_numeric(clusters["RA_ICRS"], errors="coerce")
    cluster_dec = pd.to_numeric(clusters["DE_ICRS"], errors="coerce")
    good_clusters = clusters.loc[
        clusters["_bad"].eq(False).fillna(False)
        & cluster_ra.notna()
        & cluster_dec.notna()
    ].copy()
    source_ra = _numeric(output, "ra")
    source_dec = _numeric(output, "dec")
    valid = source_ra.notna() & source_dec.notna()
    if good_clusters.empty or not valid.any():
        return output

    cluster_coords = SkyCoord(
        ra=pd.to_numeric(good_clusters["RA_ICRS"], errors="coerce").to_numpy() * u.deg,
        dec=pd.to_numeric(good_clusters["DE_ICRS"], errors="coerce").to_numpy() * u.deg,
    )
    source_coords = SkyCoord(
        ra=source_ra.loc[valid].to_numpy() * u.deg,
        dec=source_dec.loc[valid].to_numpy() * u.deg,
    )
    nearest_index, separation, _ = source_coords.match_to_catalog_sky(cluster_coords)
    nearest = good_clusters.iloc[nearest_index].reset_index(drop=True)
    valid_index = output.index[valid]

    assignments = {
        "ucc_nearest_cluster": nearest["name"].to_numpy(),
        "ucc_nearest_sep_arcmin": separation.arcmin,
        "ucc_nearest_r50_arcmin": pd.to_numeric(nearest.get("r_50"), errors="coerce").to_numpy(),
        "ucc_nearest_age_myr": pd.to_numeric(nearest.get("Age_[Myr]"), errors="coerce").to_numpy(),
        "ucc_nearest_distance_kpc": pd.to_numeric(nearest.get("Dist_[kpc]"), errors="coerce").to_numpy(),
        "ucc_nearest_uti": pd.to_numeric(nearest.get("UTI"), errors="coerce").to_numpy(),
    }
    for column, values in assignments.items():
        output.loc[valid_index, column] = values
    r50 = pd.to_numeric(output.loc[valid_index, "ucc_nearest_r50_arcmin"], errors="coerce")
    sep = pd.to_numeric(output.loc[valid_index, "ucc_nearest_sep_arcmin"], errors="coerce")
    output.loc[valid_index, "ucc_nearest_sep_r50"] = np.where(r50 > 0, sep / r50, np.nan)

    source_cluster_columns = (
        ("parallax", "Plx", "ucc_nearest_dparallax_mas"),
        ("pmra", "pmRA", "ucc_nearest_dpmra_masyr"),
        ("pmdec", "pmDE", "ucc_nearest_dpmdec_masyr"),
    )
    for source_column, cluster_column, output_column in source_cluster_columns:
        if source_column not in output or cluster_column not in nearest:
            continue
        source_values = pd.to_numeric(output.loc[valid_index, source_column], errors="coerce").to_numpy()
        cluster_values = pd.to_numeric(nearest[cluster_column], errors="coerce").to_numpy()
        output.loc[valid_index, output_column] = source_values - cluster_values
    return output


def _legacy_aliases(sources: pd.DataFrame) -> pd.DataFrame:
    output = sources.copy()
    output["cluster_name"] = output.get("ucc_cluster", pd.Series(pd.NA, index=output.index))
    output["cluster_membership_prob"] = output.get("ucc_pmem", pd.Series(np.nan, index=output.index))
    output["cluster_age_myr"] = output.get("ucc_age_myr", pd.Series(np.nan, index=output.index))
    distance_kpc = pd.to_numeric(
        output.get("ucc_distance_kpc", pd.Series(np.nan, index=output.index)), errors="coerce"
    )
    output["cluster_dist_pc"] = distance_kpc * 1000.0
    output["cluster_catalog"] = np.where(
        output.get("ucc_listed_member", pd.Series(False, index=output.index)).fillna(False),
        "UCC",
        pd.NA,
    )
    return output


def add_open_cluster_context(
    frame: pd.DataFrame,
    *,
    ucc_dir: str | Path,
    hr24_dir: str | Path | None = None,
    include_proximity: bool = True,
) -> OpenClusterMatchResult:
    """Add exact-ID UCC/HR24 membership and separate UCC proximity fields."""
    expanded = expand_feature_layers(frame)
    sources = expanded.copy().reset_index(drop=True)
    sources["_open_cluster_input_row"] = np.arange(len(sources), dtype=np.int64)
    sources["open_cluster_gaia_id"] = gaia_identifier_series(sources)
    sources["open_cluster_match_version"] = OPEN_CLUSTER_MATCH_VERSION

    ucc_associations, ucc_clusters, ucc_manifest = load_ucc_tables(ucc_dir)
    ucc_preferred, ucc_all = _match_ucc(sources, ucc_associations)
    sources = _assign_columns(sources, ucc_preferred, UCC_OUTPUT_COLUMNS)
    if include_proximity:
        sources = add_ucc_proximity(sources, ucc_clusters)

    manifests: list[dict[str, object]] = [ucc_manifest]
    all_matches = [ucc_all]
    if hr24_dir is not None:
        hr_associations, _hr_clusters, hr_manifest = load_hr24_tables(hr24_dir)
        hr_preferred, hr_all = _match_hr24(sources, hr_associations)
        sources = _assign_columns(sources, hr_preferred, HR24_OUTPUT_COLUMNS)
        manifests.append(hr_manifest)
        all_matches.append(hr_all)
    else:
        sources["hr24_match_status"] = np.where(
            sources["open_cluster_gaia_id"].isna(), "missing_gaia_id", "not_run"
        )
        sources["hr24_catalog_id"] = HR24_CATALOG_ID
        sources["hr24_n_matches"] = 0
        for column in (
            "hr24_listed_member",
            "hr24_p50_member",
            "hr24_bound_member",
            "hr24_high_quality_member",
        ):
            sources[column] = False

    sources = _legacy_aliases(sources)
    combined_matches = pd.concat(all_matches, ignore_index=True, sort=False)
    sources = sources.drop(columns=["_open_cluster_input_row"])
    combined_matches = combined_matches.drop(columns=["_open_cluster_input_row"], errors="ignore")
    manifest = {
        "match_version": OPEN_CLUSTER_MATCH_VERSION,
        "membership_definitions": {
            "ucc_listed_member": "Gaia ID appears in UCC_members",
            "ucc_p50_member": f"UCC membership probability >= {UCC_MEMBER_THRESHOLD}",
            "ucc_good_member": "ucc_p50_member and UCC bad_oc is false",
            "hr24_p50_member": f"Hunt-Reffert membership probability >= {HR24_MEMBER_THRESHOLD}",
            "hr24_bound_member": "hr24_p50_member and cluster Type is o",
            "hr24_high_quality_member": (
                "hr24_bound_member, CST > 5, and median CMD class > 0.5"
            ),
            "ucc_nearest": "nearest reliable cluster centre only; not membership",
        },
        "input_rows": int(len(frame)),
        "output_rows": int(len(sources)),
        "all_match_rows": int(len(combined_matches)),
        "catalogs": manifests,
    }
    return OpenClusterMatchResult(
        sources=sources,
        all_matches=combined_matches,
        manifest=json.loads(json.dumps(manifest, default=str)),
    )


__all__ = [
    "HR24_OUTPUT_COLUMNS",
    "LEGACY_CLUSTER_COLUMNS",
    "OPEN_CLUSTER_MATCH_VERSION",
    "OPEN_CLUSTER_OUTPUT_COLUMNS",
    "OpenClusterMatchResult",
    "UCC_OUTPUT_COLUMNS",
    "UCC_PROXIMITY_COLUMNS",
    "add_open_cluster_context",
    "add_ucc_proximity",
    "gaia_identifier_series",
    "load_hr24_tables",
    "load_ucc_tables",
]
