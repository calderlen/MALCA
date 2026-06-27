"""External survey light-curve discovery, loading, and trace assembly for review plots."""

from __future__ import annotations

from collections import OrderedDict
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from malca.config import (
    GAIA_TCB_EPOCH_JD,
    JD_OFFSET,
    KEPLER_BKJD_OFFSET,
    MJD_TO_JD,
    REVIEW_CACHE_LIMIT,
    REVIEW_MAX_EXTERNAL_POINTS,
    TESS_BTJD_OFFSET,
)
from malca.external_lc_manifest import (
    clear_external_lc_manifest_caches,
    index_external_lc_paths_from_manifest,
    normalize_external_lc_file_prefix,
)

_MAX_EXTERNAL_TRACE_POINTS = REVIEW_MAX_EXTERNAL_POINTS

EXTERNAL_LC_SPECS: dict[str, dict] = {
    "atlas": {
        "time_col": "mjd",
        "time_offset": MJD_TO_JD,
        "jd_system": "mjd",
        "bands": {
            "c": {"color": "#00ccff", "marker": "diamond", "label": "ATLAS c"},
            "o": {"color": "#ff8c42", "marker": "diamond", "label": "ATLAS o"},
        },
        "filter_col": "filter",
        "mag_col": "mag",
        "err_col": "mag_err",
    },
    "ztf": {
        "time_col": "mjd",
        "jd_system": "mjd",
        "bands": {
            "zg": {"color": "#44aa44", "marker": "triangle-up", "label": "ZTF g"},
            "zr": {"color": "#dd4444", "marker": "triangle-up", "label": "ZTF r"},
            "zi": {"color": "#8844cc", "marker": "triangle-up", "label": "ZTF i"},
        },
        "filter_col": "band",
        "mag_col": "mag",
        "err_col": "mag_err",
    },
    "gaia_epoch": {
        "time_col": "time",
        "jd_system": "bjd_gaia",  # Gaia TCB in days since J2010.0 (JD 2455197.5)
        "bands": {
            "G": {"color": "#e8c547", "marker": "star", "label": "Gaia G"},
        },
        "filter_col": "band",
        "mag_col": "mag",
        "err_col": "mag_err",
    },
    "tess": {
        "time_col": "time",
        "jd_system": "btjd",  # BTJD = BJD - 2457000.0
        "is_flux": True,
        "bands": {
            "TESS": {"color": "#cc66ff", "marker": "hexagon", "label": "TESS"},
        },
        "filter_col": None,
        "mag_col": "flux",
        "err_col": "flux_err",
    },
    "neowise": {
        "time_col": "mjd",
        "jd_system": "mjd",
        "bands": {
            "W1": {"color": "#4fa3ff", "marker": "x", "label": "NEOWISE W1", "mag_col": "w1mpro", "err_col": "w1sigmpro"},
            "W2": {"color": "#ff8c42", "marker": "x", "label": "NEOWISE W2", "mag_col": "w2mpro", "err_col": "w2sigmpro"},
        },
        "filter_col": None,
        "mag_col": "w1mpro",
        "err_col": "w1sigmpro",
    },
    "neowise_w1": {
        "time_col": "mjd",
        "jd_system": "mjd",
        "bands": {
            "W1": {"color": "#4fa3ff", "marker": "x", "label": "NEOWISE W1", "mag_col": "w1mpro", "err_col": "w1sigmpro"},
        },
        "filter_col": None,
        "mag_col": "w1mpro",
        "err_col": "w1sigmpro",
    },
    "neowise_w2": {
        "time_col": "mjd",
        "jd_system": "mjd",
        "bands": {
            "W2": {"color": "#ff8c42", "marker": "x", "label": "NEOWISE W2", "mag_col": "w2mpro", "err_col": "w2sigmpro"},
        },
        "filter_col": None,
        "mag_col": "w2mpro",
        "err_col": "w2sigmpro",
    },
    "neowise_color": {
        "time_col": "mjd",
        "jd_system": "mjd",
        "bands": {
            "W1-W2": {"color": "#a832a8", "marker": "x", "label": "NEOWISE W1-W2", "mag_col": "color_w1_w2", "err_col": "color_w1_w2_err"},
        },
        "filter_col": None,
        "mag_col": "color_w1_w2",
        "err_col": "color_w1_w2_err",
    },
    "allwise_mep": {
        "time_col": "mjd",
        "jd_system": "mjd",
        "bands": {
            "W1": {"color": "#4fa3ff", "marker": "cross", "label": "AllWISE W1", "mag_col": "w1mpro", "err_col": "w1sigmpro"},
            "W2": {"color": "#ff8c42", "marker": "cross", "label": "AllWISE W2", "mag_col": "w2mpro", "err_col": "w2sigmpro"},
            "W3": {"color": "#d89cff", "marker": "cross", "label": "AllWISE W3", "mag_col": "w3mpro", "err_col": "w3sigmpro"},
            "W4": {"color": "#ffcc66", "marker": "cross", "label": "AllWISE W4", "mag_col": "w4mpro", "err_col": "w4sigmpro"},
        },
        "filter_col": None,
        "mag_col": "w1mpro",
        "err_col": "w1sigmpro",
    },
    "kepler": {
        "time_col": "time",
        "jd_system": "bkjd",  # BKJD = BJD - 2454833.0
        "is_flux": True,
        "bands": {
            "Kepler": {"color": "#ffb6c1", "marker": "hexagon", "label": "Kepler/K2"},
        },
        "filter_col": None,
        "mag_col": "flux",
        "err_col": "flux_err",
    },
    "aavso": {
        "time_col": "mjd",
        "jd_system": "mjd",
        "bands": {
            "V": {"color": "#00ff00", "marker": "circle", "label": "AAVSO V"},
            "B": {"color": "#0000ff", "marker": "circle", "label": "AAVSO B"},
            "CV": {"color": "#aaaaaa", "marker": "circle", "label": "AAVSO CV"},
        },
        "filter_col": "filter",
        "mag_col": "mag",
        "err_col": "mag_err",
    },
    "ogle": {
        "time_col": "mjd",
        "jd_system": "mjd",
        "bands": {
            "I": {"color": "#cc3344", "marker": "diamond-open", "label": "OGLE I"},
            "V": {"color": "#44aa44", "marker": "diamond-open", "label": "OGLE V"},
        },
        "filter_col": "band",
        "mag_col": "mag",
        "err_col": "mag_err",
    },
    "stripe82": {
        "time_col": "mjd",
        "jd_system": "mjd",
        "bands": {
            "u": {"color": "#6c7cff", "marker": "square-open", "label": "Stripe 82 u"},
            "g": {"color": "#44aa44", "marker": "square-open", "label": "Stripe 82 g"},
            "r": {"color": "#dd4444", "marker": "square-open", "label": "Stripe 82 r"},
            "i": {"color": "#8844cc", "marker": "square-open", "label": "Stripe 82 i"},
            "z": {"color": "#ccaa44", "marker": "square-open", "label": "Stripe 82 z"},
        },
        "filter_col": "band",
        "mag_col": "mag",
        "err_col": "mag_err",
    },
    "vvvx_virac": {
        "time_col": "mjd",
        "jd_system": "mjd",
        "bands": {
            "z": {"color": "#5f7fff", "marker": "triangle-down-open", "label": "VVVX Z"},
            "y": {"color": "#57a773", "marker": "triangle-down-open", "label": "VVVX Y"},
            "j": {"color": "#e0b448", "marker": "triangle-down-open", "label": "VVVX J"},
            "h": {"color": "#df6f53", "marker": "triangle-down-open", "label": "VVVX H"},
            "ks": {"color": "#c45fd8", "marker": "triangle-down-open", "label": "VVVX Ks"},
        },
        "filter_col": "band",
        "mag_col": "mag",
        "err_col": "mag_err",
    },
    "ps1": {
        "time_col": "mjd",
        "jd_system": "mjd",
        "bands": {
            "g_ps": {"color": "#44aa44", "marker": "star", "label": "PS1 g"},
            "r_ps": {"color": "#dd4444", "marker": "star", "label": "PS1 r"},
            "i_ps": {"color": "#8844cc", "marker": "star", "label": "PS1 i"},
            "z_ps": {"color": "#ccaa44", "marker": "star", "label": "PS1 z"},
            "y_ps": {"color": "#aaaa33", "marker": "star", "label": "PS1 y"},
        },
        "filter_col": "filter",
        "mag_col": "mag",
        "err_col": "mag_err",
    },
    "crts": {
        "time_col": "mjd",
        "jd_system": "mjd",
        "bands": {
            "CV": {"color": "#bbbbbb", "marker": "square", "label": "CRTS CV"},
        },
        "filter_col": None,
        "mag_col": "mag",
        "err_col": "mag_err",
    },
}

EXTERNAL_SOURCE_ORDER = (
    "asassn", "atlas", "ztf", "gaia_epoch", "tess", "neowise", "neowise_w1", "neowise_w2", "neowise_color",
    "kepler", "aavso", "ogle", "stripe82", "allwise_mep", "vvvx_virac",
    "ps1", "crts",
)
EXTERNAL_SOURCE_LABELS = {
    "asassn": "ASAS-SN",
    "atlas": "ATLAS",
    "ztf": "ZTF",
    "gaia_epoch": "Gaia Epoch",
    "tess": "TESS",
    "neowise": "NEOWISE W1/W2",
    "neowise_w1": "NEOWISE W1",
    "neowise_w2": "NEOWISE W2",
    "neowise_color": "NEOWISE W1-W2",
    "kepler": "Kepler/K2",
    "aavso": "AAVSO",
    "ogle": "OGLE I/V",
    "stripe82": "SDSS Stripe 82",
    "allwise_mep": "AllWISE MEP",
    "vvvx_virac": "VVVX/VIRAC2",
    "ps1": "PS1",
    "crts": "CRTS",
}
EXTERNAL_SOURCE_VALUES = set(EXTERNAL_SOURCE_ORDER)

_EXTERNAL_LC_CACHE: OrderedDict[tuple, pd.DataFrame] = OrderedDict()


def _cache_get(cache: OrderedDict, key):
    value = cache.get(key)
    if value is None:
        return None
    cache.move_to_end(key)
    return value


def _cache_put(cache: OrderedDict, key, value) -> None:
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > REVIEW_CACHE_LIMIT:
        cache.popitem(last=False)


def coerce_external_source_values(raw_value: object) -> list[str]:
    """Normalize old single-select source values and new multi-select lists."""
    if raw_value is None:
        values = ["asassn"]
    elif isinstance(raw_value, str):
        values = [raw_value]
    elif isinstance(raw_value, (list, tuple, set, np.ndarray, pd.Series)):
        values = list(raw_value)
    else:
        values = [raw_value]

    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip().lower()
        if not text:
            continue
        if text == "all":
            return list(EXTERNAL_SOURCE_ORDER)
        
        texts = [text]
        if text in {"wise", "neowise", "wise_w1_w2"}:
            texts = ["neowise"]
        elif text in {"w1", "wise_w1"}:
            texts = ["neowise_w1"]
        elif text in {"w2", "wise_w2"}:
            texts = ["neowise_w2"]
        elif text in {"wise_color", "neowise_color", "w1_w2"}:
            texts = ["neowise_color"]
        elif text in {"k2", "kepler_k2"}:
            texts = ["kepler"]
        elif text in {"sdss_s82", "s82", "stripe_82", "sdss_stripe82"}:
            texts = ["stripe82"]
        elif text in {"allwise", "allwise_multiepoch", "wise_mep"}:
            texts = ["allwise_mep"]
        elif text in {"vvv", "vvvx", "virac", "virac2", "vvvx_virac2"}:
            texts = ["vvvx_virac"]
            
        for t in texts:
            if t not in EXTERNAL_SOURCE_VALUES or t in seen:
                continue
            out.append(t)
            seen.add(t)

    if isinstance(raw_value, str) and out and out[0] != "asassn":
        out.insert(0, "asassn")
    return out


def external_source_label(source_name: object) -> str:
    source = str(source_name or "").strip().lower()
    return EXTERNAL_SOURCE_LABELS.get(source, source.upper() if source else "External")


_external_source_label = external_source_label


def candidate_lookup_keys(candidate_id: str, payload: dict) -> list[str]:
    keys = [str(candidate_id)]
    for key in ("candidate_id", "asas_sn_id"):
        value = payload.get(key)
        if value is not None:
            keys.append(str(value))
    path_value = payload.get("path")
    if path_value:
        keys.append(Path(str(path_value)).stem)
    seen: set[str] = set()
    return [key for key in keys if key and not (key in seen or seen.add(key))]


def resolve_run_dir_from_plot_dir(plot_dir: str | Path | None) -> Path | None:
    """Infer run directory from plot-dir or run-dir style path."""
    if not plot_dir:
        return None
    cached = _resolve_run_dir_from_plot_dir_cached(str(plot_dir))
    return Path(cached) if cached else None


@lru_cache(maxsize=64)
def _resolve_run_dir_from_plot_dir_cached(plot_dir_text: str) -> str | None:
    if not plot_dir_text:
        return None
    path = Path(str(plot_dir_text)).expanduser()
    try:
        if path.exists():
            path = path.resolve()
    except Exception:
        pass
    if path.name == "plots":
        return str(path.parent)
    if (path / "plots").is_dir():
        return str(path)
    if (path / "results").is_dir():
        return str(path)
    if (path / "bundle_assets" / "lightcurves").is_dir():
        return str(path)
    if (path.parent / "results").is_dir():
        return str(path.parent)
    if (path.parent / "plots").is_dir():
        return str(path.parent)
    if (path.parent / "bundle_assets" / "lightcurves").is_dir():
        return str(path.parent)
    return None


@lru_cache(maxsize=64)
def index_external_lc_paths_from_root(root_text: str, prefix: str) -> dict[str, str]:
    """Index candidate -> external LC parquet paths for a results root."""
    return index_external_lc_paths_from_manifest(str(Path(root_text).expanduser()), prefix)


def clear_external_lc_discovery_caches() -> None:
    """Clear review-side external LC discovery caches."""
    index_external_lc_paths_from_root.cache_clear()
    clear_external_lc_manifest_caches()


def discover_external_lcs(
    candidate_id: str,
    payload: dict,
    plot_dir: Path | str | None,
    requested_sources: list[str],
    *,
    default_results_root: Path | None = None,
) -> dict[str, Path]:
    """Locate external LC parquet files for the requested survey sources."""
    lookup_keys = candidate_lookup_keys(candidate_id, payload)
    run_dir = resolve_run_dir_from_plot_dir(plot_dir)
    search_roots: list[Path] = []
    if run_dir is not None:
        search_roots.append(run_dir / "results")
    if default_results_root is not None:
        default_root = Path(default_results_root)
        if default_root not in search_roots:
            search_roots.append(default_root)

    found: dict[str, Path] = {}
    file_prefix_map: dict[str, Path] = {}
    for source_name in requested_sources:
        prefix = str(source_name).strip().lower()
        if prefix == "asassn" or prefix not in EXTERNAL_SOURCE_VALUES:
            continue
        if prefix in found:
            continue
            
        file_prefix = normalize_external_lc_file_prefix(prefix)

        if file_prefix in file_prefix_map:
            found[prefix] = file_prefix_map[file_prefix]
            continue
            
        for root in search_roots:
            index_map = index_external_lc_paths_from_root(str(root.resolve()), file_prefix)
            for key in lookup_keys:
                path_text = index_map.get(str(key))
                if path_text:
                    found[prefix] = Path(path_text)
                    file_prefix_map[file_prefix] = found[prefix]
                    break
            if prefix in found:
                break
    return found


def _rename_first_present(df: pd.DataFrame, canonical: str, aliases: tuple[str, ...]) -> pd.DataFrame:
    if canonical in df.columns:
        return df
    for alias in aliases:
        if alias in df.columns:
            return df.rename(columns={alias: canonical})
    return df


def _coerce_numeric_column(df: pd.DataFrame, column: str) -> None:
    if column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")


def _normalize_mjd_column(df: pd.DataFrame, column: str = "mjd") -> None:
    """Normalize a time column to true MJD, tolerating JD-valued inputs."""
    if column not in df.columns:
        return
    df[column] = pd.to_numeric(df[column], errors="coerce")
    finite = df[column].to_numpy()
    finite = finite[np.isfinite(finite)]
    if finite.size and float(np.nanmedian(finite)) > 1_000_000.0:
        df[column] = df[column] - MJD_TO_JD


def normalize_external_lc_dataframe(source_name: str, df_ext: pd.DataFrame) -> pd.DataFrame:
    """Normalize heterogeneous external-LC schemas to the viewer's expected columns."""
    if df_ext is None or df_ext.empty:
        return df_ext

    source = str(source_name or "").strip().lower()
    df = df_ext.copy()

    if source == "atlas":
        df = _rename_first_present(df, "mjd", ("MJD", "mjd", "JD"))
        df = _rename_first_present(df, "filter", ("F", "filter"))
        df = _rename_first_present(df, "mag", ("m", "mag"))
        df = _rename_first_present(df, "mag_err", ("dm", "mag_err", "magerr"))
        _normalize_mjd_column(df)
        if "filter" in df.columns:
            df["filter"] = df["filter"].astype(str).str.strip().str.lower()
    elif source == "ztf":
        df = _rename_first_present(df, "mjd", ("mjd", "hjd"))
        df = _rename_first_present(df, "band", ("band", "filtercode"))
        df = _rename_first_present(df, "mag", ("mag",))
        df = _rename_first_present(df, "mag_err", ("mag_err", "magerr"))
        _normalize_mjd_column(df)
        if "band" in df.columns:
            band_map = {
                "1": "zg", "1.0": "zg",
                "2": "zr", "2.0": "zr",
                "3": "zi", "3.0": "zi",
                "zg": "zg", "zr": "zr", "zi": "zi",
            }
            df["band"] = (
                df["band"]
                .astype(str)
                .str.strip()
                .str.lower()
                .map(lambda value: band_map.get(value, value))
            )
    elif source == "gaia_epoch":
        df = _rename_first_present(df, "time", ("time", "g_transit_time"))
        df = _rename_first_present(df, "band", ("band",))
        df = _rename_first_present(df, "mag", ("mag", "g_transit_mag"))
        df = _rename_first_present(df, "mag_err", ("mag_err", "mag_error", "g_transit_mag_error"))
        _coerce_numeric_column(df, "time")
        if "band" in df.columns:
            df["band"] = df["band"].astype(str).str.strip().str.upper()
    elif source == "tess":
        df = _rename_first_present(df, "time", ("time",))
        df = _rename_first_present(df, "flux", ("flux",))
        df = _rename_first_present(df, "flux_err", ("flux_err",))
        _coerce_numeric_column(df, "time")
    elif source in ("neowise", "neowise_w1", "neowise_w2", "neowise_color"):
        df = _rename_first_present(df, "mjd", ("mjd", "MJD", "JD"))
        df = _rename_first_present(df, "w1mpro", ("w1mpro", "W1", "w1", "w1_mag"))
        df = _rename_first_present(df, "w1sigmpro", ("w1sigmpro", "w1err", "w1_err", "w1_mag_err"))
        df = _rename_first_present(df, "w2mpro", ("w2mpro", "W2", "w2", "w2_mag"))
        df = _rename_first_present(df, "w2sigmpro", ("w2sigmpro", "w2err", "w2_err", "w2_mag_err"))
        _normalize_mjd_column(df)
        
        if "w1mpro" in df.columns and "w2mpro" in df.columns:
            w1 = pd.to_numeric(df["w1mpro"], errors="coerce")
            w2 = pd.to_numeric(df["w2mpro"], errors="coerce")
            df["color_w1_w2"] = w1 - w2
            if "w1sigmpro" in df.columns and "w2sigmpro" in df.columns:
                w1_err = pd.to_numeric(df["w1sigmpro"], errors="coerce")
                w2_err = pd.to_numeric(df["w2sigmpro"], errors="coerce")
                df["color_w1_w2_err"] = np.sqrt(w1_err**2 + w2_err**2)
    elif source == "allwise_mep":
        df = _rename_first_present(df, "mjd", ("mjd", "MJD", "JD"))
        for band in ("w1", "w2", "w3", "w4"):
            upper = band.upper()
            df = _rename_first_present(df, f"{band}mpro", (f"{band}mpro", f"{band}mpro_ep", upper, band, f"{band}_mag"))
            df = _rename_first_present(
                df,
                f"{band}sigmpro",
                (f"{band}sigmpro", f"{band}sigmpro_ep", f"{band}err", f"{band}_err", f"{band}_mag_err"),
            )
        _normalize_mjd_column(df)
    elif source == "kepler":
        df = _rename_first_present(df, "time", ("time",))
        df = _rename_first_present(df, "flux", ("flux",))
        df = _rename_first_present(df, "flux_err", ("flux_err",))
        _coerce_numeric_column(df, "time")
    elif source == "aavso":
        df = _rename_first_present(df, "mjd", ("mjd", "JD"))
        df = _rename_first_present(df, "filter", ("filter", "Filter"))
        df = _rename_first_present(df, "band", ("band", "Band"))
        df = _rename_first_present(df, "mag", ("mag", "Mag"))
        df = _rename_first_present(df, "mag_err", ("mag_err", "Err"))
        _normalize_mjd_column(df)
        if "filter" not in df.columns and "band" in df.columns:
            df["filter"] = df["band"]
        if "filter" in df.columns:
            df["filter"] = df["filter"].astype(str).str.strip().str.upper()
        if "band" not in df.columns and "filter" in df.columns:
            df["band"] = df["filter"]
    elif source == "ogle":
        df = _rename_first_present(df, "mjd", ("mjd", "MJD", "JD"))
        df = _rename_first_present(df, "band", ("band", "filter", "Filter"))
        df = _rename_first_present(df, "mag", ("mag", "Mag", "magnitude"))
        df = _rename_first_present(df, "mag_err", ("mag_err", "Err", "magerr", "error"))
        _normalize_mjd_column(df)
        if "band" in df.columns:
            df["band"] = df["band"].astype(str).str.strip().str.upper()
    elif source == "stripe82":
        df = _rename_first_present(df, "mjd", ("mjd", "MJD", "JD"))
        df = _rename_first_present(df, "band", ("band", "filter", "Filter"))
        df = _rename_first_present(df, "mag", ("mag", "Mag", "magnitude"))
        df = _rename_first_present(df, "mag_err", ("mag_err", "Err", "magerr", "error"))
        _normalize_mjd_column(df)
        if "band" in df.columns:
            df["band"] = df["band"].astype(str).str.strip().str.lower()
    elif source == "vvvx_virac":
        df = _rename_first_present(df, "mjd", ("mjd", "MJD", "JD", "obs_mjd"))
        df = _rename_first_present(df, "band", ("band", "filter", "Filter"))
        df = _rename_first_present(df, "mag", ("mag", "Mag", "magnitude"))
        df = _rename_first_present(df, "mag_err", ("mag_err", "Err", "magerr", "error"))
        _normalize_mjd_column(df)
        if "band" in df.columns:
            band_map = {"k": "ks", "ks": "ks", "k_s": "ks", "z": "z", "y": "y", "j": "j", "h": "h"}
            df["band"] = (
                df["band"]
                .astype(str)
                .str.strip()
                .str.lower()
                .map(lambda value: band_map.get(value, value))
            )
    elif source == "ps1":
        df = _rename_first_present(df, "mjd", ("mjd", "obsTime"))
        df = _rename_first_present(df, "filter", ("filter", "filterID"))
        df = _rename_first_present(df, "flux_psf", ("flux_psf", "psfFlux"))
        df = _rename_first_present(df, "flux_psf_err", ("flux_psf_err", "psfFluxErr"))
        df = _rename_first_present(df, "mag", ("mag",))
        df = _rename_first_present(df, "mag_err", ("mag_err", "magerr"))
        _normalize_mjd_column(df)
        if "filter" in df.columns:
            filter_map = {
                "1": "g_ps", "1.0": "g_ps",
                "2": "r_ps", "2.0": "r_ps",
                "3": "i_ps", "3.0": "i_ps",
                "4": "z_ps", "4.0": "z_ps",
                "5": "y_ps", "5.0": "y_ps",
                "g": "g_ps", "r": "r_ps", "i": "i_ps", "z": "z_ps", "y": "y_ps",
                "g_ps": "g_ps", "r_ps": "r_ps", "i_ps": "i_ps", "z_ps": "z_ps", "y_ps": "y_ps",
            }
            df["filter"] = (
                df["filter"]
                .astype(str)
                .str.strip()
                .str.lower()
                .map(lambda value: filter_map.get(value, value))
            )
        if "mag" not in df.columns and "flux_psf" in df.columns:
            flux = pd.to_numeric(df["flux_psf"], errors="coerce")
            valid_flux = flux > 0
            df["mag"] = np.nan
            df.loc[valid_flux, "mag"] = -2.5 * np.log10(flux[valid_flux]) + 8.90
        if "mag_err" not in df.columns and "flux_psf" in df.columns and "flux_psf_err" in df.columns:
            flux = pd.to_numeric(df["flux_psf"], errors="coerce")
            flux_err = pd.to_numeric(df["flux_psf_err"], errors="coerce")
            df["mag_err"] = np.nan
            valid_flux = flux > 0
            df.loc[valid_flux, "mag_err"] = 1.08 * (flux_err[valid_flux] / flux[valid_flux])
    elif source == "crts":
        df = _rename_first_present(df, "mjd", ("mjd", "ObsTime"))
        df = _rename_first_present(df, "mag", ("mag", "Mag"))
        df = _rename_first_present(df, "mag_err", ("mag_err", "magerr", "e_Mag"))
        _normalize_mjd_column(df)

    return df


def load_external_lc_frame(source_name: str, lc_path: Path) -> pd.DataFrame:
    """Load and normalize an external LC parquet with a small in-memory cache."""
    try:
        lc_path = Path(lc_path)
        stat = lc_path.stat()
        key = (str(source_name), str(lc_path.resolve()), int(stat.st_mtime_ns), int(stat.st_size))
    except Exception:
        return pd.DataFrame()

    cached = _cache_get(_EXTERNAL_LC_CACHE, key)
    if cached is not None:
        return cached.copy()

    try:
        df = normalize_external_lc_dataframe(source_name, pd.read_parquet(lc_path))
    except Exception:
        return pd.DataFrame()

    _cache_put(_EXTERNAL_LC_CACHE, key, df.copy())
    return df


def build_external_traces(
    panel_id: str,
    external_lcs: dict[str, Path],
    jd_offset: float,
    is_flux: bool,
    mag_anchor: float | None = None,
    warnings: list[str] | None = None,
) -> tuple[list[dict], set[str]]:
    """Build PlotTrace-compatible dicts for external survey overlays."""
    traces: list[dict] = []
    sources_with_traces: set[str] = set()
    plot_jd_offset = JD_OFFSET

    for source_name, lc_path in external_lcs.items():
        spec = EXTERNAL_LC_SPECS.get(source_name)
        if spec is None:
            continue
        is_flux_source = bool(spec.get("is_flux", False))
        try:
            lc_path = Path(lc_path)
            if not lc_path.exists():
                continue
            df_ext = load_external_lc_frame(source_name, lc_path)
            if df_ext.empty:
                continue
        except Exception:
            continue

        trace_count_before = len(traces)

        time_col = spec["time_col"]
        actual_time = None
        for column in df_ext.columns:
            if column.lower() == time_col.lower():
                actual_time = column
                break
        if actual_time is None:
            continue

        times = pd.to_numeric(df_ext[actual_time], errors="coerce").to_numpy()

        jd_sys = spec.get("jd_system", "mjd")
        if jd_sys == "mjd":
            finite_t = times[np.isfinite(times)]
            if finite_t.size and float(np.nanmedian(finite_t)) > 1_000_000.0:
                jd = times
            else:
                jd = times + MJD_TO_JD
            x_plot = jd - plot_jd_offset
        elif jd_sys == "bjd_gaia":
            jd = times + GAIA_TCB_EPOCH_JD
            x_plot = jd - plot_jd_offset
        elif jd_sys == "btjd":
            jd = times + TESS_BTJD_OFFSET
            x_plot = jd - plot_jd_offset
        elif jd_sys == "bkjd":
            jd = times + KEPLER_BKJD_OFFSET
            x_plot = jd - plot_jd_offset
        else:
            x_plot = times - jd_offset

        filter_col = spec.get("filter_col")
        default_mag_col = spec["mag_col"]
        default_err_col = spec.get("err_col", "")

        col_lookup = {column.lower(): column for column in df_ext.columns}
        actual_filt = col_lookup.get(filter_col.lower()) if filter_col else None

        source_transform_warned = False
        for _band_key, band_info in spec["bands"].items():
            band_mag_col = str(band_info.get("mag_col") or default_mag_col)
            band_err_col = str(band_info.get("err_col") or default_err_col or "")
            actual_mag = col_lookup.get(band_mag_col.lower())
            actual_err = col_lookup.get(band_err_col.lower()) if band_err_col else None
            if actual_mag is None:
                continue
            if actual_filt:
                mask = df_ext[actual_filt].astype(str) == _band_key
                band_df = df_ext[mask]
                band_x = x_plot[mask.to_numpy()]
            else:
                band_df = df_ext
                band_x = x_plot

            if band_df.empty:
                continue

            raw_y = pd.to_numeric(band_df[actual_mag], errors="coerce").to_numpy(dtype=float)
            good = np.isfinite(band_x) & np.isfinite(raw_y)
            if not good.any():
                continue

            display_label = str(band_info["label"])
            flux_to_relative_mag = bool(is_flux_source and not is_flux)
            y = raw_y.copy()
            ref_flux = np.nan
            anchor_mag = np.nan
            if flux_to_relative_mag:
                positive = good & (raw_y > 0)
                if not positive.any():
                    continue
                ref_flux = float(np.nanmedian(raw_y[positive]))
                if not np.isfinite(ref_flux) or ref_flux <= 0:
                    continue
                anchor_mag = float(mag_anchor) if mag_anchor is not None and np.isfinite(float(mag_anchor)) else 0.0
                y = np.full(raw_y.shape, np.nan, dtype=float)
                y[positive] = anchor_mag - 2.5 * np.log10(raw_y[positive] / ref_flux)
                good = np.isfinite(band_x) & np.isfinite(y)
                if not good.any():
                    continue
                display_label = f"{display_label} rel. Δm"
                if warnings is not None and not source_transform_warned:
                    warnings.append(
                        f"{band_info['label']} flux plotted as relative magnitude anchored to m={anchor_mag:.4f}; "
                        f"not calibrated {band_info['label']}-band magnitude."
                    )
                    source_transform_warned = True
            elif is_flux_source and is_flux:
                display_label = f"{display_label} flux"
            elif not is_flux_source and is_flux:
                y = np.power(10.0, -0.4 * y)

            err_full = None
            if actual_err and actual_err in band_df.columns:
                err_values = pd.to_numeric(band_df[actual_err], errors="coerce").to_numpy(dtype=float)
                valid_err = good & np.isfinite(err_values)
                if valid_err.any():
                    err_full = np.full(y.shape, np.nan, dtype=float)
                    if flux_to_relative_mag:
                        valid_flux_err = valid_err & (raw_y > 0)
                        err_full[valid_flux_err] = (2.5 / np.log(10.0)) * err_values[valid_flux_err] / raw_y[valid_flux_err]
                    elif not is_flux_source and is_flux:
                        err_full[valid_err] = 0.921 * y[valid_err] * err_values[valid_err]
                    else:
                        err_full[valid_err] = err_values[valid_err]

            good_idx = np.flatnonzero(good)
            if good_idx.size > _MAX_EXTERNAL_TRACE_POINTS:
                step = int(np.ceil(good_idx.size / float(_MAX_EXTERNAL_TRACE_POINTS)))
                good_idx = good_idx[::step]

            x_vals = band_x[good_idx]
            y_vals = y[good_idx]
            jd_vals = x_vals + plot_jd_offset
            err_array = err_full[good_idx] if err_full is not None else None
            if err_array is not None and not np.isfinite(err_array).any():
                err_array = None

            if flux_to_relative_mag:
                raw_flux_vals = raw_y[good_idx]
                err_vals = err_array if err_array is not None else np.full(jd_vals.shape, np.nan, dtype=float)
                custom_ext = np.column_stack(
                    [
                        jd_vals,
                        raw_flux_vals,
                        np.full(jd_vals.shape, ref_flux, dtype=float),
                        np.full(jd_vals.shape, anchor_mag, dtype=float),
                        err_vals,
                    ]
                )
                hover_ext = (
                    f"<b>{display_label}</b><br>"
                    "JD: %{customdata[0]:.5f}<br>"
                    f"JD - {int(plot_jd_offset)}: %{{x:.5f}}<br>"
                    "m<sub>rel</sub>: %{y:.4f}<br>"
                    "raw flux: %{customdata[1]:.4e}<br>"
                    "median flux: %{customdata[2]:.4e}<br>"
                    "anchor m: %{customdata[3]:.4f}<br>"
                    + ("σ<sub>m,rel</sub>: %{customdata[4]:.4f}<extra></extra>" if err_array is not None else "<extra></extra>")
                )
            elif err_array is not None:
                custom_ext = np.column_stack([jd_vals, err_array])
                hover_ext = (
                    f"<b>{display_label}</b><br>"
                    "JD: %{customdata[0]:.5f}<br>"
                    f"JD - {int(plot_jd_offset)}: %{{x:.5f}}<br>"
                    + ("F: %{y:.4e}<br>" if is_flux else "m: %{y:.4f}<br>")
                    + ("σ<sub>F</sub>: %{customdata[1]:.3e}<extra></extra>" if is_flux else "σ<sub>m</sub>: %{customdata[1]:.4f}<extra></extra>")
                )
            else:
                custom_ext = jd_vals.reshape(-1, 1)
                hover_ext = (
                    f"<b>{display_label}</b><br>"
                    "JD: %{customdata[0]:.5f}<br>"
                    f"JD - {int(plot_jd_offset)}: %{{x:.5f}}<br>"
                    + ("F: %{y:.4e}<br>" if is_flux else "m: %{y:.4f}<br>")
                    + "<extra></extra>"
                )

            traces.append(
                {
                    "panel_id": panel_id,
                    "x": x_vals,
                    "y": y_vals,
                    "yerr": err_array,
                    "color": band_info["color"],
                    "marker": band_info["marker"],
                    "label": display_label,
                    "alpha": 0.8,
                    "marker_size": 6,
                    "kind": "scatter",
                    "showlegend": True,
                    "legendgroup": source_name,
                    "customdata": custom_ext,
                    "hovertemplate": hover_ext,
                }
            )

        if len(traces) > trace_count_before:
            sources_with_traces.add(str(source_name).strip().lower())

    return traces, sources_with_traces
