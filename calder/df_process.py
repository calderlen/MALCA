import numpy as np
import pandas as pd
from pathlib import Path

def read_df_parquet(path):
    """
    load one parquet file produced by dip_finder
    """
    file = Path(path)

    df = pd.read_parquet(file).copy()

    if "mag_bin" not in df.columns:
        mag_bin = file.stem.replace("peaks_", "").replace("_", ".")
        df["mag_bin"] = mag_bin

    # provenance column to trace which parquet file each row came from
    df["source_file"] = file.name

    return df


def filter_df(
    df,
    *,
    min_rows_g=None,
    min_rows_v=None,
    min_g_n_peaks=None,
    min_v_n_peaks=None,
    min_g_dip_fraction=None,
    max_g_jump_fraction=None,
    min_v_dip_fraction=None,
    max_v_jump_fraction=None,
    min_g_n_dip_runs=None,
    min_v_n_dip_runs=None,
    min_g_max_depth=None,
    min_v_max_depth=None,
    require_g_dip_dominated=None,
    require_v_dip_dominated=None,
):
    """Return a filtered copy of a dip_finder DataFrame.

    Pass whichever thresholds you want to explore; omitted parameters are not
    enforced. ``require_*_dip_dominated`` accepts ``True``/``False`` to demand a
    specific dip-dominated classification from :func:`lc_metrics.is_dip_dominated`.
    """

    mask = pd.Series(True, index=df.index)

    def _col(name):
        if name not in df.columns:
            raise KeyError(
                f"Column '{name}' is missing; regenerate parquet with dip_finder to include it."
            )
        return df[name]

    def _gte(name, value):
        nonlocal mask
        if value is None:
            return
        mask &= _col(name).fillna(-np.inf) >= value

    def _lte(name, value):
        nonlocal mask
        if value is None:
            return
        mask &= _col(name).fillna(np.inf) <= value

    def _require_bool(name, desired):
        nonlocal mask
        if desired is None:
            return
        series = _col(name).fillna(False).astype(bool)
        mask &= (series == bool(desired))

    # point-count constraints
    _gte("n_rows_g", min_rows_g)
    _gte("n_rows_v", min_rows_v)

    # peak counts
    _gte("g_n_peaks", min_g_n_peaks)
    _gte("v_n_peaks", min_v_n_peaks)

    # dip/jump fractions
    _gte("g_dip_fraction", min_g_dip_fraction)
    _lte("g_jump_fraction", max_g_jump_fraction)
    _gte("v_dip_fraction", min_v_dip_fraction)
    _lte("v_jump_fraction", max_v_jump_fraction)

    # run counts
    _gte("g_n_dip_runs", min_g_n_dip_runs)
    _gte("v_n_dip_runs", min_v_n_dip_runs)

    # depth thresholds
    _gte("g_max_depth", min_g_max_depth)
    _gte("v_max_depth", min_v_max_depth)

    # dip-dominated requirement
    _require_bool("g_is_dip_dominated", require_g_dip_dominated)
    _require_bool("v_is_dip_dominated", require_v_dip_dominated)

    return df.loc[mask].reset_index(drop=True)

def filter_df(df):
