import numpy as np

from lc_baseline import per_camera_baseline


def find_runs(mask, max_gap=1, min_length=1):
    idx = np.flatnonzero(mask)
    runs = []

    if len(idx) == 0 or len(idx) < min_length:
        return runs
    
    start = idx[0]
    end = idx[0]

    for i in range(1, len(idx)):
        if idx[i] - end <= max_gap:
            end = idx[i]
        else:
            if end - start + 1 >= min_length:
                runs.append((start, end))
            start = idx[i]
            end = idx[i]

    if end - start + 1 >= min_length:
        runs.append((start, end))

    return runs



def run_metrics(df, dip_threshold=0.3, window=30, min_points=10):
    """
    calculates dip and jump metrics for a given lc df and returns as a dict
    """
    # Brayden just used the mean mag of the lightcurve as the baseline, but I think this can be improved upon
    # Ideally we collect from per_camera_baseline.py
    # this is still in progress, so skip for now and instead use a simple rolling median
    #df = per_camera_baseline(df)
    
    work = df.sort_values("JD").reset_index(drop=True).copy()

    work["baseline"] = (
        work["mag"]
        .rolling(window=window, min_periods=min_points, center=True)
        .median()
    )
    work["baseline"] = work["baseline"].fillna(method="bfill").fillna(method="ffill")

    dip_mask = work["mag"] >= (work["baseline"] + dip_threshold)   # dimmer than baseline
    jump_mask = work["mag"] <= (work["baseline"] - dip_threshold)  # brighter than baseline

    dip_runs = find_runs(dip_mask.to_numpy())
    jump_runs = find_runs(jump_mask.to_numpy())

    n_dip_runs = len(dip_runs)
    n_jump_runs = len(jump_runs)
    n_dip_points = int(dip_mask.sum())
    n_jump_points = int(jump_mask.sum())

    most_recent_dip = np.nan
    most_recent_jump = np.nan
    max_depth = np.nan
    max_height = np.nan
    max_dip_duration = np.nan
    max_jump_duration = np.nan

    if n_dip_points > 0:
        most_recent_dip = float(work.loc[dip_mask, "JD"].max())
        max_depth = float(np.nanmax(work.loc[dip_mask, "mag"] - work.loc[dip_mask, "baseline"]))
        if n_dip_runs > 0:
            max_dip_duration = 0.0
            for start, end in dip_runs:
                span = float(work.loc[end, "JD"] - work.loc[start, "JD"])
                max_dip_duration = max(max_dip_duration, span)

    if n_jump_points > 0:
        most_recent_jump = float(work.loc[jump_mask, "JD"].max())
        max_height = float(np.nanmax(work.loc[jump_mask, "baseline"] - work.loc[jump_mask, "mag"]))
        if n_jump_runs > 0:
            max_jump_duration = 0.0
            for start, end in jump_runs:
                span = float(work.loc[end, "JD"] - work.loc[start, "JD"])
                max_jump_duration = max(max_jump_duration, span)
    
    metrics_dict = {
        "n_dip_runs": n_dip_runs,
        "n_jump_runs": n_jump_runs,
        "n_dip_points": n_dip_points,
        "n_jump_points": n_jump_points,
        "most_recent_dip": most_recent_dip,
        "most_recent_jump": most_recent_jump,
        "max_depth": max_depth,
        "max_height": max_height,
        "max_dip_duration": max_dip_duration,
        "max_jump_duration": max_jump_duration,
        "dip_fraction": n_dip_points / len(df) if len(df) > 0 else np.nan,
        "jump_fraction": n_jump_points / len(df) if len(df) > 0 else np.nan
    }

    return metrics_dict

def run_metrics_pcb(df, **pcb_kwargs):
    """
    same metrics as run_metrics, but the baseline comes from per_camera_baseline
    """
    df_sorted = df.sort_values("JD").reset_index(drop=True)
    df_pcb = per_camera_baseline(df_sorted, **pcb_kwargs).sort_values("JD").reset_index(drop=True)

    if "baseline" not in df_pcb.columns:
        raise ValueError("per_camera_baseline must supply a 'baseline' column")

    dip_mask = df_pcb["mag"] < (df_pcb["baseline"] + 0.3)
    jump_mask = df_pcb["mag"] > (df_pcb["baseline"] - 0.3)

    dip_runs = find_runs(dip_mask.to_numpy())
    jump_runs = find_runs(jump_mask.to_numpy())

    n_dip_runs = len(dip_runs)
    n_jump_runs = len(jump_runs)
    n_dip_points = int(np.sum(dip_mask))
    n_jump_points = int(np.sum(jump_mask))

    most_recent_dip = np.nan
    most_recent_jump = np.nan
    max_depth = np.nan
    max_height = np.nan
    max_dip_duration = np.nan
    max_jump_duration = np.nan

    if n_dip_points > 0:
        most_recent_dip = float(df_pcb.loc[dip_mask, "JD"].max())
        max_depth = float(np.nanmax(df_pcb.loc[dip_mask, "baseline"] - df_pcb.loc[dip_mask, "mag"]))
        if n_dip_runs > 0:
            max_dip_duration = 0.0
            for start, end in dip_runs:
                max_dip_duration = max(max_dip_duration, float(df_pcb.loc[end, "JD"] - df_pcb.loc[start, "JD"]))

    if n_jump_points > 0:
        most_recent_jump = float(df_pcb.loc[jump_mask, "JD"].max())
        max_height = float(np.nanmax(df_pcb.loc[jump_mask, "mag"] - df_pcb.loc[jump_mask, "baseline"]))
        if n_jump_runs > 0:
            max_jump_duration = 0.0
            for start, end in jump_runs:
                max_jump_duration = max(max_jump_duration, float(df_pcb.loc[end, "JD"] - df_pcb.loc[start, "JD"]))

    return {
        "n_dip_runs": n_dip_runs,
        "n_jump_runs": n_jump_runs,
        "n_dip_points": n_dip_points,
        "n_jump_points": n_jump_points,
        "most_recent_dip": most_recent_dip,
        "most_recent_jump": most_recent_jump,
        "max_depth": max_depth,
        "max_height": max_height,
        "max_dip_duration": max_dip_duration,
        "max_jump_duration": max_jump_duration,
        "dip_fraction": n_dip_points / len(df_pcb) if len(df_pcb) > 0 else np.nan,
        "jump_fraction": n_jump_points / len(df_pcb) if len(df_pcb) > 0 else np.nan,
    }
    


def is_dip_dominated(metrics_dict, min_dip_fraction=0.67):
    """
    returns True if the the dip fraction from the metrics dict is above a certain value, currently 2/3
    """
    if ~np.isnan(metrics_dict["dip_fraction"]) and metrics_dict["dip_fraction"] >= min_dip_fraction:
            return True
    else :
        return False
 

def multi_camera_confirmation():
    pass

