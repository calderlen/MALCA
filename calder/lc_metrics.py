import numpy as np

from lc_baseline import per_camera_median_baseline, per_camera_mean_baseline, global_mean_baseline


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


def run_metrics(
    df,
    baseline_func=per_camera_median_baseline,
    dip_threshold=0.3,
    **baseline_kwargs,
):
    """
    calculates dip and jump metrics for a given lc df and returns as a dict
    """
    df_sorted = df.sort_values("JD").reset_index(drop=True)
    df_baseline = (
        baseline_func(df_sorted, **baseline_kwargs)
        .sort_values("JD")
        .reset_index(drop=True)
    )

    dip_mask = df_baseline["mag"] >= (df_baseline["baseline"] + dip_threshold)
    jump_mask = df_baseline["mag"] <= (df_baseline["baseline"] - dip_threshold)

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
        most_recent_dip = float(df_baseline.loc[dip_mask, "JD"].max())
        max_depth = float(
            np.nanmax(df_baseline.loc[dip_mask, "mag"] - df_baseline.loc[dip_mask, "baseline"])
        )
        if n_dip_runs > 0:
            max_dip_duration = 0.0
            for start, end in dip_runs:
                max_dip_duration = max(
                    max_dip_duration,
                    float(df_baseline.loc[end, "JD"] - df_baseline.loc[start, "JD"]),
                )

    if n_jump_points > 0:
        most_recent_jump = float(df_baseline.loc[jump_mask, "JD"].max())
        max_height = float(
            np.nanmax(df_baseline.loc[jump_mask, "baseline"] - df_baseline.loc[jump_mask, "mag"])
        )
        if n_jump_runs > 0:
            max_jump_duration = 0.0
            for start, end in jump_runs:
                max_jump_duration = max(
                    max_jump_duration,
                    float(df_baseline.loc[end, "JD"] - df_baseline.loc[start, "JD"]),
                )

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
        "dip_fraction": n_dip_points / len(df_baseline) if len(df_baseline) > 0 else np.nan,
        "jump_fraction": n_jump_points / len(df_baseline) if len(df_baseline) > 0 else np.nan,
    }


def is_dip_dominated(metrics_dict, min_dip_fraction=0.66):
    """
    returns True if the the dip fraction from the metrics dict is above a certain value, currently 2/3
    """
    if ~np.isnan(metrics_dict["dip_fraction"]) and metrics_dict["dip_fraction"] >= min_dip_fraction:
            return True
    else :
        return False
 

def multi_camera_confirmation():
    pass
