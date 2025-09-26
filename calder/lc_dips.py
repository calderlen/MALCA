import numpy as np


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



def run_metrics(df):
    '''
    calculates dip and jump metrics for a given lc df and returns as a dict
    '''
    # Brayden just used the mean mag of the lightcurve as the baseline, but I think this can be improved upon
    # Ideally we collect from per_camera_baseline.py
    # this is still in progress, so skip for now and instead use a simple rolling median
    #df = per_camera_baseline(df)
    
    df = df.sort_values("JD").reset_index(drop=True)

    # simple rolling median baseline
    df = df.sort_values("JD").reset_index(drop=True)
    df["baseline"] = df["mag"].rolling(window=30, min_periods=10, center=True).median()
    df["baseline"] = df["baseline"].fillna(method="bfill").filln(method="ffill")

    # find excursions wrt baseline with simple 0.3 mag threshold
    dip_mask = df["mag"] < (df["baseline"] + 0.3)
    jump_mask = df["mag"] > (df["baseline"] - 0.3)

    dip_runs = find_runs(dip_mask.to_numpy())
    jump_runs = find_runs(jump_mask.to_numpy())

    # metrics
    n_dip_runs = len(dip_runs)
    n_jump_runs = len(jump_runs)
    n_dip_points = int(np.sum(dip_mask))
    n_jump_points = int(np.sum(jump_mask))

    #most_recent_dip = np.nan
    most_recent_jump = np.nan
    max_depth = np.nan
    max_height = np.nan
    max_dip_duration = np.nan
    max_jump_duration = np.nan

    if n_dip_points > 0:
        most_recent_dip = float(df.loc[dip_mask, "JD"].max())
        max_depth = float(np.nanmax(df.loc[dip_mask, "baseline"] - df.loc[dip_mask, "mag"]))
        if n_dip_runs > 0:
            max_dip_duration = 0.
            for run in dip_runs:
                max_dip_duration = np.max([df.loc[run[1], "JD"] - df.loc[run[0], "JD"]])
        
    if n_jump_points > 0:
        most_recent_jump = float(df.loc[jump_mask, "JD"].max())
        max_height = np.max(df.loc[jump_mask, "mag"] - df.loc[jump_mask, "baseline"])
        if jump_runs > 0:
            max_jump_duration = 0.
            for run in jump_runs:
                max_jump_duration = np.max([df.loc[run[1], "JD"] - df.loc[run[0], "JD"]])
    

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

def is_dip_dominated(metrics_dict, min_dip_fraction=0.67):
    '''
    returns True if the the dip fraction from the metrics dict is above a certain value, currently 2/3
    '''
    if ~np.isnan(metrics_dict["dip_fraction"]) and metrics_dict["dip_fraction"] >= min_dip_fraction:
            return True
    else :
        return False
 

def multi_camera_confirmation():
    pass

