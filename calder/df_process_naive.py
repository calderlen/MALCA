from pathlib import Path

import pandas as pd


def read_df_csv_naive(
    csv_path,
    out_csv_path=None,
    write_csv: bool = True,
    index: bool = False,
    band: str = "either",
):
    """
    Read peaks_[mag_bin].csv and return only rows where either band has a non-zero number of peaks. Optionally, output peaks_[mag_bin]_selected_dippers.csv. Optionally search for only g band, only v band, or both.
    """

    file = Path(csv_path)
    df = pd.read_csv(file).copy()

    for col in ("g_n_peaks", "v_n_peaks"):
        if col not in df.columns:
            raise KeyError(
                f"Column '{col}' is missing; cannot select nonzero-peak rows."
            )

    df["g_n_peaks"] = pd.to_numeric(df["g_n_peaks"], errors="coerce").fillna(0)
    df["v_n_peaks"] = pd.to_numeric(df["v_n_peaks"], errors="coerce").fillna(0)

    band_key = band.lower()
    if band_key not in {"g", "v", "both", "either"}:
        raise ValueError("band must be one of 'g', 'v', 'both', 'either'")

    if band_key == "g":
        mask = df["g_n_peaks"] > 0
    elif band_key == "v":
        mask = df["v_n_peaks"] > 0
    elif band_key == "both":
        mask = (df["g_n_peaks"] > 0) & (df["v_n_peaks"] > 0)
    else:
        mask = (df["g_n_peaks"] > 0) | (df["v_n_peaks"] > 0)

    out = df.loc[mask].reset_index(drop=True)
    out["source_file"] = file.name

    if write_csv:
        dest = (
            Path(out_csv_path)
            if out_csv_path is not None
            else file.parent / f"{file.stem}_selected_dippers.csv"
        )
        out.to_csv(dest, index=index)

    return out
