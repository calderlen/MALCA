import pandas as pd

vsx_dir = '/data/poohbah/1/assassin/lenhart/code/calder/vsxcat.090525'

vsx_columns = [
    "id_vsx", "name", "UNKNOWN_FLAG", "ra", "dec", "variability_class",
    "mag", "band_mag", "amplitude_flag", "amplitude",
    "amplitude/mag_diff", "band_amplitude/band_mag_diff",
    "epoch", "period", "spectral_type"
]

# note: your dtype_map keys don't actually match vsx_columns
# here we only set dtypes that align directly
dtype_map = {
    "id_vsx": "int64",
    "name": "string",
    "UNKNOWN_FLAG": "float64",
    "ra": "float64",
    "dec": "float64",
    "variability_class": "string",
    "mag": "string",  # keep string to preserve "<" limits etc
    "band_mag": "string",
    "amplitude_flag": "string",
    "amplitude": "float64",
    "epoch": "float64",
    "period": "float64",
    "spectral_type": "string",
}

df = pd.read_fwf(
    vsx_dir,
    names=vsx_columns,
    dtype=dtype_map,
    on_bad_lines="skip",
    colspecs="infer",
    infer_nrows=10000,
    )

# Write each .unique() result to a separate text file
pd.Series(df["UNKNOWN_FLAG"].unique()).to_csv("output/unknown_flag_values.csv", index=False, header=False)
pd.Series(df["variability_class"].unique()).to_csv("output/variability_class_values.csv", index=False, header=False)
pd.Series(df["amplitude_flag"].unique()).to_csv("output/amplitude_flag_values.csv", index=False, header=False)
pd.Series(df["spectral_type"].unique()).to_csv("output/spectral_type_values.csv", index=False, header=False)