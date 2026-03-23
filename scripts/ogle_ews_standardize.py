from pathlib import Path
import pandas as pd
import csv

input_path = Path("ogle-ews-march22.txt")
raw = input_path.read_text(encoding="utf-8", errors="replace")

clean_chars = []
for ch in raw.replace("\r\n", "\n").replace("\r", "\n"):
    if ch == "\n" or ch == "\t" or ord(ch) >= 32:
        clean_chars.append(ch)
text = "".join(clean_chars)

def tokenize(line):
    parts = [p.strip() for p in line.split("\t")]
    while parts and parts[0] == "":
        parts = parts[1:]
    while parts and parts[-1] == "":
        parts = parts[:-1]
    return parts

full_cols_original = ['Event', 'Field', 'Star No', 'RA (J2000)', 'Dec (J2000)', 'Tmax (HJD)', 'Tmax (UT)', 'tau', 'Umin', 'Amax', 'Dmag', 'fbl', 'Ibl', 'I0']
old_cols_original = ['Event', 'Field', 'Star No', 'RA (J2000)', 'Dec (J2000)', 'Tmax (HJD)', 'Tmax (UT)', 'tau', 'Amax', 'Dmag', 'I0']
snake_map = {'Event': 'event', 'Field': 'field', 'Star No': 'star_no', 'RA (J2000)': 'ra_j2000', 'Dec (J2000)': 'dec_j2000', 'Tmax (HJD)': 'tmax_hjd', 'Tmax (UT)': 'tmax_ut', 'tau': 'tau', 'Umin': 'umin', 'Amax': 'amax', 'Dmag': 'dmag', 'fbl': 'fbl', 'Ibl': 'ibl', 'I0': 'i0'}

rows = []
for line in text.splitlines():
    tk = tokenize(line)
    if not tk:
        continue
    if tk[0] == "Event" and len(tk) in (11, 14):
        continue
    if len(tk) == 11:
        row = {c: pd.NA for c in full_cols_original}
        for c, v in zip(old_cols_original, tk):
            row[c] = v
    elif len(tk) == 14:
        row = {c: v for c, v in zip(full_cols_original, tk)}
    else:
        raise ValueError(f"Unexpected token count {len(tk)} in line: {line[:120]!r}")
    rows.append(row)

df = pd.DataFrame(rows, columns=full_cols_original).replace("-", pd.NA).rename(columns=snake_map)
for col in df.columns:
    df[col] = df[col].astype("string")

display_df = df.fillna("")
widths = {col: max(len(col), int(display_df[col].map(len).max())) for col in df.columns}
sep = "  "
starts = []
cursor = 0
for col in df.columns:
    starts.append(cursor)
    cursor += widths[col] + len(sep)
colspecs = [(s, s + widths[col]) for s, col in zip(starts, df.columns)]

fwf_path = Path("ogle_ews_standardized.fwf")
with fwf_path.open("w", encoding="utf-8", newline="\n") as f:
    f.write(sep.join(col.ljust(widths[col]) for col in df.columns) + "\n")
    for _, row in display_df.iterrows():
        f.write(sep.join(str(row[col]).ljust(widths[col]) for col in df.columns) + "\n")

parsed = pd.read_fwf(
    fwf_path,
    colspecs=colspecs,
    dtype="string",
    na_values=[""],
    keep_default_na=True,
)
for col in parsed.columns:
    parsed[col] = parsed[col].str.strip()
parsed = parsed.replace({"": pd.NA})

parsed.to_csv("ogle_ews_standardized.csv", index=False, quoting=csv.QUOTE_MINIMAL)
