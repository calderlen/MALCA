from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


def load_many(paths: Sequence[str | Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for p in paths:
        p = Path(p).expanduser()
        if not p.exists():
            continue
        if p.is_dir():
            for f in sorted(p.glob("*.csv")):
                frames.append(pd.read_csv(f))
        else:
            frames.append(pd.read_csv(p))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _to_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(float) != 0
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "t", "yes", "y"})


def _extract_ids(df: pd.DataFrame, id_col: str) -> pd.Series:
    if id_col in df.columns:
        return df[id_col].astype(str)
    for fallback in ("source_id", "asas_sn_id"):
        if fallback in df.columns:
            return df[fallback].astype(str)
    if "path" in df.columns:
        return df["path"].apply(lambda x: Path(str(x)).stem.replace(".dat2", "").replace(".csv", ""))
    return pd.Series([], dtype=str)


def band_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    peak_cols = [c for c in ("g_n_peaks", "v_n_peaks") if c in df.columns]
    if peak_cols:
        for band in ("g", "v"):
            col = f"{band}_n_peaks"
            if col in df.columns:
                out[f"{band}_det"] = pd.to_numeric(df[col], errors="coerce").fillna(0) > 0
        base_cols = list(out.columns)
        if base_cols:
            out["either_det"] = out[base_cols].any(axis=1)
            out["both_det"] = out[base_cols].all(axis=1) if len(base_cols) > 1 else out["either_det"]
        return out

    sig_cols = [c for c in ("dip_significant", "jump_significant") if c in df.columns]
    if sig_cols:
        if "dip_significant" in df.columns:
            out["dip_det"] = _to_bool(df["dip_significant"])
        if "jump_significant" in df.columns:
            out["jump_det"] = _to_bool(df["jump_significant"])
        base_cols = list(out.columns)
        if base_cols:
            out["either_det"] = out[base_cols].any(axis=1)
            out["both_det"] = out[base_cols].all(axis=1) if len(base_cols) > 1 else out["either_det"]
        return out

    return out


def summarize(df: pd.DataFrame, label: str) -> dict:
    if df.empty:
        return {"label": label, "n": 0}
    flags = band_flags(df)
    summary = {
        "label": label,
        "n": len(df),
        "n_mag_bins": df["mag_bin"].nunique() if "mag_bin" in df.columns else None,
        "n_g": int(flags["g_det"].sum()) if "g_det" in flags else None,
        "n_v": int(flags["v_det"].sum()) if "v_det" in flags else None,
        "n_dip": int(flags["dip_det"].sum()) if "dip_det" in flags else None,
        "n_jump": int(flags["jump_det"].sum()) if "jump_det" in flags else None,
        "n_either": int(flags["either_det"].sum()) if "either_det" in flags else None,
        "n_both": int(flags["both_det"].sum()) if "both_det" in flags else None,
    }
    if "mag_bin" in df.columns:
        summary["by_mag_bin"] = (
            df.assign(either=flags["either_det"] if "either_det" in flags else True)
            .groupby("mag_bin")["either"]
            .sum()
            .to_dict()
        )
    return summary


def retention(pre: pd.DataFrame, post: pd.DataFrame, id_col: str = "asas_sn_id") -> dict:
    if pre.empty or post.empty:
        return {
            "n_pre": len(pre),
            "n_post": len(post),
            "retained": 0,
            "retention_frac": 0.0,
        }
    pre_ids = _extract_ids(pre, id_col)
    post_ids = _extract_ids(post, id_col)
    if pre_ids.empty or post_ids.empty:
        return {
            "n_pre": len(pre),
            "n_post": len(post),
            "retained": 0,
            "retention_frac": 0.0,
        }
    retained = len(set(pre_ids) & set(post_ids))
    frac = retained / len(pre) if len(pre) > 0 else 0.0
    return {
        "n_pre": len(pre),
        "n_post": len(post),
        "retained": retained,
        "retention_frac": frac,
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="False-positive reduction summary (pre vs post filter).")
    parser.add_argument("--pre", nargs="+", required=True, help="Pre-filter CSV(s) or directory.")
    parser.add_argument("--post", nargs="+", required=True, help="Post-filter CSV(s) or directory.")
    parser.add_argument("--id-col", default="asas_sn_id", help="ID column for retention match.")
    args = parser.parse_args(argv)

    pre_df = load_many(args.pre)
    post_df = load_many(args.post)

    pre_summary = summarize(pre_df, "pre")
    post_summary = summarize(post_df, "post")
    retain = retention(pre_df, post_df, id_col=args.id_col)

    print("=== Summary ===")
    print(pre_summary)
    print(post_summary)
    print("=== Retention ===")
    print(retain)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
