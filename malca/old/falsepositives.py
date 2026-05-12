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
            for f in sorted(p.glob("*.parquet")):
                frames.append(pd.read_parquet(f))
        else:
            frames.append(pd.read_parquet(p))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def band_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for band in ("g", "v"):
        col = f"{band}_n_peaks"
        if col in df.columns:
            out[f"{band}_det"] = pd.to_numeric(df[col], errors="coerce").fillna(0) > 0
    out["either_det"] = out.any(axis=1) if not out.empty else pd.Series(dtype=bool)
    out["both_det"] = out.all(axis=1) if not out.empty else pd.Series(dtype=bool)
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
    if pre.empty or post.empty or id_col not in pre.columns or id_col not in post.columns:
        return {
            "n_pre": len(pre),
            "n_post": len(post),
            "retained": 0,
            "retention_frac": 0.0,
        }
    pre_ids = pre[id_col].astype(str)
    post_ids = post[id_col].astype(str)
    retained = len(set(pre_ids) & set(post_ids))
    frac = retained / len(pre) if len(pre) > 0 else 0.0
    return {
        "n_pre": len(pre),
        "n_post": len(post),
        "retained": retained,
        "retention_frac": frac,
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="False-positive reduction summary (tag vs post filter).")
    parser.add_argument("--pre", nargs="+", required=True, help="Tag-stage Parquet(s) or directory.")
    parser.add_argument("--post", nargs="+", required=True, help="Post-filter Parquet(s) or directory.")
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
