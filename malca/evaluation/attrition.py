from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from malca.io.table_io import read_feature_table


_TRUE_VALUES = frozenset({"true", "1", "t", "yes", "y"})
_FALSE_VALUES = frozenset({"false", "0", "f", "no", "n"})
_MISSING_ID_VALUES = frozenset({"", "nan", "none", "null", "<na>"})


def _canonical_id(value: object) -> str | None:
    """Return one stable textual identity without turning nulls into IDs."""
    if value is None or value is pd.NA:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (int, np.integer)):
        text = str(int(value))
    elif isinstance(value, (float, np.floating)) and np.isfinite(value) and float(value).is_integer():
        text = str(int(value))
    else:
        text = str(value).strip()
    return None if text.lower() in _MISSING_ID_VALUES else text


def _wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float | None, float | None]:
    if trials <= 0:
        return None, None
    p = successes / trials
    denominator = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / denominator
    half_width = z * np.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials**2)) / denominator
    return float(max(0.0, center - half_width)), float(min(1.0, center + half_width))


def load_many(paths: Sequence[str | Path]) -> pd.DataFrame:
    """Load stage tables and reject missing inputs or duplicate candidates.

    Silently skipping a missing table makes attrition look better or worse for a
    reason unrelated to the pipeline.  Every requested input is therefore part
    of the accounting contract.
    """
    frames: list[pd.DataFrame] = []
    for raw_path in paths:
        path = Path(raw_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Attrition input does not exist: {path}")
        if path.is_dir():
            files = sorted(
                child
                for pattern in ("*.parquet", "*.csv")
                for child in path.glob(pattern)
            )
            if not files:
                raise ValueError(f"Attrition input directory contains no Parquet or CSV tables: {path}")
            for child in files:
                frame = read_feature_table(child)
                frame["_attrition_input_path"] = str(child.resolve())
                frames.append(frame)
        else:
            frame = read_feature_table(path)
            frame["_attrition_input_path"] = str(path.resolve())
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _parse_bool(value: object) -> object:
    if value is None or value is pd.NA:
        return pd.NA
    try:
        if bool(pd.isna(value)):
            return pd.NA
    except (TypeError, ValueError):
        pass
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return bool(value)
    if isinstance(value, (float, np.floating)) and np.isfinite(value) and float(value) in (0.0, 1.0):
        return bool(int(value))
    text = str(value).strip().lower()
    if text in _TRUE_VALUES:
        return True
    if text in _FALSE_VALUES:
        return False
    return pd.NA


def _to_bool(series: pd.Series) -> pd.Series:
    """Parse only explicit booleans; malformed and missing values stay unknown."""
    return pd.Series((_parse_bool(value) for value in series), index=series.index, dtype="boolean")


def _extract_ids(df: pd.DataFrame, id_col: str) -> pd.Series:
    selected: pd.Series | None = None
    selected_name: str | None = None
    for candidate in (id_col, "candidate_id", "source_id", "asas_sn_id"):
        if candidate in df.columns:
            selected = df[candidate]
            selected_name = candidate
            break
    if selected is None and "path" in df.columns:
        selected = df["path"].map(lambda value: Path(str(value)).stem)
        selected_name = "path"
    if selected is None:
        raise ValueError(
            f"No candidate identity column is available; expected {id_col!r}, "
            "candidate_id, source_id, asas_sn_id, or path"
        )
    ids = selected.map(_canonical_id).astype("string")
    if bool(ids.isna().any()):
        raise ValueError(f"Blank/null candidate identities found in {selected_name!r}")
    duplicate_mask = ids.duplicated(keep=False)
    if bool(duplicate_mask.any()):
        examples = sorted(ids.loc[duplicate_mask].astype(str).unique())[:5]
        raise ValueError(f"Duplicate candidate identities found in {selected_name!r}: {examples}")
    return ids


def _nullable_any(columns: list[pd.Series], index: pd.Index) -> pd.Series:
    if not columns:
        return pd.Series(pd.NA, index=index, dtype="boolean")
    frame = pd.concat(columns, axis=1).astype("boolean")
    result = pd.Series(pd.NA, index=index, dtype="boolean")
    any_true = frame.fillna(False).any(axis=1)
    all_known = frame.notna().all(axis=1)
    all_false = all_known & ~frame.fillna(False).any(axis=1)
    result.loc[any_true] = True
    result.loc[all_false] = False
    return result


def _nullable_all(columns: list[pd.Series], index: pd.Index) -> pd.Series:
    if not columns:
        return pd.Series(pd.NA, index=index, dtype="boolean")
    frame = pd.concat(columns, axis=1).astype("boolean")
    result = pd.Series(pd.NA, index=index, dtype="boolean")
    any_false = (~frame.fillna(True)).any(axis=1)
    all_known = frame.notna().all(axis=1)
    all_true = all_known & frame.fillna(False).all(axis=1)
    result.loc[any_false] = False
    result.loc[all_true] = True
    return result


def band_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    peak_cols = [column for column in ("g_n_peaks", "v_n_peaks") if column in df.columns]
    if peak_cols:
        base: list[pd.Series] = []
        for band in ("g", "v"):
            column = f"{band}_n_peaks"
            if column not in df.columns:
                continue
            numeric = pd.to_numeric(df[column], errors="coerce")
            invalid = numeric.notna() & (numeric.lt(0) | numeric.mod(1).ne(0))
            flag = (numeric.gt(0)).astype("boolean").mask(numeric.isna() | invalid, pd.NA)
            out[f"{band}_det"] = flag
            base.append(out[f"{band}_det"])
        out["either_det"] = _nullable_any(base, df.index)
        out["both_det"] = _nullable_all(base, df.index) if len(base) > 1 else out["either_det"]
        return out

    sig_cols = [column for column in ("dip_significant", "jump_significant") if column in df.columns]
    if sig_cols:
        base = []
        for kind in ("dip", "jump"):
            column = f"{kind}_significant"
            if column not in df.columns:
                continue
            out[f"{kind}_det"] = _to_bool(df[column])
            base.append(out[f"{kind}_det"])
        out["either_det"] = _nullable_any(base, df.index)
        out["both_det"] = _nullable_all(base, df.index) if len(base) > 1 else out["either_det"]
    return out


def _flag_accounting(flag: pd.Series) -> dict[str, object]:
    parsed = flag.astype("boolean")
    known = parsed.notna()
    numerator = int(parsed.loc[known].sum())
    denominator = int(known.sum())
    low, high = _wilson_interval(numerator, denominator)
    return {
        "numerator": numerator,
        "denominator": denominator,
        "unknown": int((~known).sum()),
        "fraction": numerator / denominator if denominator else None,
        "ci95_low": low,
        "ci95_high": high,
    }


def summarize(df: pd.DataFrame, label: str) -> dict:
    if df.empty:
        return {"label": label, "n": 0, "accounting_status": "empty"}
    # Validate identity even when this function is called independently.
    _extract_ids(df, "asas_sn_id")
    flags = band_flags(df)
    summary: dict[str, object] = {
        "label": label,
        "n": int(len(df)),
        "n_mag_bins": int(df["mag_bin"].nunique()) if "mag_bin" in df.columns else None,
        "n_g": int(flags["g_det"].fillna(False).sum()) if "g_det" in flags else None,
        "n_v": int(flags["v_det"].fillna(False).sum()) if "v_det" in flags else None,
        "n_dip": int(flags["dip_det"].fillna(False).sum()) if "dip_det" in flags else None,
        "n_jump": int(flags["jump_det"].fillna(False).sum()) if "jump_det" in flags else None,
        "n_either": int(flags["either_det"].fillna(False).sum()) if "either_det" in flags else None,
        "n_both": int(flags["both_det"].fillna(False).sum()) if "both_det" in flags else None,
        "accounting_status": "ok" if "either_det" in flags else "no_detection_columns",
    }
    if "either_det" in flags:
        summary["either_detection"] = _flag_accounting(flags["either_det"])
    if "both_det" in flags:
        summary["both_detection"] = _flag_accounting(flags["both_det"])
    if "mag_bin" in df.columns:
        by_mag_bin: dict[str, int] = {}
        by_mag_bin_accounting: dict[str, dict[str, object]] = {}
        for mag_bin, indices in df.groupby("mag_bin", dropna=False).groups.items():
            if "either_det" in flags:
                accounting = _flag_accounting(flags.loc[indices, "either_det"])
                by_mag_bin[str(mag_bin)] = int(accounting["numerator"])
                by_mag_bin_accounting[str(mag_bin)] = accounting
            else:
                by_mag_bin[str(mag_bin)] = 0
                by_mag_bin_accounting[str(mag_bin)] = {
                    "numerator": 0,
                    "denominator": 0,
                    "unknown": int(len(indices)),
                    "fraction": None,
                    "ci95_low": None,
                    "ci95_high": None,
                }
        summary["by_mag_bin"] = by_mag_bin
        summary["by_mag_bin_accounting"] = by_mag_bin_accounting
    return summary


def retention(pre: pd.DataFrame, post: pd.DataFrame, id_col: str = "asas_sn_id") -> dict:
    pre_ids = _extract_ids(pre, id_col) if not pre.empty else pd.Series(dtype="string")
    post_ids = _extract_ids(post, id_col) if not post.empty else pd.Series(dtype="string")
    pre_set = set(pre_ids.astype(str))
    post_set = set(post_ids.astype(str))
    retained = len(pre_set & post_set)
    denominator = len(pre_set)
    unexpected_post = len(post_set - pre_set)
    low, high = _wilson_interval(retained, denominator)
    return {
        "n_pre": int(len(pre)),
        "n_post": int(len(post)),
        "unique_pre": denominator,
        "unique_post": len(post_set),
        "retained": retained,
        "retention_numerator": retained,
        "retention_denominator": denominator,
        "retention_frac": retained / denominator if denominator else None,
        "retention_ci95_low": low,
        "retention_ci95_high": high,
        "unexpected_post": unexpected_post,
        "accounting_status": "ok" if unexpected_post == 0 else "post_contains_unknown_candidates",
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="False-positive reduction summary (tag vs post filter).")
    parser.add_argument("--pre", nargs="+", required=True, help="Tag-stage CSV(s) or directory.")
    parser.add_argument("--post", nargs="+", required=True, help="Post-filter CSV(s) or directory.")
    parser.add_argument("--id-column", default="asas_sn_id", help="ID column for retention match.")
    args = parser.parse_args(argv)

    pre_df = load_many(args.pre)
    post_df = load_many(args.post)

    pre_summary = summarize(pre_df, "pre")
    post_summary = summarize(post_df, "post")
    retain = retention(pre_df, post_df, id_col=args.id_column)

    print("=== Summary ===")
    print(pre_summary)
    print(post_summary)
    print("=== Retention ===")
    print(retain)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
