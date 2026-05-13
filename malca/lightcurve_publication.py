from __future__ import annotations

import argparse
import hashlib
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from malca.lightcurve_io import load_lightcurve_df


JD_OFFSET = 2450000.0

BAND_COLORS = {
    "g": "#238b45",
    "V": "#6a51a3",
    "B": "#2171b5",
    "u": "#08519c",
    "r": "#cb181d",
    "i": "#d94801",
    "z": "#8c6d31",
    "all": "#252525",
    "unknown": "#737373",
}

MARKERS = ("o", "s", "D", "^", "v", "P", "X", "<", ">", "h")

PUBLICATION_STYLE = {
    "font.family": "DejaVu Serif",
    "mathtext.fontset": "dejavuserif",
    "axes.linewidth": 0.8,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

COLUMN_ALIASES = {
    "time": (
        "jd",
        "hjd",
        "bjd",
        "mjd",
        "time",
        "date",
        "julian date",
        "julian_date",
    ),
    "mag": (
        "mag",
        "magnitude",
        "m",
        "mag auto",
        "mag_auto",
        "psfmag",
        "apmag",
        "calmag",
    ),
    "mag_error": (
        "mag error",
        "mag_error",
        "magerr",
        "mag err",
        "magnitude error",
        "e mag",
        "e_mag",
        "emag",
        "error",
        "err",
        "sigma",
        "sigma mag",
        "sigma_mag",
    ),
    "flux": (
        "flux",
        "f",
        "flux density",
        "flux_density",
    ),
    "flux_error": (
        "flux error",
        "flux_error",
        "fluxerr",
        "flux err",
        "e flux",
        "e_flux",
        "eflux",
    ),
    "band": (
        "filter",
        "phot filter",
        "phot_filter",
        "passband",
        "band",
        "filter band",
        "filter_band",
        "v g band",
        "v_g_band",
        "vgband",
    ),
    "camera": (
        "camera",
        "camera#",
        "camera id",
        "camera_id",
        "camera number",
        "camera_number",
        "camera name",
        "camera_name",
        "cam",
        "cam id",
        "cam_id",
    ),
    "quality": (
        "quality",
        "quality flag",
        "quality_flag",
        "good bad",
        "good_bad",
        "goodbad",
        "flag",
    ),
    "saturated": (
        "saturated",
        "sat",
        "saturation",
    ),
}


@dataclass(frozen=True)
class PublicationLightCurve:
    df: pd.DataFrame
    source_path: Path
    time_column: str
    y_kind: str
    y_label: str
    default_invert_y: bool


def _normalize_column_name(name: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


def _column_lookup(df: pd.DataFrame) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for column in df.columns:
        lookup.setdefault(_normalize_column_name(column), str(column))
    return lookup


def _find_column(df: pd.DataFrame, aliases: Sequence[str]) -> str | None:
    lookup = _column_lookup(df)
    for alias in aliases:
        column = lookup.get(_normalize_column_name(alias))
        if column is not None:
            return column
    return None


def _format_band(value: object) -> str:
    if pd.isna(value):
        return "unknown"

    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.notna(numeric) and np.isfinite(float(numeric)):
        rounded = int(round(float(numeric)))
        if abs(float(numeric) - rounded) < 1e-9:
            if rounded == 0:
                return "g"
            if rounded == 1:
                return "V"

    text = str(value).strip()
    if not text:
        return "unknown"
    lower = text.lower()
    if lower == "v":
        return "V"
    if lower in {"g", "u", "r", "i", "z"}:
        return lower
    if lower == "b":
        return "B"
    return text


def _as_boolean_series(values: pd.Series, *, default: bool) -> pd.Series:
    if values.empty:
        return pd.Series(dtype=bool)

    numeric = pd.to_numeric(values, errors="coerce")
    non_null = values.notna()
    if non_null.any() and numeric[non_null].notna().all():
        return numeric.fillna(1 if default else 0).astype(float) != 0.0

    text = values.astype("string").fillna("").str.strip().str.upper()
    true_tokens = {"1", "TRUE", "T", "YES", "Y", "GOOD", "G", "OK", "A"}
    false_tokens = {"0", "FALSE", "F", "NO", "N", "BAD", "B"}
    out = pd.Series(default, index=values.index, dtype=bool)
    out[text.isin(true_tokens)] = True
    out[text.isin(false_tokens)] = False
    return out


def _quality_good_series(values: pd.Series | None, index: pd.Index) -> pd.Series:
    if values is None:
        return pd.Series(True, index=index, dtype=bool)
    return _as_boolean_series(pd.Series(values, index=index), default=True)


def _saturated_series(values: pd.Series | None, index: pd.Index) -> pd.Series:
    if values is None:
        return pd.Series(False, index=index, dtype=bool)
    return _as_boolean_series(pd.Series(values, index=index), default=False)


def _read_csv_frame(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, comment="#", skip_blank_lines=True)


def _load_matplotlib():
    if "MPLCONFIGDIR" not in os.environ:
        default_config = Path.home() / ".config" / "matplotlib"
        if not os.access(default_config, os.W_OK):
            cache_dir = Path(tempfile.gettempdir()) / "malca-matplotlib"
            cache_dir.mkdir(parents=True, exist_ok=True)
            os.environ["MPLCONFIGDIR"] = str(cache_dir)

    import matplotlib.pyplot as plt
    from matplotlib.ticker import AutoMinorLocator

    return plt, AutoMinorLocator


def _read_raw_lightcurve(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".dat", ".dat2", ".dat3"}:
        return load_lightcurve_df(path, file_ext=suffix[1:])
    return _read_csv_frame(path)


def normalize_lightcurve_frame(df: pd.DataFrame, source_path: str | Path) -> PublicationLightCurve:
    """Normalize a generic light-curve table to plotting columns."""
    source = Path(source_path)
    if df.empty:
        raise ValueError(f"Light curve table is empty: {source}")

    time_col = _find_column(df, COLUMN_ALIASES["time"])
    mag_col = _find_column(df, COLUMN_ALIASES["mag"])
    mag_err_col = _find_column(df, COLUMN_ALIASES["mag_error"])
    flux_col = _find_column(df, COLUMN_ALIASES["flux"])
    flux_err_col = _find_column(df, COLUMN_ALIASES["flux_error"])
    band_col = _find_column(df, COLUMN_ALIASES["band"])
    camera_col = _find_column(df, COLUMN_ALIASES["camera"])
    quality_col = _find_column(df, COLUMN_ALIASES["quality"])
    saturated_col = _find_column(df, COLUMN_ALIASES["saturated"])

    if time_col is None:
        raise ValueError(
            "Could not infer a time column. Expected one of: "
            f"{', '.join(COLUMN_ALIASES['time'])}"
        )

    if mag_col is not None:
        y_col = mag_col
        y_err_col = mag_err_col
        y_kind = "mag"
        y_label = "Magnitude [mag]"
        default_invert_y = True
    elif flux_col is not None:
        y_col = flux_col
        y_err_col = flux_err_col
        y_kind = "flux"
        y_label = "Flux"
        default_invert_y = False
    else:
        raise ValueError(
            "Could not infer a magnitude or flux column. Expected one of: "
            f"{', '.join(COLUMN_ALIASES['mag'] + COLUMN_ALIASES['flux'])}"
        )

    out = pd.DataFrame(index=df.index)
    out["time"] = pd.to_numeric(df[time_col], errors="coerce")
    out["value"] = pd.to_numeric(df[y_col], errors="coerce")
    if y_err_col is not None:
        out["value_error"] = pd.to_numeric(df[y_err_col], errors="coerce")
    else:
        out["value_error"] = np.nan

    if band_col is not None:
        out["band"] = df[band_col].map(_format_band)
    else:
        out["band"] = "all"

    if camera_col is not None:
        camera = df[camera_col].astype("string").fillna("").str.strip()
        out["camera"] = camera.where(camera != "", "unknown")
    else:
        out["camera"] = "all"

    out["good_quality"] = _quality_good_series(
        df[quality_col] if quality_col is not None else None,
        df.index,
    )
    out["saturated"] = _saturated_series(
        df[saturated_col] if saturated_col is not None else None,
        df.index,
    )

    return PublicationLightCurve(
        df=out.reset_index(drop=True),
        source_path=source,
        time_column=time_col,
        y_kind=y_kind,
        y_label=y_label,
        default_invert_y=default_invert_y,
    )


def load_lightcurve(path: str | Path) -> PublicationLightCurve:
    """Load SkyPatrol/generic CSV or ASAS-SN dat/dat2/dat3 light curves."""
    lc_path = Path(path).expanduser()
    if not lc_path.exists():
        raise FileNotFoundError(f"Light curve file not found: {lc_path}")
    return normalize_lightcurve_frame(_read_raw_lightcurve(lc_path), lc_path)


def _parse_band_filter(bands: Sequence[str] | None) -> set[str] | None:
    if not bands:
        return None
    return {_format_band(band) for band in bands}


def filter_lightcurve(
    lc: PublicationLightCurve,
    *,
    bands: Sequence[str] | None = None,
    include_bad_quality: bool = False,
    include_saturated: bool = False,
    max_error: float | None = 1.0,
) -> pd.DataFrame:
    """Apply conservative publication-plot filters."""
    df = lc.df.copy()
    mask = np.isfinite(df["time"]) & np.isfinite(df["value"])

    err = pd.to_numeric(df["value_error"], errors="coerce")
    err_valid = np.isfinite(err) & (err > 0)
    df.loc[~err_valid, "value_error"] = np.nan

    if lc.y_kind == "mag" and max_error is not None and np.isfinite(float(max_error)):
        mask &= (~err_valid) | (err <= float(max_error))

    if not include_bad_quality:
        mask &= df["good_quality"].fillna(True).astype(bool)

    if not include_saturated:
        mask &= ~df["saturated"].fillna(False).astype(bool)

    wanted_bands = _parse_band_filter(bands)
    if wanted_bands is not None:
        mask &= df["band"].isin(wanted_bands)

    return df.loc[mask].sort_values("time").reset_index(drop=True)


def _stable_color(label: str) -> str:
    digest = hashlib.md5(str(label).encode("utf-8")).hexdigest()
    hue = int(digest[:8], 16) % 10
    palette = (
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    )
    return palette[hue]


def _stable_marker(label: str) -> str:
    digest = hashlib.md5(str(label).encode("utf-8")).hexdigest()
    return MARKERS[int(digest[:8], 16) % len(MARKERS)]


def _band_sort_key(label: str) -> tuple[int, str]:
    order = {"u": 0, "B": 1, "V": 2, "g": 3, "r": 4, "i": 5, "z": 6, "all": 7}
    return order.get(label, 99), label


def _default_title(path: Path) -> str:
    stem = path.stem
    for suffix in ("-light-curves", "_light_curves", "_lightcurve", "-lightcurve"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def resolve_time_axis(
    time: pd.Series,
    *,
    source_column: str,
    offset: str = "auto",
) -> tuple[pd.Series, str]:
    """Return plotted time values and an axis label."""
    requested = str(offset).strip().lower()
    norm_col = _normalize_column_name(source_column)
    raw = pd.to_numeric(time, errors="coerce")
    finite = raw[np.isfinite(raw)]
    median = float(finite.median()) if not finite.empty else np.nan

    if requested in {"none", "raw", "0"}:
        if "mjd" in norm_col:
            return raw, "MJD"
        if "jd" in norm_col and np.isfinite(median) and median < 100000:
            return raw, "JD - 2450000"
        return raw, f"{source_column} [days]"

    if requested == "auto":
        if "mjd" in norm_col:
            return raw, "MJD"
        if "jd" in norm_col:
            if np.isfinite(median) and median > 2000000:
                return raw - JD_OFFSET, "JD - 2450000"
            return raw, "JD - 2450000"
        return raw, f"{source_column} [days]"

    try:
        numeric_offset = float(requested)
    except ValueError as exc:
        raise ValueError("--time-offset must be 'auto', 'none', or a numeric value") from exc
    return raw - numeric_offset, f"{source_column} - {numeric_offset:g}"


def _group_label(row: pd.Series, group_by: str) -> str:
    if group_by == "none":
        return "all"
    if group_by == "band":
        return str(row["band"])
    if group_by == "camera":
        return str(row["camera"])
    if group_by == "band-camera":
        return f"{row['band']} / {row['camera']}"
    raise ValueError(f"Unknown group_by value: {group_by}")


def _group_color(label: str, group_by: str) -> str:
    if group_by == "band":
        return BAND_COLORS.get(label, _stable_color(label))
    if group_by == "band-camera":
        band = label.split("/", 1)[0].strip()
        return BAND_COLORS.get(band, _stable_color(label))
    return BAND_COLORS.get(label, _stable_color(label))


def plot_lightcurve(
    lc: PublicationLightCurve,
    df: pd.DataFrame,
    *,
    output: str | Path | None = None,
    show: bool = False,
    title: str | None = None,
    figsize: tuple[float, float] = (7.0, 4.2),
    dpi: int = 300,
    group_by: str = "band",
    show_errorbars: bool = True,
    invert_y: bool | None = None,
    time_offset: str = "auto",
    legend: str = "auto",
    marker_size: float = 3.5,
    xlim: Sequence[float] | None = None,
    ylim: Sequence[float] | None = None,
) -> None:
    if df.empty:
        raise ValueError("No light-curve points remain after filtering")

    plt, auto_minor_locator = _load_matplotlib()

    plot_df = df.copy()
    plot_df["time_plot"], x_label = resolve_time_axis(
        plot_df["time"],
        source_column=lc.time_column,
        offset=time_offset,
    )
    plot_df["plot_group"] = plot_df.apply(_group_label, axis=1, group_by=group_by)

    with plt.rc_context(PUBLICATION_STYLE):
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

        groups = sorted(plot_df["plot_group"].dropna().unique(), key=_band_sort_key)
        for label in groups:
            part = plot_df[plot_df["plot_group"] == label]
            if part.empty:
                continue
            color = _group_color(str(label), group_by)
            marker = (
                _stable_marker(str(label))
                if group_by != "band"
                else _stable_marker(str(part["band"].iloc[0]))
            )
            plot_label = None if str(label) == "all" else str(label)
            yerr = part["value_error"].to_numpy()
            has_yerr = show_errorbars and np.isfinite(yerr).any()

            if has_yerr:
                ax.errorbar(
                    part["time_plot"],
                    part["value"],
                    yerr=yerr,
                    fmt=marker,
                    linestyle="none",
                    markersize=marker_size,
                    markeredgecolor="0.15",
                    markeredgewidth=0.35,
                    color=color,
                    ecolor=color,
                    elinewidth=0.55,
                    capsize=1.2,
                    alpha=0.82,
                    label=plot_label,
                )
            else:
                ax.scatter(
                    part["time_plot"],
                    part["value"],
                    s=marker_size**2,
                    marker=marker,
                    linewidths=0.35,
                    edgecolors="0.15",
                    color=color,
                    alpha=0.82,
                    label=plot_label,
                )

        if title is None:
            title = _default_title(lc.source_path)
        if title:
            ax.set_title(title, pad=8)

        ax.set_xlabel(x_label)
        ax.set_ylabel(lc.y_label)
        should_invert_y = invert_y if invert_y is not None else lc.default_invert_y
        if should_invert_y:
            ax.invert_yaxis()

        if xlim is not None:
            ax.set_xlim(float(xlim[0]), float(xlim[1]))
        if ylim is not None:
            ax.set_ylim(float(ylim[0]), float(ylim[1]))

        ax.grid(True, which="major", linewidth=0.4, alpha=0.28)
        ax.xaxis.set_minor_locator(auto_minor_locator())
        ax.yaxis.set_minor_locator(auto_minor_locator())
        ax.tick_params(which="both", direction="in", top=True, right=True)
        ax.tick_params(which="major", length=4.0, width=0.8)
        ax.tick_params(which="minor", length=2.0, width=0.6)

        handles, labels = ax.get_legend_handles_labels()
        if legend != "none" and handles:
            if legend == "outside":
                ax.legend(
                    handles,
                    labels,
                    loc="center left",
                    bbox_to_anchor=(1.02, 0.5),
                    frameon=False,
                )
            else:
                ncol = 1 if len(labels) <= 5 else 2
                ax.legend(handles, labels, loc="best", frameon=False, ncol=ncol)

        if output is not None:
            out_path = Path(output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, dpi=dpi)

        if show:
            plt.show()
        else:
            plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="malca lc-plot",
        description="Create a publication-quality light-curve plot from SkyPatrol/generic CSV or ASAS-SN dat/dat2/dat3 files.",
    )
    parser.add_argument("path", type=Path, help="Light-curve CSV/dat/dat2/dat3 path.")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output figure path. Defaults to '<input>_lightcurve.pdf' unless --show is used.")
    parser.add_argument("--show", action="store_true", help="Display the plot interactively.")
    parser.add_argument("--title", default=None, help="Figure title. Pass an empty string to suppress the title.")
    parser.add_argument("--figsize", nargs=2, type=float, metavar=("WIDTH", "HEIGHT"), default=(7.0, 4.2), help="Figure size in inches.")
    parser.add_argument("--dpi", type=int, default=300, help="Raster output DPI.")
    parser.add_argument("--group-by", choices=("band", "camera", "band-camera", "none"), default="band", help="Legend/grouping dimension.")
    parser.add_argument("--bands", nargs="+", default=None, help="Only plot these bands, e.g. --bands g V.")
    parser.add_argument("--include-bad-quality", action="store_true", help="Include points flagged as bad quality.")
    parser.add_argument("--include-saturated", action="store_true", help="Include saturated points.")
    parser.add_argument("--max-error", type=float, default=1.0, help="Drop points with uncertainty above this value. Use --no-max-error to disable.")
    parser.add_argument("--no-max-error", action="store_true", help="Disable uncertainty filtering.")
    parser.add_argument("--no-errorbars", action="store_true", help="Plot markers without error bars.")
    parser.add_argument("--no-invert-y", action="store_true", help="Do not invert the y-axis for magnitudes.")
    parser.add_argument("--time-offset", default="auto", help="Time offset to subtract: auto, none, or a numeric value.")
    parser.add_argument("--legend", choices=("auto", "outside", "none"), default="auto", help="Legend placement.")
    parser.add_argument("--markersize", type=float, default=3.5, help="Marker size in points.")
    parser.add_argument("--xlim", nargs=2, type=float, metavar=("MIN", "MAX"), default=None, help="X-axis limits after time offset.")
    parser.add_argument("--ylim", nargs=2, type=float, metavar=("MIN", "MAX"), default=None, help="Y-axis limits.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    lc = load_lightcurve(args.path)
    max_error = None if args.no_max_error else args.max_error
    plot_df = filter_lightcurve(
        lc,
        bands=args.bands,
        include_bad_quality=args.include_bad_quality,
        include_saturated=args.include_saturated,
        max_error=max_error,
    )
    if plot_df.empty:
        raise SystemExit("No finite light-curve points remain after filtering.")

    output = args.output
    if output is None and not args.show:
        output = args.path.with_name(f"{args.path.stem}_lightcurve.pdf")

    invert_y = False if args.no_invert_y else None
    plot_lightcurve(
        lc,
        plot_df,
        output=output,
        show=args.show,
        title=args.title,
        figsize=(float(args.figsize[0]), float(args.figsize[1])),
        dpi=args.dpi,
        group_by=args.group_by,
        show_errorbars=not args.no_errorbars,
        invert_y=invert_y,
        time_offset=args.time_offset,
        legend=args.legend,
        marker_size=args.markersize,
        xlim=args.xlim,
        ylim=args.ylim,
    )

    band_summary = ", ".join(sorted(plot_df["band"].dropna().astype(str).unique(), key=_band_sort_key))
    print(
        f"Plotted {len(plot_df)}/{len(lc.df)} points from {args.path}"
        + (f" in bands: {band_summary}" if band_summary else "")
    )
    if output is not None:
        print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
