#!/usr/bin/env python
"""Build a visually curated, morphology-grouped PDF for July 1 dippers."""
from __future__ import annotations

import argparse
import math
import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import plot_march18_review_lightcurve_grids as lightcurve_grids
from malca.review.lightcurve_sources import load_external_lc_frame


RUN_ROOT = Path("output/runs/dat3-full-extended_2026-07-01-v4")
DEFAULT_REVIEW_DB = RUN_ROOT / "review" / "review.db"
DEFAULT_OUTPUT_PDF = Path("output/pdf/july1_dippers_grouped_morphology.pdf")
DEFAULT_OUTPUT_MANIFEST = Path("output/pdf/july1_dippers_grouped_morphology_manifest.csv")
NEW_ONLY_OUTPUT_DIR = Path("output/pdf/new_no_external_yso_or_dipper")
NEW_ONLY_OUTPUT_PDF = NEW_ONLY_OUTPUT_DIR / "july1_dippers_no_external_yso_or_dipper.pdf"
NEW_ONLY_OUTPUT_MANIFEST = (
    NEW_ONLY_OUTPUT_DIR / "july1_dippers_no_external_yso_or_dipper_manifest.csv"
)

ROWS = 5
COLS = 3
SINGLE_EPISODE_ROWS = 8
SINGLE_EPISODE_COLS = 6
RECURRENT_ROWS = 5
RECURRENT_COLS = 6
COMPLEX_ROWS = 4
COMPLEX_COLS = 2
AMBIGUOUS_ROWS = 3
AMBIGUOUS_COLS = 2
LANDSCAPE_PAGE_WIDTH = 11.0
LANDSCAPE_PAGE_HEIGHT = 8.5
NEOWISE_W1_COLOR = "#2474b5"
NEOWISE_W2_COLOR = "#e67e22"


FAMILIES: tuple[dict[str, object], ...] = (
    {
        "key": "single_dominant_dimming_episode",
        "label": "A. Single dominant dimming episode",
        "description": "One dominant event, from compact V-shaped dip to extended low state.",
        "candidate_ids": (
            "stv_111670309173",
            "stv_120259148405",
            "stv_120259356690",
            "stv_17181027305",
            "stv_197569226514",
            "stv_240518717560",
            "stv_266288257407",
            "stv_283468165807",
            "stv_360777826205",
            "stv_369367304600",
            "stv_42950519514",
            "stv_472446832201",
            "stv_532576256705",
            "stv_566936725849",
            "stv_601295730966",
            "stv_601295761416",
            "stv_635655520796",
            "stv_94489594805",
            "stv_300648087617",
            "stv_498216222923",
            "stv_8591303502",
            "stv_292059016008",
            "stv_369367234804",
            "stv_541166985810",
            "stv_60130141761",
            "stv_68720714610",
            "stv_206158525635",
            "stv_566936751170",
            "stv_180388903123",
            "stv_369368258528",
            "stv_403727513981",
            "stv_17180437911",
            "stv_223338997633",
            "stv_283467842509",
            "stv_420907788679",
            "stv_541166181486",
            "stv_609885850038",
            "stv_627065319105",
            "stv_94489437356",
            "stv_249109213616",
            "stv_68719676517",
            "stv_489626538045",
            "stv_111669557747",
            "stv_197569238413",
            "stv_403727411589",
            "stv_523986354332",
            "stv_618475663505",
            "stv_644245359876",
            # User-curated single-dominant assignments, 2026-07-26.
            "stv_197569146752",  # J084053+024045
            "stv_283468931829",  # J012333-734458
            "stv_446677541838",  # J072651-503443
            "stv_635655213015",  # J112109-603213
            "stv_652835553348",  # J100825-551449
            "stv_103079502524",  # J211600+541527
            "stv_120259384073",  # J003526+621516
            "stv_463857379909",  # J073911-054434
            "stv_8590787268",  # J221345+372653
            "stv_111670444883",  # J035431+345506
            "stv_17180004254",  # J033439+683106
            "stv_214748985701",  # J190017+171438
            "stv_317828613008",  # J100417+850811
            "stv_446676962403",  # J010513-722906
            "stv_515396599194",  # J071008-231640
            "stv_111669305159",  # J013331+615404
            "stv_146029052740",  # J225106+573321
            "stv_146029595645",  # J003305+603202
            "stv_214748665650",  # J054604+272730
            "stv_395137536331",  # J175159+115949
            "stv_8591170248",  # J052443+354004
            "stv_111669455609",  # J213239+483602
        ),
    },
    {
        "key": "recurrent_or_stochastic",
        "label": "B. Recurrent / stochastic multi-dip behavior",
        "description": "Two or more dips, multi-depth events, or persistent irregular downward excursions.",
        "candidate_ids": (
            "stv_111669291649",
            "stv_128850429575",
            "stv_180388640882",
            "stv_240519504803",
            "stv_300648040390",
            "stv_300648890329",
            "stv_463856750690",
            "stv_463857647562",
            "stv_481036839646",
            "stv_489626566903",
            "stv_60130131403",
            "stv_163209415214",
            "stv_240518636016",
            "stv_25770316308",
            "stv_395137575008",
            "stv_446676921101",
            "stv_549755992463",
            "stv_652835964994",
            "stv_111669273145",
            "stv_146029419304",
            "stv_17180374105",
            "stv_25771086021",
            "stv_369367489518",
            "stv_446677119900",
            "stv_481036586933",
            "stv_515396131751",
            # User-curated recurrent/stochastic assignments, 2026-07-26.
            "stv_386547548488",  # J073531-274243
            "stv_103080542701",  # J223555+570510
            "stv_154618944199",  # J194839+300241
            "stv_386547717669",  # J193030+142854
            "stv_515396303780",  # J074300-200120
            "stv_566936811310",  # J073834-315957
            "stv_609885931971",  # J125207-615636
            "stv_352187778169",  # J192536-131107
            "stv_369368019182",  # J064401-254201
            "stv_395137147332",  # J072525-033530
            "stv_609885930304",  # J114360-615627
            "stv_644245387906",  # J181752-580749
            "stv_111670547411",  # J223832+583831
            "stv_137440061640",  # J042228+294721
            "stv_360777789109",  # J063929+113207
            "stv_463856558214",  # J010207-714116
            "stv_601296234211",  # J162745-520406
            "stv_188979142258",  # J211656+280302
            "stv_266289132701",  # J114534-292211
            "stv_481036753007",  # J180307-395154
            "stv_558346776813",  # J142018-233455
            "stv_438087432036",  # J083744-401622
            "stv_429497765184",  # J175713-273841
            "stv_111669512802",  # J000154+330250
        ),
    },
    {
        "key": "structured_dimming",
        "label": "C. Structured dimming",
        "description": "A long-term trend combined with dips, multiple states, or strongly structured events.",
        "candidate_ids": (
            "stv_163209590869",
            "stv_523987067704",
            "stv_584116028406",
            "stv_592705538522",
            "stv_154620038181",
            "stv_395138016907",
            "stv_592705518006",
            "stv_644245286164",
        ),
    },
    {
        "key": "ambiguous_or_low_snr",
        "label": "D. Ambiguous / low-SNR morphology",
        "description": "No stable shape assignment from the survey light curve alone; retain for follow-up.",
        "candidate_ids": (
            "stv_532576054353",
            "stv_446677131304",
            "stv_498217396542",
            "stv_77309980503",
            "stv_94489786439",
        ),
    },
)

# The entries above are a frozen visual grouping of the original reviewed
# cohort.  Later reviewed dippers must remain visible in the atlas, but are
# deliberately not auto-assigned to one of those science-facing families.
UNCURATED_FAMILY = {
    "key": "not_yet_morphology_curated",
    "label": "E. Newly added - morphology not curated",
    "description": (
        "Reviewed dippers added after the manual visual grouping; shown here "
        "without a morphology-family assignment."
    ),
}

USER_CURATED_ATLAS_ASSIGNMENTS_20260726 = frozenset(
    FAMILIES[0]["candidate_ids"][-22:] + FAMILIES[1]["candidate_ids"][-24:]
)

EXTERNAL_CLASS_COLUMNS: tuple[tuple[str, str], ...] = (
    ("ASAS-SN", "asassn_var_type"),
    ("VSX", "vsx_class"),
    ("Gaia", "gaia_var_class"),
    ("SIMBAD", "simbad_otype"),
    ("ZTF", "ztf_var_type"),
)
EXTERNAL_YOUNG_DIPPER_TOKENS = {
    "CTTS",
    "DIP",
    "DIPPER",
    "DYPER",
    "TTS",
    "UXOR",
    "WTTS",
    "YSO",
}
SIMBAD_YOUNG_DIPPER_TYPES = {"OR*", "TT*", "Y*O"}


def _read_dippers(review_db: Path) -> pd.DataFrame:
    query = """
        SELECT
            r.candidate_id,
            lower(r.event_class) AS event_class,
            r.classification_confidence,
            r.morphology_primary,
            r.morphology_secondary,
            r.morphology_secondary_json,
            c.asassn_var_type,
            c.vsx_class,
            c.gaia_var_class,
            c.simbad_otype,
            c.ztf_var_type,
            c.lc_path,
            c.asas_sn_id,
            coalesce(
                c.ra,
                json_extract(c.payload_json, '$.ra'),
                json_extract(json_extract(c.payload_json, '$.payload_json'), '$.ra')
            ) AS ra,
            coalesce(
                c.dec,
                json_extract(c.payload_json, '$.dec'),
                json_extract(json_extract(c.payload_json, '$.payload_json'), '$.dec')
            ) AS dec,
            c.dip_best_t0,
            c.jump_best_t0
        FROM reviews AS r
        JOIN candidates AS c USING(candidate_id)
        WHERE lower(coalesce(r.event_class, '')) = 'dipper'
        ORDER BY coalesce(r.classification_confidence, 0) DESC, r.candidate_id
    """
    with sqlite3.connect(review_db) as conn:
        rows = pd.read_sql_query(query, conn)
    rows["candidate_id"] = rows["candidate_id"].astype(str)
    rows["event_class"] = rows["event_class"].astype(str)
    return rows


def _normalized_external_class(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null", "unknown", "-", "--"}:
        return ""
    return text.upper()


def _is_external_yso_or_dipper_label(column: str, value: object) -> bool:
    text = _normalized_external_class(value)
    if not text:
        return False
    if column == "simbad_otype":
        return text in SIMBAD_YOUNG_DIPPER_TYPES
    tokens = {token for token in re.split(r"[^A-Z0-9]+", text) if token}
    return bool(tokens & EXTERNAL_YOUNG_DIPPER_TOKENS)


def _external_yso_or_dipper_evidence(row: pd.Series) -> str:
    evidence: list[str] = []
    for catalog, column in EXTERNAL_CLASS_COLUMNS:
        value = _normalized_external_class(row.get(column))
        if _is_external_yso_or_dipper_label(column, value):
            evidence.append(f"{catalog}:{value}")
    return "; ".join(evidence)


def _annotate_external_yso_or_dipper_evidence(rows: pd.DataFrame) -> pd.DataFrame:
    rows = rows.copy()
    evidence = rows.apply(_external_yso_or_dipper_evidence, axis=1)
    rows["external_yso_or_dipper_evidence"] = evidence
    rows["has_external_yso_or_dipper_label"] = evidence.ne("")
    return rows


def _family_assignment() -> tuple[dict[str, str], dict[str, int]]:
    assignment: dict[str, str] = {}
    family_rank: dict[str, int] = {}
    duplicates: list[str] = []
    for rank, family in enumerate(FAMILIES):
        key = str(family["key"])
        family_rank[key] = rank
        for candidate_id in family["candidate_ids"]:
            candidate_id = str(candidate_id)
            if candidate_id in assignment:
                duplicates.append(candidate_id)
            assignment[candidate_id] = key
    if duplicates:
        raise ValueError(f"Candidates assigned to multiple families: {sorted(set(duplicates))}")
    return assignment, family_rank


def _families_for_live_rows(
    rows: pd.DataFrame,
    assignment: dict[str, str],
) -> tuple[tuple[dict[str, object], ...], dict[str, int]]:
    live_ids = set(rows["candidate_id"])
    assigned_ids = set(assignment)
    uncurated_ids = tuple(sorted(live_ids - assigned_ids))

    families = [
        {
            **family,
            "candidate_ids": tuple(
                candidate_id
                for candidate_id in family["candidate_ids"]
                if candidate_id in live_ids
            ),
        }
        for family in FAMILIES
    ]
    families = [family for family in families if family["candidate_ids"]]
    if uncurated_ids:
        families.append({**UNCURATED_FAMILY, "candidate_ids": uncurated_ids})

    for candidate_id in uncurated_ids:
        assignment[candidate_id] = str(UNCURATED_FAMILY["key"])

    family_rank = {
        str(family["key"]): rank for rank, family in enumerate(families)
    }
    return tuple(families), family_rank


def _read_neowise_paths(run_root: Path) -> dict[str, Path]:
    results_root = run_root / "results"
    manifest_path = results_root / "external_lc_manifest.parquet"
    if not manifest_path.exists():
        return {}
    manifest = pd.read_parquet(manifest_path)
    required = {"candidate_id", "file_prefix"}
    if not required.issubset(manifest.columns):
        return {}
    prefix = manifest["file_prefix"].astype(str).str.strip().str.lower()
    manifest = manifest[prefix.isin({"neowise", "neowise_w1", "neowise_w2"})].copy()
    if "updated_unix" in manifest.columns:
        manifest = manifest.sort_values("updated_unix")
    manifest = manifest.drop_duplicates("candidate_id", keep="last")

    paths: dict[str, Path] = {}
    for _, row in manifest.iterrows():
        candidate_id = str(row.get("candidate_id") or "").strip()
        if not candidate_id:
            continue
        path_value = row.get("path")
        path = Path(str(path_value)).expanduser() if pd.notna(path_value) else Path()
        if not path.is_file() and "path_relative" in manifest.columns:
            relative_value = row.get("path_relative")
            if pd.notna(relative_value):
                path = results_root / str(relative_value)
        if path.is_file():
            paths[candidate_id] = path.resolve()
    return paths


def _plot_neowise_overlay(
    ax,
    *,
    path: Path | None,
    compact: bool,
) -> dict[str, object]:
    status: dict[str, object] = {
        "neowise_status": "missing",
        "neowise_path": "" if path is None else str(path),
        "neowise_w1_points": 0,
        "neowise_w2_points": 0,
        "neowise_relative_scale_mag": np.nan,
    }
    if path is None or not path.is_file():
        return status

    frame = load_external_lc_frame("neowise", path)
    if frame.empty or "mjd" not in frame.columns:
        status["neowise_status"] = "empty"
        return status

    mjd = pd.to_numeric(frame["mjd"], errors="coerce").to_numpy(dtype=float)
    if np.isfinite(mjd).any() and float(np.nanmedian(mjd)) > 1_000_000.0:
        x = mjd - 2_458_000.0
    else:
        x = mjd - 57_999.5

    series: list[tuple[str, str, str, np.ndarray, np.ndarray, np.ndarray | None]] = []
    all_centered: list[np.ndarray] = []
    for band, color, marker in (
        ("W1", NEOWISE_W1_COLOR, "o"),
        ("W2", NEOWISE_W2_COLOR, "s"),
    ):
        key = band.lower()
        mag_col = f"{key}mpro"
        err_col = f"{key}sigmpro"
        if mag_col not in frame.columns:
            continue
        mag = pd.to_numeric(frame[mag_col], errors="coerce").to_numpy(dtype=float)
        good = np.isfinite(x) & np.isfinite(mag)
        if not np.any(good):
            continue
        x_band = x[good]
        mag_band = mag[good]
        centered = mag_band - float(np.nanmedian(mag_band))
        err_band: np.ndarray | None = None
        if err_col in frame.columns:
            err = pd.to_numeric(frame[err_col], errors="coerce").to_numpy(dtype=float)[good]
            if np.isfinite(err).any():
                err_band = err
        status[f"neowise_{key}_points"] = int(len(centered))
        series.append((band, color, marker, x_band, centered, err_band))
        all_centered.append(centered)

    if not series:
        status["neowise_status"] = "empty"
        return status

    ir_ax = ax.twinx()
    ir_ax.patch.set_visible(False)
    ir_ax.set_zorder(ax.get_zorder() + 1)
    marker_size = 1.15 if compact else 2.1
    for band, color, marker, x_band, centered, err_band in series:
        if err_band is not None:
            valid_err = np.isfinite(err_band) & (err_band > 0)
            if np.any(valid_err):
                ir_ax.errorbar(
                    x_band[valid_err],
                    centered[valid_err],
                    yerr=err_band[valid_err],
                    fmt="none",
                    ecolor=color,
                    elinewidth=0.14 if compact else 0.25,
                    capsize=0.35 if compact else 0.65,
                    capthick=0.14 if compact else 0.25,
                    alpha=0.45,
                    zorder=4,
                )
        ir_ax.scatter(
            x_band,
            centered,
            s=marker_size,
            marker=marker,
            color=color,
            linewidths=0,
            alpha=0.82,
            rasterized=False,
            zorder=5,
        )

    centered_values = np.concatenate(all_centered)
    scale = max(0.08, 1.08 * float(np.nanmax(np.abs(centered_values))))
    ir_ax.set_ylim(scale, -scale)
    status["neowise_relative_scale_mag"] = scale

    finite_x = np.concatenate([item[3][np.isfinite(item[3])] for item in series])
    if finite_x.size:
        optical_xlim = ax.get_xlim()
        xmin = min(float(optical_xlim[0]), float(np.nanmin(finite_x)))
        xmax = max(float(optical_xlim[1]), float(np.nanmax(finite_x)))
        xpad = max(3.0, 0.005 * (xmax - xmin))
        ax.set_xlim(xmin - xpad, xmax + xpad)

    ir_ax.set_yticks([])
    ir_ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
    for spine in ir_ax.spines.values():
        spine.set_visible(False)
    status["neowise_status"] = "ok"
    return status


def _add_survey_legend(
    fig,
    *,
    morphology_title: str,
    compact: bool,
) -> None:
    fig.text(
        0.5,
        0.997,
        morphology_title,
        ha="center",
        va="top",
        fontsize=5.8 if compact else 6.7,
        color="0.10",
    )
    handles = (
        Line2D([], [], color="black", marker="o", linestyle="None", markersize=3.2),
        Line2D([], [], color=NEOWISE_W1_COLOR, marker="o", linestyle="None", markersize=3.2),
        Line2D([], [], color=NEOWISE_W2_COLOR, marker="s", linestyle="None", markersize=3.2),
    )
    legend = fig.legend(
        handles,
        ("ASAS-SN", "NEOWISE W1", "NEOWISE W2"),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.980),
        ncol=3,
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        edgecolor="black",
        fontsize=5.1 if compact else 5.8,
        handlelength=0.7,
        handletextpad=0.35,
        columnspacing=0.9,
        borderpad=0.35,
    )
    legend.get_frame().set_linewidth(0.55)


def _write_group_page(
    pdf,
    page_rows: pd.DataFrame,
    *,
    family: dict[str, object],
    pdf_page: int,
    total_pdf_pages: int,
    run_root: Path,
    rows: int,
    cols: int,
    page_width: float,
    page_height: float,
    compact: bool,
    neowise_paths: dict[str, Path],
) -> list[dict[str, object]]:
    fig, statuses = lightcurve_grids._make_page_figure(
        page_rows,
        event_class="dipper",
        page_number=pdf_page,
        n_pages=total_pdf_pages,
        rows=rows,
        cols=cols,
        run_root=run_root,
        page_width=page_width,
        page_height=page_height,
        min_active_rows=rows,
    )
    subplot_left = 0.047 if compact else lightcurve_grids.SUBPLOT_LEFT
    subplot_bottom = 0.057 if compact else lightcurve_grids.SUBPLOT_BOTTOM
    subplot_top = 0.905 if compact else 0.908
    subplot_wspace = 0.10 if compact else lightcurve_grids.SUBPLOT_WSPACE
    subplot_hspace = 0.16 if compact else lightcurve_grids.SUBPLOT_HSPACE
    fig.subplots_adjust(
        left=subplot_left,
        right=lightcurve_grids.SUBPLOT_RIGHT,
        bottom=subplot_bottom,
        top=subplot_top,
        wspace=subplot_wspace,
        hspace=subplot_hspace,
    )
    primary_axes = list(fig.axes)
    if compact:
        for ax in primary_axes:
            ax.tick_params(
                which="major",
                labelsize=3.15,
                length=2.2,
                width=0.42,
                pad=0.7,
            )
            ax.tick_params(which="minor", length=1.2, width=0.35)
            for spine in ax.spines.values():
                spine.set_linewidth(0.45)
            for text in ax.texts:
                text.set_fontsize(3.0)
            for collection in ax.collections:
                if hasattr(collection, "get_sizes"):
                    sizes = collection.get_sizes()
                    if len(sizes):
                        collection.set_sizes(sizes * 0.42)
                collection.set_linewidth(0.16)
            for line in ax.lines:
                line.set_linewidth(min(line.get_linewidth(), 0.35))
    for ax, (_, row), status in zip(
        primary_axes[: len(page_rows)],
        page_rows.iterrows(),
        statuses,
        strict=True,
    ):
        candidate_id = str(row["candidate_id"])
        status.update(
            _plot_neowise_overlay(
                ax,
                path=neowise_paths.get(candidate_id),
                compact=compact,
            )
        )
    for text in fig.texts:
        if text.get_text() == "JD - 2458000 [d]":
            text.set_position((0.525, 0.016 if compact else 0.022))
            if compact:
                text.set_fontsize(6.2)
        elif text.get_text() == "m [mag]":
            text.set_position((lightcurve_grids.Y_LABEL_X, 0.50))
            if compact:
                text.set_fontsize(6.2)
    _add_survey_legend(
        fig,
        morphology_title=str(family["label"]),
        compact=compact,
    )
    pdf.savefig(fig, dpi=300)
    lightcurve_grids.plt.close(fig)
    return statuses


def _family_layout(family_key: str) -> dict[str, object]:
    if family_key == "single_dominant_dimming_episode":
        return {
            "rows": SINGLE_EPISODE_ROWS,
            "cols": SINGLE_EPISODE_COLS,
            "page_width": LANDSCAPE_PAGE_WIDTH,
            "page_height": LANDSCAPE_PAGE_HEIGHT,
            "compact": True,
        }
    if family_key == "recurrent_or_stochastic":
        return {
            "rows": RECURRENT_ROWS,
            "cols": RECURRENT_COLS,
            "page_width": LANDSCAPE_PAGE_WIDTH,
            "page_height": LANDSCAPE_PAGE_HEIGHT,
            "compact": True,
        }
    if family_key == "structured_dimming":
        return {
            "rows": COMPLEX_ROWS,
            "cols": COMPLEX_COLS,
            "page_width": LANDSCAPE_PAGE_WIDTH,
            "page_height": LANDSCAPE_PAGE_HEIGHT,
            "compact": False,
        }
    if family_key == "ambiguous_or_low_snr":
        return {
            "rows": AMBIGUOUS_ROWS,
            "cols": AMBIGUOUS_COLS,
            "page_width": LANDSCAPE_PAGE_WIDTH,
            "page_height": LANDSCAPE_PAGE_HEIGHT,
            "compact": False,
        }
    return {
        "rows": ROWS,
        "cols": COLS,
        "page_width": lightcurve_grids.DEFAULT_FIG_WIDTH,
        "page_height": lightcurve_grids.DEFAULT_FIG_HEIGHT,
        "compact": False,
    }


def build_grouped_pdf(
    *,
    review_db: Path,
    run_root: Path,
    output_pdf: Path,
    output_manifest: Path,
    new_only: bool = False,
) -> pd.DataFrame:
    rows = _read_dippers(review_db)
    rows = _annotate_external_yso_or_dipper_evidence(rows)
    if new_only:
        rows = rows.loc[~rows["has_external_yso_or_dipper_label"]].copy()
        if rows.empty:
            raise ValueError("No reviewed dippers remain after the external YSO/dipper filter.")
    assignment, _ = _family_assignment()
    families, family_rank = _families_for_live_rows(rows, assignment)
    neowise_paths = _read_neowise_paths(run_root)

    rows["morphology_family"] = rows["candidate_id"].map(assignment)
    rows["family_rank"] = rows["morphology_family"].map(family_rank)
    rows = rows.sort_values(
        ["family_rank", "classification_confidence", "candidate_id"],
        ascending=[True, False, True],
    ).reset_index(drop=True)

    grid_pages = 0
    for family in families:
        layout = _family_layout(str(family["key"]))
        family_page_size = int(layout["rows"]) * int(layout["cols"])
        family_rows = rows[rows["morphology_family"] == str(family["key"])]
        grid_pages += math.ceil(len(family_rows) / family_page_size)
    total_pdf_pages = grid_pages
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, object]] = []
    pdf_page = 0
    with lightcurve_grids.PdfPages(
        output_pdf,
        metadata={
            "Title": (
                "July 1 dippers without external YSO/dipper labels"
                if new_only
                else "July 1 dippers grouped by light-curve morphology"
            ),
            "Author": "MALCA",
            "Subject": (
                "Manual morphology grouping of the original reviewed dipper cohort; "
                "later additions are explicitly uncurated"
            ),
            "Creator": "MALCA",
        },
    ) as pdf:
        for family in families:
            family_key = str(family["key"])
            family_rows = rows[rows["morphology_family"] == family_key].copy()
            layout = _family_layout(family_key)
            family_page_size = int(layout["rows"]) * int(layout["cols"])
            for family_page, start in enumerate(
                range(0, len(family_rows), family_page_size), start=1
            ):
                pdf_page += 1
                page_rows = family_rows.iloc[start : start + family_page_size]
                statuses = _write_group_page(
                    pdf,
                    page_rows,
                    family=family,
                    pdf_page=pdf_page,
                    total_pdf_pages=total_pdf_pages,
                    run_root=run_root,
                    rows=int(layout["rows"]),
                    cols=int(layout["cols"]),
                    page_width=float(layout["page_width"]),
                    page_height=float(layout["page_height"]),
                    compact=bool(layout["compact"]),
                    neowise_paths=neowise_paths,
                )
                for panel, ((_, row), status) in enumerate(
                    zip(page_rows.iterrows(), statuses, strict=True), start=1
                ):
                    manifest_rows.append(
                        {
                            "candidate_id": row["candidate_id"],
                            "morphology_family": family_key,
                            "family_label": family["label"],
                            "family_description": family["description"],
                            "family_page": family_page,
                            "pdf_page": pdf_page,
                            "panel": panel,
                            "classification_confidence": row["classification_confidence"],
                            "existing_morphology_secondary": row["morphology_secondary"],
                            "existing_morphology_secondary_json": row[
                                "morphology_secondary_json"
                            ],
                            "assignment_basis": (
                                "user_curated_atlas_assignment_2026-07-26"
                                if row["candidate_id"]
                                in USER_CURATED_ATLAS_ASSIGNMENTS_20260726
                                else "manual_visual_review_2026-07-21"
                                if family_key != str(UNCURATED_FAMILY["key"])
                                else "not_yet_morphology_curated"
                            ),
                            "selection": (
                                "no_direct_external_yso_or_dipper_label"
                                if new_only
                                else "all_reviewed_dippers"
                            ),
                            "external_yso_or_dipper_evidence": row[
                                "external_yso_or_dipper_evidence"
                            ],
                            **status,
                        }
                    )

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(output_manifest, index=False)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-db", type=Path, default=DEFAULT_REVIEW_DB)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--output-pdf", type=Path)
    parser.add_argument("--output-manifest", type=Path)
    parser.add_argument(
        "--new-only",
        action="store_true",
        help=(
            "Export only reviewed dippers without a direct external YSO/dipper-like "
            "class in ASAS-SN, VSX, Gaia, SIMBAD, or ZTF."
        ),
    )
    args = parser.parse_args()

    output_pdf = args.output_pdf or (
        NEW_ONLY_OUTPUT_PDF if args.new_only else DEFAULT_OUTPUT_PDF
    )
    output_manifest = args.output_manifest or (
        NEW_ONLY_OUTPUT_MANIFEST if args.new_only else DEFAULT_OUTPUT_MANIFEST
    )

    manifest = build_grouped_pdf(
        review_db=args.review_db,
        run_root=args.run_root,
        output_pdf=output_pdf,
        output_manifest=output_manifest,
        new_only=args.new_only,
    )
    counts = manifest.groupby("morphology_family").size().to_dict()
    print(f"Wrote grouped dipper PDF: {output_pdf}")
    print(f"Wrote grouping manifest: {output_manifest}")
    print(f"Candidate count: {len(manifest)}")
    print(f"Family counts: {counts}")


if __name__ == "__main__":
    main()
