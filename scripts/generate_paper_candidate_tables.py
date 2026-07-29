#!/usr/bin/env python
"""Export publication tables for reviewed dipper, LTV, and microlensing candidates.

The default output is a combined, class-sectioned LaTeX ``longtable`` plus one
standalone LaTeX table per class.  Matching CSV files retain identifiers and
provenance columns that are useful for checking the paper values.

Examples
--------
Generate the current July 1 paper tables::

    conda run -n malca python scripts/generate_paper_candidate_tables.py

Use a different review database and output directory::

    conda run -n malca python scripts/generate_paper_candidate_tables.py \
        --db output/runs/my_run/review/review.db \
        --output-dir output/runs/my_run/results/paper_candidate_tables
"""

from __future__ import annotations

import argparse
import math
import re
import sqlite3
import sys
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from malca.ltv.cmd import dustmaps_cmd_from_fields  # noqa: E402
from malca.review.coordinate_labels import format_j_designation, payload_ra_dec  # noqa: E402
from malca.review.paper_candidates import build_publication_cohort  # noqa: E402
from malca.review.store import get_candidate_payload  # noqa: E402


DEFAULT_DB = (
    REPO_ROOT
    / "output"
    / "runs"
    / "dat3-full-extended_2026-07-01-v4"
    / "review"
    / "review.db"
)
DEFAULT_OUTPUT_DIR = DEFAULT_DB.parents[1] / "results" / "paper_candidate_tables"

CLASS_ORDER = ("dipper", "ltv", "microlensing")
CLASS_TITLES = {
    "dipper": "Dipper",
    "ltv": "Long-term-variable",
    "microlensing": "Microlensing",
}

LATEX_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("source", "Source", ""),
    ("ra_deg", r"R.A.", r"[deg]"),
    ("dec_deg", r"Decl.", r"[deg]"),
    ("search_method", "Search method", ""),
    ("aligned_asassn_mean_mag", r"$\langle m_{\rm ASAS\text{-}SN}\rangle$", r"[mag]"),
    ("absolute_g_mag", r"$M_G$", r"[mag]"),
    ("bp_rp_mag", r"$G_{\rm BP}-G_{\rm RP}$", r"[mag]"),
    ("rv_amplitude_kms", r"$RV_{\rm amp}$", r"[km s$^{-1}$]"),
    ("ruwe", "RUWE", ""),
    ("distance_pc", "Distance", r"[pc]"),
)
LATEX_ALIGNMENT = "lrrlrrrrrr"


def finite_number(value: object) -> float | None:
    """Return a finite float, treating common serialized missing values as null."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "nan", "none", "null", "<na>"}:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def first_finite(payload: Mapping[str, object], keys: Iterable[str]) -> float | None:
    for key in keys:
        value = finite_number(payload.get(key))
        if value is not None:
            return value
    return None


def clean_identifier(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null", "<na>"}:
        return None
    return text


def select_distance(payload: Mapping[str, object]) -> tuple[float | None, str | None]:
    """Choose a paper distance without making a network query.

    Existing posterior distance products take precedence, followed by the
    distance used by the dust calculation.  A naive inverse-parallax estimate
    is deliberately not manufactured here: it is biased/unstable at modest
    signal-to-noise and has no faithful uncertainty in this table.
    """
    choices = (
        ("bj_r_med_photogeo", "Bailer-Jones photogeometric"),
        ("bj_r_med_geo", "Bailer-Jones geometric"),
        ("distance_gspphot", "Gaia GSP-Phot"),
        ("dust_distance_pc", "dust-distance input"),
        ("distance_pc", "stored distance"),
        ("dist_pc", "stored distance"),
    )
    for key, label in choices:
        distance = finite_number(payload.get(key))
        if distance is not None and distance > 0:
            return distance, label

    return None, None


def select_distance_with_uncertainty(payload: Mapping[str, object]) -> dict[str, object]:
    """Select a distance and carry its matching interval/provenance fields."""
    distance, source = select_distance(payload)
    lower = upper = None
    uncertainty_source = None
    if source == "Bailer-Jones photogeometric":
        lower = first_finite(payload, ("bj_r_lo_photogeo", "r_lo_photogeo"))
        upper = first_finite(payload, ("bj_r_hi_photogeo", "r_hi_photogeo"))
        uncertainty_source = "Bailer-Jones photogeometric interval" if lower is not None or upper is not None else None
    elif source == "Bailer-Jones geometric":
        lower = first_finite(payload, ("bj_r_lo_geo", "r_lo_geo"))
        upper = first_finite(payload, ("bj_r_hi_geo", "r_hi_geo"))
        uncertainty_source = "Bailer-Jones geometric interval" if lower is not None or upper is not None else None
    elif source == "Gaia GSP-Phot":
        lower = first_finite(payload, ("distance_gspphot_lower", "distance_gspphot_lo"))
        upper = first_finite(payload, ("distance_gspphot_upper", "distance_gspphot_hi"))
        uncertainty_source = "Gaia GSP-Phot interval" if lower is not None or upper is not None else None
    elif source in {"dust-distance input", "stored distance"}:
        error = first_finite(payload, ("distance_error", "distance_error_pc", "dist_error"))
        if distance is not None and error is not None and error >= 0:
            lower = max(0.0, distance - error)
            upper = distance + error
            uncertainty_source = "stored symmetric distance error"
    return {
        "distance_pc": distance,
        "distance_source": source,
        "distance_lower_pc": lower,
        "distance_upper_pc": upper,
        "distance_uncertainty_source": uncertainty_source,
    }


def observed_bp_rp(payload: Mapping[str, object]) -> float | None:
    color = finite_number(payload.get("bp_rp"))
    if color is not None:
        return color
    bp = finite_number(payload.get("phot_bp_mean_mag"))
    rp = finite_number(payload.get("phot_rp_mean_mag"))
    return bp - rp if bp is not None and rp is not None else None


def build_source_row(
    payload: Mapping[str, object],
    *,
    search_method: str = "Pipeline",
) -> dict[str, object]:
    """Build one auditable source-property row from a merged review payload."""
    candidate_id = clean_identifier(payload.get("candidate_id")) or "unknown"
    coords = payload_ra_dec(dict(payload))
    if coords is None:
        ra_deg = dec_deg = None
        source = candidate_id
    else:
        ra_deg, dec_deg = coords
        source = format_j_designation(ra_deg, dec_deg)

    distance_info = select_distance_with_uncertainty(payload)
    distance_pc = finite_number(distance_info["distance_pc"])
    bp_rp = observed_bp_rp(payload)
    gaia_g = finite_number(payload.get("phot_g_mean_mag"))
    cmd = dustmaps_cmd_from_fields(
        g_mag=gaia_g,
        bp_rp=bp_rp,
        dist_pc=distance_pc,
        a_v_3d=payload.get("A_v_3d"),
        bp_mag=payload.get("phot_bp_mean_mag"),
        rp_mag=payload.get("phot_rp_mean_mag"),
        parallax_mas=payload.get("parallax"),
    )

    absolute_g = finite_number(cmd.get("cmd_mag"))
    if absolute_g is None:
        absolute_g = first_finite(payload, ("mg0", "mg"))
    table_color = finite_number(cmd.get("cmd_color"))
    if table_color is None:
        table_color = first_finite(payload, ("bprp0", "bp_rp"))

    return {
        "candidate_class": clean_identifier(payload.get("event_class")),
        "candidate_id": candidate_id,
        "source": source,
        "gaia_dr3_source_id": (
            clean_identifier(payload.get("source_id"))
            or clean_identifier(payload.get("gaia_id"))
        ),
        "ra_deg": ra_deg,
        "dec_deg": dec_deg,
        "search_method": search_method,
        "aligned_asassn_mean_mag": first_finite(
            payload,
            (
                "stats_photometry_mean_mag",
                "ltv_median",
                "baseline_mag",
            ),
        ),
        "gaia_g_mag": gaia_g,
        "absolute_g_mag": absolute_g,
        "bp_rp_mag": table_color,
        "rv_amplitude_kms": first_finite(
            payload,
            ("rv_amplitude_robust", "ms_rv_amplitude_robust"),
        ),
        "ruwe": finite_number(payload.get("ruwe")),
        "distance_pc": distance_pc,
        **distance_info,
        "cmd_coordinate_source": clean_identifier(cmd.get("cmd_coordinate_source")),
        "review_classification_confidence": finite_number(payload.get("classification_confidence")),
        "review_updated_at": clean_identifier(payload.get("review_updated_at")),
    }


def _reviewed_sql(include_nonreviewed: bool) -> str:
    if include_nonreviewed:
        return ""
    return "AND lower(trim(COALESCE(r.workflow_status, r.status, ''))) = 'reviewed'"


def load_review_candidate_payloads(
    db_path: Path | str,
    *,
    classes: Sequence[str] = CLASS_ORDER,
    include_nonreviewed: bool = False,
) -> list[dict[str, object]]:
    """Read selected review classes and return GUI-equivalent merged payloads."""
    selected = tuple(str(value).strip().lower() for value in classes)
    unsupported = sorted(set(selected) - set(CLASS_ORDER))
    if unsupported:
        raise ValueError(
            "Unsupported class(es): "
            + ", ".join(unsupported)
            + ". This exporter intentionally excludes eclipsing binaries."
        )
    if not selected:
        return []

    path = Path(db_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    uri = f"file:{path.as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True, timeout=30.0) as conn:
        conn.execute("PRAGMA query_only = ON")
        review_columns = {
            str(row[1]) for row in conn.execute("PRAGMA table_info(reviews)").fetchall()
        }
        optional = {
            "workflow_status": "status",
            "status": None,
            "disposition": None,
            "duplicate_of": None,
            "classification_confidence": None,
            "updated_at": None,
        }
        select_parts = ["r.candidate_id", "lower(trim(r.event_class)) AS event_class"]
        for column, fallback in optional.items():
            if column in review_columns:
                select_parts.append(f"r.{column} AS {column}")
            elif fallback and fallback in review_columns:
                select_parts.append(f"r.{fallback} AS {column}")
            else:
                select_parts.append(f"NULL AS {column}")
        review_frame = pd.read_sql_query(
            "SELECT " + ", ".join(select_parts) + " FROM reviews AS r JOIN candidates AS c USING (candidate_id)",
            conn,
        )
        review_frame = review_frame.loc[
            review_frame["event_class"].astype("string").str.lower().isin(selected)
        ].copy()
        if not include_nonreviewed:
            bucket_map = {
                "dipper": "Dipper",
                "ltv": "LTV",
                "microlensing": "Microlensing",
            }
            review_frame = build_publication_cohort(
                review_frame,
                buckets=[bucket_map[value] for value in selected],
            )
            review_frame = review_frame.loc[review_frame["publication_selected"]].copy()
        review_frame = review_frame.sort_values(["updated_at", "candidate_id"], na_position="last")
        payloads: list[dict[str, object]] = []
        for row in review_frame.to_dict(orient="records"):
            candidate_id = row["candidate_id"]
            payload = get_candidate_payload(conn, str(candidate_id))
            payload["candidate_id"] = str(candidate_id)
            payload["event_class"] = str(row["event_class"])
            payload["classification_confidence"] = row.get("classification_confidence")
            payload["review_updated_at"] = row.get("updated_at")
            if not include_nonreviewed:
                payload["publication_cohort_version"] = row.get("publication_cohort_version")
            payloads.append(payload)
    return payloads


def build_candidate_table(
    payloads: Sequence[Mapping[str, object]],
    *,
    search_method: str = "Pipeline",
    sort_by: str = "ra",
) -> pd.DataFrame:
    rows = [build_source_row(payload, search_method=search_method) for payload in payloads]
    table = pd.DataFrame(rows)
    if table.empty:
        return table

    table["candidate_class"] = table["candidate_class"].astype("string").str.lower()
    class_rank = {name: rank for rank, name in enumerate(CLASS_ORDER)}
    table["_class_rank"] = table["candidate_class"].map(class_rank).fillna(len(class_rank))
    if sort_by == "candidate-id":
        sort_columns = ["_class_rank", "candidate_id"]
    elif sort_by == "review-date":
        sort_columns = ["_class_rank", "review_updated_at", "candidate_id"]
    else:
        sort_columns = ["_class_rank", "ra_deg", "dec_deg", "candidate_id"]
    return table.sort_values(sort_columns, na_position="last").drop(columns="_class_rank").reset_index(drop=True)


_LATEX_ESCAPE_MAP = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def latex_escape(value: object) -> str:
    return "".join(_LATEX_ESCAPE_MAP.get(character, character) for character in str(value))


def format_latex_value(column: str, value: object) -> str:
    if column in {"source", "search_method"}:
        return latex_escape(value) if clean_identifier(value) is not None else r"\ldots"
    number = finite_number(value)
    if number is None:
        return r"\ldots"
    precision = {
        "ra_deg": 5,
        "dec_deg": 5,
        "aligned_asassn_mean_mag": 2,
        "absolute_g_mag": 2,
        "bp_rp_mag": 2,
        "rv_amplitude_kms": 2,
        "ruwe": 2,
        "distance_pc": 0,
    }[column]
    return f"{number:.{precision}f}"


def _latex_header_lines() -> list[str]:
    labels = " & ".join(label for _column, label, _unit in LATEX_COLUMNS) + r" \\"
    units = " & ".join(unit for _column, _label, unit in LATEX_COLUMNS) + r" \\"
    return [labels, units]


def _latex_body_lines(table: pd.DataFrame, *, include_sections: bool) -> list[str]:
    lines: list[str] = []
    groups: list[tuple[str | None, pd.DataFrame]]
    if include_sections:
        groups = [
            (candidate_class, table.loc[table["candidate_class"].eq(candidate_class)])
            for candidate_class in CLASS_ORDER
            if table["candidate_class"].eq(candidate_class).any()
        ]
    else:
        groups = [(None, table)]

    for group_index, (candidate_class, group) in enumerate(groups):
        if group_index:
            lines.append(r"\addlinespace[0.35em]")
        if candidate_class is not None:
            title = latex_escape(f"{CLASS_TITLES[candidate_class]} candidates")
            lines.append(
                rf"\multicolumn{{{len(LATEX_COLUMNS)}}}{{l}}{{\textit{{{title}}}}} \\"
            )
        for _index, row in group.iterrows():
            values = [format_latex_value(column, row.get(column)) for column, _label, _unit in LATEX_COLUMNS]
            lines.append(" & ".join(values) + r" \\")
    return lines


def render_longtable(
    table: pd.DataFrame,
    *,
    caption: str,
    label: str,
    include_sections: bool = False,
) -> str:
    """Render a complete multipage LaTeX table using longtable and booktabs."""
    header = _latex_header_lines()
    continued_caption = latex_escape(caption + " (continued)")
    caption_tex = latex_escape(caption)
    label_tex = re.sub(r"[^A-Za-z0-9:.-]+", "-", label).strip("-")
    n_columns = len(LATEX_COLUMNS)
    lines = [
        r"% Requires \usepackage{booktabs,longtable}",
        rf"\begin{{longtable}}{{{LATEX_ALIGNMENT}}}",
        rf"\caption{{{caption_tex}}}\label{{{label_tex}}} \\",
        r"\toprule",
        *header,
        r"\midrule",
        r"\endfirsthead",
        rf"\caption[]{{{continued_caption}}} \\",
        r"\toprule",
        *header,
        r"\midrule",
        r"\endhead",
        r"\midrule",
        rf"\multicolumn{{{n_columns}}}{{r}}{{Continued on next page}} \\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
        *_latex_body_lines(table, include_sections=include_sections),
        r"\end{longtable}",
        "",
    ]
    return "\n".join(lines)


def render_single_page_table(
    table: pd.DataFrame,
    *,
    caption: str,
    label: str,
    include_sections: bool = False,
) -> str:
    """Render a nonbreaking, full-width table scaled to a single float page."""
    header = _latex_header_lines()
    caption_tex = latex_escape(caption)
    label_tex = re.sub(r"[^A-Za-z0-9:.-]+", "-", label).strip("-")
    lines = [
        r"% Requires \usepackage{booktabs,adjustbox}",
        r"\clearpage",
        r"\begin{table*}[p]",
        r"\centering",
        rf"\caption{{{caption_tex}}}",
        rf"\label{{{label_tex}}}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.2pt}",
        r"\renewcommand{\arraystretch}{0.72}",
        r"\begin{adjustbox}{max totalsize={\textwidth}{0.82\textheight},center}",
        rf"\begin{{tabular}}{{{LATEX_ALIGNMENT}}}",
        r"\toprule",
        *header,
        r"\midrule",
        *_latex_body_lines(table, include_sections=include_sections),
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{adjustbox}",
        r"\end{table*}",
        r"\clearpage",
        "",
    ]
    return "\n".join(lines)


def _caption_for_class(candidate_class: str) -> str:
    return (
        f"Source properties of the reviewed MALCA {CLASS_TITLES[candidate_class].lower()} candidates. "
        "The Gaia absolute magnitude and color are extinction-corrected where a line-of-sight "
        "extinction is available."
    )


def export_tables(table: pd.DataFrame, output_dir: Path | str) -> list[Path]:
    """Write combined and per-class CSV/LaTeX tables and return their paths."""
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    combined_csv = out_dir / "candidate_source_properties.csv"
    combined_tex = out_dir / "candidate_source_properties.tex"
    combined_single_page_tex = out_dir / "candidate_source_properties_single_page.tex"
    table.to_csv(combined_csv, index=False)
    combined_caption = (
        "Source properties of the reviewed MALCA dipper, long-term-variable, and "
        "microlensing candidates. The Gaia absolute magnitude and color are "
        "extinction-corrected where a line-of-sight extinction is available."
    )
    combined_tex.write_text(
        render_longtable(
            table,
            caption=combined_caption,
            label="tab:malca-candidate-source-properties",
            include_sections=True,
        )
    )
    combined_single_page_tex.write_text(
        render_single_page_table(
            table,
            caption=combined_caption,
            label="tab:malca-candidate-source-properties",
            include_sections=True,
        )
    )
    written.extend((combined_csv, combined_tex, combined_single_page_tex))

    single_page_tables: list[str] = []
    for candidate_class in CLASS_ORDER:
        subset = table.loc[table["candidate_class"].eq(candidate_class)].copy()
        if subset.empty:
            continue
        stem = f"{candidate_class}_candidates"
        csv_path = out_dir / f"{stem}.csv"
        tex_path = out_dir / f"{stem}.tex"
        single_page_tex_path = out_dir / f"{stem}_single_page.tex"
        subset.to_csv(csv_path, index=False)
        tex_path.write_text(
            render_longtable(
                subset,
                caption=_caption_for_class(candidate_class),
                label=f"tab:{candidate_class}-candidates",
            )
        )
        single_page_table = render_single_page_table(
            subset,
            caption=_caption_for_class(candidate_class),
            label=f"tab:{candidate_class}-candidates",
        )
        single_page_tex_path.write_text(single_page_table)
        single_page_tables.append(single_page_table)
        written.extend((csv_path, tex_path, single_page_tex_path))

    three_tables_path = out_dir / "candidate_source_properties_three_tables.tex"
    three_tables_path.write_text("\n".join(single_page_tables))
    written.append(three_tables_path)
    return written


def completeness_summary(table: pd.DataFrame) -> pd.DataFrame:
    fields = ("ra_deg", "aligned_asassn_mean_mag", "absolute_g_mag", "bp_rp_mag", "distance_pc")
    rows: list[dict[str, object]] = []
    for candidate_class in CLASS_ORDER:
        subset = table.loc[table["candidate_class"].eq(candidate_class)]
        if subset.empty:
            continue
        row: dict[str, object] = {"class": candidate_class, "rows": len(subset)}
        for field in fields:
            row[field] = int(pd.to_numeric(subset[field], errors="coerce").notna().sum())
        rows.append(row)
    return pd.DataFrame(rows)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="Input MALCA review SQLite database.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the generated .tex and .csv files.",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        choices=CLASS_ORDER,
        default=list(CLASS_ORDER),
        help="Candidate classes to export. Eclipsing binaries are intentionally unavailable.",
    )
    parser.add_argument(
        "--include-nonreviewed",
        action="store_true",
        help="Include selected-class rows that are not in reviewed workflow status.",
    )
    parser.add_argument(
        "--search-method",
        default="Pipeline",
        help="Text placed in the Search method column (default: Pipeline).",
    )
    parser.add_argument(
        "--sort-by",
        choices=("ra", "candidate-id", "review-date"),
        default="ra",
        help="Row ordering within each class (default: ra).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payloads = load_review_candidate_payloads(
        args.db,
        classes=args.classes,
        include_nonreviewed=args.include_nonreviewed,
    )
    table = build_candidate_table(
        payloads,
        search_method=args.search_method,
        sort_by=args.sort_by,
    )
    if table.empty:
        raise SystemExit("No matching candidate rows found.")

    written = export_tables(table, args.output_dir)
    print(completeness_summary(table).to_string(index=False))
    print(f"\nWrote {len(written)} files to {Path(args.output_dir).expanduser().resolve()}")
    for path in written:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
