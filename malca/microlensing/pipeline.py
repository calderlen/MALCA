"""CLI pipeline for joint ASAS-SN, ATLAS, and ZTF microlensing fits."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

from malca.config import DEFAULT_OUTPUT_DIR, PARQUET_CACHE_COMPRESSION
from malca.review.native_lightcurve import resolve_lightcurve_path
from malca.review.store import get_candidate_payload

from .datasets import load_candidate_datasets
from .diagnostics import candidate_result_row, dataset_result_rows
from .joint_fit import fit_individual_dataset_pspl, fit_joint_pspl, fit_leave_one_survey_out
from .parallax import fit_joint_parallax
from .plotting import plot_joint_fit
from .schema import MICROLENSING_JOINT_COLUMN_SPECS, MICROLENSING_JOINT_VERSION


@dataclass
class _CandidateTask:
    candidate_id: str
    asas_sn_id: str
    ra_deg: float | None
    dec_deg: float | None
    asassn_path: str | None
    external_lc_dir: str | None
    surveys: tuple[str, ...]
    output_dir: str
    parallax: bool
    parallax_mcmc: bool
    plot: bool


def _finite_optional(value: object) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def _failure_candidate_row(candidate_id: str, status: str, error: str | None = None) -> dict[str, object]:
    row: dict[str, object] = {
        "candidate_id": candidate_id,
        **{name: np.nan for name, _dtype, _kind in MICROLENSING_JOINT_COLUMN_SPECS},
        "microlensing_joint_version": MICROLENSING_JOINT_VERSION,
        "microlensing_joint_status": status,
    }
    if error:
        row["microlensing_joint_error"] = error
    return row


def _fit_candidate(task: _CandidateTask) -> tuple[dict[str, object], list[dict[str, object]]]:
    try:
        datasets = load_candidate_datasets(
            task.candidate_id,
            asassn_path=task.asassn_path,
            external_lc_dir=task.external_lc_dir,
            surveys=task.surveys,
            candidate_aliases=(task.asas_sn_id,),
        )
        if not datasets:
            return _failure_candidate_row(task.candidate_id, "no_datasets"), []
        fit = fit_joint_pspl(datasets)
        individual = fit_individual_dataset_pspl(datasets, joint_seed=fit)
        leave_one_out = fit_leave_one_survey_out(datasets, joint_seed=fit)
        parallax_result = None
        if task.parallax:
            parallax_result = fit_joint_parallax(
                datasets,
                fit,
                ra_deg=task.ra_deg,
                dec_deg=task.dec_deg,
                run_mcmc=task.parallax_mcmc,
            )
        candidate_row = candidate_result_row(task.candidate_id, datasets, fit, parallax=parallax_result)
        dataset_rows = dataset_result_rows(
            task.candidate_id,
            datasets,
            fit,
            individual_fits=individual,
            leave_one_survey_out=leave_one_out,
        )
        if task.plot and fit.success:
            candidate_row["microlensing_joint_plot_path"] = str(
                plot_joint_fit(task.candidate_id, datasets, fit, Path(task.output_dir) / "plots")
            )
        return candidate_row, dataset_rows
    except Exception as exc:
        return _failure_candidate_row(
            task.candidate_id,
            "error",
            f"{type(exc).__name__}: {exc}",
        ), []


def _review_candidates(review_db: Path, candidate_ids: list[str] | None) -> list[dict[str, object]]:
    with sqlite3.connect(review_db) as connection:
        query = """
            SELECT c.candidate_id, c.asas_sn_id, c.ra, c.dec
            FROM candidates c
            JOIN reviews r ON r.candidate_id = c.candidate_id
            WHERE r.event_class = 'microlensing'
        """
        params: list[object] = []
        if candidate_ids:
            placeholders = ",".join("?" for _ in candidate_ids)
            query += f" AND c.candidate_id IN ({placeholders})"
            params.extend(candidate_ids)
        query += " ORDER BY c.candidate_id"
        rows = connection.execute(query, params).fetchall()
        output: list[dict[str, object]] = []
        for candidate_id, asas_sn_id, ra, dec in rows:
            payload = get_candidate_payload(connection, str(candidate_id))
            payload.setdefault("candidate_id", candidate_id)
            payload.setdefault("asas_sn_id", asas_sn_id)
            lc_path = resolve_lightcurve_path(payload, review_db.parent)
            output.append(
                {
                    "candidate_id": str(candidate_id),
                    "asas_sn_id": "" if asas_sn_id is None else str(asas_sn_id),
                    "ra_deg": _finite_optional(ra if ra is not None else payload.get("ra")),
                    "dec_deg": _finite_optional(dec if dec is not None else payload.get("dec")),
                    "asassn_path": str(lc_path) if lc_path is not None else None,
                }
            )
    return output


def run_pipeline(
    *,
    review_db: Path | str,
    external_lc_dir: Path | str | None,
    surveys: tuple[str, ...],
    output_dir: Path | str,
    candidate_ids: list[str] | None = None,
    fit_workers: int = 1,
    parallax: bool = False,
    parallax_mcmc: bool = False,
    plot: bool = False,
    merge_review_db: bool = False,
) -> tuple[Path, Path]:
    review_db = Path(review_db).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = _review_candidates(review_db, candidate_ids)
    if not candidates:
        raise ValueError("No reviewed microlensing candidates matched the requested selection")
    tasks = [
        _CandidateTask(
            candidate_id=str(row["candidate_id"]),
            asas_sn_id=str(row["asas_sn_id"]),
            ra_deg=row["ra_deg"],
            dec_deg=row["dec_deg"],
            asassn_path=row["asassn_path"],
            external_lc_dir=str(Path(external_lc_dir).expanduser().resolve()) if external_lc_dir else None,
            surveys=surveys,
            output_dir=str(output_dir),
            parallax=parallax,
            parallax_mcmc=parallax_mcmc,
            plot=plot,
        )
        for row in candidates
    ]
    if fit_workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=fit_workers) as executor:
            fitted = list(executor.map(_fit_candidate, tasks))
    else:
        fitted = [_fit_candidate(task) for task in tasks]

    candidate_rows = [item[0] for item in fitted]
    dataset_rows = [row for item in fitted for row in item[1]]
    candidate_table = pd.DataFrame(candidate_rows)
    dataset_table = pd.DataFrame(dataset_rows)
    candidate_path = output_dir / "microlensing_joint_results.parquet"
    dataset_path = output_dir / "microlensing_dataset_results.parquet"
    candidate_table.to_parquet(candidate_path, index=False, compression=PARQUET_CACHE_COMPRESSION)
    dataset_table.to_parquet(dataset_path, index=False, compression=PARQUET_CACHE_COMPRESSION)

    if merge_review_db and not candidate_table.empty:
        from malca.review.store import db_connect, init_db, merge_candidate_results

        summary_columns = ["candidate_id"] + [
            name for name, _dtype, _kind in MICROLENSING_JOINT_COLUMN_SPECS if name in candidate_table.columns
        ]
        with db_connect(review_db) as connection:
            init_db(connection)
            merge_candidate_results(
                connection,
                candidate_table[summary_columns],
                clear_columns=summary_columns[1:],
            )

    return candidate_path, dataset_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fit shared-geometry PSPL models to cached multi-survey photometry")
    parser.add_argument("--review-db", type=Path, required=True, help="Review SQLite database containing microlensing labels")
    parser.add_argument("--external-lc-dir", type=Path, default=None, help="Results root containing external_lc_manifest.parquet and cached light curves")
    parser.add_argument(
        "--surveys",
        nargs="+",
        default=["asassn", "atlas", "ztf", "ztf_forced"],
        choices=["asassn", "atlas", "ztf", "ztf_forced"],
    )
    parser.add_argument("--output-dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR) / "microlensing" / "joint")
    parser.add_argument("--candidate-id", action="append", dest="candidate_ids", help="Fit only this reviewed candidate; repeat as needed")
    parser.add_argument("--fit-workers", type=int, default=1)
    parser.add_argument("--parallax", action="store_true", help="Attempt annual parallax for eligible long events")
    parser.add_argument("--parallax-mcmc", action="store_true", help="Also run the existing-style short Metropolis diagnostic")
    parser.add_argument("--plot", action="store_true", help="Write one joint-fit PDF per candidate")
    parser.add_argument("--merge-review-db", action="store_true", help="Merge only the compact candidate summary into Review")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    candidate_path, dataset_path = run_pipeline(
        review_db=args.review_db,
        external_lc_dir=args.external_lc_dir,
        surveys=tuple(args.surveys),
        output_dir=args.output_dir,
        candidate_ids=args.candidate_ids,
        fit_workers=max(1, int(args.fit_workers)),
        parallax=bool(args.parallax),
        parallax_mcmc=bool(args.parallax_mcmc),
        plot=bool(args.plot),
        merge_review_db=bool(args.merge_review_db),
    )
    print(candidate_path)
    print(dataset_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
