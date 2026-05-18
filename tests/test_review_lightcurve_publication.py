from __future__ import annotations

from pathlib import Path
import sys
import types


if "celerite2" not in sys.modules:
    sys.modules["celerite2"] = types.SimpleNamespace(
        GaussianProcess=object,
        terms=types.SimpleNamespace(SHOTerm=object),
    )

from malca.review.lightcurve_publication import build_review_lightcurve_publication_pdf


def _write_dat2(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "2458000.0 14.10 0.02 1 4 0 0 ba/F1",
                "2458001.0 14.18 0.02 1 4 0 0 ba/F1",
                "2458002.0 14.55 0.03 1 4 0 0 ba/F1",
                "2458003.0 14.15 0.02 1 4 0 0 ba/F1",
                "2458000.5 13.95 0.02 1 5 1 0 bb/F1",
                "2458001.5 14.02 0.02 1 5 1 0 bb/F1",
                "2458002.5 14.34 0.03 1 5 1 0 bb/F1",
                "2458003.5 14.00 0.02 1 5 1 0 bb/F1",
            ]
        ),
        encoding="ascii",
    )


def test_review_lightcurve_publication_pdf_uses_native_matplotlib_data_path(tmp_path: Path) -> None:
    lc_path = tmp_path / "123.dat2"
    _write_dat2(lc_path)
    payload = {
        "candidate_id": "123",
        "asas_sn_id": "123",
        "lc_path": str(lc_path),
        "dip_best_t0": 2458002.0,
        "dip_best_width_param": 0.4,
        "dip_bayes_factor": 20.0,
        "dip_run_count": 1,
        "period_consensus_days": 2.0,
    }

    pdf = build_review_lightcurve_publication_pdf(
        payload,
        plot_dir=None,
        selected_cameras=[],
        selected_bands=["g", "V"],
        filter_bad_cameras=False,
        show_baseline=True,
        show_event_markers=True,
        show_residuals=True,
        show_phase_fold=True,
        show_raw_mag=True,
        override_period=None,
        show_diagnostics=True,
        confidence_colors=True,
        run_params={"baseline_func": "global_median"},
        yaxis_mode="mag",
    )

    assert pdf.startswith(b"%PDF")
    assert len(pdf) > 1000
