from __future__ import annotations

import importlib.util
from io import BytesIO
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from malca.enrich.spectrum_fetch import SpectrumData


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "plot_dipper_spectra_context.py"
)
SPEC = importlib.util.spec_from_file_location(
    "plot_dipper_spectra_context",
    SCRIPT_PATH,
)
assert SPEC is not None and SPEC.loader is not None
PLOTTER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PLOTTER
SPEC.loader.exec_module(PLOTTER)


def test_select_spectrum_rows_uses_nearest_unique_survey_record() -> None:
    spectra = pd.DataFrame(
        {
            "candidate_id": ["a", "a", "a", "b", "c"],
            "survey": ["lamost_dr7", "lamost_dr7", "apogee_dr16", "desi_dr1", "rave_dr6"],
            "sep_arcsec": [0.8, 0.2, 0.4, 0.1, 0.3],
            "record": ["far", "near", "apogee", "desi", "rave"],
        }
    )

    selected = PLOTTER.select_spectrum_rows(spectra, ["b", "a"])

    assert selected[["candidate_id", "survey"]].values.tolist() == [
        ["b", "desi_dr1"],
        ["a", "apogee_dr16"],
        ["a", "lamost_dr7"],
    ]
    assert selected.loc[selected["survey"].eq("lamost_dr7"), "record"].item() == "near"


def test_select_spectrum_rows_rejects_metadata_only_rows() -> None:
    spectra = pd.DataFrame(
        {
            "candidate_id": ["a", "a"],
            "survey": ["sdss2_sn", "lamost_dr7"],
            "sep_arcsec": [0.1, 0.2],
            "spectrum_record_status": ["metadata_only", "available"],
        }
    )

    selected = PLOTTER.select_spectrum_rows(spectra, ["a"])

    assert selected["survey"].tolist() == ["lamost_dr7"]


def test_median_center_is_independent_by_band() -> None:
    frame = pd.DataFrame(
        {
            "band": ["g", "g", "V", "V"],
            "mag": [12.0, 14.0, 8.0, 10.0],
        }
    )

    centered = PLOTTER._median_center(frame)

    assert centered["centered_mag"].tolist() == [-1.0, 1.0, -1.0, 1.0]


def test_draw_lightcurve_context_only_plots_asassn_v_and_g(
    tmp_path,
    monkeypatch,
) -> None:
    candidate_id = "stv_test_lc"
    lightcurve_path = tmp_path / "lightcurve.csv"
    lightcurve_path.write_text("placeholder\n")
    candidates = pd.DataFrame(
        {
            "candidate_id": [candidate_id],
            "lc_path": [str(lightcurve_path)],
            "payload_json": ["{}"],
        }
    ).set_index("candidate_id", drop=False)
    context = PLOTTER.CandidateContext(
        run_dir=tmp_path,
        candidate_rows=candidates,
        sed_rows={},
        sed_curves={},
        sed_fits={},
        sed_points={},
    )
    monkeypatch.setattr(PLOTTER, "load_lightcurve", lambda _path: object())
    monkeypatch.setattr(
        PLOTTER,
        "filter_lightcurve",
        lambda _frame, max_error: pd.DataFrame(
            {
                "time": [2_458_001.0, 2_458_002.0, 2_458_003.0],
                "value": [12.0, 12.1, 8.0],
                "value_error": [0.02, 0.03, 0.04],
                "band": ["V", "g", "infrared"],
            }
        ),
    )

    fig, ax = plt.subplots()
    try:
        status = PLOTTER.draw_lightcurve_context(ax, context, candidate_id)
        assert [text.get_text() for text in ax.get_legend().get_texts()] == ["V", "g"]
        assert ax.get_title(loc="left") == "ASAS-SN (bands median centered)"
        assert status["asassn_n_points"] == 2
        assert set(status) == {"asassn_status", "asassn_n_points", "asassn_path"}
    finally:
        plt.close(fig)


def test_draw_spectrum_context_handles_detector_gaps() -> None:
    wave = np.r_[np.linspace(15150.0, 15800.0, 500), np.linspace(15870.0, 16420.0, 450)]
    flux = 1.0 + 0.02 * np.sin(wave / 5.0)
    flux -= 0.18 * np.exp(-0.5 * ((wave - 15650.0) / 1.2) ** 2)
    spectrum = SpectrumData(
        wavelength=wave,
        flux=flux,
        flux_err=np.full_like(flux, 0.01),
    )
    fig, (raw_ax, residual_ax) = plt.subplots(2, 1, sharex=True)
    try:
        result = PLOTTER.draw_spectrum_context(
            raw_ax,
            residual_ax,
            spectrum,
            survey="apogee_dr16",
        )
        assert 0 < result["spectrum_n_points"] <= len(wave)
        assert result["spectrum_wavelength_min_angstrom"] >= wave.min()
        assert result["spectrum_wavelength_max_angstrom"] <= wave.max()
        assert len(raw_ax.lines) >= 4
        assert residual_ax.get_xlabel()
    finally:
        plt.close(fig)


def test_build_candidate_page_is_vector_pdf_ready(monkeypatch) -> None:
    candidate_id = "stv_test_1"
    candidates = pd.DataFrame(
        {
            "candidate_id": [candidate_id],
            "ra": [30.0],
            "dec": [15.0],
            "lc_path": [""],
            "payload_json": ["{}"],
        }
    ).set_index("candidate_id", drop=False)
    context = PLOTTER.CandidateContext(
        run_dir=Path("."),
        candidate_rows=candidates,
        sed_rows={candidate_id: pd.DataFrame()},
        sed_curves={candidate_id: pd.DataFrame()},
        sed_fits={candidate_id: pd.DataFrame()},
        sed_points={candidate_id: pd.DataFrame()},
    )
    monkeypatch.setattr(
        PLOTTER,
        "draw_lightcurve_context",
        lambda ax, _context, _candidate_id: {
            "asassn_status": "ok",
            "asassn_n_points": 2,
        },
    )
    monkeypatch.setattr(
        PLOTTER,
        "draw_sed_context",
        lambda ax, _context, _candidate_id: {
            "sed_status": "ok",
            "sed_n_points": 4,
        },
    )
    monkeypatch.setattr(
        PLOTTER,
        "CONTEXT_STYLE",
        {**PLOTTER.CONTEXT_STYLE, "text.usetex": False},
    )
    wave = np.linspace(4000.0, 8000.0, 1000)
    spectrum = SpectrumData(
        wavelength=wave,
        flux=1.0 + 0.03 * np.sin(wave / 10.0),
    )

    fig, status = PLOTTER.build_candidate_page(
        context,
        candidate_id,
        spectrum,
        survey="lamost_dr7",
    )
    try:
        buffer = BytesIO()
        fig.savefig(buffer, format="pdf")
        assert buffer.getvalue().startswith(b"%PDF")
        assert status["candidate_id"] == candidate_id
        assert status["survey"] == "lamost_dr7"
        assert len(fig.axes) == 4
    finally:
        plt.close(fig)
