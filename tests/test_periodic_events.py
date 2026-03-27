from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from malca.periodic_events import run_periodic_events


def _write_dat3(path: Path, times: np.ndarray, mags: np.ndarray) -> None:
    lines: list[str] = []
    for idx, (time_value, mag_value) in enumerate(zip(times, mags, strict=True)):
        camera = 1 + (idx % 2)
        lines.append(
            f"{float(time_value):.6f} {float(mag_value):.6f} 0.030000 1 {camera:d} 0 0 cam{camera}/field1"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def _periodic_eclipse_lightcurve() -> tuple[np.ndarray, np.ndarray]:
    times = np.linspace(0.0, 120.0, 240)
    phase = np.mod(times, 4.0) / 4.0
    eclipse = ((phase < 0.08) | (phase > 0.92)).astype(float)
    mags = 14.0 + 0.75 * eclipse + 0.02 * np.sin(2.0 * np.pi * times / 4.0)
    return times, mags


def _flat_lightcurve() -> tuple[np.ndarray, np.ndarray]:
    times = np.linspace(0.0, 120.0, 240)
    mags = np.full_like(times, 14.0)
    return times, mags


def test_run_periodic_events_finds_significant_phase_folded_dip(tmp_path: Path) -> None:
    lc_path = tmp_path / "periodic.dat3"
    times, mags = _periodic_eclipse_lightcurve()
    _write_dat3(lc_path, times, mags)

    df = pd.DataFrame(
        {
            "source_id": ["periodic"],
            "dat_path": [str(lc_path)],
            "pre_periodicity_selected_period": [4.0],
        }
    )
    out = run_periodic_events(df, path_col="dat_path", workers=1, show_tqdm=False)

    assert "path_x" not in out.columns
    assert bool(out.loc[0, "dip_significant"]) is True
    assert out.loc[0, "analysis_branch"] == "periodic"
    assert out.loc[0, "event_model"] == "phase_folded_dip"
    assert float(out.loc[0, "phase_dip_depth_mag"]) > 0.2
    assert int(out.loc[0, "phase_dip_support_cycles"]) >= 3


def test_run_periodic_events_rejects_flat_profile(tmp_path: Path) -> None:
    lc_path = tmp_path / "flat.dat3"
    times, mags = _flat_lightcurve()
    _write_dat3(lc_path, times, mags)

    df = pd.DataFrame(
        {
            "source_id": ["flat"],
            "dat_path": [str(lc_path)],
            "pre_periodicity_selected_period": [4.0],
        }
    )
    out = run_periodic_events(df, path_col="dat_path", workers=1, show_tqdm=False)

    assert bool(out.loc[0, "dip_significant"]) is False
    assert str(out.loc[0, "phase_profile_reason"]) in {
        "no_positive_phase_dip",
        "low_depth_snr",
        "weak_phase_dip",
    }
