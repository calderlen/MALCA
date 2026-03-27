from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from malca.periodicity_gate import apply_pre_periodicity_gate


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


def _single_dip_lightcurve() -> tuple[np.ndarray, np.ndarray]:
    times = np.linspace(0.0, 120.0, 240)
    mags = np.full_like(times, 14.0)
    dip_mask = np.abs(times - 60.0) <= 1.0
    mags[dip_mask] += 0.9
    return times, mags


def test_apply_pre_periodicity_gate_flags_strong_periodic_lightcurve(tmp_path: Path) -> None:
    periodic_path = tmp_path / "periodic.dat3"
    times, mags = _periodic_eclipse_lightcurve()
    _write_dat3(periodic_path, times, mags)

    df = pd.DataFrame({"source_id": ["periodic"], "dat_path": [str(periodic_path)]})
    out = apply_pre_periodicity_gate(
        df,
        n_periods=800,
        n_bootstrap=24,
        significance_level=0.05,
        strong_single_sig=0.02,
        min_period=0.5,
        max_period=10.0,
        workers=1,
        show_tqdm=False,
    )

    assert bool(out.loc[0, "pre_periodic_flag"]) is True
    assert out.loc[0, "pre_periodicity_label"] == "periodic"
    assert float(out.loc[0, "pre_periodicity_selected_period"]) > 0.0


def test_apply_pre_periodicity_gate_keeps_single_dip_in_stochastic_branch(tmp_path: Path) -> None:
    dip_path = tmp_path / "single_dip.dat3"
    times, mags = _single_dip_lightcurve()
    _write_dat3(dip_path, times, mags)

    df = pd.DataFrame({"source_id": ["single_dip"], "dat_path": [str(dip_path)]})
    out = apply_pre_periodicity_gate(
        df,
        n_periods=800,
        n_bootstrap=24,
        significance_level=0.05,
        strong_single_sig=0.02,
        min_period=0.5,
        max_period=10.0,
        workers=1,
        show_tqdm=False,
    )

    assert bool(out.loc[0, "pre_periodic_flag"]) is False
    assert out.loc[0, "pre_periodicity_label"] in {"ambiguous", "non_periodic"}
