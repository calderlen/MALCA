from __future__ import annotations

import sys
import types

import pandas as pd
import pytest

if "iar.IARModel" not in sys.modules:
    fake_iar_pkg = types.ModuleType("iar")
    fake_iar_model = types.ModuleType("iar.IARModel")

    def _dummy_iar(*args, **kwargs):
        _ = args, kwargs
        return float("nan")

    fake_iar_model.IARphikalman = _dummy_iar
    fake_iar_pkg.IARModel = fake_iar_model
    sys.modules["iar"] = fake_iar_pkg
    sys.modules["iar.IARModel"] = fake_iar_model

import malca.core.stats as stats


def _identity_baseline(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    _ = kwargs
    out = df.copy()
    out["resid"] = pd.to_numeric(out["mag"], errors="coerce")
    return out


def test_string_length_ignores_time_spacing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(stats, "per_camera_gp_baseline", _identity_baseline)

    df = pd.DataFrame(
        {
            "JD": [2450000.0, 2450001.0, 2451000.0],
            "mag": [0.0, 1.0, 2.0],
            "error": [0.1, 0.1, 0.1],
            "camera_name": ["camA", "camA", "camA"],
        }
    )

    out = stats.baseline_subtracted_string_length(df)
    assert out["string_length_total"] == pytest.approx(2.0)
    assert out["string_length_mean_step"] == pytest.approx(1.0)
    assert out["string_length_n_steps"] == pytest.approx(2.0)


def test_string_length_higher_for_jittery_curve(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(stats, "per_camera_gp_baseline", _identity_baseline)

    base = {
        "JD": [1.0, 2.0, 3.0, 4.0, 5.0],
        "error": [0.1, 0.1, 0.1, 0.1, 0.1],
        "camera_name": ["camA", "camA", "camA", "camA", "camA"],
    }
    smooth = pd.DataFrame({**base, "mag": [0.0, 0.1, 0.2, 0.3, 0.4]})
    jitter = pd.DataFrame({**base, "mag": [0.0, 1.0, -1.0, 1.0, -1.0]})

    smooth_out = stats.baseline_subtracted_string_length(smooth)
    jitter_out = stats.baseline_subtracted_string_length(jitter)

    assert jitter_out["string_length_total"] > smooth_out["string_length_total"]
    assert jitter_out["string_length_mean_step"] > smooth_out["string_length_mean_step"]


def test_string_length_handles_missing_camera_columns(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(stats, "per_camera_gp_baseline", _identity_baseline)

    df = pd.DataFrame(
        {
            "JD": [10.0, 11.0, 12.0],
            "mag": [1.0, 1.5, 1.0],
            "error": [0.1, 0.1, 0.1],
        }
    )

    out = stats.baseline_subtracted_string_length(df)
    assert out["string_length_total"] == pytest.approx(1.0)
    assert out["string_length_n_steps"] == pytest.approx(2.0)


def test_enrich_row_worker_flattens_field_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_compute_stats(*args, **kwargs):
        _ = args, kwargs
        return pd.DataFrame(), {
            "asassn_field_key": "field1",
            "asassn_fields": "field1,field2",
            "asassn_field_count": 2,
            "asassn_field_key_fraction": 0.75,
            "camera_name_key": "cam1",
            "camera_names": "cam1,cam2",
            "camera_name_count": 2,
            "camera_name_key_fraction": 0.5,
            "by_field": pd.DataFrame({"field": ["field1"]}),
        }

    monkeypatch.setattr(stats, "compute_stats", fake_compute_stats)

    row = stats._enrich_row_worker(({"path": "lc/1001.dat2"}, "1001", "lc", False))

    assert row["stats_asassn_field_key"] == "field1"
    assert row["stats_asassn_fields"] == "field1,field2"
    assert row["stats_asassn_field_count"] == 2
    assert row["stats_camera_name_key"] == "cam1"
    assert "stats_by_field" not in row


def test_enrich_row_worker_passes_file_extension(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    def fake_compute_stats(*args, **kwargs):
        seen["kwargs"] = kwargs
        return pd.DataFrame(), {"n_points_total": 5}

    monkeypatch.setattr(stats, "compute_stats", fake_compute_stats)

    row = stats._enrich_row_worker(({"path": "lc/1001.dat2"}, "1001", "lc", False, "dat2"))

    assert seen["kwargs"]["file_ext"] == "dat2"
    assert row["stats_n_points_total"] == 5
