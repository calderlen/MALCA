from __future__ import annotations

from pathlib import Path
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


def test_enrich_row_worker_prefers_exact_lc_path_and_its_suffix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exact_path = tmp_path / "ASASSN-source-with-hyphens.dat7"
    exact_path.write_text("placeholder", encoding="ascii")
    seen: dict[str, object] = {}

    def fake_compute_stats(asassn_id, path, **kwargs):
        seen.update(
            {
                "asassn_id": asassn_id,
                "path": path,
                "file_ext": kwargs.get("file_ext"),
            }
        )
        return pd.DataFrame(), {"n_points_total": 5}

    monkeypatch.setattr(stats, "compute_stats", fake_compute_stats)

    row = stats._enrich_row_worker(
        (
            {"lc_path": str(exact_path), "candidate_id": "stv-not-the-file-stem"},
            "truncated",
            str(tmp_path),
            False,
            "dat3",
        )
    )

    assert seen == {
        "asassn_id": "ASASSN-source-with-hyphens",
        "path": str(exact_path),
        "file_ext": "dat7",
    }
    assert row["stats_n_points_total"] == 5


def test_compute_stats_reads_exact_mixed_suffix_path_without_reconstructing_id(
    tmp_path: Path,
) -> None:
    exact_path = tmp_path / "ASASSN-source-with-hyphens.dat7"
    exact_path.write_text(
        "\n".join(
            [
                "1000.0 14.0 0.05 1 1 0 0 cam1/field1",
                "1010.0 14.1 0.05 1 1 0 0 cam1/field1",
                "1020.0 14.2 0.05 1 1 0 0 cam1/field1",
            ]
        )
        + "\n",
        encoding="ascii",
    )

    _frame, summary = stats.compute_stats(
        "incorrect-truncated-id",
        exact_path,
        compute_ls=False,
        file_ext="dat3",
    )

    assert summary["compute_status"] == "ok"
    assert summary["file_points_total"] == 3
    assert summary["time_span_days"] == pytest.approx(20.0)
    assert summary["variability_sokolovsky_v"] == pytest.approx(0.1 / 28.2)
