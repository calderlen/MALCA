from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

import malca.detect as detect_module


def test_detect_periodic_only_pregate_writes_canonical_outputs(tmp_path: Path, monkeypatch) -> None:
    out_dir = tmp_path / "run"
    lc_path = tmp_path / "lightcurves" / "periodic_source.dat3"
    lc_path.parent.mkdir(parents=True, exist_ok=True)
    lc_path.write_text("dummy\n", encoding="ascii")

    def fake_build_manifest(*_args: object, **_kwargs: object) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "source_id": ["src-1"],
                "dat_exists": [True],
                "path": [str(lc_path)],
                "dat_path": [str(lc_path)],
                "mag_bin": ["13_13.5"],
            }
        )

    def fake_apply_tags(df: pd.DataFrame, **_kwargs: object) -> pd.DataFrame:
        out = df.copy()
        out["failed_sparse"] = False
        out["failed_multi_camera"] = False
        out["failed_mag_range"] = False
        return out

    def fake_apply_pre_periodicity_gate(df: pd.DataFrame, **_kwargs: object) -> pd.DataFrame:
        out = df.copy()
        out["excluded_cameras"] = "2"
        out["pre_periodicity_label"] = "periodic"
        out["pre_periodic_flag"] = True
        out["pre_periodicity_selected_period"] = 6.0
        out["pre_periodicity_method"] = "pdm"
        out["pre_periodicity_pdm_method"] = "plavchan"
        return out

    seen: dict[str, object] = {}

    def fake_subprocess_run(cmd: list[str], check: bool = False) -> object:
        _ = check
        output_path = Path(cmd[cmd.index("--output") + 1])
        metadata_path = Path(cmd[cmd.index("--metadata-csv") + 1])
        baseline_indices = [idx for idx, token in enumerate(cmd) if token == "--baseline-func"]
        baseline_func = cmd[baseline_indices[-1] + 1]
        meta = pd.read_csv(metadata_path)
        seen["baseline_func"] = baseline_func
        seen["metadata_cols"] = meta.columns.tolist()

        df_out = meta.copy()
        df_out["dip_significant"] = True
        df_out["jump_significant"] = True
        df_out["baseline_source"] = "phase_template" if baseline_func == "phase_template" else "gp_masked"
        df_out["n_points"] = 120
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_out.to_parquet(output_path, index=False)

        class _Result:
            returncode = 0

        return _Result()

    def fake_apply_filters(df: pd.DataFrame, **_kwargs: object) -> pd.DataFrame:
        out = df.copy()
        out["failed_any"] = False
        return out

    monkeypatch.setattr(detect_module, "build_manifest", fake_build_manifest)
    monkeypatch.setattr(detect_module, "apply_tags", fake_apply_tags)
    monkeypatch.setattr(detect_module, "apply_pre_periodicity_gate", fake_apply_pre_periodicity_gate)
    monkeypatch.setattr(detect_module, "apply_filters", fake_apply_filters)
    monkeypatch.setattr(detect_module.subprocess, "run", fake_subprocess_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "malca.detect",
            "--mag-bin",
            "13_13.5",
            "--out-dir",
            str(out_dir),
            "--skip-vsx",
            "--skip-camera-median",
            "--apply-pre-periodicity-gate",
            "--workers",
            "1",
            "--pre-periodicity-workers",
            "1",
            "--skip-gaia-ruwe-validation",
            "--skip-gaia-pm-validation",
            "--skip-periodic-catalog-validation",
            "--no-run-characterize",
            "--no-run-classify",
            "--no-run-enrich",
            "--no-run-neighbor-enrich",
            "--no-run-spectra-enrich",
            "--no-run-vetting",
            "--no-export-bundle",
        ],
    )

    detect_module.main()

    results_dir = out_dir / "results"
    tagged_results = results_dir / "lc_events_results_13_13.5.parquet"
    tagged_filtered = results_dir / "lc_events_filtered_13_13.5.parquet"

    assert tagged_results.exists()
    assert tagged_filtered.exists()
    assert (results_dir / "lc_events_results.parquet").exists()
    assert (results_dir / "lc_events_filtered.parquet").exists()
    assert not (results_dir / "lc_periodic_events_results_13_13.5.parquet").exists()

    out = pd.read_parquet(tagged_results)
    assert len(out) == 1
    assert out.loc[0, "baseline_source"] == "phase_template"
    assert out.loc[0, "pre_periodicity_label"] == "periodic"
    assert bool(out.loc[0, "pre_periodic_flag"]) is True
    assert float(out.loc[0, "pre_periodicity_selected_period"]) == 6.0
    assert out.loc[0, "pre_periodicity_method"] == "pdm"

    assert seen["baseline_func"] == "phase_template"
    assert seen["metadata_cols"] == [
        "path",
        "excluded_cameras",
        "pre_periodicity_label",
        "pre_periodic_flag",
        "pre_periodicity_selected_period",
        "pre_periodicity_method",
    ]
