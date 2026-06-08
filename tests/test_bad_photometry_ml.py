from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from malca.meta_analysis.ml import bad_photometry as bp
from malca.table_io import write_feature_table


def _write_dat3(path: Path, rows: list[tuple[float, float, float, int, int, int, int, str]]) -> None:
    path.write_text(
        "\n".join(
            f"{jd:12.7f} {mag:.6f} {err:.6f} {good:d} {camera:d} {band:d} {sat:d} {field}"
            for jd, mag, err, good, camera, band, sat, field in rows
        )
        + "\n",
        encoding="ascii",
    )


def test_build_dropout_dataset_labels_v1_minus_v2(tmp_path: Path) -> None:
    v1 = tmp_path / "v1.parquet"
    v2 = tmp_path / "v2.parquet"
    out = tmp_path / "dataset.parquet"
    write_feature_table(
        pd.DataFrame(
            {
                "candidate_id": ["stv_100", "stv_200", "stv_300"],
                "timescale": ["stv", "stv", "stv"],
                "asas_sn_id": ["100", "200", "300"],
                "dipper_score": [10.0, 2.0, 5.0],
            }
        ),
        v1,
    )
    write_feature_table(
        pd.DataFrame(
            {
                "candidate_id": ["stv_100", "stv_300"],
                "timescale": ["stv", "stv"],
                "asas_sn_id": ["100", "300"],
            }
        ),
        v2,
    )

    dataset = bp.build_dropout_dataset(v1, v2, output=out, key="asas_sn_id")

    assert out.exists()
    by_id = dataset.set_index("asas_sn_id")
    assert by_id.loc["100", "bad_photometry"] == 0
    assert by_id.loc["200", "bad_photometry"] == 1
    assert bool(by_id.loc["300", "survived_reprocessing"]) is True
    assert dataset["dropout_source_id"].tolist() == ["100", "200", "300"]


def test_build_dropout_dataset_rejects_v2_outside_v1(tmp_path: Path) -> None:
    v1 = tmp_path / "v1.csv"
    v2 = tmp_path / "v2.csv"
    pd.DataFrame({"candidate_id": ["A"]}).to_csv(v1, index=False)
    pd.DataFrame({"candidate_id": ["A", "B"]}).to_csv(v2, index=False)

    with pytest.raises(ValueError, match="v2 contains IDs absent from v1"):
        bp.build_dropout_dataset(v1, v2)


def test_make_stratified_group_split_keeps_duplicate_ids_together() -> None:
    df = pd.DataFrame(
        {
            "dropout_source_id": [f"src-{idx // 2}" for idx in range(40)],
            "candidate_id": [f"row-{idx}" for idx in range(40)],
            "bad_photometry": [0 if idx < 20 else 1 for idx in range(40)],
        }
    )

    out = bp.make_stratified_group_split(df, seed=7)

    assert set(out["split"]) == {"train", "val", "test"}
    per_group = out.groupby("dropout_source_id")["split"].nunique()
    assert int(per_group.max()) == 1
    counts = out.groupby(["split", "bad_photometry"]).size()
    assert counts.loc[("train", 0)] > 0
    assert counts.loc[("train", 1)] > 0


def test_binary_metrics_and_calibration_are_rank_based() -> None:
    y = np.array([0, 1, 1, 0])
    score = np.array([0.1, 0.9, 0.8, 0.2])

    metrics = bp.binary_metrics(y, score, top_fractions=(0.5,))
    calibration = bp.calibration_by_decile(y, score)

    assert metrics["pr_auc"] == pytest.approx(1.0)
    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["top_50pct_precision"] == pytest.approx(1.0)
    assert metrics["top_50pct_recall"] == pytest.approx(1.0)
    assert {"risk_decile", "n", "mean_score", "observed_rate"}.issubset(calibration.columns)


def test_preprocess_raw_lightcurve_builds_full_and_event_views(tmp_path: Path) -> None:
    lc = tmp_path / "123.dat3"
    _write_dat3(
        lc,
        [
            (100.0, 13.0, 0.02, 1, 1, 0, 0, "ba/F1"),
            (110.0, 13.1, 0.02, 1, 1, 0, 0, "ba/F1"),
            (120.0, 13.8, 0.03, 1, 2, 1, 0, "bb/F1"),
            (130.0, 13.2, 0.02, 1, 2, 1, 0, "bb/F1"),
            (500.0, 13.0, 0.02, 1, 3, 0, 0, "bc/F2"),
            (510.0, 13.1, 0.02, 1, 3, 0, 0, "bc/F2"),
        ],
    )
    row = {
        "dipper_score": 5.0,
        "jumper_score": 1.0,
        "dip_best_t0": 120.0,
        "dip_max_run_duration": 5.0,
    }
    config = bp.RawPreprocessConfig(full_max_points=4, event_max_points=3)

    processed = bp.preprocess_raw_lightcurve(lc, row, config=config)

    assert processed["full_x"].shape == (4, len(bp.RAW_FEATURE_NAMES))
    assert processed["event_x"].shape == (3, len(bp.RAW_FEATURE_NAMES))
    assert int(processed["full_mask"].sum()) == 4
    assert int(processed["event_mask"].sum()) == 3
    assert bool(processed["raw_available"]) is True
    assert bool(processed["event_window_available"]) is True
    assert processed["flags"].tolist() == [1.0, 1.0]


def test_raw_model_forward_tiny_config() -> None:
    torch = pytest.importorskip("torch")
    config = bp.RawModelConfig(
        d_model=8,
        residual_blocks=1,
        transformer_layers=1,
        attention_heads=2,
        dropout=0.0,
        embedding_dim=8,
    )
    model = bp.create_raw_model(input_dim=len(bp.RAW_FEATURE_NAMES), config=config)
    full_x = torch.randn(2, 5, len(bp.RAW_FEATURE_NAMES))
    event_x = torch.randn(2, 3, len(bp.RAW_FEATURE_NAMES))
    full_mask = torch.tensor([[True, True, True, False, False], [True, True, False, False, False]])
    event_mask = torch.tensor([[True, True, False], [True, False, False]])
    flags = torch.ones(2, 2)

    logits, embedding = model(full_x, full_mask, event_x, event_mask, flags, return_embedding=True)

    assert logits.shape == (2,)
    assert embedding.shape == (2, 8)
