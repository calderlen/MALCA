from __future__ import annotations

import numpy as np
import pandas as pd

from malca.lightcurve_publication import (
    filter_lightcurve,
    load_lightcurve,
    plot_lightcurve,
    resolve_time_axis,
)


def test_generic_csv_aliases_are_normalized(tmp_path):
    csv_path = tmp_path / "generic.csv"
    pd.DataFrame(
        {
            "mjd": [59000.0, 59001.0, 59002.0, 59003.0],
            "magnitude": [14.1, 14.2, 14.3, 14.4],
            "mag_err": [0.02, 0.03, 5.0, 0.02],
            "passband": ["g", "V", "g", "V"],
            "camera_id": ["a", "a", "b", "b"],
            "quality": ["G", "G", "G", "B"],
        }
    ).to_csv(csv_path, index=False)

    lc = load_lightcurve(csv_path)
    filtered = filter_lightcurve(lc)

    assert lc.time_column == "mjd"
    assert lc.y_kind == "mag"
    assert filtered["band"].tolist() == ["g", "V"]
    assert filtered["camera"].tolist() == ["a", "a"]


def test_dat2_file_uses_existing_loader(tmp_path):
    dat_path = tmp_path / "123.dat2"
    dat_path.write_text(
        "\n".join(
            [
                "7479.8 14.10 0.02 1 4 0 0 ba/F1",
                "7480.8 14.20 0.03 1 5 1 0 bb/F1",
            ]
        )
    )

    lc = load_lightcurve(dat_path)
    filtered = filter_lightcurve(lc)

    assert filtered["band"].tolist() == ["g", "V"]
    assert filtered["camera"].tolist() == ["4", "5"]


def test_resolve_time_axis_auto_for_full_jd():
    plotted, label = resolve_time_axis(
        pd.Series([2457000.0, 2457001.0]),
        source_column="JD",
        offset="auto",
    )

    assert label == "JD - 2450000"
    assert np.allclose(plotted, [7000.0, 7001.0])


def test_plot_lightcurve_writes_output(tmp_path):
    csv_path = tmp_path / "sky.csv"
    pd.DataFrame(
        {
            "JD": [2457000.0, 2457001.0, 2457002.0],
            "Mag": [13.0, 13.1, 13.05],
            "Mag Error": [0.01, 0.02, 0.02],
            "Filter": ["g", "g", "V"],
            "Quality": ["G", "G", "G"],
            "Camera": ["ba", "ba", "bb"],
        }
    ).to_csv(csv_path, index=False)
    lc = load_lightcurve(csv_path)
    filtered = filter_lightcurve(lc)
    output = tmp_path / "plot.png"

    plot_lightcurve(lc, filtered, output=output, title="", group_by="band")

    assert output.exists()
    assert output.stat().st_size > 0
