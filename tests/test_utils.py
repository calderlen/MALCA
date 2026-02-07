from __future__ import annotations

import numpy as np
import pandas as pd

from malca.utils import filter_bad_cameras


def _make_camera_df(seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    n = 120
    for cam in (1, 2):
        jd = 2458000.0 + np.arange(n, dtype=float)
        mag = 14.0 + rng.normal(0.0, 0.02, size=n)
        err = np.full(n, 0.02)
        for t, m, e in zip(jd, mag, err):
            rows.append({"JD": t, "mag": m, "error": e, "camera#": cam})
    return pd.DataFrame(rows)


def test_filter_bad_cameras_flags_isolated_catastrophic_camera():
    df = _make_camera_df()

    # Inject a single catastrophic outlier in camera 1 only.
    idx = df.index[(df["camera#"] == 1) & (df["JD"] == 2458060.0)]
    assert len(idx) == 1
    df.loc[idx[0], "mag"] = 18.2

    df_filtered, bad = filter_bad_cameras(
        df,
        filter_scatter=False,
        filter_offset=False,
        filter_catastrophic=True,
        catastrophic_mag_excursion=3.0,
        catastrophic_min_count=1,
        catastrophic_max_fraction=0.05,
    )

    assert 1 in bad
    assert (df_filtered["camera#"] == 1).sum() == 0
    assert (df_filtered["camera#"] == 2).sum() > 0
