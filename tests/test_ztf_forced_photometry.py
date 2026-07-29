from __future__ import annotations

from pathlib import Path

import pandas as pd

from malca.enrichment.ztf_forced_photometry import (
    ZTF_FORCED_STATUS_URL,
    parse_ztf_forced_result,
    query_ztf_forced_phot,
)


RAW_PRODUCT = """# Requested input R.A. = 12.3456789 degrees
# Requested input Dec. = -4.5000000 degrees
# index, field, filter, jd, forcediffimflux, forcediffimfluxunc, procstatus
0 1 ZTF_g 2459000.0 10.0 2.0 0
1 1 ZTF_r 2459001.0 -1.0 3.0 56
"""


class _Response:
    def __init__(self, text: str = "", status_code: int = 200):
        self.text = text
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _Session:
    def __init__(self, *, ready: bool):
        self.ready = ready
        self.posts: list[dict] = []

    def post(self, _url, **kwargs):
        self.posts.append(kwargs)
        return _Response("accepted")

    def get(self, url, **_kwargs):
        if url == ZTF_FORCED_STATUS_URL:
            return _Response('/ztf/ops/forcedphot/lc/batchfp/x/batchfp_req0001_lc.txt' if self.ready else "queued")
        return _Response(RAW_PRODUCT)


def test_parse_ztf_forced_result_preserves_raw_columns() -> None:
    parsed = parse_ztf_forced_result(RAW_PRODUCT)
    assert list(parsed.columns) == ["index", "field", "filter", "jd", "forcediffimflux", "forcediffimfluxunc", "procstatus"]
    assert len(parsed) == 2


def test_ztf_forced_client_submits_then_downloads_without_resubmit(tmp_path: Path) -> None:
    candidates = pd.DataFrame({"candidate_id": ["C1"], "ra": [12.3456789], "dec": [-4.5]})
    first = _Session(ready=False)
    query_ztf_forced_phot(
        candidates, email="person@example.edu", userpass="secret", output_dir=tmp_path,
        jd_start=2458194.5, jd_end=2459002.5, session=first,
    )
    assert len(first.posts) == 1
    assert "secret" not in (tmp_path / "ztf_forced_phot_tasks.parquet").read_bytes().decode("latin1", errors="ignore")

    second = _Session(ready=True)
    out = query_ztf_forced_phot(
        candidates, email="person@example.edu", userpass="secret", output_dir=tmp_path,
        jd_start=2458194.5, jd_end=2459002.5, session=second,
    )
    assert not second.posts
    assert out.loc[0, "ztf_forced_lc_n_epochs"] == 2
    assert out.loc[0, "ztf_forced_lc_n_good"] == 1
    assert (tmp_path / "ztf_forced_lc_C1.parquet").exists()
