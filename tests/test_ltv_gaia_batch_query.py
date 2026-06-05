from __future__ import annotations

import pandas as pd
import pytest

from malca import utils


class _FakeResult:
    def __init__(self, frame: pd.DataFrame):
        self.frame = frame

    def to_table(self):
        return self

    def to_pandas(self) -> pd.DataFrame:
        return self.frame


def _coords(n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "_idx": list(range(n)),
            "ra": [10.0 + idx for idx in range(n)],
            "dec": [-5.0 - idx for idx in range(n)],
        }
    )


def test_batch_gaia_cone_query_retries_transient_failure(monkeypatch) -> None:
    calls = {"n": 0}

    class FakeService:
        def __init__(self, url: str):
            self.url = url

        def run_async(self, query: str, uploads: dict):
            calls["n"] += 1
            if calls["n"] == 1:
                raise TimeoutError("temporary TAP timeout")
            return _FakeResult(
                pd.DataFrame(
                    {
                        "_idx": [0],
                        "source_id": [123],
                        "pmra": [1.0],
                        "sep_arcsec": [0.2],
                    }
                )
            )

    monkeypatch.setattr(utils.pyvo.dal, "TAPService", FakeService)

    out = utils.batch_gaia_cone_query(
        _coords(1),
        select_cols="g.pmra",
        match_radius_arcsec=3.0,
        chunk_size=1,
        n_workers=1,
        max_attempts=2,
        retry_base_sleep=0,
        raise_on_failed_chunk=True,
    )

    assert calls["n"] == 2
    assert out["_idx"].tolist() == [0]
    assert out["source_id"].tolist() == [123]


def test_batch_gaia_cone_query_raises_when_all_chunks_fail(monkeypatch) -> None:
    class FakeService:
        def __init__(self, url: str):
            self.url = url

        def run_async(self, query: str, uploads: dict):
            raise TimeoutError("Gaia unavailable")

    monkeypatch.setattr(utils.pyvo.dal, "TAPService", FakeService)

    with pytest.raises(RuntimeError, match="all chunk queries failed"):
        utils.batch_gaia_cone_query(
            _coords(1),
            select_cols="g.pmra",
            match_radius_arcsec=3.0,
            chunk_size=1,
            n_workers=1,
            max_attempts=2,
            retry_base_sleep=0,
            raise_on_all_failed=True,
        )


def test_batch_gaia_cone_query_raises_on_partial_failure_when_strict(monkeypatch) -> None:
    class FakeService:
        def __init__(self, url: str):
            self.url = url

        def run_async(self, query: str, uploads: dict):
            idx = int(uploads["upload_table"]["_idx"][0])
            if idx == 0:
                return _FakeResult(
                    pd.DataFrame(
                        {
                            "_idx": [0],
                            "source_id": [123],
                            "pmra": [1.0],
                            "sep_arcsec": [0.2],
                        }
                    )
                )
            raise TimeoutError("second chunk failed")

    monkeypatch.setattr(utils.pyvo.dal, "TAPService", FakeService)

    with pytest.raises(RuntimeError, match=r"1/2 chunk"):
        utils.batch_gaia_cone_query(
            _coords(2),
            select_cols="g.pmra",
            match_radius_arcsec=3.0,
            chunk_size=1,
            n_workers=1,
            max_attempts=1,
            retry_base_sleep=0,
            raise_on_failed_chunk=True,
        )
