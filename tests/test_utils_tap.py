from __future__ import annotations

import concurrent.futures

import pandas as pd
from astropy.table import Table

import malca.utils as utils


def _coords_df(n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "_idx": list(range(n)),
            "ra": [10.0 + 0.01 * i for i in range(n)],
            "dec": [-5.0 - 0.01 * i for i in range(n)],
        }
    )


def test_batch_tap_crossmatch_retries_async_result_404(monkeypatch) -> None:
    class FakeAsyncJob:
        def __init__(self) -> None:
            self.calls = 0

        def wait_for_job_end(self, *, verbose: bool = False):
            _ = verbose
            return None, "COMPLETED"

        def get_phase(self, *, update: bool = False):
            _ = update
            return "COMPLETED"

        def get_results(self):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError('404 Error 404: No result identified with "result" in the job "123"!')
            return Table.from_pandas(
                pd.DataFrame({"_idx": [0], "main_id": ["SIMBAD 1"], "sep_arcsec": [0.2]})
            )

    job = FakeAsyncJob()

    class FakeTapPlus:
        def __init__(self, url: str) -> None:
            self.url = url

        def launch_job_async(self, query: str, **kwargs):
            _ = (query, kwargs)
            return job

        def launch_job(self, *args, **kwargs):
            raise AssertionError("sync fallback should not be used")

    monkeypatch.setattr(utils, "TapPlus", FakeTapPlus)
    monkeypatch.setattr(utils.time, "sleep", lambda _s: None)

    result = utils.batch_tap_crossmatch(
        _coords_df(51),
        tap_url="https://example.test/tap",
        catalog_table="basic",
        select_cols="c.main_id",
        chunk_size=51,
        n_workers=1,
        raise_on_all_failed=True,
    )

    assert job.calls == 2
    assert result.loc[0, "main_id"] == "SIMBAD 1"


def test_batch_tap_crossmatch_falls_back_to_sync_subchunks(monkeypatch) -> None:
    class FakeAsyncJob:
        def wait_for_job_end(self, *, verbose: bool = False):
            _ = verbose
            return None, "COMPLETED"

        def get_phase(self, *, update: bool = False):
            _ = update
            return "COMPLETED"

        def get_results(self):
            raise RuntimeError('404 Error 404: No result identified with "result" in the job "123"!')

    class FakeSyncJob:
        def __init__(self, upload_resource: Table) -> None:
            self.upload_resource = upload_resource

        def get_results(self):
            idx = list(self.upload_resource["_idx"])
            return Table.from_pandas(
                pd.DataFrame(
                    {
                        "_idx": idx,
                        "main_id": [f"SIMBAD {i}" for i in idx],
                        "sep_arcsec": [0.3] * len(idx),
                    }
                )
            )

    class FakeTapPlus:
        def __init__(self, url: str) -> None:
            self.url = url

        def launch_job_async(self, query: str, **kwargs):
            _ = (query, kwargs)
            return FakeAsyncJob()

        def launch_job(self, query: str, **kwargs):
            _ = query
            return FakeSyncJob(kwargs["upload_resource"])

    monkeypatch.setattr(utils, "TapPlus", FakeTapPlus)
    monkeypatch.setattr(utils.time, "sleep", lambda _s: None)

    result = utils.batch_tap_crossmatch(
        _coords_df(51),
        tap_url="https://example.test/tap",
        catalog_table="basic",
        select_cols="c.main_id",
        chunk_size=51,
        n_workers=1,
        raise_on_all_failed=True,
    )

    assert len(result) == 51
    assert result["main_id"].iloc[0] == "SIMBAD 0"
    assert result["main_id"].iloc[-1] == "SIMBAD 50"


def test_batch_tap_crossmatch_raises_when_all_chunks_fail(monkeypatch) -> None:
    class FakeAsyncJob:
        def wait_for_job_end(self, *, verbose: bool = False):
            _ = verbose
            return None, "COMPLETED"

        def get_phase(self, *, update: bool = False):
            _ = update
            return "COMPLETED"

        def get_results(self):
            raise RuntimeError('404 Error 404: No result identified with "result" in the job "123"!')

    class FakeTapPlus:
        def __init__(self, url: str) -> None:
            self.url = url

        def launch_job_async(self, query: str, **kwargs):
            _ = (query, kwargs)
            return FakeAsyncJob()

        def launch_job(self, query: str, **kwargs):
            _ = (query, kwargs)
            raise RuntimeError("sync failed")

    monkeypatch.setattr(utils, "TapPlus", FakeTapPlus)
    monkeypatch.setattr(utils.time, "sleep", lambda _s: None)

    try:
        utils.batch_tap_crossmatch(
            _coords_df(51),
            tap_url="https://example.test/tap",
            catalog_table="basic",
            select_cols="c.main_id",
            chunk_size=51,
            n_workers=1,
            raise_on_all_failed=True,
        )
    except RuntimeError as exc:
        assert "all chunk queries failed" in str(exc)
    else:
        raise AssertionError("expected batch_tap_crossmatch to raise")


def test_batch_tap_crossmatch_handles_concurrent_futures_timeout(monkeypatch) -> None:
    class FakeFuture:
        def __init__(self) -> None:
            self._done = False
            self.cancelled = False

        def done(self) -> bool:
            return self._done

        def cancel(self) -> None:
            self.cancelled = True

    class FakeExecutor:
        def __init__(self, max_workers: int) -> None:
            self.max_workers = max_workers
            self.futures: list[FakeFuture] = []

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def submit(self, fn, chunk):
            _ = (fn, chunk)
            fut = FakeFuture()
            self.futures.append(fut)
            return fut

    def fake_as_completed(futures, timeout=None):
        _ = (futures, timeout)
        raise concurrent.futures.TimeoutError("6 (of 10) futures unfinished")

    monkeypatch.setattr(utils, "ThreadPoolExecutor", FakeExecutor)
    monkeypatch.setattr(utils, "as_completed", fake_as_completed)

    try:
        utils.batch_tap_crossmatch(
            _coords_df(3),
            tap_url="https://example.test/tap",
            catalog_table="basic",
            select_cols="c.main_id",
            chunk_size=1,
            n_workers=1,
            raise_on_all_failed=True,
        )
    except RuntimeError as exc:
        assert "all chunk queries failed" in str(exc)
        assert "timed out" in str(exc)
    else:
        raise AssertionError("expected batch_tap_crossmatch to convert timeout into RuntimeError")
