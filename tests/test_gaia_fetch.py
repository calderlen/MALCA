from __future__ import annotations

from pathlib import Path

import pandas as pd
from astropy.table import Table

import malca.gaia_fetch as gaia_fetch


def _mk_chunk_df(ids: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_id": ids,
            "ra": [10.0] * len(ids),
            "dec": [-10.0] * len(ids),
        }
    )


def test_fetch_gaia_catalog_resumes_from_chunk_checkpoints(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "gaia_catalog.parquet"

    # Avoid real network setup.
    monkeypatch.setattr(gaia_fetch.pyvo.dal, "TAPService", lambda *_args, **_kwargs: object())

    # First run: first chunk succeeds, second chunk fails.
    first_calls: list[str] = []

    def first_fetch(_tap, chunk_ids: list[str]):
        key = ",".join(chunk_ids)
        first_calls.append(key)
        if key == "1,2":
            return _mk_chunk_df(["1", "2"])
        if key == "3,4":
            return None
        raise AssertionError(f"unexpected chunk key {key}")

    monkeypatch.setattr(gaia_fetch, "_fetch_chunk", first_fetch)
    first_df = gaia_fetch.fetch_gaia_catalog(["1", "2", "3", "4"], output_path=output, chunk_size=2)

    assert first_calls == ["1,2", "3,4"]
    assert sorted(first_df["source_id"].astype(str).tolist()) == ["1", "2"]

    ckpt_dir = gaia_fetch._checkpoint_dir_for_output(output)
    assert (ckpt_dir / f"{gaia_fetch._chunk_key(['1', '2'])}.parquet").exists()
    assert not (ckpt_dir / f"{gaia_fetch._chunk_key(['3', '4'])}.parquet").exists()

    # Second run: only missing chunk should be queried.
    second_calls: list[str] = []

    def second_fetch(_tap, chunk_ids: list[str]):
        key = ",".join(chunk_ids)
        second_calls.append(key)
        if key == "3,4":
            return _mk_chunk_df(["3", "4"])
        raise AssertionError(f"unexpected chunk key {key}")

    monkeypatch.setattr(gaia_fetch, "_fetch_chunk", second_fetch)
    second_df = gaia_fetch.fetch_gaia_catalog(["1", "2", "3", "4"], output_path=output, chunk_size=2)

    assert second_calls == ["3,4"]
    assert sorted(second_df["source_id"].astype(str).tolist()) == ["1", "2", "3", "4"]


def test_zero_row_chunk_creates_done_marker_and_skips_rerun(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "gaia_catalog.parquet"

    monkeypatch.setattr(gaia_fetch.pyvo.dal, "TAPService", lambda *_args, **_kwargs: object())

    first_calls: list[str] = []

    def first_fetch(_tap, chunk_ids: list[str]):
        key = ",".join(chunk_ids)
        first_calls.append(key)
        if key == "1,2":
            return _mk_chunk_df([])
        if key == "3,4":
            return _mk_chunk_df(["3", "4"])
        raise AssertionError(f"unexpected chunk key {key}")

    monkeypatch.setattr(gaia_fetch, "_fetch_chunk", first_fetch)
    gaia_fetch.fetch_gaia_catalog(["1", "2", "3", "4"], output_path=output, chunk_size=2)

    ckpt_dir = gaia_fetch._checkpoint_dir_for_output(output)
    done_marker = ckpt_dir / f"{gaia_fetch._chunk_key(['1', '2'])}.done"
    assert done_marker.exists()

    second_calls: list[str] = []

    def second_fetch(_tap, chunk_ids: list[str]):
        second_calls.append(",".join(chunk_ids))
        return _mk_chunk_df(["99"])

    monkeypatch.setattr(gaia_fetch, "_fetch_chunk", second_fetch)
    second_df = gaia_fetch.fetch_gaia_catalog(["1", "2", "3", "4"], output_path=output, chunk_size=2)

    # Nothing should be re-queried: both chunks were checkpointed in run 1.
    assert second_calls == []
    assert sorted(second_df["source_id"].astype(str).tolist()) == ["3", "4"]


def test_resume_works_when_chunk_size_changes(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "gaia_catalog.parquet"

    monkeypatch.setattr(gaia_fetch.pyvo.dal, "TAPService", lambda *_args, **_kwargs: object())

    first_calls: list[str] = []

    def first_fetch(_tap, chunk_ids: list[str]):
        first_calls.append(",".join(chunk_ids))
        # First run with chunk_size=2: both chunks succeed.
        return _mk_chunk_df(chunk_ids)

    monkeypatch.setattr(gaia_fetch, "_fetch_chunk", first_fetch)
    gaia_fetch.fetch_gaia_catalog(["1", "2", "3", "4"], output_path=output, chunk_size=2)

    assert first_calls == ["1,2", "3,4"]

    second_calls: list[str] = []

    def second_fetch(_tap, chunk_ids: list[str]):
        second_calls.append(",".join(chunk_ids))
        return _mk_chunk_df(chunk_ids)

    monkeypatch.setattr(gaia_fetch, "_fetch_chunk", second_fetch)
    # Rerun with a different chunk size should perform no network calls.
    second_df = gaia_fetch.fetch_gaia_catalog(["1", "2", "3", "4"], output_path=output, chunk_size=3)

    assert second_calls == []
    assert sorted(second_df["source_id"].astype(str).tolist()) == ["1", "2", "3", "4"]


def test_fetch_chunk_uses_async_tap_upload() -> None:
    class FakeResult:
        def __init__(self, df: pd.DataFrame) -> None:
            self._df = df

        def to_table(self) -> Table:
            return Table.from_pandas(self._df)

    class FakeTap:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, object]]] = []

        def run_async(self, query: str, uploads: dict[str, object]) -> FakeResult:
            self.calls.append((query, uploads))
            return FakeResult(pd.DataFrame({"SOURCE_ID": [1], "RA": [123.4]}))

    tap = FakeTap()
    df = gaia_fetch._fetch_chunk(tap, ["1", "2"])

    assert df is not None
    assert list(df.columns) == ["source_id", "ra"]
    assert len(tap.calls) == 1

    query, uploads = tap.calls[0]
    assert "TAP_UPLOAD.upload_table" in query
    assert "upload_table" in uploads
    upload = uploads["upload_table"]
    assert isinstance(upload, Table)
    assert list(upload["source_id"]) == [1, 2]


def test_fetch_chunk_retries_async_errors(monkeypatch) -> None:
    class FakeResult:
        def __init__(self, df: pd.DataFrame) -> None:
            self._df = df

        def to_table(self) -> Table:
            return Table.from_pandas(self._df)

    class FlakyTap:
        def __init__(self) -> None:
            self.n_calls = 0

        def run_async(self, _query: str, uploads: dict[str, object]) -> FakeResult:
            _ = uploads
            self.n_calls += 1
            if self.n_calls < 2:
                raise RuntimeError("temporary tap failure")
            return FakeResult(pd.DataFrame({"source_id": [42]}))

    monkeypatch.setattr(gaia_fetch.time, "sleep", lambda _s: None)

    tap = FlakyTap()
    df = gaia_fetch._fetch_chunk(tap, ["42"])

    assert df is not None
    assert tap.n_calls == 2
    assert list(df["source_id"].astype(int)) == [42]
