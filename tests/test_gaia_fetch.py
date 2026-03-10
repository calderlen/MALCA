from __future__ import annotations

import json
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


def test_zero_row_chunk_is_retried_on_rerun(tmp_path: Path, monkeypatch) -> None:
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
    assert not done_marker.exists()

    second_calls: list[str] = []

    def second_fetch(_tap, chunk_ids: list[str]):
        key = ",".join(chunk_ids)
        second_calls.append(key)
        if key == "1,2":
            return _mk_chunk_df(["1", "2"])
        raise AssertionError(f"unexpected chunk key {key}")

    monkeypatch.setattr(gaia_fetch, "_fetch_chunk", second_fetch)
    second_df = gaia_fetch.fetch_gaia_catalog(["1", "2", "3", "4"], output_path=output, chunk_size=2)

    # The empty chunk should be retried because it was not checkpointed.
    assert second_calls == ["1,2"]
    assert sorted(second_df["source_id"].astype(str).tolist()) == ["1", "2", "3", "4"]


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
    assert df.columns[0] == "source_id"
    assert "ra" in df.columns
    assert "tmass_j" in df.columns
    assert "unwise_w1" in df.columns
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


def test_all_failed_chunks_do_not_write_empty_cache(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "gaia_catalog.parquet"

    monkeypatch.setattr(gaia_fetch.pyvo.dal, "TAPService", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(gaia_fetch, "_fetch_chunk", lambda *_args, **_kwargs: None)

    try:
        gaia_fetch.fetch_gaia_catalog(["1", "2"], output_path=output, chunk_size=2)
    except RuntimeError as exc:
        assert "no valid rows" in str(exc).lower()
    else:
        raise AssertionError("expected fetch_gaia_catalog to fail")

    assert not output.exists()


def test_existing_valid_cache_is_not_overwritten_on_failed_refresh(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "gaia_catalog.parquet"
    original = gaia_fetch._ensure_gaia_schema(_mk_chunk_df(["1", "2"]))
    original.to_parquet(output, index=False)

    monkeypatch.setattr(gaia_fetch.pyvo.dal, "TAPService", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(gaia_fetch, "_fetch_chunk", lambda *_args, **_kwargs: None)

    out = gaia_fetch.fetch_gaia_catalog(["1", "2", "3"], output_path=output, chunk_size=2)

    assert sorted(out["source_id"].astype(str).tolist()) == ["1", "2"]
    reloaded = pd.read_parquet(output)
    assert sorted(reloaded["source_id"].astype(str).tolist()) == ["1", "2"]


def test_stale_done_marker_does_not_block_refetch(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "gaia_catalog.parquet"
    ckpt_dir = gaia_fetch._checkpoint_dir_for_output(output)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    stale_marker = ckpt_dir / f"{gaia_fetch._chunk_key(['1', '2'])}.done"
    stale_marker.write_text(json.dumps({"row_count": 0, "ids": ["1", "2"]}), encoding="utf-8")

    monkeypatch.setattr(gaia_fetch.pyvo.dal, "TAPService", lambda *_args, **_kwargs: object())

    calls: list[str] = []

    def fake_fetch(_tap, chunk_ids: list[str]):
        calls.append(",".join(chunk_ids))
        return _mk_chunk_df(chunk_ids)

    monkeypatch.setattr(gaia_fetch, "_fetch_chunk", fake_fetch)

    out = gaia_fetch.fetch_gaia_catalog(["1", "2"], output_path=output, chunk_size=2)

    assert calls == ["1,2"]
    assert sorted(out["source_id"].astype(str).tolist()) == ["1", "2"]
