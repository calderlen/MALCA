from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from malca.enrichment.atlas_forced_photometry import (
    ATLAS_PREPROCESS_VERSION,
    ATLAS_SUMMARY_COLUMNS,
    parse_atlas_result,
    query_atlas_forced_phot,
    summarize_atlas_lc,
)


ATLAS_TEXT = """###MJD m dm uJy duJy F err chi/N RA Dec x y maj min phi apfit mag5sig Sky Obs
59000 15.0 0.1 3.0 0.2 c 0 1.0 1.0 2.0 500 500 2 2 0 -0.5 19 20 obs1
59001 16.0 0.1 2.0 0.2 c 0 1.0 1.0 2.0 500 500 2 2 0 -0.5 19 20 obs2
59002 14.0 0.1 4.0 0.2 o 0 1.0 1.0 2.0 500 500 2 2 0 -0.5 19 20 obs3
"""
ATLAS_HEADER_ONLY = ATLAS_TEXT.splitlines()[0]


class FakeResponse:
    def __init__(
        self,
        status_code: int,
        payload: object = None,
        *,
        text: str = "",
        headers: dict[str, str] | None = None,
        json_error: Exception | None = None,
    ) -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text
        self.headers = headers or {}
        self._json_error = json_error

    def json(self):
        if self._json_error is not None:
            raise self._json_error
        return self._payload


def _frame(n: int = 1) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "candidate_id": [f"C{i}" for i in range(n)],
            "ra": [i / 2.0 + 1.0 for i in range(n)],
            "dec": [2.0] * n,
        }
    )


def _task_items(data: dict[str, object], first_id: int = 1) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    for offset, line in enumerate(str(data["radeclist"]).splitlines()):
        ra, dec = (float(value) for value in line.split(","))
        task_id = first_id + offset
        items.append(
            {
                "url": f"https://fallingstar-data.com/forcedphot/queue/{task_id}/",
                "id": task_id,
                "ra": ra,
                "dec": dec,
                "mjd_min": float(data["mjd_min"]),
                "mjd_max": data.get("mjd_max"),
                "use_reduced": bool(data["use_reduced"]),
            }
        )
    return items


def test_parse_preserves_atlas_header_and_full_table() -> None:
    result = parse_atlas_result(ATLAS_TEXT)

    assert result.columns.tolist() == [
        "MJD", "m", "dm", "uJy", "duJy", "F", "err", "chi/N", "RA", "Dec",
        "x", "y", "maj", "min", "phi", "apfit", "mag5sig", "Sky", "Obs",
    ]
    assert len(result) == 3
    result["atlas_image_type"] = "reduced"
    assert summarize_atlas_lc(result) == {
        "atlas_has_phot": True,
        "atlas_n_det_cyan": 2,
        "atlas_n_det_orange": 1,
        "atlas_cyan_range": 0.4402,
        "atlas_orange_range": pytest.approx(float("nan"), nan_ok=True),
        "atlas_preprocess_version": ATLAS_PREPROCESS_VERSION,
        "atlas_n_raw": 3,
        "atlas_n_good": 3,
        "atlas_n_rejected": 0,
    }
    empty = parse_atlas_result(ATLAS_HEADER_ONLY)
    assert empty.empty
    assert list(summarize_atlas_lc(empty)) == list(ATLAS_SUMMARY_COLUMNS)


def test_bulk_submission_batches_100_plus_54_and_resume_never_reposts(tmp_path: Path) -> None:
    class SubmitSession:
        def __init__(self) -> None:
            self.posts: list[dict[str, object]] = []
            self.next_id = 1

        def post(self, _url: str, **kwargs):
            data = kwargs["data"]
            self.posts.append(data)
            items = _task_items(data, self.next_id)
            self.next_id += len(items)
            return FakeResponse(201, items)

        def get(self, *_args, **_kwargs):
            raise AssertionError("submit_only must not poll")

    root = tmp_path / "results"
    lc_dir = root / "external_lcs"
    session = SubmitSession()
    submitted = query_atlas_forced_phot(
        _frame(154),
        token="secret",
        output_dir=lc_dir,
        results_root=root,
        submit_only=True,
        session=session,
        progress=lambda _message: None,
    )

    assert [len(str(data["radeclist"]).splitlines()) for data in session.posts] == [100, 54]
    assert all(data["use_reduced"] is True for data in session.posts)
    assert all(data["send_email"] is False for data in session.posts)
    assert submitted["atlas_has_phot"].isna().all()
    ledger = pd.read_parquet(lc_dir / "atlas_forced_phot_tasks.parquet")
    assert len(ledger) == 154
    assert set(ledger["status"]) == {"queued"}
    assert ledger["task_url"].str.endswith("/").all()

    class NoNetworkSession:
        def post(self, *_args, **_kwargs):
            raise AssertionError("resume attempted a duplicate POST")

        def get(self, *_args, **_kwargs):
            raise AssertionError("submit_only must not poll")

    resumed = query_atlas_forced_phot(
        _frame(154),
        token="secret",
        output_dir=lc_dir,
        results_root=root,
        submit_only=True,
        session=NoNetworkSession(),
        progress=lambda _message: None,
    )
    assert resumed["atlas_has_phot"].isna().all()


def test_completed_result_writes_raw_provenance_and_parent_manifest(tmp_path: Path) -> None:
    class CompleteSession:
        def post(self, _url: str, **_kwargs):
            # URL-only objects are valid because list response order is stable.
            return FakeResponse(201, [{"url": "/forcedphot/queue/7/", "id": 7}])

        def get(self, url: str, **_kwargs):
            if "/queue/7/" in url:
                return FakeResponse(
                    200,
                    {
                        "id": 7,
                        "starttimestamp": "start",
                        "finishtimestamp": "finish",
                        "result_url": "https://example.test/result.txt",
                        "error_msg": None,
                    },
                )
            return FakeResponse(200, text=ATLAS_TEXT)

    root = tmp_path / "results"
    lc_dir = root / "external_lcs"
    out = query_atlas_forced_phot(
        _frame(),
        token="secret",
        output_dir=lc_dir,
        results_root=root,
        poll_interval=0,
        session=CompleteSession(),
        progress=lambda _message: None,
    )

    assert bool(out.loc[0, "atlas_has_phot"])
    path = lc_dir / "atlas_lc_C0.parquet"
    raw = pd.read_parquet(path)
    assert {
        "MJD", "m", "dm", "F", "mjd", "mag", "mag_err", "filter",
        "candidate_id", "request_key", "task_id", "task_url", "result_url",
        "atlas_image_type",
    } <= set(raw.columns)
    assert raw["candidate_id"].eq("C0").all()
    assert raw.attrs["atlas_image_types"] == ["reduced"]
    manifest = pd.read_parquet(root / "external_lc_manifest.parquet")
    assert manifest[["candidate_id", "path_relative"]].to_dict("records") == [
        {"candidate_id": "C0", "path_relative": "external_lcs/atlas_lc_C0.parquet"}
    ]


def test_429_invalid_poll_json_and_expired_result_url_are_retried(tmp_path: Path) -> None:
    class RecoverySession:
        def __init__(self) -> None:
            self.post_count = 0
            self.task_poll_count = 0

        def post(self, _url: str, **kwargs):
            self.post_count += 1
            if self.post_count == 1:
                return FakeResponse(429, {"detail": "Expected available in 2 seconds."})
            return FakeResponse(201, _task_items(kwargs["data"]))

        def get(self, url: str, **_kwargs):
            if "/queue/1/" in url:
                self.task_poll_count += 1
                if self.task_poll_count == 1:
                    return FakeResponse(200, json_error=ValueError("temporary proxy body"))
                result_url = (
                    "https://example.test/expired.txt"
                    if self.task_poll_count == 2
                    else "https://example.test/fresh.txt"
                )
                return FakeResponse(
                    200,
                    {
                        "id": 1,
                        "starttimestamp": "start",
                        "finishtimestamp": "finish",
                        "result_url": result_url,
                        "error_msg": None,
                    },
                )
            if url.endswith("expired.txt"):
                return FakeResponse(404, text="expired")
            return FakeResponse(200, text=ATLAS_TEXT)

    sleeps: list[float] = []
    root = tmp_path / "results"
    session = RecoverySession()
    out = query_atlas_forced_phot(
        _frame(),
        token="secret",
        output_dir=root / "external_lcs",
        results_root=root,
        poll_interval=0,
        session=session,
        sleep_func=sleeps.append,
        progress=lambda _message: None,
    )

    assert sleeps == [2.0]
    assert session.task_poll_count == 3
    assert bool(out.loc[0, "atlas_has_phot"])


def test_submitting_batch_reconciles_remote_url_without_duplicate_post(tmp_path: Path) -> None:
    class LostResponseSession:
        def post(self, *_args, **_kwargs):
            raise ConnectionError("connection lost after server acceptance")

    root = tmp_path / "results"
    lc_dir = root / "external_lcs"
    frame = _frame()
    with pytest.raises(RuntimeError, match="submission failed"):
        query_atlas_forced_phot(
            frame,
            token="secret",
            output_dir=lc_dir,
            results_root=root,
            session=LostResponseSession(),
            progress=lambda _message: None,
        )
    ledger = pd.read_parquet(lc_dir / "atlas_forced_phot_tasks.parquet")
    assert ledger.loc[0, "status"] == "submitting"
    comment = ledger.loc[0, "batch_comment"]

    class ReconcileSession:
        def __init__(self) -> None:
            self.posts = 0

        def post(self, *_args, **_kwargs):
            self.posts += 1
            raise AssertionError("reconciliation attempted a duplicate POST")

        def get(self, url: str, **_kwargs):
            assert "pagesize=500" in url
            return FakeResponse(
                200,
                {
                    "next": None,
                    "results": [
                        {
                            "url": "https://fallingstar-data.com/forcedphot/queue/44/",
                            "id": 44,
                            "comment": comment,
                            "ra": 1.0,
                            "dec": 2.0,
                            "mjd_min": 57000.0,
                            "mjd_max": None,
                            "use_reduced": True,
                        }
                    ],
                },
            )

    session = ReconcileSession()
    out = query_atlas_forced_phot(
        frame,
        token="secret",
        output_dir=lc_dir,
        results_root=root,
        submit_only=True,
        session=session,
        progress=lambda _message: None,
    )
    assert session.posts == 0
    assert out["atlas_has_phot"].isna().all()
    recovered = pd.read_parquet(lc_dir / "atlas_forced_phot_tasks.parquet")
    assert recovered.loc[0, "status"] == "queued"
    assert recovered.loc[0, "task_url"].endswith("/44/")


def test_header_only_result_is_terminal_no_data_with_mode_provenance(tmp_path: Path) -> None:
    class EmptySession:
        def post(self, _url: str, **kwargs):
            return FakeResponse(201, _task_items(kwargs["data"]))

        def get(self, url: str, **_kwargs):
            if "/queue/1/" in url:
                return FakeResponse(
                    200,
                    {
                        "id": 1,
                        "starttimestamp": "start",
                        "finishtimestamp": "finish",
                        "result_url": "https://example.test/empty.txt",
                        "error_msg": None,
                    },
                )
            return FakeResponse(200, text=ATLAS_HEADER_ONLY)

    root = tmp_path / "results"
    lc_dir = root / "external_lcs"
    out = query_atlas_forced_phot(
        _frame(),
        token="secret",
        output_dir=lc_dir,
        results_root=root,
        poll_interval=0,
        session=EmptySession(),
        progress=lambda _message: None,
    )

    assert out.loc[0, "atlas_has_phot"] is False
    assert out.loc[0, "atlas_n_det_cyan"] == 0
    empty = pd.read_parquet(lc_dir / "atlas_lc_C0.parquet")
    assert empty.empty
    assert empty.attrs["atlas_image_types"] == ["reduced"]
    ledger = pd.read_parquet(lc_dir / "atlas_forced_phot_tasks.parquet")
    assert ledger.loc[0, "status"] == "no_data"
