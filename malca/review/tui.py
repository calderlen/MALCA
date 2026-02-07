from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from malca.review.store import (
    DEFAULT_DB_PATH,
    INTEREST_REASON_TAGS,
    STATUS_OPTIONS,
    db_connect,
    export_reviews,
    get_candidate_payload,
    get_review,
    import_candidates,
    load_candidates_file,
    query_queue,
    save_review,
)
from malca.review.metadata import extract_review_metadata


def _print_candidate(row: pd.Series, review: dict) -> None:
    cid = str(row["candidate_id"])
    print(f"\n[{cid}] score={review['interest_score']} pass={review['review_pass']} status={review['status']}")
    if review["interest_reason"]:
        print("  reasons:", ", ".join(review["interest_reason"]))
    print(
        "  lsp: power={0} p={1} score={2}".format(
            row.get("lsp_power", "-"), row.get("lsp_bootstrap_sig", "-"), row.get("periodicity_score", "-")
        )
    )


def main_with_args(*, db: Path | None = None, input_path: Path | None = None, reviewer: str = "") -> None:
    db_resolved = (db or Path(DEFAULT_DB_PATH)).expanduser()
    conn = db_connect(db_resolved)

    if input_path:
        src = input_path.expanduser()
        n_rows, n_new = import_candidates(conn, load_candidates_file(src), str(src))
        print(f"Imported {n_rows} rows ({n_new} new candidate IDs)")

    idx = 0

    while True:
        queue_df = query_queue(
            conn,
            only_unreviewed=False,
            require_failed_any_false=False,
            periodic_flag_mode="Any",
            catalog_match_mode="Any",
            high_ruwe_mode="Any",
            min_periodicity_score=None,
            max_lsp_bootstrap_sig=None,
            min_lsp_power=None,
            sort_col="candidate_id",
            sort_desc=False,
        )
        if queue_df.empty:
            print("No candidates in DB. Use --input to import.")
            return

        idx = max(0, min(idx, len(queue_df) - 1))
        row = queue_df.iloc[idx]
        cid = str(row["candidate_id"])
        review = get_review(conn, cid)
        _print_candidate(row, review)

        cmd = input("review> ").strip()
        if not cmd:
            continue
        parts = cmd.split()
        op = parts[0].lower()

        if op in {"q", "quit", "exit"}:
            break
        if op in {"n", "next"}:
            idx = min(len(queue_df) - 1, idx + 1)
            continue
        if op in {"p", "prev"}:
            idx = max(0, idx - 1)
            continue
        if op == "goto" and len(parts) > 1:
            try:
                idx = int(parts[1])
            except Exception:
                matches = queue_df.index[queue_df["candidate_id"].astype(str) == parts[1]].tolist()
                if matches:
                    idx = int(matches[0])
            continue
        if op == "show":
            payload = get_candidate_payload(conn, cid)
            for label, value in extract_review_metadata(payload):
                print(f"  {label}: {value}")
            continue

        # edits
        score = int(review["interest_score"])
        reasons = list(review["interest_reason"])
        review_pass = int(review["review_pass"])
        status = str(review["status"])
        notes = str(review["notes"])

        if op == "score" and len(parts) > 1:
            score = max(0, min(5, int(parts[1])))
            save_review(
                conn,
                candidate_id=cid,
                interest_score=score,
                interest_reason=reasons,
                review_pass=review_pass,
                notes=notes,
                status=status,
                reviewer=reviewer,
                event_type="score",
            )
            print(f"saved score={score}")
            continue
        if op == "reason" and len(parts) >= 3:
            sub = parts[1].lower()
            tag = parts[2]
            if sub == "add" and tag not in reasons:
                reasons.append(tag)
            elif sub in {"rm", "remove"} and tag in reasons:
                reasons.remove(tag)
            save_review(
                conn,
                candidate_id=cid,
                interest_score=score,
                interest_reason=reasons,
                review_pass=review_pass,
                notes=notes,
                status=status,
                reviewer=reviewer,
                event_type="reason",
            )
            print("saved reasons")
            continue
        if op == "pass" and len(parts) > 1:
            review_pass = max(1, int(parts[1]))
            save_review(
                conn,
                candidate_id=cid,
                interest_score=score,
                interest_reason=reasons,
                review_pass=review_pass,
                notes=notes,
                status=status,
                reviewer=reviewer,
                event_type="pass",
            )
            print(f"saved review_pass={review_pass}")
            continue
        if op == "status" and len(parts) > 1:
            val = parts[1]
            if val in STATUS_OPTIONS:
                status = val
                save_review(
                    conn,
                    candidate_id=cid,
                    interest_score=score,
                    interest_reason=reasons,
                    review_pass=review_pass,
                    notes=notes,
                    status=status,
                    reviewer=reviewer,
                    event_type="status",
                )
                print(f"saved status={status}")
            else:
                print(f"status must be one of: {', '.join(STATUS_OPTIONS)}")
            continue
        if op == "note":
            notes = cmd[len("note"):].strip()
            save_review(
                conn,
                candidate_id=cid,
                interest_score=score,
                interest_reason=reasons,
                review_pass=review_pass,
                notes=notes,
                status=status,
                reviewer=reviewer,
                event_type="note",
            )
            print("saved note")
            continue
        if op == "save":
            save_review(
                conn,
                candidate_id=cid,
                interest_score=score,
                interest_reason=reasons,
                review_pass=review_pass,
                notes=notes,
                status=status,
                reviewer=reviewer,
                event_type="save",
            )
            print("saved")
            if len(parts) > 1 and parts[1].lower() == "next":
                idx = min(len(queue_df) - 1, idx + 1)
            continue
        if op == "tags":
            print("available reason tags:", ", ".join(INTEREST_REASON_TAGS))
            continue
        if op == "export" and len(parts) > 1:
            out = Path(parts[1]).expanduser()
            export_reviews(conn, out, only_reviewed=True)
            print(f"exported: {out}")
            continue

        print("commands: next/prev/goto/show/score/reason/pass/status/note/save [next]/tags/export/quit")


def main() -> None:
    parser = argparse.ArgumentParser(description="Terminal triage interface for MALCA review")
    parser.add_argument("--db", type=Path, default=Path(DEFAULT_DB_PATH), help="SQLite DB path")
    parser.add_argument("--input", type=Path, default=None, help="Optional CSV/Parquet to import before start")
    parser.add_argument("--reviewer", type=str, default="", help="Reviewer name")
    args = parser.parse_args()
    main_with_args(db=args.db, input_path=args.input, reviewer=args.reviewer)


if __name__ == "__main__":
    main()
