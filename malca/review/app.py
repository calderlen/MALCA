from __future__ import annotations

"""Streamlit review app for MALCA candidate triage."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from malca.review.store import (
    DEFAULT_DB_PATH,
    INTEREST_REASON_TAGS,
    STATUS_OPTIONS,
    count_progress,
    db_connect,
    export_reviews,
    find_plot_image,
    get_candidate_payload,
    get_review,
    import_candidates,
    load_app_state,
    load_candidates_file,
    query_queue,
    recent_history,
    save_app_state,
    save_review,
)
from malca.review.metadata import extract_review_metadata
from malca.config.config_paths import VSX_CROSSMATCH_PATH, GAIA_CACHE_FILE
from malca.config.config_characterize import GAIA_CHUNK_SIZE


DEFAULT_XMATCH = str(VSX_CROSSMATCH_PATH)
DEFAULT_GAIA_CACHE = str(GAIA_CACHE_FILE)


def main() -> None:
    st.set_page_config(page_title="MALCA Candidate Review", layout="wide")
    st.title("MALCA Candidate Review")

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--db", default=DEFAULT_DB_PATH)
    args, _ = parser.parse_known_args(sys.argv[1:])

    db_path_str = st.sidebar.text_input("SQLite DB path", value=str(args.db))
    conn = db_connect(Path(db_path_str).expanduser())

    last_file = load_app_state(conn, "last_input_file", "")
    input_path_str = st.sidebar.text_input("Candidates file (CSV/Parquet)", value=last_file)
    st.sidebar.markdown("**Characterize on import**")
    characterize_crossmatch = st.sidebar.text_input("Crossmatch CSV", value=DEFAULT_XMATCH)
    characterize_cache = st.sidebar.text_input("Gaia cache", value=DEFAULT_GAIA_CACHE)
    characterize_chunk_size = st.sidebar.number_input("Gaia chunk size", min_value=1, value=GAIA_CHUNK_SIZE, step=100)
    characterize_dust = st.sidebar.checkbox("Enable dustmaps3d", value=True)
    characterize_starhorse = st.sidebar.text_input("StarHorse", value="tap")

    if st.sidebar.button("Import / Refresh Candidates", type="primary"):
        if not input_path_str.strip():
            st.sidebar.error("Provide a file path first.")
        else:
            try:
                src = Path(input_path_str).expanduser()
                n_rows, n_new = import_candidates(
                    conn,
                    load_candidates_file(src),
                    str(src),
                    characterize_before_import=True,
                    characterize_crossmatch=Path(characterize_crossmatch).expanduser(),
                    characterize_chunk_size=int(characterize_chunk_size),
                    characterize_cache=Path(characterize_cache).expanduser(),
                    characterize_dust=bool(characterize_dust),
                    characterize_starhorse=(characterize_starhorse.strip() or None),
                )
                save_app_state(conn, "last_input_file", str(src))
                st.sidebar.success(f"Imported {n_rows} rows ({n_new} new candidate IDs)")
            except Exception as e:
                st.sidebar.error(f"Import failed: {e}")

    reviewed, total = count_progress(conn)
    st.sidebar.metric("Reviewed", f"{reviewed}/{total}")
    st.sidebar.progress(0.0 if total == 0 else reviewed / total)

    st.sidebar.subheader("Queue Filters")
    only_unreviewed = st.sidebar.checkbox("Only unreviewed", value=True)
    require_failed_any_false = st.sidebar.checkbox("Require failed_any = False", value=True)
    periodic_flag_mode = st.sidebar.selectbox("periodic_flag", ["Any", "True", "False"], index=0)
    catalog_match_mode = st.sidebar.selectbox("catalog_match", ["Any", "True", "False"], index=0)
    high_ruwe_mode = st.sidebar.selectbox("high_ruwe_flag", ["Any", "True", "False"], index=0)
    min_periodicity_score = st.sidebar.number_input("Min periodicity_score", value=np.nan, format="%.3f")
    max_lsp_bootstrap_sig = st.sidebar.number_input("Max lsp_bootstrap_sig", value=np.nan, format="%.6f")
    min_lsp_power = st.sidebar.number_input("Min lsp_power", value=np.nan, format="%.4f")
    sort_col = st.sidebar.selectbox(
        "Sort by",
        [
            "candidate_id",
            "periodicity_score",
            "lsp_bootstrap_sig",
            "lsp_power",
            "dip_best_log_bf",
            "jump_best_log_bf",
            "interest_score",
            "review_pass",
            "updated_at",
        ],
        index=0,
    )
    sort_desc = st.sidebar.checkbox("Sort descending", value=False)

    queue_df = query_queue(
        conn,
        only_unreviewed=only_unreviewed,
        require_failed_any_false=require_failed_any_false,
        periodic_flag_mode=periodic_flag_mode,
        catalog_match_mode=catalog_match_mode,
        high_ruwe_mode=high_ruwe_mode,
        min_periodicity_score=None if np.isnan(min_periodicity_score) else float(min_periodicity_score),
        max_lsp_bootstrap_sig=None if np.isnan(max_lsp_bootstrap_sig) else float(max_lsp_bootstrap_sig),
        min_lsp_power=None if np.isnan(min_lsp_power) else float(min_lsp_power),
        sort_col=sort_col,
        sort_desc=sort_desc,
    )
    if queue_df.empty:
        st.warning("No candidates in queue. Import candidates and/or loosen filters.")
        st.stop()

    if "queue_idx" not in st.session_state:
        st.session_state.queue_idx = 0
    st.session_state.queue_idx = max(0, min(st.session_state.queue_idx, len(queue_df) - 1))

    nav_col1, nav_col2, nav_col3, nav_col4 = st.columns([1, 1, 2, 2])
    with nav_col1:
        if st.button("Previous"):
            st.session_state.queue_idx = max(0, st.session_state.queue_idx - 1)
    with nav_col2:
        if st.button("Next"):
            st.session_state.queue_idx = min(len(queue_df) - 1, st.session_state.queue_idx + 1)
    with nav_col3:
        st.session_state.queue_idx = st.number_input(
            "Queue index", min_value=0, max_value=len(queue_df) - 1, value=st.session_state.queue_idx, step=1
        )
    with nav_col4:
        st.caption(f"Queue size: {len(queue_df)}")

    sel = queue_df.iloc[int(st.session_state.queue_idx)]
    candidate_id = str(sel["candidate_id"])
    payload = get_candidate_payload(conn, candidate_id)
    payload["candidate_id"] = candidate_id
    review = get_review(conn, candidate_id)

    st.subheader(f"Candidate: {candidate_id}")
    met1, met2, met3, met4, met5 = st.columns(5)
    met1.metric("periodicity_score", f"{sel['periodicity_score']:.3f}" if pd.notna(sel["periodicity_score"]) else "-")
    met2.metric("lsp_bootstrap_sig", f"{sel['lsp_bootstrap_sig']:.4g}" if pd.notna(sel["lsp_bootstrap_sig"]) else "-")
    met3.metric("lsp_period", f"{sel['lsp_period']:.4f}" if pd.notna(sel["lsp_period"]) else "-")
    met4.metric("dip_best_log_bf", f"{sel['dip_best_log_bf']:.3f}" if pd.notna(sel["dip_best_log_bf"]) else "-")
    met5.metric("jump_best_log_bf", f"{sel['jump_best_log_bf']:.3f}" if pd.notna(sel["jump_best_log_bf"]) else "-")

    left, right = st.columns([2, 1])
    with left:
        st.markdown("**Candidate Context**")
        summary = {k: v for k, v in extract_review_metadata(payload)}
        st.dataframe(pd.DataFrame([summary]).T.rename(columns={0: "value"}), use_container_width=True)

        st.markdown("**Light Curve Plot**")
        default_plot_dir = load_app_state(conn, "last_plot_dir", "")
        plot_dir_str = st.text_input("Plot directory (optional)", value=default_plot_dir)
        if plot_dir_str.strip():
            save_app_state(conn, "last_plot_dir", plot_dir_str)
            plot_path = find_plot_image(payload, Path(plot_dir_str).expanduser())
            if plot_path is not None:
                if plot_path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                    st.image(str(plot_path), caption=str(plot_path))
                else:
                    st.info(f"Found plot: {plot_path}")
            else:
                st.info("No matching plot found in directory.")

    with right:
        st.markdown("**Review**")
        reviewer_default = load_app_state(conn, "reviewer", "")
        reviewer = st.text_input("Reviewer", value=review["reviewer"] or reviewer_default)
        if reviewer:
            save_app_state(conn, "reviewer", reviewer)

        with st.form("review_form"):
            interest_score = st.slider("Interest score", min_value=0, max_value=5, value=int(review["interest_score"]), step=1)
            interest_reason = st.multiselect("Interest reason", INTEREST_REASON_TAGS, default=review["interest_reason"])
            review_pass = st.number_input("Review pass", min_value=1, value=int(review["review_pass"]), step=1)
            status = st.selectbox("Status", STATUS_OPTIONS, index=STATUS_OPTIONS.index(review["status"]))
            notes = st.text_area("Notes", value=review["notes"], height=160)
            save_only = st.form_submit_button("Save")
            save_next = st.form_submit_button("Save + Next")

        if save_only or save_next:
            save_review(
                conn,
                candidate_id=candidate_id,
                interest_score=int(interest_score),
                interest_reason=list(interest_reason),
                review_pass=int(review_pass),
                notes=notes,
                status=status,
                reviewer=reviewer,
                event_type="save_next" if save_next else "save",
            )
            st.success("Saved")
            if save_next:
                st.session_state.queue_idx = min(len(queue_df) - 1, st.session_state.queue_idx + 1)
            st.rerun()

        if review["updated_at"]:
            st.caption(f"Last updated: {review['updated_at']}")

    st.markdown("**Recent Activity**")
    hist = recent_history(conn, limit=5)
    if hist.empty:
        st.caption("No reviews yet.")
    else:
        st.dataframe(hist, use_container_width=True, hide_index=True)

    st.sidebar.subheader("Export")
    export_path_str = st.sidebar.text_input("Export path", value="output/review/reviewed_candidates.parquet")
    export_only_reviewed = st.sidebar.checkbox("Export only reviewed", value=True)
    if st.sidebar.button("Export Reviews"):
        try:
            export_reviews(conn, Path(export_path_str).expanduser(), only_reviewed=export_only_reviewed)
            st.sidebar.success(f"Exported to {export_path_str}")
        except Exception as e:
            st.sidebar.error(f"Export failed: {e}")


if __name__ == "__main__":
    main()
