# This file was mechanically split from malca.review.app; preserve behavior when editing.
def _review_db_state_signature(db_path: str | Path | None = None) -> str:
    """Return a cache signature changed only by candidate/review writes."""
    return review_content_signature(db_path or DB_PATH)


def _diagnostic_background_signature(db_path: str | Path | None = None) -> str:
    """Return a cache signature for diagnostic background data."""
    return _review_db_state_signature(db_path or DB_PATH)


def _diagnostic_background_cache_key(signature: str) -> str:
    """Return diskcache key used for diagnostic background blobs."""
    return f"diagnostic-background:{signature}"


def _get_cached_diagnostic_background(signature: str | None) -> dict | None:
    """Load cached diagnostic background data from diskcache."""
    if not signature:
        return None
    try:
        cached = _bc_cache.get(_diagnostic_background_cache_key(signature))
    except Exception:
        return None
    return cached if isinstance(cached, dict) else None


def _store_cached_diagnostic_background(signature: str, background: dict) -> None:
    """Persist diagnostic background data to diskcache."""
    if not signature:
        return
    _bc_cache.set(_diagnostic_background_cache_key(signature), background)


def _load_or_cache_diagnostic_background(signature: str | None = None) -> tuple[dict, bool]:
    """Load diagnostic background from cache, falling back to the review DB."""
    resolved_signature = signature or _diagnostic_background_signature(DB_PATH)
    cached = _get_cached_diagnostic_background(resolved_signature)
    if cached is not None:
        return cached, True

    with closing(db_connect(Path(DB_PATH))) as conn:
        background = get_diagnostic_background(conn)
    _store_cached_diagnostic_background(resolved_signature, background)
    return background, False


@lru_cache(maxsize=512)
def _candidate_context_cached(
    db_path_text: str,
    db_signature: str,
    candidate_id: str,
) -> tuple[str, str | None, str | None]:
    """Load one candidate payload + local path context from SQLite, cached by DB state."""
    _ = db_signature
    with closing(db_connect(Path(db_path_text))) as conn:
        payload = get_candidate_payload(conn, candidate_id) or {}
        row = conn.execute(
            "SELECT lc_path, source_path FROM candidates WHERE candidate_id = ?",
            (candidate_id,),
        ).fetchone()

    if not isinstance(payload, dict):
        payload = {}
    if not payload.get("candidate_id"):
        payload["candidate_id"] = str(candidate_id)

    payload_json = json.dumps(payload, sort_keys=True, default=str)
    stored_lc_path = None
    source_path = None
    if row:
        stored_lc_path = str(row[0]).strip() if row[0] not in (None, "") else None
        source_path = str(row[1]).strip() if row[1] not in (None, "") else None
    return payload_json, stored_lc_path, source_path


def _candidate_context(candidate_id: str | None) -> tuple[dict, str | None, str | None]:
    """Return payload plus stored lc/source paths for the active DB."""
    cid = str(candidate_id or "").strip()
    if not cid:
        return {}, None, None

    payload_json, stored_lc_path, source_path = _candidate_context_cached(
        str(Path(DB_PATH).expanduser()),
        _review_db_state_signature(DB_PATH),
        cid,
    )
    try:
        payload = json.loads(payload_json) if payload_json else {}
    except Exception:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    if not payload.get("candidate_id"):
        payload["candidate_id"] = cid
    if stored_lc_path and not payload.get("lc_path"):
        payload["lc_path"] = stored_lc_path
    if source_path and not payload.get("source_path"):
        payload["source_path"] = source_path
    return payload, stored_lc_path, source_path


@lru_cache(maxsize=16)
def _progress_counts_cached(db_path_text: str, db_signature: str) -> tuple[int, int]:
    """Load reviewed/total progress counts, cached by DB state."""
    _ = db_signature
    with closing(db_connect(Path(db_path_text))) as conn:
        return count_progress(conn)


def _progress_counts() -> tuple[int, int]:
    """Return reviewed/total counts for the active review DB."""
    return _progress_counts_cached(
        str(Path(DB_PATH).expanduser()),
        _review_db_state_signature(DB_PATH),
    )


def _current_eda_frame() -> pd.DataFrame:
    """Load the EDA dataframe for the active review DB."""
    return load_review_eda_frame(DB_PATH, _review_db_state_signature(DB_PATH))


def _queue_candidate_ids(queue_data: object) -> list[str]:
    if not isinstance(queue_data, dict):
        return []
    return [str(value) for value in (queue_data.get('candidate_ids') or [])]


def _clear_review_state_caches() -> None:
    """Clear in-process caches derived from the review DB."""
    _candidate_context_cached.cache_clear()
    _progress_counts_cached.cache_clear()

