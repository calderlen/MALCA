# This file was mechanically split from malca.review.app; preserve behavior when editing.
def _resolve_run_dir_from_plot_dir(plot_dir: str | None) -> Path | None:
    """Infer run directory from plot-dir or run-dir style path."""
    if not plot_dir:
        return None
    cached = _resolve_run_dir_from_plot_dir_cached(str(plot_dir))
    return Path(cached) if cached else None


@lru_cache(maxsize=64)
def _resolve_run_dir_from_plot_dir_cached(plot_dir_text: str) -> str | None:
    """Cached implementation for run-dir inference from plot-dir text."""
    if not plot_dir_text:
        return None
    p = Path(str(plot_dir_text)).expanduser()
    try:
        if p.exists():
            p = p.resolve()
    except Exception:
        pass
    if p.name == "plots":
        return str(p.parent)
    if (p / "plots").is_dir():
        return str(p)
    if (p / "results").is_dir():
        return str(p)
    if (p / "bundle_assets" / "lightcurves").is_dir():
        return str(p)
    if (p.parent / "results").is_dir():
        return str(p.parent)
    if (p.parent / "plots").is_dir():
        return str(p.parent)
    if (p.parent / "bundle_assets" / "lightcurves").is_dir():
        return str(p.parent)
    return None


def _resolve_run_dir_from_db_path(db_path: str | Path | None) -> Path | None:
    """Infer run directory from a review DB path or standalone bundled DB path."""
    if not db_path:
        return None
    cached = _resolve_run_dir_from_db_path_cached(str(db_path))
    return Path(cached) if cached else None


@lru_cache(maxsize=16)
def _resolve_run_dir_from_db_path_cached(db_path_text: str) -> str | None:
    """Cached implementation for run-dir inference from a review DB path."""
    if not db_path_text:
        return None
    p = Path(str(db_path_text)).expanduser()
    try:
        if p.exists():
            p = p.resolve()
    except Exception:
        pass
    if p.suffix.lower() != ".db":
        return None
    if (p.parent / "results").is_dir() or (p.parent / "plots").is_dir() or (p.parent / "bundle_assets" / "lightcurves").is_dir():
        return str(p.parent)
    if p.parent.name != "review":
        return None
    run_dir = p.parent.parent
    if (run_dir / "results").is_dir() or (run_dir / "plots").is_dir() or (run_dir / "bundle_assets" / "lightcurves").is_dir():
        return str(run_dir)
    return None


def _review_db_for_plot_dir(plot_dir: str | None) -> Path | None:
    """Return the sibling run-local review DB for a plot dir, if present."""
    run_dir = _resolve_run_dir_from_plot_dir(plot_dir)
    if run_dir is None:
        return None
    review_dir = run_dir / "review"
    candidate = review_dir / "review.db"
    if candidate.exists() and _count_candidates_in_db(candidate) > 0:
        return candidate.resolve()
    if review_dir.is_dir():
        populated = [
            db_path
            for db_path in sorted(review_dir.glob("*.db"))
            if _count_candidates_in_db(db_path) > 0
        ]
        if populated:
            return max(
                populated,
                key=lambda db_path: (_count_candidates_in_db(db_path), db_path.stat().st_size),
            ).resolve()
    if candidate.exists():
        return candidate.resolve()
    return None


def _db_plot_mismatch_warning(db_path: str | Path | None, plot_dir: str | None) -> str:
    """Describe likely DB/plot-dir mismatches that would hide candidates."""
    if not plot_dir or not db_path:
        return ""

    selected = Path(str(db_path)).expanduser().resolve()
    sibling = _review_db_for_plot_dir(plot_dir)
    if sibling is None or sibling == selected:
        return ""

    selected_count = _count_candidates_in_db(selected)
    sibling_count = _count_candidates_in_db(sibling)
    if sibling_count < 0:
        return ""

    if selected_count == 0 and sibling_count > 0:
        return (
            f"Selected DB {selected} has 0 candidates, but the run-local DB for "
            f"{Path(str(plot_dir)).expanduser().resolve()} is {sibling} with {sibling_count} candidates. "
            f"Use --review-db {sibling} or omit --review-db to use the run-local DB automatically."
        )

    return ""


def _project_root() -> Path:
    """Repository root inferred from this file location."""
    return _APP_REPO_ROOT


def _configured_plot_dir() -> Path | None:
    """Return the configured plot directory, if any."""
    if not PLOT_DIR:
        return None
    cached = _configured_plot_dir_cached(str(PLOT_DIR))
    return Path(cached) if cached else None


@lru_cache(maxsize=8)
def _configured_plot_dir_cached(plot_dir_text: str) -> str | None:
    if not plot_dir_text:
        return None
    p = Path(str(plot_dir_text)).expanduser()
    try:
        if p.exists():
            p = p.resolve()
    except Exception:
        pass
    return str(p)


@lru_cache(maxsize=512)
def _existing_run_dir_from_path_text(path_text: str) -> str | None:
    """Infer a run dir from an existing local path without resolving stale roots."""
    text = str(path_text or "").strip()
    if not text or "://" in text:
        return None
    p = Path(text).expanduser()
    candidates = (p, p.parent, p.parent.parent)
    for candidate in candidates:
        try:
            if (candidate / "results").is_dir() or (candidate / "bundle_assets" / "lightcurves").is_dir():
                try:
                    return str(candidate.resolve())
                except Exception:
                    return str(candidate)
        except Exception:
            continue
    return None


def _run_dir_from_source_path(source_path: object = None) -> Path | None:
    """Infer a run directory from a candidate source path when possible."""
    text = str(source_path or "").strip()
    if text:
        source_run = _existing_run_dir_from_path_text(text)
        if source_run:
            return Path(source_run)

    return _resolve_run_dir_from_db_path(DB_PATH)


def _review_plot_dir_for_context(source_path: object = None) -> Path | None:
    """Return a plot-dir anchor for the active review context.

    Native plotting resolves bundled light curves relative to a run's plots
    directory.  A review DB sitting inside ``<run>/review`` is enough context to
    infer that anchor, even when the run has no rendered PNG plots.
    """
    plot_dir = _configured_plot_dir()
    if plot_dir is not None:
        return plot_dir

    run_dir = _run_dir_from_source_path(source_path)
    if run_dir is None:
        return None

    if (
        (run_dir / "bundle_assets" / "lightcurves").is_dir()
        or (run_dir / "plots").is_dir()
        or (run_dir / "run_params.json").exists()
    ):
        return run_dir / "plots"
    return None


def _effective_local_lc_path(
    payload: dict | None,
    *,
    stored_lc_path: object = None,
    source_path: object = None,
) -> str | None:
    """Return the best local/bundled LC path for review UI display and lookup."""
    payload_dict = dict(payload) if isinstance(payload, dict) else {}

    explicit_local_paths: list[str] = []
    for raw in (stored_lc_path, payload_dict.get("lc_path")):
        text = str(raw or "").strip()
        if not text or text in explicit_local_paths:
            continue
        explicit_local_paths.append(text)
        try:
            if Path(text).expanduser().exists():
                return text
        except Exception:
            continue

    if explicit_local_paths and not payload_dict.get("lc_path"):
        payload_dict["lc_path"] = explicit_local_paths[0]

    plot_dir = _review_plot_dir_for_context(source_path)

    try:
        resolved = resolve_lightcurve_path(payload_dict, plot_dir)
    except Exception:
        resolved = None

    cluster_lc_path = str(payload_dict.get("path") or "").strip()
    if resolved is not None:
        resolved_text = str(resolved)
        if resolved_text and resolved_text != cluster_lc_path:
            return resolved_text

    if explicit_local_paths and cluster_lc_path:
        return explicit_local_paths[0]

    return None


def _display_lc_paths(
    payload: dict | None,
    *,
    stored_lc_path: object = None,
    source_path: object = None,
) -> tuple[str | None, str | None]:
    """Return display-friendly (cluster_path, local_path) values for the footer/search UI."""
    payload_dict = dict(payload) if isinstance(payload, dict) else {}
    cluster_lc_path = str(payload_dict.get("path") or "").strip() or None
    local_lc_path = _effective_local_lc_path(
        payload_dict,
        stored_lc_path=stored_lc_path,
        source_path=source_path,
    )

    stored_lc_text = str(stored_lc_path or "").strip() or None
    if cluster_lc_path:
        return cluster_lc_path, local_lc_path

    # LTV standalone imports may only have the original raw-lc path in lc_path.
    # If it cannot be resolved locally, show it as the source/cluster path rather
    # than mislabeling it as a usable local review copy.
    if local_lc_path is None and stored_lc_text:
        return stored_lc_text, None

    return None, local_lc_path


def _plot_asset_root() -> Path:
    """Root used for locating and serving static plot files."""
    plot_dir = _configured_plot_dir()
    return plot_dir if plot_dir is not None else _project_root()


def _plot_url_for_path(plot_path: Path) -> str:
    """Return a `/plots/...` URL for a discovered plot path."""
    root = _plot_asset_root().resolve()
    candidate = Path(plot_path).expanduser().resolve()
    try:
        rel_path = candidate.relative_to(root)
    except ValueError:
        return ""
    return f"/plots/{rel_path.as_posix()}"


def _plot_file_from_src(src: str) -> Path | None:
    """Resolve a static plot URL back to an on-disk file path."""
    text = str(src or "")
    if not text.startswith('/plots/'):
        return None

    rel = text[len('/plots/'):]
    suffix = Path(rel).suffix.lower()
    if suffix not in _PLOT_STATIC_EXTENSIONS:
        return None

    root = _plot_asset_root().resolve()
    candidate = (root / rel).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate


def _plot_search_root_for_payload(payload: dict | None) -> Path | None:
    """Return the best plot directory to search for a candidate payload."""
    plot_dir = _configured_plot_dir()
    if plot_dir is not None:
        return plot_dir

    inferred_plot_dir = _review_plot_dir_for_context((payload or {}).get("source_path"))
    if inferred_plot_dir is not None and inferred_plot_dir.is_dir():
        return inferred_plot_dir

    for key in ("plot_path", "png_path"):
        raw_path = (payload or {}).get(key)
        if not raw_path:
            continue
        candidate = Path(str(raw_path)).expanduser()
        if candidate.suffix.lower() in _PLOT_STATIC_EXTENSIONS and candidate.exists():
            try:
                return candidate.resolve().parent
            except Exception:
                return candidate.parent

        run_dir_text = _existing_run_dir_from_path_text(str(candidate.parent))
        if run_dir_text:
            plot_candidate = Path(run_dir_text) / "plots"
            if plot_candidate.is_dir():
                return plot_candidate

    for key in ("path", "lc_path"):
        raw_path = (payload or {}).get(key)
        if not raw_path:
            continue
        run_dir_text = _existing_run_dir_from_path_text(str(raw_path))
        if run_dir_text:
            plot_candidate = Path(run_dir_text) / "plots"
            if plot_candidate.is_dir():
                return plot_candidate

    return None


def _candidate_plot_src(payload: dict | None) -> str:
    """Return a static plot URL for *payload*, if one can be located."""
    plot_root = _plot_search_root_for_payload(payload)
    if plot_root is None:
        return ""

    plot_path = find_plot_image(payload or {}, plot_root)
    if plot_path and plot_path.exists():
        return _plot_url_for_path(plot_path)
    return ""


def _count_candidates_in_db(path: Path) -> int:
    """Return number of candidates in DB, or -1 when unavailable."""
    try:
        with closing(sqlite3.connect(str(path), timeout=30.0)) as conn:
            row = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='candidates'"
            ).fetchone()
            if not row:
                return -1
            return int(conn.execute("SELECT COUNT(*) FROM candidates").fetchone()[0])
    except Exception:
        return -1


def _db_path_with_appended_suffix(path: Path) -> Path:
    """Return the sibling path produced by appending '.db' to the filename."""
    return path.with_name(f"{path.name}.db")


def _prefer_populated_db_sibling(path: Path) -> Path:
    """Prefer a populated '<name>.db' sibling over an empty suffixless DB."""
    if path.suffix.lower() == ".db":
        return path

    sibling = _db_path_with_appended_suffix(path)
    if not sibling.exists():
        return path

    selected_count = _count_candidates_in_db(path)
    sibling_count = _count_candidates_in_db(sibling)
    if sibling_count > 0 and selected_count <= 0:
        return sibling
    return path


def _resolve_db_cli_path(raw_path: str) -> Path:
    """Resolve --review-db robustly for both cwd-relative and repo-relative usage."""
    p = Path(raw_path).expanduser()
    if p.is_absolute():
        return _prefer_populated_db_sibling(p.resolve()).resolve()

    cwd_candidate = (Path.cwd() / p).resolve()
    repo_candidate = (_project_root() / p).resolve()
    existing = [x for x in (cwd_candidate, repo_candidate) if x.exists()]

    if len(existing) == 2:
        ranked = sorted(
            existing,
            key=lambda x: (_count_candidates_in_db(x), x.stat().st_size),
            reverse=True,
        )
        return _prefer_populated_db_sibling(ranked[0]).resolve()
    if len(existing) == 1:
        return _prefer_populated_db_sibling(existing[0]).resolve()
    return cwd_candidate


def _resolve_plot_cli_path(raw_path: str) -> Path:
    """Resolve --plot-dir robustly for both cwd-relative and repo-relative usage."""
    p = Path(raw_path).expanduser()
    if p.is_absolute():
        return p.resolve()

    cwd_candidate = (Path.cwd() / p).resolve()
    repo_candidate = (_project_root() / p).resolve()

    if cwd_candidate.exists() and cwd_candidate.is_dir():
        return cwd_candidate
    if repo_candidate.exists() and repo_candidate.is_dir():
        return repo_candidate
    return cwd_candidate


def _extract_bundle_scope(path_text: str | None) -> str:
    """Extract output_bundle_* token from a path-like string."""
    if not path_text:
        return ""
    text = str(path_text)
    m = re.search(r"(output_bundle_[^/\\]+)", text)
    return m.group(1) if m else ""


def _split_import_sources(path_text: str | None) -> list[str]:
    """Split import-path text into individual path tokens."""
    if not path_text:
        return []
    tokens: list[str] = []
    for line in str(path_text).replace(";", "\n").splitlines():
        for piece in line.split(","):
            text = piece.strip()
            if text:
                tokens.append(text)
    return tokens


def _candidate_files_for_run_dir(run_dir: Path) -> list[Path]:
    """Return candidate files for a run/results directory, including multi-bin outputs."""
    def _sort_key(path: Path) -> tuple[float, float, str]:
        stem = path.stem
        match = re.search(r"_([0-9]+(?:\.[0-9]+)?)_([0-9]+(?:\.[0-9]+)?)$", stem)
        if match:
            return (float(match.group(1)), float(match.group(2)), stem)
        return (float('inf'), float('inf'), stem)

    root = run_dir / "results"
    if run_dir.name == "results":
        root = run_dir
    if not root.exists() or not root.is_dir():
        return []

    exact_vetted = root / "lc_events_vetted.parquet"
    if exact_vetted.exists():
        return [exact_vetted.resolve()]

    tagged_vetted = sorted(root.glob("lc_events_vetted_*.parquet"), key=_sort_key)
    if tagged_vetted:
        return [p.resolve() for p in tagged_vetted]

    fallback_names = (
        "lc_events_spectra.parquet",
        "lc_events_neighbors.parquet",
        "lc_events_classified.parquet",
        "lc_events_enriched.parquet",
        "lc_events_characterized.parquet",
        "lc_events_filtered.parquet",
    )
    matches = [root / name for name in fallback_names if (root / name).exists()]
    return [p.resolve() for p in matches]


def _resolve_import_sources(path_text: str | None, *, allow_run_dirs: bool = True) -> list[Path]:
    """Resolve import-path text into one or more source files."""
    resolved: list[Path] = []
    seen: set[str] = set()
    for token in _split_import_sources(path_text):
        matches: list[Path] = []
        if any(ch in token for ch in "*?[]"):
            matches = [Path(p).expanduser().resolve() for p in sorted(globlib.glob(token, recursive=True))]
        else:
            raw = Path(token).expanduser()
            candidates = [raw]
            if not raw.is_absolute():
                candidates.append((_project_root() / raw).expanduser())
            for candidate in candidates:
                try:
                    if candidate.exists():
                        matches = [candidate.resolve()]
                        break
                except Exception:
                    continue
        if not matches:
            continue
        for match in matches:
            sources = _candidate_files_for_run_dir(match) if (allow_run_dirs and match.is_dir()) else [match]
            for source in sources:
                key = str(source)
                if key in seen:
                    continue
                seen.add(key)
                resolved.append(source)
    return resolved


def _summarize_source_paths(paths: list[Path]) -> str:
    """Return a compact human-readable label for a source list."""
    if not paths:
        return ""
    names = [p.name for p in paths]
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]}, {names[1]}"
    return f"{names[0]}, {names[1]} (+{len(names) - 2} more)"


def _extract_bundle_scopes(path_text: str | None) -> list[str]:
    """Extract all bundle tokens from import text."""
    scopes: list[str] = []
    seen: set[str] = set()
    for token in _split_import_sources(path_text):
        scope = _extract_bundle_scope(token)
        if scope and scope not in seen:
            seen.add(scope)
            scopes.append(scope)
    return scopes


def _queue_scope_from_import_text(path_text: str | None) -> object:
    """Build queue scoping metadata from import-path text."""
    paths = _resolve_import_sources(path_text)
    if paths:
        return {
            'source_paths': [str(p) for p in paths],
            'label': _summarize_source_paths(paths),
        }
    scopes = _extract_bundle_scopes(path_text)
    if scopes:
        return {
            'source_path_like_any': scopes,
            'label': ", ".join(scopes),
        }
    return ''


def _source_path_for_queue_filter(path_str: str) -> str:
    """Convert an import path (e.g. run dir or results file) to the value stored in candidates.source_path (run dir)."""
    path_str = str(path_str).strip()
    if not path_str:
        return path_str
    # DB stores run directory; import path may be a file under run_dir/results/
    if "/results/" in path_str:
        return path_str.split("/results/")[0]
    return path_str


def _source_path_fallback_tokens(paths: list[str]) -> list[str]:
    tokens: list[str] = []
    seen: set[str] = set()
    for path in paths:
        token = Path(str(path)).name.strip()
        if not token or token in seen:
            continue
        seen.add(token)
        tokens.append(token)
    return tokens


def _queue_scope_filter_kwargs(scope_value: object) -> dict[str, object]:
    """Translate queue-source store payload into DB filter kwargs."""
    if isinstance(scope_value, dict):
        if scope_value.get('source_paths'):
            # Normalize so file paths (e.g. .../results/foo.parquet) become run dirs to match candidates.source_path
            normalized = [_source_path_for_queue_filter(p) for p in scope_value['source_paths']]
            kwargs: dict[str, object] = {'source_paths': normalized}
            fallback_tokens = _source_path_fallback_tokens(normalized)
            if fallback_tokens:
                kwargs['source_path_fallback_like_any'] = fallback_tokens
            return kwargs
        if scope_value.get('source_path_like_any'):
            return {'source_path_like_any': list(scope_value['source_path_like_any'])}
        return {}
    if scope_value:
        return {'source_path_like': str(scope_value)}
    return {}


def _queue_scope_label(scope_value: object) -> str:
    """Return a short label for queue scoping metadata."""
    if isinstance(scope_value, dict):
        label = scope_value.get('label')
        if label:
            return str(label)
        paths = scope_value.get('source_paths') or []
        if paths:
            return _summarize_source_paths([Path(str(p)) for p in paths])
        likes = scope_value.get('source_path_like_any') or []
        if likes:
            return ", ".join(str(v) for v in likes)
        return ''
    return str(scope_value or '')


def _vetting_mode_for_sources(path_text: str | None) -> str:
    """Summarize vetting-mode status for one or more import sources."""
    paths = _resolve_import_sources(path_text)
    if not paths:
        return _vetting_mode_for_input(path_text)
    modes = [_vetting_mode_for_input(p) for p in paths]
    if len(modes) == 1:
        return modes[0]
    counts: dict[str, int] = {}
    for mode in modes:
        counts[mode] = counts.get(mode, 0) + 1
    return "; ".join(
        f"{count} {mode.lower()}" if count != 1 else mode
        for mode, count in sorted(counts.items(), key=lambda item: item[0])
    )


def _vetting_mode_for_input(input_path: str | Path | None) -> str:
    """Classify how import vetting will be satisfied for a source file."""
    if not input_path:
        return "re-vetting needed"

    p = Path(str(input_path)).expanduser()
    try:
        p = p.resolve()
    except Exception:
        pass

    if "vetted" in p.stem.lower():
        return "Using vetted input"

    cache_path = Path(str(p) + ".vetting_cache.parquet")
    if cache_path.exists():
        return "cache hit"

    return "re-vetting needed"


def _load_spectra_rows(candidate_id: str, run_dir: Path | None) -> pd.DataFrame:
    """Load spectra matches for one candidate if local enrichment exists."""
    if run_dir is None:
        return pd.DataFrame()
    spectra_long = run_dir / "results" / "spectra_enrichment" / "spectra_long.parquet"
    if not spectra_long.exists():
        return pd.DataFrame()

    cid = str(candidate_id)
    cols = ["candidate_id", "survey", "catalog", "sep_arcsec", "spectrum_redshift", "spectrum_spectral_type", "link"]
    try:
        df = pd.read_parquet(spectra_long, columns=cols, filters=[("candidate_id", "==", cid)])
    except Exception:
        try:
            df = pd.read_parquet(spectra_long)
        except Exception:
            return pd.DataFrame()
        if "candidate_id" in df.columns:
            df = df[df["candidate_id"].astype(str) == cid]
    for c in cols:
        if c not in df.columns:
            df[c] = "" if c in {"survey", "catalog", "spectrum_spectral_type", "link"} else pd.NA
    return df[cols].reset_index(drop=True)


@lru_cache(maxsize=4)
def _index_neowise_paths(run_dir_text: str) -> dict[str, str]:
    """Index candidate->NEOWISE parquet paths once per run directory."""
    return _index_external_lc_paths(run_dir_text, "neowise")


def _build_neowise_figure(df_neowise: pd.DataFrame) -> go.Figure:
    """Build a compact NEOWISE light-curve panel."""
    return _build_neowise_figure_with_theme(df_neowise, DEFAULT_THEME)


def _external_followup_theme(theme: str | None) -> dict[str, object]:
    """Theme tokens for external follow-up cards and mini plots."""
    mode = str(theme or DEFAULT_THEME).strip().lower()
    if mode == "white":
        return {
            "card_style": {
                'border': '1px solid #c5d0da',
                'borderRadius': '6px',
                'padding': '8px 10px',
                'background': '#ffffff',
                'color': '#1c2733',
            },
            "muted": '#5a6b7b',
            "error": '#a53a3a',
            "paper_bg": '#ffffff',
            "plot_bg": '#ffffff',
            "font": '#1c2733',
            "grid": 'rgba(104, 128, 149, 0.18)',
            "legend_bg": 'rgba(255, 255, 255, 0.92)',
            "legend_border": 'rgba(120, 140, 158, 0.35)',
        }
    if mode == "gray":
        return {
            "card_style": {
                'border': '1px solid #4c566a',
                'borderRadius': '6px',
                'padding': '8px 10px',
                'background': '#3b4252',
                'color': '#d8dee9',
            },
            "muted": '#aab6c7',
            "error": '#f29f9f',
            "paper_bg": '#2e3440',
            "plot_bg": '#2e3440',
            "font": '#d8dee9',
            "grid": 'rgba(129, 161, 193, 0.15)',
            "legend_bg": 'rgba(59, 66, 82, 0.9)',
            "legend_border": 'rgba(129, 161, 193, 0.3)',
        }
    return {
        "card_style": {
            'border': '1px solid #2a2a2a',
            'borderRadius': '6px',
            'padding': '8px 10px',
            'background': '#0d0d0d',
            'color': '#e0e0e0',
        },
        "muted": '#9fb6cb',
        "error": '#dd8080',
        "paper_bg": '#0d0d0d',
        "plot_bg": '#0d0d0d',
        "font": '#dce8f2',
        "grid": 'rgba(96, 116, 130, 0.22)',
        "legend_bg": 'rgba(13, 13, 13, 0.88)',
        "legend_border": 'rgba(113, 140, 160, 0.3)',
    }


def _convert_external_times_to_review_axis(values, jd_system: str = "mjd") -> np.ndarray:
    """Convert source-native time values into the review axis: JD - JD_OFFSET."""
    t = pd.to_numeric(values, errors="coerce").to_numpy()
    finite_t = t[np.isfinite(t)]
    if jd_system == "mjd":
        if finite_t.size and float(np.nanmedian(finite_t)) > 1_000_000.0:
            jd = t
        else:
            jd = t + MJD_TO_JD
    elif jd_system == "bjd_gaia":
        jd = t + GAIA_TCB_EPOCH_JD
    elif jd_system == "btjd":
        jd = t + TESS_BTJD_OFFSET
    elif jd_system == "bkjd":
        jd = t + KEPLER_BKJD_OFFSET
    else:
        jd = t
    return jd - JD_OFFSET


def _apply_external_figure_layout(
    fig: go.Figure,
    *,
    title: str,
    theme: str,
    yaxis_label: str,
    reverse_y: bool,
    height: int = 240,
) -> go.Figure:
    """Apply a consistent themed layout for external mini plots."""
    spec = _external_followup_theme(theme)
    fig.update_layout(
        height=height,
        margin=dict(l=42, r=10, t=34, b=32),
        title=title,
        legend=dict(
            orientation="h",
            x=0.0,
            y=1.1,
            bgcolor=spec["legend_bg"],
            bordercolor=spec["legend_border"],
            borderwidth=1,
            font=dict(color=spec["font"], family=PUBLICATION_PLOTLY_FONT),
        ),
        paper_bgcolor=spec["paper_bg"],
        plot_bgcolor=spec["plot_bg"],
        font=dict(color=spec["font"], family=PUBLICATION_PLOTLY_FONT),
    )
    fig.update_xaxes(title="JD - 2458000 [d]", gridcolor=spec["grid"], zeroline=False)
    fig.update_yaxes(
        title=yaxis_label,
        autorange="reversed" if reverse_y else True,
        gridcolor=spec["grid"],
        zeroline=False,
    )
    return fig


def _plot_title_text(fig: go.Figure | dict | None, fallback: str) -> str:
    if fig is None:
        return fallback
    try:
        title = go.Figure(fig).layout.title.text
    except Exception:
        title = None
    text = str(title or "").strip()
    return text or fallback


def _exportable_graph(
    fig: go.Figure,
    *,
    panel: str,
    name: str,
    height: str = "250px",
) -> html.Div:
    safe_panel = slugify_token(panel, fallback="panel")
    safe_name = slugify_token(name, fallback="plot")
    graph_id = {"type": "mini-plot-export-graph", "panel": safe_panel, "name": safe_name}
    button_id = {"type": "mini-plot-export-btn", "panel": safe_panel, "name": safe_name}
    return html.Div(
        [
            html.Div(
                html.Button("Export PDF", id=button_id, n_clicks=0, className="compact-btn"),
                style={"display": "flex", "justifyContent": "flex-end", "marginBottom": "4px"},
            ),
            dcc.Graph(
                id=graph_id,
                figure=fig,
                mathjax=True,
                config=graph_config_without_image_export({'displayModeBar': False}),
                style={'height': height},
            ),
        ],
        style={"display": "grid", "gap": "2px"},
    )


def _exportable_plot_card(
    fig: go.Figure,
    *,
    panel: str,
    name: str,
    card_style: dict,
    height: str = "280px",
) -> html.Div:
    return html.Div(
        _exportable_graph(fig, panel=panel, name=name, height=height),
        style=card_style,
    )


def _build_neowise_figure_with_theme(df_neowise: pd.DataFrame, theme: str) -> go.Figure:
    """Build a compact NEOWISE light-curve panel."""
    fig = go.Figure()
    if df_neowise is None or df_neowise.empty:
        return _apply_external_figure_layout(
            fig,
            title="NEOWISE",
            theme=theme,
            yaxis_label="m [mag]",
            reverse_y=True,
            height=220,
        )

    time_col = "mjd" if "mjd" in df_neowise.columns else ("MJD" if "MJD" in df_neowise.columns else None)
    if time_col is None:
        return _apply_external_figure_layout(
            fig,
            title="NEOWISE (missing MJD column)",
            theme=theme,
            yaxis_label="m [mag]",
            reverse_y=True,
            height=220,
        )

    x = _convert_external_times_to_review_axis(df_neowise[time_col], "mjd")
    band_specs = [
        ("W1", "w1mpro", "w1sigmpro", "#4fa3ff"),
        ("W2", "w2mpro", "w2sigmpro", "#ff8c42"),
    ]
    added = 0
    for name, mag_col, err_col, color in band_specs:
        if mag_col not in df_neowise.columns:
            continue
        y = pd.to_numeric(df_neowise[mag_col], errors="coerce")
        good = np.isfinite(x) & np.isfinite(y)
        if not bool(good.any()):
            continue
        err_vals = None
        if err_col in df_neowise.columns:
            ev = pd.to_numeric(df_neowise[err_col], errors="coerce")
            if np.isfinite(ev[good]).any():
                err_vals = ev[good]
        fig.add_trace(
            go.Scattergl(
                x=x[good],
                y=y[good],
                mode="markers",
                name=name,
                marker=dict(size=5, color=color, opacity=0.85),
                error_y=dict(type="data", array=err_vals, visible=err_vals is not None, thickness=0.7),
            )
        )
        added += 1

    _apply_external_figure_layout(
        fig,
        title="NEOWISE Light Curve",
        theme=theme,
        yaxis_label="m [mag]",
        reverse_y=True,
    )
    if added == 0:
        fig.add_annotation(text="No finite W1/W2 points", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    return fig


@lru_cache(maxsize=32)
def _index_external_lc_paths(run_dir_text: str, prefix: str) -> dict[str, str]:
    """Index candidate -> external LC parquet paths for a given prefix."""
    root = Path(run_dir_text) / "results"
    return _index_external_lc_paths_from_root(str(root), prefix)


@lru_cache(maxsize=64)
def _index_external_lc_paths_from_root(root_text: str, prefix: str) -> dict[str, str]:
    """Index candidate -> external LC parquet paths for a results root."""
    return shared_index_external_lc_paths_from_manifest(str(Path(root_text).expanduser()), prefix)


def _build_external_lc_figure(
    df_lc: pd.DataFrame,
    title: str,
    band_specs: list[tuple[str, str, str, str]],
    time_col: str = "mjd",
    yaxis_label: str = "m [mag]",
    reverse_y: bool = True,
    filter_col: str | None = None,
    source_name: str | None = None,
    theme: str | None = None,
    jd_system: str = "mjd",
) -> go.Figure:
    """Build a compact LC panel for any external source.

    *band_specs* is a list of (band_value, mag_col, err_col, color) tuples.
    When *filter_col* is set, only rows where ``df[filter_col] == band_value``
    are plotted for each band.
    """
    fig = go.Figure()
    if df_lc is None or df_lc.empty:
        return _apply_external_figure_layout(
            fig,
            title=title,
            theme=theme,
            yaxis_label=yaxis_label,
            reverse_y=reverse_y,
            height=220,
        )

    if source_name:
        df_lc = normalize_external_lc_dataframe(source_name, df_lc)
        if df_lc is None or df_lc.empty:
            return _apply_external_figure_layout(
                fig,
                title=title,
                theme=theme,
                yaxis_label=yaxis_label,
                reverse_y=reverse_y,
                height=220,
            )

    # Resolve time column (case-insensitive)
    col_lookup = {c.lower(): c for c in df_lc.columns}
    actual_time_col = col_lookup.get(time_col.lower())
    if actual_time_col is None:
        return _apply_external_figure_layout(
            fig,
            title=f"{title} (missing {time_col})",
            theme=theme,
            yaxis_label=yaxis_label,
            reverse_y=reverse_y,
            height=220,
        )

    added = 0
    for band_value, mag_col, err_col, color in band_specs:
        actual_filter_col = col_lookup.get(filter_col.lower()) if filter_col else None
        actual_mag_col = col_lookup.get(mag_col.lower())
        actual_err_col = col_lookup.get(err_col.lower()) if err_col else None
        # Filter rows for this band if filter_col is specified
        if actual_filter_col:
            subset = df_lc[df_lc[actual_filter_col].astype(str) == band_value]
        else:
            subset = df_lc
        if subset.empty or actual_mag_col is None:
            continue

        x = _convert_external_times_to_review_axis(subset[actual_time_col], jd_system)
        y = pd.to_numeric(subset[actual_mag_col], errors="coerce")
        good = np.isfinite(x) & np.isfinite(y)
        if not bool(good.any()):
            continue
        err_vals = None
        if actual_err_col and actual_err_col in subset.columns:
            ev = pd.to_numeric(subset[actual_err_col], errors="coerce")
            if np.isfinite(ev[good]).any():
                err_vals = ev[good]
        fig.add_trace(
            go.Scattergl(
                x=x[good],
                y=y[good],
                mode="markers",
                name=band_value,
                marker=dict(size=5, color=color, opacity=0.85),
                error_y=dict(type="data", array=err_vals, visible=err_vals is not None, thickness=0.7),
            )
        )
        added += 1

    _apply_external_figure_layout(
        fig,
        title=f"{title} Light Curve",
        theme=theme,
        yaxis_label=yaxis_label,
        reverse_y=reverse_y,
    )
    if added == 0:
        fig.add_annotation(text="No finite data points", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    return fig


def _coerce_bool(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, (list, tuple, dict, set, np.ndarray, pd.Series)):
        return bool(len(value))
    try:
        if pd.isna(value):
            return False
    except Exception:
        pass
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value) != 0.0
    s = str(value).strip().lower()
    return s in {"1", "true", "t", "yes", "y"}
