# This file was mechanically split from malca.review.app; preserve behavior when editing.
def main():
    """Main entry point."""
    global DB_PATH, PLOT_DIR, INITIAL_CANDIDATE_QUERY

    parser = argparse.ArgumentParser(description="MALCA Dash Review App")
    parser.add_argument('--review-db', default=None, help="Review SQLite database path (default: standalone.db without --plot-dir, review.db with --plot-dir)")
    parser.add_argument('--plot-dir', help="Plot directory path (auto-detects ./plots if not specified)")
    parser.add_argument('--host', default='127.0.0.1', help="Host")
    parser.add_argument('--port', default=8050, type=int, help="Port")
    parser.add_argument('--candidate', default=None, help="Candidate ID / ASAS-SN ID / Gaia ID / LC stem to open on startup")
    parser.add_argument('--no-browser', action='store_true', help="Do not auto-open a browser tab/window on startup")
    parser.add_argument('--debug', action='store_true', help="Debug mode")
    parser.add_argument('--verbose-http', action='store_true',
                        help="Show Flask/Werkzeug per-request access logs")
    args = parser.parse_args()
    INITIAL_CANDIDATE_QUERY = str(args.candidate).strip() if args.candidate not in (None, '') else None

    # Auto-detect plot directory if not specified
    if args.plot_dir:
        PLOT_DIR = str(_resolve_plot_cli_path(args.plot_dir))
        if not Path(PLOT_DIR).exists() or not Path(PLOT_DIR).is_dir():
            print(f"Error: plot directory does not exist: {PLOT_DIR}")
            print("Use an existing run bundle plots directory, for example:")
            print("  malca review --review-db output/runs/stv/20250121_143052/review/review.db --plot-dir output/runs/stv/20250121_143052/plots")
            sys.exit(1)
    else:
        # Try current directory first
        if Path('./plots').is_dir():
            PLOT_DIR = str(Path('./plots').resolve())
            print(f"Auto-detected plot directory: {Path(PLOT_DIR).resolve()}")
        else:
            # Standalone mode — no plot dir required
            PLOT_DIR = None
            print("Running in standalone mode (no --plot-dir)")

    inferred_plot_db = _review_db_for_plot_dir(PLOT_DIR)

    # Choose DB: explicit --review-db overrides; otherwise standalone gets its own DB
    if args.review_db is not None:
        DB_PATH = str(_resolve_db_cli_path(args.review_db))
    elif PLOT_DIR is None:
        # Standalone mode: use a separate DB so pipeline candidates don't bleed in
        DB_PATH = str(_resolve_db_cli_path(str(DEFAULT_STANDALONE_DB_PATH)))
    elif inferred_plot_db is not None:
        DB_PATH = str(inferred_plot_db)
    else:
        DB_PATH = str(_resolve_db_cli_path(str(DEFAULT_DB_PATH)))

    # Publish runtime paths so spawned background workers inherit the same config.
    os.environ[_REVIEW_DB_ENV] = str(DB_PATH)
    if PLOT_DIR:
        os.environ[_REVIEW_PLOT_ENV] = str(PLOT_DIR)
    else:
        os.environ.pop(_REVIEW_PLOT_ENV, None)

    mismatch_warning = _db_plot_mismatch_warning(DB_PATH, PLOT_DIR)
    if mismatch_warning:
        print(f"Warning: {mismatch_warning}")

    print(f"Starting MALCA Review App...")
    print(f"  Database:  {DB_PATH}")
    print(f"  Plot dir:  {PLOT_DIR}")
    print(f"  Server:    http://{args.host}:{args.port}")
    print(f"\nKeyboard shortcuts:")
    print("  [D]ipper [M]icrolensing [F]lare [Y]so [U]nknown [I]nstrumental [O]ther | [1-4] Confidence | [.] Save | [Enter] Done | [Backspace] Back | [Esc] Sidebar | [?] Help")
    print("")

    # Auto-open browser
    url = f"http://{args.host}:{args.port}"
    if not args.no_browser:
        Timer(0.1, lambda: webbrowser.open(url)).start()

    if not args.verbose_http:
        # Keep explicit pipeline/status prints, but hide the noisy per-request
        # development-server access lines so long-running actions are readable.
        logging.getLogger("werkzeug").setLevel(logging.ERROR)
        app.server.logger.setLevel(logging.ERROR)

    try:
        # Single-threaded local serving keeps concurrent request/socket churn bounded
        # during long review sessions.
        app.run(debug=args.debug, host=args.host, port=args.port, threaded=False)
    except KeyboardInterrupt:
        pass
    finally:
        _cleanup_background_resources()


if __name__ == '__main__':
    main()
