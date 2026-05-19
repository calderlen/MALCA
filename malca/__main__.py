"""
MALCA - Multi-timescale ASAS-SN Light Curve Analysis

Unified command-line interface for the MALCA pipeline.
Run 'malca --help' for grouped commands; 'malca <command> --help' for options.
"""
import argparse
import importlib
import os
from pathlib import Path
import sys

# Command groups for grouped --help (order = importance)
GROUP_ORDER = [
    "Discovery",
    "Review",
    "LTV",
    "Evaluation",
    "Enrichment",
    "Other",
]
COMMAND_GROUPS = {
    "manifest": "Discovery",
    "pipeline": "Discovery",
    "detect": "Discovery",
    "filter": "Discovery",
    "tag": "Discovery",
    "events": "Discovery",
    "plot": "Discovery",
    "lc-plot": "Discovery",
    "gaia-fetch": "Discovery",
    "characterize": "Discovery",
    "classify": "Discovery",
    "review": "Review",
    "review-refresh": "Review",
    "review-merge": "Review",
    "review-explore": "Review",
    "review-sync": "Review",
    "review-taxonomy": "Review",
    "review-maint": "Review",
    "review-qt": "Review",
    "ltv-core": "LTV",
    "ltv-build": "LTV",
    "ltv-pipeline": "LTV",
    "ltv-ingest": "LTV",
    "ltv-injection": "LTV",
    "ltv-bundle": "LTV",
    "injection": "Evaluation",
    "detection-rate": "Evaluation",
    "validate": "Evaluation",
    "attrition": "Evaluation",
    "reproduce": "Evaluation",
    "audit": "Evaluation",
    "false-positive": "Evaluation",
    "neighbors": "Enrichment",
    "spectra": "Enrichment",
    "vsx-filter": "Enrichment",
    "vsx-crossmatch": "Enrichment",
    "external-lcs": "Enrichment",
    "multi-survey-features": "Enrichment",
    "sed-photometry": "Enrichment",
    "vetting": "Other",
    "dev": "Other",
}

MALCA_EPILOG = """
Run 'malca <command> --help' for per-command options.

Common workflows:
  Discovery   malca pipeline  then  malca review --plot-dir <run>/plots
  LTV         malca ltv-pipeline  then  malca review --review-db output/runs/ltv/review/review.db
"""


class _GroupedHelpFormatter(argparse.RawDescriptionHelpFormatter):
    """Format subparsers in groups (Discovery, Review, LTV, etc.) instead of one flat list."""

    def _format_action(self, action):
        if not isinstance(action, argparse._SubParsersAction):
            return super()._format_action(action)
        by_group = {g: [] for g in GROUP_ORDER}
        for name, choice_parser in action.choices.items():
            group = COMMAND_GROUPS.get(name, "Other")
            if group not in by_group:
                by_group[group] = []
            help_str = choice_parser.description or ""
            by_group[group].append((name, help_str))
        parts = []
        for group in GROUP_ORDER:
            items = by_group.get(group)
            if not items:
                continue
            parts.append(f"  {group}:")
            for name, help_str in items:
                parts.append(f"    {name:<20} {help_str}")
            parts.append("")
        return "\n".join(parts) + "\n"


def _run_module_main(module_name: str, remaining_args: list[str]) -> None:
    mod = importlib.import_module(module_name)
    sys.argv = [sys.argv[0]] + remaining_args
    mod.main()





def main():
    # Avoid CoreFoundation fork crash on macOS when subprocesses (e.g. TAP) run from within Dash
    os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")
    # Check if user is calling a subcommand with --help
    # If so, forward directly to the submodule
    if len(sys.argv) >= 2 and sys.argv[1] in [
        "manifest", "pipeline", "detect", "reproduce", "injection",
        "detection-rate", "validate", "plot", "audit",
        "events", "lc-plot", "gaia-fetch", "characterize", "classify", "filter", "tag",
        "attrition", "review", "review-qt", "review-refresh", "review-merge", "review-explore", "review-sync", "review-taxonomy", "review-maint",
        "neighbors", "spectra", "false-positive", "vsx-filter", "vsx-crossmatch", "external-lcs", "multi-survey-features", "sed-photometry",
        "vetting",
        "ltv-core", "ltv-build", "ltv-pipeline", "ltv-injection", "ltv-ingest", "ltv-bundle",
        "dev",
    ]:
        command = sys.argv[1]
        remaining = sys.argv[2:]
        
        # Dispatch to appropriate module (--help will be handled by that module)
        if command == "manifest":
            _run_module_main("malca.manifest", remaining)
        elif command in {"pipeline", "detect"}:
            _run_module_main("malca.detect", remaining)
        elif command == "reproduce":
            reproduce = importlib.import_module("malca.evaluation.reproduce")
            sys.argv = [sys.argv[0]] + remaining
            reproduce.main()
        elif command == "injection":
            injection = importlib.import_module("malca.evaluation.injection")
            sys.argv = [sys.argv[0]] + remaining
            injection.main()
        elif command == "detection-rate":
            detection_rate_mod = importlib.import_module("malca.evaluation.detection_rate")
            sys.argv = [sys.argv[0]] + remaining
            detection_rate_mod.main()
        elif command == "attrition":
            attrition = importlib.import_module("malca.evaluation.attrition")
            sys.argv = [sys.argv[0]] + remaining
            attrition.main()
        elif command == "audit":
            _run_module_main("malca.audit", remaining)
        elif command == "plot":
            _run_module_main("malca.plot", remaining)
        elif command == "lc-plot":
            _run_module_main("malca.lightcurve_publication", remaining)
        elif command == "events":
            _run_module_main("malca.events", remaining)
        elif command == "gaia-fetch":
            _run_module_main("malca.gaia_fetch", remaining)
        elif command == "characterize":
            _run_module_main("malca.characterize", remaining)
        elif command == "classify":
            _run_module_main("malca.classify", remaining)
        elif command == "filter":
            _run_module_main("malca.filter", remaining)
        elif command == "tag":
            _run_module_main("malca.tag", remaining)
        elif command == "review":
            _run_module_main("malca.review.app", remaining)
        elif command == "review-qt":
            review_qt = importlib.import_module("malca.review_qt.main")
            sys.argv = [sys.argv[0]] + remaining
            return review_qt.main()
        elif command == "review-refresh":
            _run_module_main("malca.review.refresh", remaining)
        elif command == "review-merge":
            _run_module_main("malca.review.merge", remaining)
        elif command == "review-explore":
            _run_module_main("malca.review.explorer", remaining)
        elif command == "review-sync":
            _run_module_main("malca.review.sync", remaining)
        elif command == "review-taxonomy":
            _run_module_main("malca.review.taxonomy", remaining)
        elif command == "review-maint":
            _run_module_main("malca.review.maintenance", remaining)
        elif command == "validate":
            validation = importlib.import_module("malca.evaluation.validation")
            sys.argv = [sys.argv[0]] + remaining
            validation.main()
        elif command == "neighbors":
            _run_module_main("malca.enrich.neighbor", remaining)
        elif command == "spectra":
            _run_module_main("malca.enrich.spectra", remaining)
        elif command == "false-positive":
            fp = importlib.import_module("malca.evaluation.false_positive")
            sys.argv = [sys.argv[0]] + remaining
            fp.main()
        elif command == "vsx-filter":
            sys.argv = [sys.argv[0]] + remaining
            vsx_filter = importlib.import_module("malca.vsx.filter")
            vsx_filter.cli()
        elif command == "vsx-crossmatch":
            sys.argv = [sys.argv[0]] + remaining
            vsx_crossmatch = importlib.import_module("malca.vsx.crossmatch")
            vsx_crossmatch.cli()
        elif command == "external-lcs":
            _run_module_main("malca.external_lcs", remaining)
        elif command == "multi-survey-features":
            _run_module_main("malca.multi_survey_features", remaining)
        elif command == "sed-photometry":
            _run_module_main("malca.sed_photometry", remaining)
        elif command == "vetting":
            _run_module_main("malca.vetting", remaining)
        elif command == "ltv-core":
            _run_module_main("malca.ltv.core", remaining)
        elif command == "ltv-build":
            ltv_pipeline = importlib.import_module("malca.ltv.pipeline")
            sys.argv = [sys.argv[0]] + remaining
            ltv_pipeline.run_pipeline_cli(
                ltv_pipeline.add_pipeline_args(argparse.ArgumentParser()).parse_args()
            )
        elif command == "ltv-pipeline":
            ltv_pipeline = importlib.import_module("malca.ltv.pipeline")
            ltv_review = importlib.import_module("malca.ltv.review")
            ltv_paths = importlib.import_module("malca.ltv.paths")
            parser = ltv_pipeline.add_pipeline_args(argparse.ArgumentParser(
                prog="malca ltv-pipeline",
                description="Full LTV workflow: build then ingest into review DB (run malca review separately to open GUI).",
            ))
            parser.add_argument(
                "--review-db",
                type=str,
                default=None,
                help="Path to LTV review SQLite DB for ingest (default: <run-dir>/review/review.db)",
            )
            parser.add_argument(
                "--skip-characterize",
                action="store_true",
                help="Skip Gaia/dust characterization during ingest",
            )
            parser.add_argument(
                "--run-vetting",
                action="store_true",
                help="Run STV vetting during ingest",
            )
            parser.add_argument(
                "--skip-stats",
                action="store_true",
                help="Skip compute_stats enrichment during ingest",
            )
            parser.add_argument(
                "--stats-compute-ls",
                action="store_true",
                help="Also compute Lomb-Scargle during compute_stats ingest enrichment",
            )
            parser.add_argument(
                "--index-file",
                type=str,
                default=None,
                help="Path to ASASSN index parquet for Gaia ID lookup during ingest",
            )
            args = parser.parse_args(remaining)
            run_dir = Path(args.run_dir).expanduser()
            if args.review_db is None:
                args.review_db = str(ltv_paths.ltv_review_db_path(run_dir))
            df = ltv_pipeline.run_pipeline_cli(args)
            ltv_review.ingest_ltv_results(
                args.review_db,
                df,
                run_characterize=not args.skip_characterize,
                run_vetting=args.run_vetting,
                run_stats=not args.skip_stats,
                stats_compute_ls=args.stats_compute_ls,
                n_workers=args.workers,
                index_path=args.index_file,
                source_path=run_dir,
                verbose=args.verbose,
            )
        elif command == "dev":
            if not remaining:
                raise SystemExit("usage: malca dev {score,stats} ...")
            dev_command, dev_args = remaining[0], remaining[1:]
            if dev_command == "score":
                _run_module_main("malca.score", dev_args)
            elif dev_command == "stats":
                _run_module_main("malca.stats", dev_args)
            else:
                raise SystemExit(f"unknown dev command: {dev_command}")
        elif command == "ltv-injection":
            _run_module_main("malca.ltv.injection", remaining)
        elif command == "ltv-ingest":
            _run_module_main("malca.ltv.review", remaining)
        elif command == "ltv-bundle":
            _run_module_main("malca.ltv.bundle", remaining)
        return 0
    
    # If no subcommand or just --help for main, show main help
    parser = argparse.ArgumentParser(
        prog="malca",
        description="MALCA: Multi-timescale ASAS-SN Light Curve Analysis",
        formatter_class=_GroupedHelpFormatter,
        epilog=MALCA_EPILOG,
    )
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")

    # Register in group order (Discovery, Review, LTV, Evaluation, Enrichment, Other)
    # Discovery
    subparsers.add_parser("manifest", description="Build manifest (source_id → path index)")
    subparsers.add_parser("pipeline", description="Run full discovery pipeline")
    subparsers.add_parser("detect", description="Alias for pipeline")
    subparsers.add_parser("filter", description="Apply candidate filters")
    subparsers.add_parser("tag", description="Apply tagging filters to candidate tables")
    subparsers.add_parser("events", description="Run event detection directly")
    subparsers.add_parser("plot", description="Plot light curves with events")
    subparsers.add_parser("lc-plot", description="Create a publication-quality light-curve figure")
    subparsers.add_parser("gaia-fetch", description="Download Gaia DR3 data for candidates (AIP TAP mirror)")
    subparsers.add_parser("characterize", description="Characterize candidates with external catalogs")
    subparsers.add_parser("classify", description="Classify candidates by variability type")
    # Review
    subparsers.add_parser("review", description="Launch Dash review GUI (keyboard-driven, fast)")
    subparsers.add_parser("review-refresh", description="Refresh review DB stats from a run or bundle")
    subparsers.add_parser("review-merge", description="Merge reviewed subset DB content into a master review DB")
    subparsers.add_parser("review-explore", description="Launch unified EDA and light-curve explorer")
    subparsers.add_parser("review-sync", description="Import/export Git-trackable review bundle files")
    subparsers.add_parser("review-taxonomy", description="Migrate legacy review DBs to taxonomy schema")
    subparsers.add_parser("review-maint", description="Review DB maintenance commands")
    # LTV
    subparsers.add_parser("ltv-core", description="Compute seasonal trends for long-term variability detection")
    subparsers.add_parser("ltv-build", description="Build LTV candidate table (filters + crossmatch + NEOWISE + extinction)")
    subparsers.add_parser("ltv-pipeline", description="Full LTV workflow up to review: build then ingest into review DB (run malca review separately to open GUI)")
    subparsers.add_parser("ltv-ingest", description="Ingest LTV build output into a review DB")
    subparsers.add_parser("ltv-injection", description="Run LTV rejection-recovery injections and plots")
    subparsers.add_parser("ltv-bundle", description="Bundle light curve files for LTV candidates passing slope/diff filters")
    # Evaluation
    subparsers.add_parser("injection", description="Run injection-recovery tests")
    subparsers.add_parser("detection-rate", description="Measure detection rate")
    subparsers.add_parser("validate", description="Validate results against known candidates")
    subparsers.add_parser("attrition", description="Summarize pre/filter attrition")
    subparsers.add_parser("reproduce", description="Re-run detection on known objects (needs raw data)")
    subparsers.add_parser("audit", description="Audit result tables, LTV status, and baseline comparison commands")
    subparsers.add_parser("false-positive", description="Run false-positive contaminant benchmark")
    # Enrichment
    subparsers.add_parser("neighbors", description="Run bulk nearest-neighbor enrichment")
    subparsers.add_parser("spectra", description="Run bulk spectra-availability enrichment")
    subparsers.add_parser("vsx-filter", description="Build cleaned ASAS-SN index and filtered VSX catalog")
    subparsers.add_parser("vsx-crossmatch", description="Crossmatch ASAS-SN catalog with VSX catalog")
    subparsers.add_parser("external-lcs", description="Fetch external light curves for candidate tables")
    subparsers.add_parser("multi-survey-features", description="Compute event-relative multi-survey features")
    subparsers.add_parser("sed-photometry", description="Fetch and normalize SED photometry for candidate tables")
    # Other
    subparsers.add_parser("vetting", description="Run post-review vetting (SIMBAD, Gaia, ASAS-SN, ZTF, TNS, eROSITA, ...)")
    subparsers.add_parser("dev", description="Developer diagnostics (score, stats)")

    if len(sys.argv) == 1:
        parser.print_help()
    else:
        parser.parse_args()
    return 0


if __name__ == "__main__":
    sys.exit(main())
