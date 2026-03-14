"""
MALCA - Multi-timescale ASAS-SN Light Curve Analysis

Unified command-line interface for the MALCA pipeline.
Run 'malca --help' for grouped commands; 'malca <command> --help' for options.
"""
import argparse
import importlib
import os
import sys

# Command groups for grouped --help (order = importance)
GROUP_ORDER = [
    "Discovery",
    "Review",
    "LTV",
    "Evaluation",
    "Enrichment",
    "ML",
    "Other",
]
COMMAND_GROUPS = {
    "manifest": "Discovery",
    "pipeline": "Discovery",
    "filter": "Discovery",
    "tag": "Discovery",
    "events": "Discovery",
    "plot": "Discovery",
    "score": "Discovery",
    "stats": "Discovery",
    "gaia-fetch": "Discovery",
    "characterize": "Discovery",
    "classify": "Discovery",
    "review": "Review",
    "review-refresh": "Review",
    "review-merge": "Review",
    "review-explore": "Review",
    "ltv-core": "LTV",
    "ltv-build": "LTV",
    "ltv-pipeline": "LTV",
    "ltv-ingest": "LTV",
    "ltv-pca": "LTV",
    "ltv-injection": "LTV",
    "ltv-bundle": "LTV",
    "injection": "Evaluation",
    "detection_rate": "Evaluation",
    "validate": "Evaluation",
    "attrition": "Evaluation",
    "reproduce": "Evaluation",
    "false_positive": "Evaluation",
    "neighbors": "Enrichment",
    "spectra": "Enrichment",
    "vsx-filter": "Enrichment",
    "vsx-crossmatch": "Enrichment",
    "ml_train": "ML",
    "ml_predict": "ML",
    "vetting": "Other",
}

MALCA_EPILOG = """
Run 'malca <command> --help' for per-command options.

Common workflows:
  Discovery   malca pipeline  then  malca review --plot-dir <run>/plots
  LTV         malca ltv-pipeline  then  malca review --db ltv_candidates.db
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
        "manifest", "pipeline", "reproduce", "injection",
        "detection_rate", "validate", "plot",
        "events", "gaia-fetch", "characterize", "classify", "filter", "tag", "score",
        "stats", "attrition", "review", "review-refresh", "review-merge", "review-explore",
        "neighbors", "spectra", "false_positive", "ml_train", "ml_predict", "vsx-filter", "vsx-crossmatch",
        "vetting",
        "ltv-core", "ltv-build", "ltv-pipeline", "ltv-injection", "ltv-pca", "ltv-ingest", "ltv-bundle",
    ]:
        command = sys.argv[1]
        remaining = sys.argv[2:]
        
        # Dispatch to appropriate module (--help will be handled by that module)
        if command == "manifest":
            _run_module_main("malca.manifest", remaining)
        elif command == "pipeline":
            _run_module_main("malca.detect", remaining)
        elif command == "reproduce":
            reproduce = importlib.import_module("malca.evaluation.reproduce")
            sys.argv = [sys.argv[0]] + remaining
            reproduce.main()
        elif command == "injection":
            injection = importlib.import_module("malca.evaluation.injection")
            sys.argv = [sys.argv[0]] + remaining
            injection.main()
        elif command == "detection_rate":
            detection_rate_mod = importlib.import_module("malca.evaluation.detection_rate")
            sys.argv = [sys.argv[0]] + remaining
            detection_rate_mod.main()
        elif command == "attrition":
            attrition = importlib.import_module("malca.evaluation.attrition")
            sys.argv = [sys.argv[0]] + remaining
            attrition.main()
        elif command == "plot":
            _run_module_main("malca.plot", remaining)
        elif command == "events":
            _run_module_main("malca.events", remaining)
        elif command == "gaia-fetch":
            _run_module_main("malca.gaia_fetch", remaining)
        elif command == "characterize":
            _run_module_main("malca.characterize", remaining)
        elif command == "classify":
            _run_module_main("malca.classify", remaining)
        elif command == "stats":
            _run_module_main("malca.stats", remaining)
        elif command == "filter":
            _run_module_main("malca.filter", remaining)
        elif command == "tag":
            _run_module_main("malca.tag", remaining)
        elif command == "score":
            _run_module_main("malca.score", remaining)
        elif command == "review":
            _run_module_main("malca.review.app", remaining)
        elif command == "review-refresh":
            _run_module_main("malca.review.refresh", remaining)
        elif command == "review-merge":
            _run_module_main("malca.review.merge", remaining)
        elif command == "review-explore":
            _run_module_main("malca.review.explorer", remaining)
        elif command == "validate":
            validation = importlib.import_module("malca.evaluation.validation")
            sys.argv = [sys.argv[0]] + remaining
            validation.main()
        elif command == "neighbors":
            _run_module_main("malca.enrich.neighbor", remaining)
        elif command == "spectra":
            _run_module_main("malca.enrich.spectra", remaining)
        elif command == "false_positive":
            fp = importlib.import_module("malca.evaluation.false_positive")
            sys.argv = [sys.argv[0]] + remaining
            fp.main()
        elif command == "ml_train":
            _run_module_main("malca.ml.train", remaining)
        elif command == "ml_predict":
            _run_module_main("malca.ml.predict", remaining)
        elif command == "vsx-filter":
            sys.argv = [sys.argv[0]] + remaining
            vsx_filter = importlib.import_module("malca.vsx.filter")
            vsx_filter.cli()
        elif command == "vsx-crossmatch":
            sys.argv = [sys.argv[0]] + remaining
            vsx_crossmatch = importlib.import_module("malca.vsx.crossmatch")
            vsx_crossmatch.cli()
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
            parser = ltv_pipeline.add_pipeline_args(argparse.ArgumentParser(
                prog="malca ltv-pipeline",
                description="Full LTV workflow: build then ingest into review DB (run malca review separately to open GUI).",
            ))
            parser.add_argument(
                "--db",
                type=str,
                default="ltv_candidates.db",
                help="Path to LTV review SQLite DB for ingest (default: ltv_candidates.db)",
            )
            args = parser.parse_args(remaining)
            df = ltv_pipeline.run_pipeline_cli(args)
            ltv_review.ingest_ltv_results(
                args.db,
                df,
                run_characterize=True,
                run_vetting=False,
                run_stats=True,
                stats_compute_ls=False,
                verbose=args.verbose,
            )
        elif command == "ltv-injection":
            _run_module_main("malca.ltv.injection", remaining)
        elif command == "ltv-pca":
            _run_module_main("malca.ltv.pca", remaining)
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

    # Register in group order (Discovery, Review, LTV, Evaluation, Enrichment, ML, Other)
    # Discovery
    subparsers.add_parser("manifest", description="Build manifest (source_id → path index)")
    subparsers.add_parser("pipeline", description="Run full discovery pipeline")
    subparsers.add_parser("filter", description="Apply candidate filters")
    subparsers.add_parser("tag", description="Apply tagging filters to candidate tables")
    subparsers.add_parser("events", description="Run event detection directly")
    subparsers.add_parser("plot", description="Plot light curves with events")
    subparsers.add_parser("score", description="Compute event score for one light curve table")
    subparsers.add_parser("stats", description="Compute light-curve statistics")
    subparsers.add_parser("gaia-fetch", description="Download Gaia DR3 data for candidates (AIP TAP mirror)")
    subparsers.add_parser("characterize", description="Characterize candidates with external catalogs")
    subparsers.add_parser("classify", description="Classify candidates by variability type")
    # Review
    subparsers.add_parser("review", description="Launch Dash review GUI (keyboard-driven, fast)")
    subparsers.add_parser("review-refresh", description="Refresh review DB stats from a run or bundle")
    subparsers.add_parser("review-merge", description="Merge reviewed subset DB content into a master review DB")
    subparsers.add_parser("review-explore", description="Launch unified EDA and light-curve explorer")
    # LTV
    subparsers.add_parser("ltv-core", description="Compute seasonal trends for long-term variability detection")
    subparsers.add_parser("ltv-build", description="Build LTV candidate table (filters + crossmatch + NEOWISE + extinction)")
    subparsers.add_parser("ltv-pipeline", description="Full LTV workflow up to review: build then ingest into review DB (run malca review separately to open GUI)")
    subparsers.add_parser("ltv-ingest", description="Ingest LTV build output into a review DB")
    subparsers.add_parser("ltv-pca", description="Fit/apply LTV PCA (fit-apply | apply)")
    subparsers.add_parser("ltv-injection", description="Run LTV rejection-recovery injections and plots")
    subparsers.add_parser("ltv-bundle", description="Bundle .dat2 files for LTV candidates passing slope/diff filters")
    # Evaluation
    subparsers.add_parser("injection", description="Run injection-recovery tests")
    subparsers.add_parser("detection_rate", description="Measure detection rate")
    subparsers.add_parser("validate", description="Validate results against known candidates")
    subparsers.add_parser("attrition", description="Summarize pre/filter attrition")
    subparsers.add_parser("reproduce", description="Re-run detection on known objects (needs raw data)")
    subparsers.add_parser("false_positive", description="Run false-positive contaminant benchmark")
    # Enrichment
    subparsers.add_parser("neighbors", description="Run bulk nearest-neighbor enrichment")
    subparsers.add_parser("spectra", description="Run bulk spectra-availability enrichment")
    subparsers.add_parser("vsx-filter", description="Build cleaned ASAS-SN index and filtered VSX catalog")
    subparsers.add_parser("vsx-crossmatch", description="Crossmatch ASAS-SN catalog with VSX catalog")
    # ML
    subparsers.add_parser("ml_train", description="Train baseline ML classifier on reviewed labels")
    subparsers.add_parser("ml_predict", description="Score candidates with a trained ML model")
    # Other
    subparsers.add_parser("vetting", description="Run post-review vetting (SIMBAD, Gaia, ASAS-SN, ZTF, TNS, eROSITA, ...)")

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
