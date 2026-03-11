"""
MALCA - Multi-timescale ASAS-SN Light Curve Analysis

Unified command-line interface for the MALCA pipeline.

Usage:
    malca manifest [options]       # Build source_id → path index
    malca pipeline [options]       # Run full pipeline
    malca validate [options]       # Validate results against known candidates
    malca plot [options]           # Plot light curves
    malca filter [options]         # Apply candidate filters
    malca gaia-fetch [options]     # Download Gaia DR3 data for candidates
    malca characterize [options]   # Multi-wavelength characterization
    malca classify [options]       # Dipper classification
    malca injection [options]      # Injection-recovery tests
    malca detection_rate [options] # Measure detection rate
    malca review [options]         # Launch Dash review GUI (keyboard-driven)
    malca review-refresh [options] # Refresh review DB stats from a run/bundle
    malca review-mini [options]    # Launch lightweight click-through LC viewer
    malca review-explore [options] # Unified EDA + light-curve explorer
    malca stats [options]          # Light-curve statistics
    malca attrition [options]      # Pre/filter attrition summary
    malca vsx-filter [options]     # Build cleaned ASAS-SN index & filtered VSX catalog
    malca vsx-crossmatch [options] # Crossmatch ASAS-SN with VSX
    malca vetting [options]        # Run post-review vetting
    malca reproduce [options]      # Re-run detection on known objects
    malca events [options]         # Run event detection directly (low-level)
    malca tag [options]            # Apply tagging filters (low-level)
    malca score [options]          # Compute event score (low-level)
    malca ml_train [options]       # Train ML classifier on reviewed labels
    malca ml_predict [options]     # Score candidates with a trained ML model
    malca ltv-core [options]       # Compute seasonal trends for LTV (long-term variability)
    malca ltv-pipeline [options]   # Run full LTV pipeline (filters + crossmatch + NEOWISE)
    malca ltv-ingest [options]     # Ingest LTV pipeline results into a review DB
"""
import argparse
import importlib
import os
import sys


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
        "stats", "attrition", "review", "review-refresh", "review-mini", "review-explore",
        "neighbors", "spectra", "false_positive", "ml_train", "ml_predict", "vsx-filter", "vsx-crossmatch",
        "vetting",
        "ltv-core", "ltv-pipeline", "ltv-ingest",
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
        elif command == "review-mini":
            _run_module_main("malca.review.mini_viewer", remaining)
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
        elif command == "ltv-pipeline":
            ltv_pipeline = importlib.import_module("malca.ltv.pipeline")
            sys.argv = [sys.argv[0]] + remaining
            ltv_pipeline.run_pipeline_cli(
                ltv_pipeline.add_pipeline_args(argparse.ArgumentParser()).parse_args()
            )
        elif command == "ltv-ingest":
            _run_module_main("malca.ltv.review", remaining)
        return 0
    
    # If no subcommand or just --help for main, show main help
    parser = argparse.ArgumentParser(
        prog="malca",
        description="MALCA: Multi-timescale ASAS-SN Light Curve Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    subparsers.add_parser("manifest", help="Build manifest (source_id → path index)")
    subparsers.add_parser("pipeline", help="Run full discovery pipeline")
    subparsers.add_parser("reproduce", help="Re-run detection on known objects (needs raw data)")
    subparsers.add_parser("injection", help="Run injection-recovery tests")
    subparsers.add_parser("detection_rate", help="Measure detection rate")
    subparsers.add_parser("validate", help="Validate results against known candidates")
    subparsers.add_parser("plot", help="Plot light curves with events")
    subparsers.add_parser("filter", help="Apply candidate filters")
    subparsers.add_parser("tag", help="Apply tagging filters to candidate tables")
    subparsers.add_parser("events", help="Run event detection directly")
    subparsers.add_parser("gaia-fetch", help="Download Gaia DR3 data for candidates (AIP TAP mirror)")
    subparsers.add_parser("characterize", help="Characterize candidates with external catalogs")
    subparsers.add_parser("classify", help="Classify candidates by variability type")
    subparsers.add_parser("score", help="Compute event score for one light curve table")
    subparsers.add_parser("stats", help="Compute light-curve statistics")
    subparsers.add_parser("attrition", help="Summarize pre/filter attrition")
    subparsers.add_parser("review", help="Launch Dash review GUI (keyboard-driven, fast)")
    subparsers.add_parser("review-refresh", help="Refresh review DB stats from a run or bundle")
    subparsers.add_parser("review-mini", help="Launch lightweight click-through light-curve viewer")
    subparsers.add_parser("review-explore", help="Launch unified EDA and light-curve explorer")
    subparsers.add_parser("false_positive", help="Run false-positive contaminant benchmark")
    subparsers.add_parser("ml_train", help="Train baseline ML classifier on reviewed labels")
    subparsers.add_parser("ml_predict", help="Score candidates with a trained ML model")
    subparsers.add_parser("neighbors", help="Run bulk nearest-neighbor enrichment")
    subparsers.add_parser("spectra", help="Run bulk spectra-availability enrichment")
    subparsers.add_parser("vsx-filter", help="Build cleaned ASAS-SN index and filtered VSX catalog")
    subparsers.add_parser("vsx-crossmatch", help="Crossmatch ASAS-SN catalog with VSX catalog")
    subparsers.add_parser("vetting", help="Run post-review vetting (SIMBAD, Gaia, ASAS-SN, ZTF, TNS, eROSITA, ...)")
    subparsers.add_parser("ltv-core", help="Compute seasonal trends for long-term variability detection")
    subparsers.add_parser("ltv-pipeline", help="Run full LTV pipeline (filters + crossmatch + NEOWISE + extinction)")
    subparsers.add_parser("ltv-ingest", help="Ingest LTV pipeline results into a review DB")

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
