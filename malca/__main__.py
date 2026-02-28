"""
MALCA - Multi-timescale ASAS-SN Light Curve Analysis

Unified command-line interface for the MALCA pipeline.

Usage:
    malca manifest [options]       # Build source_id → path index
    malca pipeline [options]       # Run full pipeline
    malca validate [options]       # Validate results against known candidates
    malca plot [options]           # Plot light curves
    malca filter [options]         # Apply signal-amplitude filter
    malca post_filter [options]    # Apply quality post-filters
    malca gaia-fetch [options]     # Download Gaia DR3 data for candidates
    malca characterize [options]   # Multi-wavelength characterization
    malca classify [options]       # Dipper classification
    malca injection [options]      # Injection-recovery tests
    malca detection_rate [options] # Measure detection rate
    malca review [options]         # Launch Dash review GUI (keyboard-driven)
    malca stats [options]          # Light-curve statistics
    malca attrition [options]      # Pre/post-filter attrition summary
    malca vsx-filter [options]     # Build cleaned ASAS-SN index & filtered VSX catalog
    malca vsx-crossmatch [options] # Crossmatch ASAS-SN with VSX
    malca vetting [options]        # Run post-review vetting
    malca reproduce [options]      # Re-run detection on known objects
    malca events [options]         # Run event detection directly (low-level)
    malca pre_filter [options]     # Apply pre-filters (low-level)
    malca pre_tag [options]        # Apply pre-tagging filters (alias)
    malca score [options]          # Compute event score (low-level)
    malca ltv-core [options]       # Compute seasonal trends for LTV (long-term variability)
    malca ltv-pipeline [options]   # Run full LTV pipeline (filters + crossmatch + NEOWISE)
    malca ltv-ingest [options]     # Ingest LTV pipeline results into a review DB
"""

import os
import sys
import argparse
import importlib


def main():
    # Avoid CoreFoundation fork crash on macOS when subprocesses (e.g. TAP) run from within Dash
    os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")
    # Check if user is calling a subcommand with --help
    # If so, forward directly to the submodule
    if len(sys.argv) >= 2 and sys.argv[1] in [
        "manifest", "pipeline", "reproduce", "injection",
        "detection_rate", "validate", "plot", "post_filter",
        "events", "gaia-fetch", "characterize", "classify", "filter", "pre_filter", "pre_tag", "score",
        "stats", "attrition", "review",
        "neighbors", "spectra", "false_positive", "ml_train", "vsx-filter", "vsx-crossmatch",
        "vetting",
        "ltv-core", "ltv-pipeline", "ltv-ingest",
    ]:
        command = sys.argv[1]
        remaining = sys.argv[2:]
        
        # Dispatch to appropriate module (--help will be handled by that module)
        if command == "manifest":
            from malca import manifest
            sys.argv = [sys.argv[0]] + remaining
            manifest.main()
        elif command == "pipeline":
            from malca import detect
            sys.argv = [sys.argv[0]] + remaining
            detect.main()
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
            from malca import plot
            sys.argv = [sys.argv[0]] + remaining
            plot.main()
        elif command == "post_filter":
            from malca import post_filter
            sys.argv = [sys.argv[0]] + remaining
            post_filter.main()
        elif command == "events":
            from malca import events
            sys.argv = [sys.argv[0]] + remaining
            events.main()
        elif command == "gaia-fetch":
            from malca import gaia_fetch
            sys.argv = [sys.argv[0]] + remaining
            gaia_fetch.main()
        elif command == "characterize":
            from malca import characterize
            sys.argv = [sys.argv[0]] + remaining
            characterize.main()
        elif command == "classify":
            from malca import classify
            sys.argv = [sys.argv[0]] + remaining
            classify.main()
        elif command == "stats":
            from malca import stats
            sys.argv = [sys.argv[0]] + remaining
            stats.main()
        elif command == "filter":
            from malca import filter as signal_filter
            sys.argv = [sys.argv[0]] + remaining
            signal_filter.main()
        elif command == "pre_filter":
            from malca import pre_filter
            sys.argv = [sys.argv[0]] + remaining
            pre_filter.main()
        elif command == "pre_tag":
            from malca import pre_tag
            sys.argv = [sys.argv[0]] + remaining
            pre_tag.main()
        elif command == "score":
            from malca import score
            sys.argv = [sys.argv[0]] + remaining
            score.main()
        elif command == "review":
            from malca.review import app
            sys.argv = [sys.argv[0]] + remaining
            app.main()
        elif command == "validate":
            validation = importlib.import_module("malca.evaluation.validation")
            sys.argv = [sys.argv[0]] + remaining
            validation.main()
        elif command == "neighbors":
            from malca.enrich import neighbor as neighbor_enrich
            sys.argv = [sys.argv[0]] + remaining
            neighbor_enrich.main()
        elif command == "spectra":
            from malca.enrich import spectra as spectra_enrich
            sys.argv = [sys.argv[0]] + remaining
            spectra_enrich.main()
        elif command == "false_positive":
            fp = importlib.import_module("malca.evaluation.false_positive")
            sys.argv = [sys.argv[0]] + remaining
            fp.main()
        elif command == "ml_train":
            from malca.ml import train as ml_train
            sys.argv = [sys.argv[0]] + remaining
            ml_train.main()
        elif command == "vsx-filter":
            from malca.vsx import filter as vsx_filter
            sys.argv = [sys.argv[0]] + remaining
            vsx_filter.cli()
        elif command == "vsx-crossmatch":
            from malca.vsx import crossmatch as vsx_crossmatch
            sys.argv = [sys.argv[0]] + remaining
            vsx_crossmatch.cli()
        elif command == "vetting":
            from malca import vetting
            sys.argv = [sys.argv[0]] + remaining
            vetting.main()
        elif command == "ltv-core":
            from malca.ltv import core as ltv_core
            sys.argv = [sys.argv[0]] + remaining
            ltv_core.main()
        elif command == "ltv-pipeline":
            from malca.ltv import pipeline as ltv_pipeline
            sys.argv = [sys.argv[0]] + remaining
            ltv_pipeline.run_pipeline_cli(
                ltv_pipeline.add_pipeline_args(argparse.ArgumentParser()).parse_args()
            )
        elif command == "ltv-ingest":
            from malca.ltv import review as ltv_review
            sys.argv = [sys.argv[0]] + remaining
            ltv_review.main()
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
    subparsers.add_parser("post_filter", help="Apply quality post-filters")
    subparsers.add_parser("filter", help="Apply signal-amplitude filter")
    subparsers.add_parser("pre_filter", help="Apply pre-filters to candidate tables")
    subparsers.add_parser("pre_tag", help="Apply pre-tagging filters to candidate tables (alias of pre_filter)")
    subparsers.add_parser("events", help="Run event detection directly")
    subparsers.add_parser("gaia-fetch", help="Download Gaia DR3 data for candidates (AIP TAP mirror)")
    subparsers.add_parser("characterize", help="Characterize candidates with external catalogs")
    subparsers.add_parser("classify", help="Classify candidates by variability type")
    subparsers.add_parser("score", help="Compute event score for one light curve table")
    subparsers.add_parser("stats", help="Compute light-curve statistics")
    subparsers.add_parser("attrition", help="Summarize pre/post-filter attrition")
    subparsers.add_parser("review", help="Launch Dash review GUI (keyboard-driven, fast)")
    subparsers.add_parser("false_positive", help="Run false-positive contaminant benchmark")
    subparsers.add_parser("ml_train", help="Train baseline ML classifier on reviewed labels")
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
