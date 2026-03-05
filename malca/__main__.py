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

from malca import characterize
from malca import classify
from malca import detect
from malca import events
from malca import filter as filter_mod
from malca import gaia_fetch
from malca import manifest
from malca import plot
from malca import score
from malca import stats
from malca import tag
from malca import vetting
from malca.enrich import neighbor as neighbor_enrich
from malca.enrich import spectra as spectra_enrich
from malca.ltv import core as ltv_core
from malca.ltv import pipeline as ltv_pipeline
from malca.ltv import review as ltv_review
from malca.ml import predict as ml_predict
from malca.ml import train as ml_train
from malca.review import app
from malca.vsx import crossmatch as vsx_crossmatch
from malca.vsx import filter as vsx_filter





def main():
    # Avoid CoreFoundation fork crash on macOS when subprocesses (e.g. TAP) run from within Dash
    os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")
    # Check if user is calling a subcommand with --help
    # If so, forward directly to the submodule
    if len(sys.argv) >= 2 and sys.argv[1] in [
        "manifest", "pipeline", "reproduce", "injection",
        "detection_rate", "validate", "plot",
        "events", "gaia-fetch", "characterize", "classify", "filter", "tag", "score",
        "stats", "attrition", "review",
        "neighbors", "spectra", "false_positive", "ml_train", "ml_predict", "vsx-filter", "vsx-crossmatch",
        "vetting",
        "ltv-core", "ltv-pipeline", "ltv-ingest",
    ]:
        command = sys.argv[1]
        remaining = sys.argv[2:]
        
        # Dispatch to appropriate module (--help will be handled by that module)
        if command == "manifest":

            sys.argv = [sys.argv[0]] + remaining
            manifest.main()
        elif command == "pipeline":

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

            sys.argv = [sys.argv[0]] + remaining
            plot.main()
        elif command == "events":

            sys.argv = [sys.argv[0]] + remaining
            events.main()
        elif command == "gaia-fetch":

            sys.argv = [sys.argv[0]] + remaining
            gaia_fetch.main()
        elif command == "characterize":

            sys.argv = [sys.argv[0]] + remaining
            characterize.main()
        elif command == "classify":

            sys.argv = [sys.argv[0]] + remaining
            classify.main()
        elif command == "stats":

            sys.argv = [sys.argv[0]] + remaining
            stats.main()
        elif command == "filter":

            sys.argv = [sys.argv[0]] + remaining
            filter_mod.main()
        elif command == "tag":

            sys.argv = [sys.argv[0]] + remaining
            tag.main()
        elif command == "score":

            sys.argv = [sys.argv[0]] + remaining
            score.main()
        elif command == "review":

            sys.argv = [sys.argv[0]] + remaining
            app.main()
        elif command == "validate":
            validation = importlib.import_module("malca.evaluation.validation")
            sys.argv = [sys.argv[0]] + remaining
            validation.main()
        elif command == "neighbors":

            sys.argv = [sys.argv[0]] + remaining
            neighbor_enrich.main()
        elif command == "spectra":

            sys.argv = [sys.argv[0]] + remaining
            spectra_enrich.main()
        elif command == "false_positive":
            fp = importlib.import_module("malca.evaluation.false_positive")
            sys.argv = [sys.argv[0]] + remaining
            fp.main()
        elif command == "ml_train":

            sys.argv = [sys.argv[0]] + remaining
            ml_train.main()
        elif command == "ml_predict":

            sys.argv = [sys.argv[0]] + remaining
            ml_predict.main()
        elif command == "vsx-filter":

            sys.argv = [sys.argv[0]] + remaining
            vsx_filter.cli()
        elif command == "vsx-crossmatch":

            sys.argv = [sys.argv[0]] + remaining
            vsx_crossmatch.cli()
        elif command == "vetting":

            sys.argv = [sys.argv[0]] + remaining
            vetting.main()
        elif command == "ltv-core":

            sys.argv = [sys.argv[0]] + remaining
            ltv_core.main()
        elif command == "ltv-pipeline":

            sys.argv = [sys.argv[0]] + remaining
            ltv_pipeline.run_pipeline_cli(
                ltv_pipeline.add_pipeline_args(argparse.ArgumentParser()).parse_args()
            )
        elif command == "ltv-ingest":

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
