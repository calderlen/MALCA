"""
MALCA - Multi-timescale ASAS-SN Light Curve Analysis

Unified command-line interface for the MALCA pipeline.

Usage:
    malca manifest [options]       # Build source_id → path index
    malca detect [options]         # Run full detection pipeline
    malca validate [options]       # Validate results against known candidates
    malca plot [options]           # Plot light curves
    malca filter [options]         # Apply signal-amplitude filter
    malca post_filter [options]    # Apply quality post-filters
    malca characterize [options]   # Multi-wavelength characterization
    malca classify [options]       # Dipper classification
    malca injection [options]      # Injection-recovery tests
    malca detection_rate [options] # Measure detection rate
    malca review [options]         # Launch review GUI + TUI
    malca review.tui [options]     # Launch terminal review interface
    malca stats [options]          # Light-curve statistics
    malca attrition [options]      # Pre/post-filter attrition summary
    malca reproduce [options]      # Re-run detection on known objects
    malca events [options]         # Run event detection directly (low-level)
    malca pre_filter [options]     # Apply pre-filters (low-level)
    malca score [options]          # Compute event score (low-level)
"""

import sys
import argparse
import importlib


def main():
    # Check if user is calling a subcommand with --help
    # If so, forward directly to the submodule
    if len(sys.argv) >= 2 and sys.argv[1] in [
        "manifest", "detect", "reproduce", "injection",
        "detection_rate", "validate", "plot", "post_filter",
        "events", "characterize", "classify", "filter", "pre_filter", "score",
        "stats", "attrition", "review.tui", "review",
        "neighbors", "spectra", "false_positive", "ml_train"
    ]:
        command = sys.argv[1]
        remaining = sys.argv[2:]
        
        # Dispatch to appropriate module (--help will be handled by that module)
        if command == "manifest":
            from malca import manifest
            sys.argv = [sys.argv[0]] + remaining
            manifest.main()
        elif command == "detect":
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
            rate = importlib.import_module("malca.evaluation.rate")
            sys.argv = [sys.argv[0]] + remaining
            rate.main()
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
        elif command == "score":
            from malca import score
            sys.argv = [sys.argv[0]] + remaining
            score.main()
        elif command == "review.tui":
            from malca.review import tui as review_tui
            sys.argv = [sys.argv[0]] + remaining
            review_tui.main()
        elif command == "review":
            from malca.review import dual as review_dual
            sys.argv = [sys.argv[0]] + remaining
            review_dual.main()
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
    subparsers.add_parser("detect", help="Run event detection pipeline")
    subparsers.add_parser("reproduce", help="Re-run detection on known objects (needs raw data)")
    subparsers.add_parser("injection", help="Run injection-recovery tests")
    subparsers.add_parser("detection_rate", help="Measure detection rate")
    subparsers.add_parser("validate", help="Validate results against known candidates")
    subparsers.add_parser("plot", help="Plot light curves with events")
    subparsers.add_parser("post_filter", help="Apply quality post-filters")
    subparsers.add_parser("filter", help="Apply signal-amplitude filter")
    subparsers.add_parser("pre_filter", help="Apply pre-filters to candidate tables")
    subparsers.add_parser("events", help="Run event detection directly")
    subparsers.add_parser("characterize", help="Characterize candidates with external catalogs")
    subparsers.add_parser("classify", help="Classify candidates by variability type")
    subparsers.add_parser("score", help="Compute event score for one light curve table")
    subparsers.add_parser("stats", help="Compute light-curve statistics")
    subparsers.add_parser("attrition", help="Summarize pre/post-filter attrition")
    subparsers.add_parser("review.tui", help="Launch terminal review interface")
    subparsers.add_parser("review", help="Launch review GUI + TUI together")
    subparsers.add_parser("false_positive", help="Run false-positive contaminant benchmark")
    subparsers.add_parser("ml_train", help="Train baseline ML classifier on reviewed labels")
    subparsers.add_parser("neighbors", help="Run bulk nearest-neighbor enrichment")
    subparsers.add_parser("spectra", help="Run bulk spectra-availability enrichment")
    
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
