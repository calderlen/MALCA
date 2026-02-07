"""
MALCA - Multi-timescale ASAS-SN Light Curve Analysis

Unified command-line interface for the MALCA pipeline.

Usage:
    malca manifest [options]    # Build source_id → path index
    malca detect [options]      # Run event detection
    malca validate [options]    # Validate on known objects
    malca injection [options]   # Run injection-recovery tests
    malca detection_rate [options]  # Measure detection rate
    malca plot [options]        # Plot light curves
    malca post_filter [options] # Apply post-filters
    malca postprocess [options] # Plot passing candidates
    malca filter [options]      # Apply signal-amplitude filter
    malca pre_filter [options]  # Apply pre-filters to manifest/candidate tables
    malca score [options]       # Compute event score for one light curve table
"""

import sys
import argparse


def main():
    # Check if user is calling a subcommand with --help
    # If so, forward directly to the submodule
    if len(sys.argv) >= 2 and sys.argv[1] in [
        "manifest", "detect", "reproduce", "injection",
        "detection_rate", "validate", "plot", "postprocess", "post_filter",
        "events", "characterize", "classify", "filter", "pre_filter", "score",
        "stats", "attrition", "review.plot", "review.gui", "review.tui", "review"
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
            from malca import reproduce
            sys.argv = [sys.argv[0]] + remaining
            reproduce.main()
        elif command == "injection":
            from malca.analysis import injection
            sys.argv = [sys.argv[0]] + remaining
            injection.main()
        elif command == "detection_rate":
            from malca.analysis import rate
            sys.argv = [sys.argv[0]] + remaining
            rate.main()
        elif command == "attrition":
            from malca.analysis import attrition
            sys.argv = [sys.argv[0]] + remaining
            attrition.main()
        elif command == "plot":
            from malca import plot
            sys.argv = [sys.argv[0]] + remaining
            plot.main()
        elif command == "postprocess":
            from malca import postprocess
            sys.argv = [sys.argv[0]] + remaining
            postprocess.main()
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
        elif command == "review.plot":
            from malca import plot_candidates as review_plot
            sys.argv = [sys.argv[0]] + remaining
            review_plot.main()
        elif command == "review.gui":
            from malca.review import dual as review_dual
            sys.argv = [sys.argv[0]] + remaining
            review_dual.main()
        elif command == "review.tui":
            from malca.review import dual as review_dual
            sys.argv = [sys.argv[0]] + remaining
            review_dual.main()
        elif command == "review":
            from malca.review import dual as review_dual
            sys.argv = [sys.argv[0]] + remaining
            review_dual.main()
        elif command == "validate":
            from malca import validation
            sys.argv = [sys.argv[0]] + remaining
            validation.main()
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
    subparsers.add_parser("postprocess", help="Plot passing candidates from post-filter output")
    subparsers.add_parser("post_filter", help="Apply quality post-filters")
    subparsers.add_parser("filter", help="Apply signal-amplitude filter")
    subparsers.add_parser("pre_filter", help="Apply pre-filters to candidate tables")
    subparsers.add_parser("events", help="Run event detection directly")
    subparsers.add_parser("characterize", help="Characterize candidates with external catalogs")
    subparsers.add_parser("classify", help="Classify candidates by variability type")
    subparsers.add_parser("score", help="Compute event score for one light curve table")
    subparsers.add_parser("stats", help="Compute light-curve statistics")
    subparsers.add_parser("attrition", help="Summarize pre/post-filter attrition")
    subparsers.add_parser("review.plot", help="Plot shortlist candidates for review")
    subparsers.add_parser("review.gui", help="Launch review GUI + TUI together")
    subparsers.add_parser("review.tui", help="Launch review GUI + TUI together")
    subparsers.add_parser("review", help="Launch review GUI + TUI together")
    
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
