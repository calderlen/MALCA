"""
MALCA - Multi-timescale ASAS-SN Light Curve Analysis

Unified command-line interface for the MALCA pipeline.
Run 'malca --help' for grouped commands; 'malca <command> --help' for options.
"""
import argparse
import importlib
import os
import sys
import textwrap

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
    "stv-pipeline": "Discovery",
    "stv-filter": "Discovery",
    "stv-tag": "Discovery",
    "stv-events": "Discovery",
    "stv-plot": "Discovery",
    "lc-plot": "Discovery",
    "gaia-fetch": "Discovery",
    "gaia-id-repair": "Discovery",
    "characterize": "Discovery",
    "classify": "Discovery",
    "review": "Review",
    "review-refresh": "Review",
    "review-merge": "Review",
    "review-sync": "Review",
    "review-taxonomy": "Review",
    "review-maint": "Review",
    "ltv-pipeline": "LTV",
    "ltv-injection": "LTV",
    "dip-injection": "Evaluation",
    "detection-rate": "Evaluation",
    "validate": "Evaluation",
    "attrition": "Evaluation",
    "reproduce": "Evaluation",
    "audit": "Evaluation",
    "bad-photometry": "Evaluation",
    "false-positive": "Evaluation",
    "neighbors": "Enrichment",
    "spectra": "Enrichment",
    "vsx-filter": "Enrichment",
    "vsx-crossmatch": "Enrichment",
    "external-lcs": "Enrichment",
    "multi-survey-features": "Enrichment",
    "feature-layers": "Enrichment",
    "migrate": "Enrichment",
    "sed-photometry": "Enrichment",
    "nuclear-context": "Enrichment",
    "vetting": "Other",
    "dev": "Other",
    "malcat-train": "Other",
}

MALCA_EPILOG = """
Run 'malca <command> --help' for per-command options.

Common workflows:
  STV           malca stv-pipeline  then  malca review --review-db <run>/review/review.db
  STV extended  malca stv-pipeline --stage full-extended
  LTV           malca ltv-pipeline --mag-bin 13_13.5  then  malca review --review-db <run>/review/review.db
  LTV extended  malca ltv-pipeline --stage full-extended --full-bundle
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
                prefix = f"    {name:<20} "
                wrapped = textwrap.wrap(
                    help_str,
                    width=max(32, self._width - len(prefix)),
                    subsequent_indent=" " * len(prefix),
                    break_on_hyphens=False,
                )
                if wrapped:
                    parts.append(prefix + wrapped[0])
                    parts.extend(wrapped[1:])
                else:
                    parts.append(prefix)
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
        "manifest", "stv-pipeline", "reproduce", "dip-injection",
        "detection-rate", "validate", "stv-plot", "audit",
        "bad-photometry",
        "stv-events", "lc-plot", "gaia-fetch", "gaia-id-repair", "characterize", "classify", "stv-filter", "stv-tag",
        "attrition", "review", "review-refresh", "review-merge", "review-sync", "review-taxonomy", "review-maint",
        "neighbors", "spectra", "false-positive", "vsx-filter", "vsx-crossmatch", "external-lcs", "multi-survey-features", "feature-layers", "migrate", "sed-photometry", "nuclear-context",
        "vetting",
        "ltv-pipeline", "ltv-injection",
        "dev", "malcat-train",
    ]:
        command = sys.argv[1]
        remaining = sys.argv[2:]
        
        # Dispatch to appropriate module (--help will be handled by that module)
        if command == "manifest":
            _run_module_main("malca.io.manifest", remaining)
        elif command == "stv-pipeline":
            _run_module_main("malca.stv.pipeline", remaining)
        elif command == "reproduce":
            reproduce = importlib.import_module("malca.evaluation.reproduce")
            sys.argv = [sys.argv[0]] + remaining
            reproduce.main()
        elif command == "dip-injection":
            injection = importlib.import_module("malca.evaluation.dip_injection")
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
            _run_module_main("malca.evaluation.audit", remaining)
        elif command == "bad-photometry":
            _run_module_main("malca.meta_analysis.ml.bad_photometry", remaining)
        elif command == "stv-plot":
            _run_module_main("malca.stv.plot", remaining)
        elif command == "lc-plot":
            _run_module_main("malca.plotting.lightcurve_publication", remaining)
        elif command == "stv-events":
            _run_module_main("malca.stv.events", remaining)
        elif command == "gaia-fetch":
            _run_module_main("malca.catalogs.gaia_fetch", remaining)
        elif command == "gaia-id-repair":
            _run_module_main("malca.catalogs.gaia_id_repair", remaining)
        elif command == "characterize":
            _run_module_main("malca.enrichment.characterize", remaining)
        elif command == "classify":
            _run_module_main("malca.enrichment.classify", remaining)
        elif command == "stv-filter":
            _run_module_main("malca.stv.filter", remaining)
        elif command == "stv-tag":
            _run_module_main("malca.stv.tag", remaining)
        elif command == "review":
            _run_module_main("malca.review.app", remaining)
        elif command == "review-refresh":
            _run_module_main("malca.review.refresh", remaining)
        elif command == "review-merge":
            _run_module_main("malca.review.merge", remaining)
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
            _run_module_main("malca.enrichment.external_lcs", remaining)
        elif command == "multi-survey-features":
            _run_module_main("malca.enrichment.multi_survey_features", remaining)
        elif command == "feature-layers":
            _run_module_main("malca.products.feature_layers", remaining)
        elif command == "migrate":
            _run_module_main("malca.migration.cli", remaining)
        elif command == "sed-photometry":
            _run_module_main("malca.enrichment.sed_photometry", remaining)
        elif command == "nuclear-context":
            _run_module_main("malca.nuclear.cmd", remaining)
        elif command == "vetting":
            _run_module_main("malca.enrichment.vetting", remaining)
        elif command == "ltv-pipeline":
            _run_module_main("malca.ltv.pipeline", remaining)
        elif command == "dev":
            if not remaining:
                raise SystemExit("usage: malca dev {score,stats} ...")
            dev_command, dev_args = remaining[0], remaining[1:]
            if dev_command == "score":
                _run_module_main("malca.stv.score", dev_args)
            elif dev_command == "stats":
                _run_module_main("malca.core.stats", dev_args)
            else:
                raise SystemExit(f"unknown dev command: {dev_command}")
        elif command == "ltv-injection":
            _run_module_main("malca.ltv.injection", remaining)
        elif command == "malcat-train":
            _run_module_main("malcat.train", remaining)
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
    subparsers.add_parser(
        "stv-pipeline",
        description="Run STV discovery pipeline; --stage full-extended adds external LCs and multi-survey features",
    )
    subparsers.add_parser("stv-filter", description="Apply STV candidate filters")
    subparsers.add_parser("stv-tag", description="Apply STV tagging filters to candidate tables")
    subparsers.add_parser("stv-events", description="Run STV event detection directly")
    subparsers.add_parser("stv-plot", description="Plot STV light curves with events")
    subparsers.add_parser("lc-plot", description="Create a publication-quality light-curve figure")
    subparsers.add_parser("gaia-fetch", description="Download Gaia DR3 data for candidates (AIP TAP mirror)")
    subparsers.add_parser("gaia-id-repair", description="Canonicalize stale Gaia DR2 IDs in review artifacts")
    subparsers.add_parser("characterize", description="Characterize candidates with external catalogs")
    subparsers.add_parser("classify", description="Classify candidates by variability type")
    # Review
    subparsers.add_parser("review", description="Launch Dash review GUI (keyboard-driven, fast)")
    subparsers.add_parser("review-refresh", description="Refresh review DB stats from a run or bundle")
    subparsers.add_parser("review-merge", description="Merge reviewed subset DB content into a master review DB")
    subparsers.add_parser("review-sync", description="Import/export Git-trackable review bundle files")
    subparsers.add_parser("review-taxonomy", description="Migrate legacy review DBs to taxonomy schema")
    subparsers.add_parser("review-maint", description="Review DB maintenance commands")
    # LTV
    subparsers.add_parser(
        "ltv-pipeline",
        description="Run LTV workflow; --stage full-extended adds external LCs/multi-survey and --full-bundle includes LC assets",
    )
    subparsers.add_parser("ltv-injection", description="Run LTV rejection-recovery injections and plots")
    # Evaluation
    subparsers.add_parser("dip-injection", description="Run dip injection-recovery tests (skew-normal/step dips)")
    subparsers.add_parser("detection-rate", description="Measure detection rate")
    subparsers.add_parser("validate", description="Validate results against known candidates")
    subparsers.add_parser("attrition", description="Summarize pre/filter attrition")
    subparsers.add_parser("reproduce", description="Re-run detection on known objects (needs raw data)")
    subparsers.add_parser("audit", description="Audit result tables, LTV status, and baseline comparison commands")
    subparsers.add_parser("bad-photometry", description="Train/apply v1-v2 bad-photometry dropout models")
    subparsers.add_parser("false-positive", description="Run false-positive contaminant benchmark")
    # Enrichment
    subparsers.add_parser("neighbors", description="Run bulk nearest-neighbor enrichment")
    subparsers.add_parser("spectra", description="Run bulk spectra-availability enrichment")
    subparsers.add_parser("vsx-filter", description="Build cleaned ASAS-SN index and filtered VSX catalog")
    subparsers.add_parser("vsx-crossmatch", description="Crossmatch ASAS-SN catalog with VSX catalog")
    subparsers.add_parser("external-lcs", description="Fetch external light curves for candidate tables")
    subparsers.add_parser("multi-survey-features", description="Compute event-relative multi-survey features")
    subparsers.add_parser("feature-layers", description="Materialize lc/external/derived feature layers for candidate tables")
    subparsers.add_parser("migrate", description="Mirror-copy outputs into the three-layer product structure")
    subparsers.add_parser("sed-photometry", description="Fetch and normalize SED photometry for candidate tables")
    subparsers.add_parser("nuclear-context", description="Run nuclear context enrichment and AGN/TDE/CLAGN scores")
    # Other
    subparsers.add_parser("vetting", description="Run post-review vetting (SIMBAD, Gaia, ASAS-SN, ZTF, TNS, eROSITA, ...)")
    subparsers.add_parser("dev", description="Developer diagnostics (score, stats)")
    subparsers.add_parser("malcat-train", description="Train the MALCAT light-curve Transformer")

    if len(sys.argv) == 1:
        parser.print_help()
    else:
        parser.parse_args()
    return 0


if __name__ == "__main__":
    sys.exit(main())
