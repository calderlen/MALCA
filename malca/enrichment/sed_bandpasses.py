"""CLI for synchronizing exact SED filter response curves."""

from __future__ import annotations

import argparse
from pathlib import Path

from malca.enrichment.synthetic_photometry import (
    CACHE_FORMAT_VERSION,
    SED_BANDPASS_CACHE_DIR,
    build_response_map,
    fetch_filter_response,
    load_cached_filter_response,
    save_filter_response,
)
from malca.review.sed import SED_BANDPASSES


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="malca sed-bandpasses",
        description="Download and validate SVO throughput curves used by MALCA synthetic photometry.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=SED_BANDPASS_CACHE_DIR,
        help=f"Response cache directory (default: {SED_BANDPASS_CACHE_DIR})",
    )
    parser.add_argument(
        "--filters",
        default="all",
        help="Comma-separated SVO filter IDs, or 'all' for every registered SED band.",
    )
    parser.add_argument("--force", action="store_true", help="Refresh responses even if cached.")
    return parser


def run(args: argparse.Namespace) -> int:
    requested = str(args.filters or "all").strip()
    if requested.lower() == "all":
        registrations = [
            (str(bandpass.svo_filter_id or ""), str(bandpass.mag_system or ""))
            for bandpass in SED_BANDPASSES.values()
            if bandpass.svo_filter_id
        ]
        pairs = sorted(set(registrations))
    else:
        filter_ids = {item.strip() for item in requested.split(",") if item.strip()}
        systems_by_id: dict[str, str] = {
            str(bandpass.svo_filter_id): str(bandpass.mag_system or "")
            for bandpass in SED_BANDPASSES.values()
            if bandpass.svo_filter_id
        }
        pairs = [(filter_id, systems_by_id.get(filter_id, "")) for filter_id in sorted(filter_ids)]
        registrations = list(pairs)

    loader = None
    if bool(args.force):
        loader = lambda filter_id, mag_system: fetch_filter_response(
            filter_id,
            mag_system,
            cache_dir=args.cache_dir,
            force=True,
        )
    responses, failures = build_response_map(
        pairs,
        cache_dir=args.cache_dir,
        allow_download=True,
        response_loader=loader,
        progress_callback=lambda message: print(message, flush=True),
    )
    for (filter_id, mag_system), response in responses.items():
        # Rewrite legacy calibration-keyed cache entries into the v3
        # filter-only identity.  Repeated systems merge metadata into one
        # physical throughput artifact.
        response_to_save = response
        if mag_system.strip().casefold() == "jy":
            legacy_vega = load_cached_filter_response(filter_id, "Vega", args.cache_dir)
            if legacy_vega is not None and legacy_vega.zero_point_jy is not None:
                response_to_save = legacy_vega
        save_filter_response(
            response_to_save,
            args.cache_dir,
            requested_mag_system=mag_system,
            refresh_provenance=(
                "legacy_cache_migration"
                if response_to_save.cache_format_version < CACHE_FORMAT_VERSION
                else response_to_save.refresh_provenance or "cache_metadata_refresh"
            ),
        )
    print(f"Cached {len(responses)}/{len(pairs)} requested response registrations in {args.cache_dir}")
    for (filter_id, mag_system), message in sorted(failures.items()):
        print(f"FAILED {filter_id} [{mag_system}]: {message}")

    # Prefetch success is not enough: validate that every requested
    # registration, including native-Jy aliases, can be served with downloads
    # disabled.  This catches cache-key and incomplete-write regressions before
    # a large SED backfill starts.
    offline_responses, offline_failures = build_response_map(
        registrations,
        cache_dir=args.cache_dir,
        allow_download=False,
    )
    unique_curves = {response.response_hash for response in offline_responses.values()}
    provenance_modes = sorted({
        response.refresh_provenance or "unspecified"
        for response in offline_responses.values()
    })
    print(
        f"Offline validation: {len(offline_responses)}/{len(pairs)} registrations, "
        f"{len(unique_curves)} unique throughput curves, "
        f"{len(registrations)} catalog band definitions; "
        f"provenance={','.join(provenance_modes)}"
    )
    for (filter_id, mag_system), message in sorted(offline_failures.items()):
        print(f"OFFLINE FAILED {filter_id} [{mag_system}]: {message}")
    return 1 if failures or offline_failures else 0


def main() -> None:
    raise SystemExit(run(build_arg_parser().parse_args()))


if __name__ == "__main__":
    main()
