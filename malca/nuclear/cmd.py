from __future__ import annotations

import argparse
from pathlib import Path

from malca.nuclear.context import NuclearContextConfig, run_nuclear_context
from malca.table_io import read_parquet_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Run nuclear context enrichment and AGN/TDE/CLAGN scoring")
    parser.add_argument("--input", type=Path, required=True, help="Input candidate Parquet table")
    parser.add_argument("--run-dir", type=Path, default=Path("output") / "runs" / "nuclear_context")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=250)
    parser.add_argument("--atlas-token", default=None)
    parser.add_argument("--tns-api-key", default=None)
    parser.add_argument("--clagn-catalog", action="append", default=[], help="Known-CLAGN catalog as source=path or path")
    parser.add_argument("--no-remote", action="store_true", help="Skip remote/catalog-query stages and compute local scores only")
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--show-progress", action="store_true")
    args = parser.parse_args()

    clagn_paths: dict[str, Path] = {}
    for item in args.clagn_catalog:
        if "=" in item:
            source, path = item.split("=", 1)
            clagn_paths[source] = Path(path)
        else:
            path = Path(item)
            clagn_paths[path.stem] = path

    remote = not args.no_remote
    config = NuclearContextConfig(
        run_dir=args.run_dir,
        cache_dir=args.cache_dir,
        checkpoint_dir=args.checkpoint_dir,
        workers=args.workers,
        chunk_size=args.chunk_size,
        atlas_token=args.atlas_token,
        tns_api_key=args.tns_api_key,
        clagn_catalog_paths=clagn_paths or None,
        refresh_cache=args.refresh_cache,
        show_progress=args.show_progress,
        run_characterize=remote,
        run_ltv_crossmatch=remote,
        run_vetting=remote,
        run_external_lcs=remote,
        run_spectra=remote,
        run_host=remote,
        run_radio=remote,
        run_swift=remote,
    )
    df = read_parquet_table(args.input)
    out = run_nuclear_context(df, config)
    print(f"Nuclear context written to {config.results_dir / 'nuclear_context.parquet'} ({len(out)} rows)")


if __name__ == "__main__":
    main()
