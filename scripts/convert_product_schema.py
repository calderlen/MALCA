#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import shutil

from malca.schema_migration import (
    PreferPolicy,
    _iter_product_paths,
    convert_product_file,
    convert_run_tree,
    scan_product,
    write_migration_report,
)


def _backup_product(product: Path, input_root: Path, backup_dir: Path) -> Path:
    if input_root.is_file():
        rel = Path(product.name)
    else:
        try:
            rel = product.relative_to(input_root)
        except ValueError:
            rel = Path(product.name)
    dest = backup_dir / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    if product.is_dir():
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(product, dest)
    else:
        shutil.copy2(product, dest)
    return dest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert legacy MALCA STV/LTV parquet products to the canonical product schema.",
    )
    parser.add_argument("input", type=Path, help="Parquet file, chunked parquet dataset, results directory, or run directory.")
    parser.add_argument("--timescale", choices=["stv", "ltv"], default=None, help="Override timescale detection.")
    parser.add_argument("--write", action="store_true", help="Write converted products. Without this flag, run a dry scan.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Write converted products under this directory.")
    parser.add_argument("--in-place", action="store_true", help="Rewrite products in place. Requires --backup-dir.")
    parser.add_argument("--backup-dir", type=Path, default=None, help="Backup destination for --in-place migration.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs in --output-dir.")
    parser.add_argument(
        "--prefer",
        choices=["fail", "canonical", "legacy"],
        default="fail",
        help="Conflict policy when legacy and canonical columns both exist and differ.",
    )
    parser.add_argument("--report", type=Path, default=None, help="Path for schema_migration_report.json.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_root = args.input.expanduser()

    if args.in_place and args.output_dir is not None:
        raise SystemExit("--in-place and --output-dir are mutually exclusive")
    if args.in_place and args.backup_dir is None:
        raise SystemExit("--in-place requires --backup-dir")
    if args.write and not args.in_place and args.output_dir is None:
        raise SystemExit("--write requires either --output-dir or --in-place")

    if not args.write:
        scans = [scan_product(path, timescale=args.timescale) for path in _iter_product_paths(input_root)]
        report_path = args.report or Path("schema_migration_report.json")
        write_migration_report(scans, report_path)
        for scan in scans:
            status = "ERROR" if scan.error else ("convert" if scan.needs_conversion else "ok")
            print(f"{status}: {scan.path}")
        print(f"Wrote report: {report_path}")
        return 1 if any(scan.error for scan in scans) else 0

    prefer: PreferPolicy = args.prefer
    if args.in_place:
        backup_dir = args.backup_dir.expanduser()
        results = []
        for product in _iter_product_paths(input_root):
            _backup_product(product, input_root, backup_dir)
            results.append(
                convert_product_file(
                    product,
                    product,
                    timescale=args.timescale,
                    overwrite=True,
                    prefer=prefer,
                )
            )
    else:
        results = convert_run_tree(
            input_root,
            args.output_dir.expanduser(),
            timescale=args.timescale,
            overwrite=bool(args.overwrite),
            prefer=prefer,
        )

    report_path = args.report
    if report_path is None:
        if args.in_place:
            report_path = input_root / "schema_migration_report.json" if input_root.is_dir() else input_root.with_name("schema_migration_report.json")
        else:
            report_path = args.output_dir.expanduser() / "schema_migration_report.json"
    write_migration_report(results, report_path)

    for result in results:
        status = "ERROR" if result.error else "wrote"
        target = result.output_path or ""
        print(f"{status}: {result.input_path} -> {target}")
        if result.error:
            print(f"  {result.error}")

    print(f"Wrote report: {report_path}")
    return 1 if any(result.error for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
