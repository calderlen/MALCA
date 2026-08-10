"""Download version-pinned open-cluster catalogues for MALCA.

UCC files are pinned to a specific Zenodo record and verified against the
publisher-provided MD5 checksums.  The static CDS Hunt & Reffert catalogue is
downloaded from its catalogue identifier and recorded with SHA-256 hashes.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
from urllib.request import Request, urlopen


UCC_RELEASE = "260615"
UCC_RECORD = "20705026"
HR24_CATALOG_ID = "J/A+A/686/A42"


@dataclass(frozen=True)
class DownloadSpec:
    catalog: str
    filename: str
    url: str
    md5: str | None = None


UCC_FILES = (
    DownloadSpec(
        "ucc",
        "README.txt",
        f"https://zenodo.org/records/{UCC_RECORD}/files/README.txt?download=1",
        "2b7ee76fb08e7f21976ee03f81ad4b83",
    ),
    DownloadSpec(
        "ucc",
        "UCC_cat.csv",
        f"https://zenodo.org/records/{UCC_RECORD}/files/UCC_cat.csv?download=1",
        "b09c51f12865df0233df8fd90e4a8443",
    ),
    DownloadSpec(
        "ucc",
        "UCC_members.parquet",
        f"https://zenodo.org/records/{UCC_RECORD}/files/UCC_members.parquet?download=1",
        "39d5affdf141bd4f2dbad6ae00435e15",
    ),
)

_HR24_BASE = "https://cdsarc.cds.unistra.fr/ftp/J/A+A/686/A42"
HR24_FILES = (
    DownloadSpec("hr24", "ReadMe", f"{_HR24_BASE}/ReadMe"),
    DownloadSpec("hr24", "clusters.dat.gz", f"{_HR24_BASE}/clusters.dat.gz"),
    DownloadSpec("hr24", "members.dat.gz", f"{_HR24_BASE}/members.dat.gz"),
)


def _hashes(path: Path) -> tuple[str, str]:
    md5 = hashlib.md5(usedforsecurity=False)
    sha256 = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            md5.update(chunk)
            sha256.update(chunk)
    return md5.hexdigest(), sha256.hexdigest()


def _download(spec: DownloadSpec, target: Path, *, force: bool) -> dict[str, object]:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not force:
        md5, sha256 = _hashes(target)
        if spec.md5 is not None and md5 != spec.md5:
            raise ValueError(
                f"Existing {target} has MD5 {md5}, expected {spec.md5}; use --force to replace it"
            )
        return {
            "filename": target.name,
            "path": str(target.resolve()),
            "url": spec.url,
            "size_bytes": int(target.stat().st_size),
            "md5": md5,
            "sha256": sha256,
            "downloaded": False,
        }

    request = Request(spec.url, headers={"User-Agent": "MALCA-open-cluster-fetch/1"})
    with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".part", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            with urlopen(request, timeout=120) as response:
                while chunk := response.read(1024 * 1024):
                    handle.write(chunk)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    md5, sha256 = _hashes(temporary)
    if spec.md5 is not None and md5 != spec.md5:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"Downloaded {spec.filename} has MD5 {md5}, expected {spec.md5}")
    temporary.replace(target)
    return {
        "filename": target.name,
        "path": str(target.resolve()),
        "url": spec.url,
        "size_bytes": int(target.stat().st_size),
        "md5": md5,
        "sha256": sha256,
        "downloaded": True,
    }


def download_catalogues(
    output_root: Path,
    *,
    catalog: str = "all",
    force: bool = False,
) -> dict[str, object]:
    """Download selected catalogues and return a machine-readable manifest."""
    selected: list[tuple[DownloadSpec, Path]] = []
    if catalog in {"all", "ucc"}:
        selected.extend((spec, output_root / "ucc" / UCC_RELEASE) for spec in UCC_FILES)
    if catalog in {"all", "hr24"}:
        selected.extend((spec, output_root / "hr24") for spec in HR24_FILES)
    if not selected:
        raise ValueError(f"Unknown catalogue selection: {catalog}")

    records = [_download(spec, directory / spec.filename, force=force) for spec, directory in selected]
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "ucc_release": UCC_RELEASE,
        "ucc_zenodo_record": UCC_RECORD,
        "hr24_catalog_id": HR24_CATALOG_ID,
        "files": records,
    }
    manifest_path = output_root / "download_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=manifest_path.parent, suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(manifest_path)
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Download pinned UCC and Hunt-Reffert open-cluster catalogues."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/external/open_clusters"),
        help="Catalogue root (default: data/external/open_clusters)",
    )
    parser.add_argument("--catalog", choices=("all", "ucc", "hr24"), default="all")
    parser.add_argument("--force", action="store_true", help="Replace existing files")
    args = parser.parse_args(argv)
    manifest = download_catalogues(
        args.output_root.expanduser(),
        catalog=args.catalog,
        force=bool(args.force),
    )
    downloaded = sum(bool(record["downloaded"]) for record in manifest["files"])
    print(f"Open-cluster catalogues ready: {len(manifest['files'])} files ({downloaded} downloaded)")
    print(f"Manifest: {(args.output_root.expanduser() / 'download_manifest.json').resolve()}")


if __name__ == "__main__":
    main()
