#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections.abc import Iterable
import json
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "malca"
    / "review"
    / "data"
    / "classification_labels"
    / "catalog_class_labels.json"
)
SIMBAD_OTYPE_NODES_URL = "https://simbad.cds.unistra.fr/guide/otypes/json/otype_nodes.json"


def _read_bundle(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    data.setdefault("maps", {})
    return data


def _write_bundle(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _records(value: object) -> list[dict[str, Any]]:
    if value is None:
        return []
    if hasattr(value, "to_dict"):
        try:
            records = value.to_dict(orient="records")
            if isinstance(records, list):
                return [dict(row) for row in records if isinstance(row, dict)]
        except TypeError:
            pass
    if isinstance(value, dict):
        for key in ("items", "results", "data", "classifiers", "classes"):
            nested = value.get(key)
            if isinstance(nested, list):
                return [dict(row) for row in nested if isinstance(row, dict)]
        return [dict(value)]
    if isinstance(value, list):
        return [dict(row) for row in value if isinstance(row, dict)]
    return []


def _first_text(row: dict[str, Any], keys: Iterable[str]) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def refresh_gaia() -> dict[str, str]:
    from astroquery.gaia import Gaia

    job = Gaia.launch_job(
        """
        SELECT class_name, class_description
        FROM gaiadr3.vari_classifier_class_definition
        """
    )
    table = job.get_results()
    out: dict[str, str] = {}
    for row in table:
        name = str(row["class_name"]).strip()
        description = str(row["class_description"]).strip()
        if name and description:
            out[name] = description
    return out


def refresh_alerce() -> tuple[dict[str, str], dict[str, str]]:
    from alerce.core import Alerce

    client = Alerce()
    class_map: dict[str, str] = {}
    stamp_map: dict[str, str] = {}
    for classifier in _records(client.query_classifiers()):
        name = _first_text(classifier, ("classifier_name", "name", "classifier"))
        version = _first_text(classifier, ("classifier_version", "version"))
        if not name or not version:
            continue
        target = stamp_map if "stamp" in name.lower() else class_map
        for row in _records(client.query_classes(name, version)):
            class_name = _first_text(row, ("class_name", "name", "class"))
            description = _first_text(
                row,
                ("class_description", "description", "label", "class_name", "name", "class"),
            )
            if class_name and description:
                target[class_name] = description
    return class_map, stamp_map


def _walk_simbad_nodes(node: object) -> Iterable[dict[str, Any]]:
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _walk_simbad_nodes(value)
    elif isinstance(node, list):
        for value in node:
            yield from _walk_simbad_nodes(value)


def refresh_simbad() -> dict[str, str]:
    import requests

    response = requests.get(SIMBAD_OTYPE_NODES_URL, timeout=30)
    response.raise_for_status()
    payload = response.json()
    out: dict[str, str] = {}
    for row in _walk_simbad_nodes(payload):
        code = _first_text(row, ("code", "otype", "id", "name", "label"))
        description = _first_text(
            row,
            ("description", "text", "long_description", "longname", "title"),
        )
        if code and description and code != description:
            out[code] = description
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Refresh bundled MALCA review classification label maps."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--source",
        action="append",
        choices=("gaia", "alerce", "simbad"),
        help="Source to refresh. May be passed more than once. Defaults to all refreshable sources.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    sources = set(args.source or ("gaia", "alerce", "simbad"))
    data = _read_bundle(args.output)
    maps = data.setdefault("maps", {})

    if "gaia" in sources:
        maps["gaia"] = refresh_gaia()
    if "alerce" in sources:
        alerce_map, alerce_stamp_map = refresh_alerce()
        if alerce_map:
            maps["alerce"] = alerce_map
        if alerce_stamp_map:
            maps["alerce_stamp"] = alerce_stamp_map
    if "simbad" in sources:
        refreshed = refresh_simbad()
        if refreshed:
            maps["simbad"] = refreshed

    if args.dry_run:
        print(json.dumps({source: len(maps.get(source, {})) for source in sorted(maps)}, indent=2))
        return 0

    _write_bundle(args.output, data)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

