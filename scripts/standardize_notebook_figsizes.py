#!/usr/bin/env python3
"""One-shot migration: replace legacy notebook figsize literals with publication constants."""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

LITERAL_REPLACEMENTS: list[tuple[str, str]] = [
    ("figsize=(10, 8)", "figsize=FIG_SINGLE_COL_HEATMAP"),
    ("figsize=(10,8)", "figsize=FIG_SINGLE_COL_HEATMAP"),
    ("figsize=(14, 8)", "figsize=FIG_TWO_COL_STANDARD"),
    ("figsize=(14,8)", "figsize=FIG_TWO_COL_STANDARD"),
    ("figsize=(12, 8)", "figsize=FIG_TWO_COL_STANDARD"),
    ("figsize=(12,8)", "figsize=FIG_TWO_COL_STANDARD"),
    ("figsize=(10, 6)", "figsize=FIG_SINGLE_COL_LC_WIDE"),
    ("figsize=(10,6)", "figsize=FIG_SINGLE_COL_LC_WIDE"),
    ("figsize=(9, 7)", "figsize=FIG_SINGLE_COL_HEATMAP"),
    ("figsize=(9,7)", "figsize=FIG_SINGLE_COL_HEATMAP"),
    ("figsize=(8, 6)", "figsize=FIG_SINGLE_COL_MEDIUM"),
    ("figsize=(8,6)", "figsize=FIG_SINGLE_COL_MEDIUM"),
    ("figsize=(7, 5)", "figsize=FIG_SINGLE_COL_PORTRAIT"),
    ("figsize=(7,5)", "figsize=FIG_SINGLE_COL_PORTRAIT"),
    ("figsize=(6, 5)", "figsize=FIG_SINGLE_COL_COMPACT"),
    ("figsize=(6,5)", "figsize=FIG_SINGLE_COL_COMPACT"),
    ("figsize=(13, 6)", "figsize=FIG_TWO_COL_LC_WIDE"),
    ("figsize=(13,6)", "figsize=FIG_TWO_COL_LC_WIDE"),
    ("figsize=(18, 6)", "figsize=FIG_TWO_COL_TRIPLE"),
    ("figsize=(18,6)", "figsize=FIG_TWO_COL_TRIPLE"),
    ("figsize=(16, 7)", "figsize=figsize_from_legacy(16, 7)"),
    ("figsize=(16,7)", "figsize=figsize_from_legacy(16, 7)"),
    ("figsize=(12, 4.5)", "figsize=figsize_from_legacy(12, 4.5)"),
    ("figsize=(12,4.5)", "figsize=figsize_from_legacy(12, 4.5)"),
    ("figsize=(12, 6)", "figsize=FIG_TWO_COL_LC_WIDE"),
    ("figsize=(12,6)", "figsize=FIG_TWO_COL_LC_WIDE"),
    ("figsize=(9, 10)", "figsize=figsize_from_legacy(9, 10)"),
    ("figsize=(9,10)", "figsize=figsize_from_legacy(9, 10)"),
    ("figsize=(7, 4)", "figsize=FIG_LC_SINGLE_COL"),
    ("figsize=(7,4)", "figsize=FIG_LC_SINGLE_COL"),
    ("figsize=(12, 4.5)", "figsize=figsize_from_legacy(12, 4.5)"),
]

REGEX_REPLACEMENTS: list[tuple[str, str]] = [
    (r"figsize=\(10, 3\.4 \* (\w+)\)", r"figsize=figsize_two_col_grid(1, \1, row_height=3.4 * FIG_TWO_COL_WIDTH / 10)"),
    (r"figsize=\(15, 6 \* (\w+)\)", r"figsize=figsize_two_col_grid(1, \1, row_height=6.0 * FIG_TWO_COL_WIDTH / 15)"),
    (r"figsize=\(14, 3 \* (\w+)\)", r"figsize=figsize_two_col_grid(2, \1, row_height=3.0 * FIG_TWO_COL_WIDTH / 14)"),
    (r"figsize=\(3 \* (\w+), 2\.5 \* (\w+)\)", r"figsize=figsize_feature_grid(\1, \2)"),
    (r"figsize=\(5\.3 \* (\w+), 3\.8 \* (\w+)\)", r"figsize=figsize_feature_grid(\1, \2, row_height=3.8)"),
    (r"figsize=\(6\.1 \* (\w+), 4\.25 \* (\w+)\)", r"figsize=figsize_feature_grid(\1, \2, row_height=4.25)"),
    (r"figsize=\(12, max\(8, 0\.28 \* len\((\w+)\)\)\)\)", r"figsize=figsize_heatmap_two_col(len(\1), row_height=0.28)"),
    (r"plt\.figure\(figsize=\(12, 10\)\)", "plt.figure(figsize=figsize_from_legacy(12, 10))"),
    (r"figsize=\(15, 10\)", "figsize=figsize_from_legacy(15, 10)"),
    (r"figsize=\(15, 5\)", "figsize=figsize_from_legacy(15, 5)"),
    (r"figsize=\(17, 5\)", "figsize=figsize_from_legacy(17, 5)"),
    (r"figsize=\(14, 5\)", "figsize=figsize_from_legacy(14, 5)"),
    (r"figsize=\(11, 4\)", "figsize=figsize_from_legacy(11, 4)"),
    (r"figsize=\(10, 5\)", "figsize=figsize_from_legacy(10, 5)"),
    (r"figsize=\(10, 4\.5\)", "figsize=figsize_from_legacy(10, 4.5)"),
    (r"figsize=\(10, 7\)", "figsize=figsize_from_legacy(10, 7)"),
    (r"figsize=\(5\.5, 5\.5\)", "figsize=FIG_SINGLE_COL_SQUARE"),
    (r"figsize=\(6, 4\)", "figsize=FIG_SINGLE_COL_MEDIUM"),
    (r"figsize=\(6, 3\.5\)", "figsize=figsize_from_legacy(6, 3.5)"),
    (r"figsize=\(5, 3\.5\)", "figsize=figsize_from_legacy(5, 3.5)"),
    (r"figsize=\(4\.8, 2\.8\)", "figsize=FIG_SINGLE_COL_LC_WIDE"),
    (r"figsize=\(5\.5, 3\.2\)", "figsize=figsize_from_legacy(5.5, 3.2)"),
    (r"figsize=\(6\.5, 5\.5\)", "figsize=figsize_from_legacy(6.5, 5.5)"),
    (r"figsize=\(7\.5, 4\)", "figsize=FIG_LC_SINGLE_COL"),
    (r"figsize=\(9, 4\.5\)", "figsize=figsize_from_legacy(9, 4.5)"),
    (r"figsize=\(8, max\(2\.5, 0\.2 \* len\((\w+)\)\)\)", r"figsize=figsize_heatmap_two_col(len(\1), row_height=0.2)"),
    (r"figsize=\(8, 4\)", "figsize=FIG_SINGLE_COL_LC_WIDE"),
    (r"figsize=\(6, 4\)", "figsize=FIG_SINGLE_COL_MEDIUM"),
    (r"figsize=\(8, 8\)", "figsize=FIG_SINGLE_COL_SQUARE"),
    (r"figsize=\(12, max\(8, 0\.28 \* len\((\w+)\)\)\)", r"figsize=figsize_heatmap_two_col(len(\1), row_height=0.28)"),
    (r"figsize=\(2, 2, figsize=\(14, 10\)\)", "figsize=figsize_from_legacy(14, 10)"),
    (r"figsize=\(2, 2, figsize=\(12, 10\)\)", "figsize=figsize_from_legacy(12, 10)"),
    (r"figsize=\(2, 3, figsize=\(16, 10\)\)", "figsize=figsize_from_legacy(16, 10)"),
    (r"figsize=\(1, 2, figsize=\(16, 6\)\)", "figsize=figsize_from_legacy(16, 6)"),
    (r"figsize=\(1, 2, figsize=\(12, 5\)\)", "figsize=figsize_from_legacy(12, 5)"),
    (r"figsize=\(2, 2, figsize=\(14, 10\)\)", "figsize=figsize_from_legacy(14, 10)"),
    (r"figsize=\(2, 2, figsize=\(12, 10\)\)", "figsize=figsize_from_legacy(12, 10)"),
    (r"figsize=\(2, len\((\w+)\), figsize=\(4\.2 \* len\(\1\), 8\)\)", r"figsize=figsize_two_col_grid(len(\1), 2, row_height=4.0)"),
    (r"figsize=\((\w+), (\w+), figsize=\(18, 4\.8 \* \2\)\)", r"figsize=figsize_two_col_grid(\1, \2, row_height=4.8 * FIG_TWO_COL_WIDTH / 18)"),
    (r"figsize=\((\w+), (\w+), figsize=\(15, 4\*\2\)\)", r"figsize=figsize_two_col_grid(\1, \2, row_height=4.0 * FIG_TWO_COL_WIDTH / 15)"),
    (r"plt\.figure\(figsize=\(14, 12\)\)", "plt.figure(figsize=figsize_from_legacy(14, 12))"),
    (r"figsize=\(2, 2, figsize=\(14, 10\)\)", "figsize=figsize_from_legacy(14, 10)"),
    (r"figsize=\(2, 3, figsize=\(18, 12\)\)", "figsize=figsize_from_legacy(18, 12)"),
    (r"figsize=\(12, 10\)", "figsize=figsize_from_legacy(12, 10)"),
    (r"figsize=\(14, 10\)", "figsize=figsize_from_legacy(14, 10)"),
    (r"figsize=\(12, 5\)", "figsize=figsize_from_legacy(12, 5)"),
    (r"figsize=\(16, 10\)", "figsize=figsize_from_legacy(16, 10)"),
    (r"figsize=\(16, 6\)", "figsize=figsize_from_legacy(16, 6)"),
    (r"figsize=\(2, 2, figsize=\(12, 10\)\)", "figsize=figsize_from_legacy(12, 10)"),
]

REPLACEMENTS = LITERAL_REPLACEMENTS

SYMBOL_TO_MODULE = {
    "FIG_SINGLE_COL_SQUARE": "malca.lightcurve_publication",
    "FIG_SINGLE_COL_HEATMAP": "malca.lightcurve_publication",
    "FIG_SINGLE_COL_LC_WIDE": "malca.lightcurve_publication",
    "FIG_SINGLE_COL_MEDIUM": "malca.lightcurve_publication",
    "FIG_SINGLE_COL_PORTRAIT": "malca.lightcurve_publication",
    "FIG_SINGLE_COL_COMPACT": "malca.lightcurve_publication",
    "FIG_TWO_COL_STANDARD": "malca.lightcurve_publication",
    "FIG_TWO_COL_LC_WIDE": "malca.lightcurve_publication",
    "FIG_TWO_COL_TRIPLE": "malca.lightcurve_publication",
    "FIG_LC_SINGLE_COL": "malca.lightcurve_publication",
    "FIG_ROC_PR_TWO_COL": "malca.lightcurve_publication",
    "FIG_TWO_COL_WIDTH": "malca.lightcurve_publication",
    "figsize_from_legacy": "malca.lightcurve_publication",
    "figsize_feature_grid": "malca.lightcurve_publication",
    "figsize_heatmap_two_col": "malca.lightcurve_publication",
    "figsize_two_col_grid": "malca.lightcurve_publication",
}


def _needed_symbols(text: str) -> set[str]:
    needed: set[str] = set()
    for symbol in SYMBOL_TO_MODULE:
        if re.search(rf"\b{re.escape(symbol)}\b", text):
            needed.add(symbol)
    return needed


def _ensure_imports(source: str, needed: set[str]) -> str:
    if not needed:
        return source
    if "from malca.lightcurve_publication import" in source:
        return source
    symbols = ",\n    ".join(sorted(needed))
    import_line = f"from malca.lightcurve_publication import (\n    {symbols},\n)\n"
    if source.startswith("#"):
        lines = source.splitlines(keepends=True)
        idx = 0
        while idx < len(lines) and (lines[idx].startswith("#") or not lines[idx].strip()):
            idx += 1
        return "".join(lines[:idx]) + import_line + "".join(lines[idx:])
    return import_line + source


def migrate_notebook(path: Path) -> bool:
    data = json.loads(path.read_text(encoding="utf-8"))
    changed = False
    for cell in data.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        original = src
        for old, new in REPLACEMENTS:
            src = src.replace(old, new)
        for pattern, repl in REGEX_REPLACEMENTS:
            src = re.sub(pattern, repl, src)
        needed = _needed_symbols(src)
        if needed:
            src = _ensure_imports(src, needed)
        if src != original:
            cell["source"] = [line if line.endswith("\n") else line + "\n" for line in src.splitlines()]
            if cell["source"] and not cell["source"][-1].endswith("\n"):
                cell["source"][-1] += "\n"
            changed = True
    if changed:
        path.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return changed


def main() -> None:
    roots = [REPO / "malca" / "notebooks", REPO / "malca" / "nuclear", REPO / "malca" / "clagn"]
    updated: list[str] = []
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.ipynb")):
            try:
                if migrate_notebook(path):
                    updated.append(str(path.relative_to(REPO)))
            except Exception as exc:
                print(f"Skipped {path.relative_to(REPO)}: {exc}")
    print(f"Updated {len(updated)} notebooks:")
    for item in updated:
        print(f"  - {item}")


if __name__ == "__main__":
    main()
