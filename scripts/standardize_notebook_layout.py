#!/usr/bin/env python3
"""One-shot migration: replace ad-hoc tight_layout/bbox_inches patterns with finalize/save helpers."""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

LAYOUT_IMPORT = "finalize_publication_figure"
SAVE_IMPORT = "save_publication_figure"
MODULE = "malca.plotting.lightcurve_publication"


def _transform_cell(src: str) -> str:
    """Apply layout transformations to a single cell source string."""
    original = src

    # Replace fig.tight_layout() / plt.tight_layout() with finalize_publication_figure(fig)
    src = re.sub(
        r"(\s*)fig\.tight_layout\([^)]*\)",
        r"\1finalize_publication_figure(fig)",
        src,
    )
    src = re.sub(
        r"(\s*)plt\.tight_layout\([^)]*\)",
        r"\1finalize_publication_figure(plt.gcf())",
        src,
    )

    # Remove constrained_layout=True from subplots() calls
    src = re.sub(r",\s*constrained_layout\s*=\s*True", "", src)
    src = re.sub(r"constrained_layout\s*=\s*True\s*,\s*", "", src)

    # Strip bbox_inches="tight" / bbox_inches='tight' from savefig calls
    src = re.sub(r""",\s*bbox_inches\s*=\s*["']tight["']""", "", src)
    src = re.sub(r"""bbox_inches\s*=\s*["']tight["']\s*,\s*""", "", src)

    if src != original:
        src = _ensure_layout_imports(src)

    return src


def _ensure_layout_imports(src: str) -> str:
    """Add finalize/save imports if not already present."""
    needed: set[str] = set()
    if LAYOUT_IMPORT in src and f"import {LAYOUT_IMPORT}" not in src and f"{LAYOUT_IMPORT}" not in _existing_imports(src):
        needed.add(LAYOUT_IMPORT)
    if SAVE_IMPORT in src and f"import {SAVE_IMPORT}" not in src and f"{SAVE_IMPORT}" not in _existing_imports(src):
        needed.add(SAVE_IMPORT)
    if not needed:
        return src

    # Try to extend existing import from malca.plotting.lightcurve_publication
    pattern = r"(from malca\.lightcurve_publication import\s*\()(.*?)(\))"
    match = re.search(pattern, src, re.DOTALL)
    if match:
        existing_block = match.group(2)
        for symbol in sorted(needed):
            if symbol not in existing_block:
                existing_block = existing_block.rstrip().rstrip(",") + f",\n    {symbol},\n"
        src = src[: match.start()] + match.group(1) + existing_block + match.group(3) + src[match.end() :]
        return src

    # Try single-line import
    match_single = re.search(r"(from malca\.lightcurve_publication import )(.+)", src)
    if match_single:
        existing_names = match_single.group(2).strip()
        for symbol in sorted(needed):
            if symbol not in existing_names:
                existing_names += f", {symbol}"
        src = src[: match_single.start()] + match_single.group(1) + existing_names + src[match_single.end() :]
        return src

    # No existing import — add one at the top (after any comments/magic)
    symbols = ", ".join(sorted(needed))
    import_line = f"from {MODULE} import {symbols}\n"
    lines = src.splitlines(keepends=True)
    idx = 0
    while idx < len(lines) and (lines[idx].startswith("#") or lines[idx].startswith("%") or not lines[idx].strip()):
        idx += 1
    return "".join(lines[:idx]) + import_line + "".join(lines[idx:])


def _existing_imports(src: str) -> str:
    """Return the text of any existing malca.plotting.lightcurve_publication import block."""
    match = re.search(r"from malca\.lightcurve_publication import.*?(?:\n(?!\s)|$)", src, re.DOTALL)
    return match.group(0) if match else ""


def migrate_notebook(path: Path) -> bool:
    data = json.loads(path.read_text(encoding="utf-8"))
    changed = False
    for cell in data.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        new_src = _transform_cell(src)
        if new_src != src:
            cell["source"] = [line if line.endswith("\n") else line + "\n" for line in new_src.splitlines()]
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
            if ".ipynb_checkpoints" in str(path):
                continue
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
