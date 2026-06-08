from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _source(rel_path: str) -> str:
    return (ROOT / rel_path).read_text(encoding="utf-8")


def test_old_product_schema_helpers_are_migration_only() -> None:
    forbidden = (
        "flatten_layer_first_frame",
        "flatten_feature_layers",
        "append_feature_layers",
        "schema_migration",
        "flatten_tree",
        "--flatten",
    )
    allowed_prefixes = {
        "malca/migration/",
        "tests/test_three_layer_compatibility.py",
    }

    for path in sorted((ROOT / "malca").rglob("*.py")) + sorted((ROOT / "scripts").rglob("*.py")):
        rel = path.relative_to(ROOT).as_posix()
        if any(rel.startswith(prefix) for prefix in allowed_prefixes):
            continue
        text = path.read_text(encoding="utf-8")
        for snippet in forbidden:
            assert snippet not in text, f"{rel} still references old schema helper {snippet!r}"


def test_feature_table_api_has_no_flat_compatibility_switches() -> None:
    text = _source("malca/table_io.py")
    assert "flatten:" not in text
    assert "layer_first:" not in text
    assert "not layer-first" in text
