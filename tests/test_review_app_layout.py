from __future__ import annotations

from malca.review.app import app


def _component_ids_in_order(node: object) -> list[object]:
    ids: list[object] = []

    def walk(item: object) -> None:
        if isinstance(item, (list, tuple)):
            for child in item:
                walk(child)
            return
        if item is None or isinstance(item, (str, int, float, bool)):
            return
        cid = getattr(item, "id", None)
        if cid is not None:
            ids.append(cid)
        walk(getattr(item, "children", None))

    layout = node() if callable(node) else node
    walk(layout)
    return ids


def test_external_and_diagnostic_panels_are_above_long_metadata() -> None:
    ids = _component_ids_in_order(app.layout)

    external_idx = ids.index("external-followup-details")
    diagnostic_idx = ids.index("diagnostic-plots-details")
    metadata_idx = ids.index("candidate-info-grid")
    run_config_idx = ids.index("run-config-details")

    assert external_idx < metadata_idx
    assert diagnostic_idx < metadata_idx
    assert metadata_idx < run_config_idx
