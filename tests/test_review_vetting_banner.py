from __future__ import annotations

from malca.review.app import _render_vetting_banner


def _component_text(node) -> list[str]:
    if node is None:
        return []
    if isinstance(node, (str, int, float)):
        return [str(node)]
    if isinstance(node, (list, tuple)):
        texts: list[str] = []
        for child in node:
            texts.extend(_component_text(child))
        return texts
    return _component_text(getattr(node, "children", None))


def test_vetting_banner_does_not_mark_gaia_flag_only_as_known() -> None:
    banner = _render_vetting_banner(
        {
            "vetting_likely_known": False,
            "gaia_var_flag": True,
            "gaia_var_class": "",
        }
    )

    text = " ".join(_component_text(banner))

    assert "POTENTIALLY NEW" in text
    assert "Gaia DR3" not in text


def test_vetting_banner_marks_gaia_class_as_known_without_summary_flag() -> None:
    banner = _render_vetting_banner(
        {
            "gaia_var_flag": False,
            "gaia_var_class": "LPV",
            "gaia_var_score": 0.91,
        }
    )

    text = " ".join(_component_text(banner))

    assert "KNOWN VARIABLE" in text
    assert "Gaia DR3" in text
    assert "LPV" in text
