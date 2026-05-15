from __future__ import annotations

from malca.review.metadata import bracket_unit_label, markdown_literal_unit_label


def test_unit_labels_use_visible_literal_brackets() -> None:
    assert bracket_unit_label("Period (d)") == "Period [d]"
    assert bracket_unit_label("Amplitude (mag)") == "Amplitude [mag]"


def test_markdown_unit_labels_escape_brackets_without_changing_visible_text() -> None:
    assert markdown_literal_unit_label("Period (d)") == r"Period \[d\]"
    assert markdown_literal_unit_label("Amplitude (mag)") == r"Amplitude \[mag\]"
