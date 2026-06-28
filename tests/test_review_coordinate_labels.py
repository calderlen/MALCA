from __future__ import annotations

from malca.review.coordinate_labels import (
    format_j_designation,
    format_ra_dec_degrees_label,
    publication_coordinate_headers,
)


def test_format_j_designation_negative_dec() -> None:
  # J094848-545959 from ~09:48:48, -54:59:59
    label = format_j_designation(147.2000, -54.9997)
    assert label.startswith("J")
    assert "-" in label[7:]


def test_format_j_designation_positive_dec() -> None:
    label = format_j_designation(65.5583, 15.4253)
    assert label.startswith("J")
    assert "+" in label


def test_publication_coordinate_headers_from_payload() -> None:
    payload = {"ra": 147.2, "dec": -54.9997}
    left, right = publication_coordinate_headers(payload)
    assert left is not None and left.startswith("J")
    assert right == "ra=147.20000, dec=-54.99970"


def test_publication_coordinate_headers_missing_coords() -> None:
    assert publication_coordinate_headers({}) == (None, None)
