from __future__ import annotations

from malca.review.session import _active_filter_specs


def test_catalog_neighbor_policy_provenance_uses_boolean_filter() -> None:
    specs = _active_filter_specs(
        {
            "catalog_neighbor_radius_arcsec": 30.0,
            "exclude_known_catalog_neighbors": True,
            "exclude_dipper_catalog_neighbors": False,
            "select_filter_mode": "exclude",
            "select_filter_logic": "or",
        }
    )

    assert specs == [
        {
            "key": "exclude_known_catalog_neighbors",
            "label": "Exclude catalog-neighbor known variables within 30 arcsec",
            "params": {
                "exclude_known_catalog_neighbors": True,
                "catalog_neighbor_radius_arcsec": 30.0,
            },
        }
    ]


def test_catalog_neighbor_radius_is_not_a_standalone_filter() -> None:
    assert _active_filter_specs({"catalog_neighbor_radius_arcsec": 30.0}) == [
        {
            "key": "catalog_neighbor_radius_arcsec",
            "label": "catalog_neighbor_radius_arcsec = 30",
            "params": {"catalog_neighbor_radius_arcsec": 30.0},
        }
    ]
