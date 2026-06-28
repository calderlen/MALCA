from __future__ import annotations

import numpy as np
from astropy.table import Table

from malca.vsx import nearby


def setup_function() -> None:
    nearby._find_nearby_vsx_cached.cache_clear()


def test_find_nearby_vsx_queries_vizier_and_normalizes_top_three(monkeypatch) -> None:
    class FakeVizier:
        instances = []

        def __init__(self, *, columns, row_limit):
            self.columns = columns
            self.row_limit = row_limit
            self.TIMEOUT = None
            FakeVizier.instances.append(self)

        def query_region(self, target, *, radius, catalog):
            self.target = target
            self.radius = radius
            self.catalog = catalog
            return [
                Table(
                    rows=[
                        (303, "Far", 10.0, 20.003, "L", np.nan),
                        (101, "Near", 10.0, 20.001, "ROT", np.nan),
                        (404, "Outside", 10.0, 20.030, "EA", 1.2),
                        (202, "Mid", 10.0, 20.002, "EA", 2.5),
                    ],
                    names=["OID", "Name", "RAJ2000", "DEJ2000", "Type", "Period"],
                )
            ]

    monkeypatch.setattr(nearby, "Vizier", FakeVizier)

    neighbors = nearby.find_nearby_vsx(10.0, 20.0, radius_arcsec=20.0)

    assert [item.name for item in neighbors] == ["Near", "Mid", "Far"]
    assert neighbors[0].sep_arcsec < neighbors[1].sep_arcsec < neighbors[2].sep_arcsec
    assert neighbors[0].url == "https://vsx.aavso.org/index.php?view=detail.top&oid=101"
    assert neighbors[0].period_days is None
    assert neighbors[1].period_days == 2.5
    assert "Rotational variable" in neighbors[0].type_label

    instance = FakeVizier.instances[0]
    assert instance.columns == nearby.VSX_NEIGHBOR_COLUMNS
    assert instance.row_limit == 3
    assert instance.catalog == nearby.VSX_VIZIER_CATALOG


def test_find_nearby_vsx_returns_empty_on_query_failure(monkeypatch) -> None:
    class FailingVizier:
        def __init__(self, *, columns, row_limit):
            pass

        def query_region(self, target, *, radius, catalog):
            raise RuntimeError("offline")

    monkeypatch.setattr(nearby, "Vizier", FailingVizier)

    assert nearby.find_nearby_vsx(10.0, 20.0) == []


def test_find_nearby_vsx_rejects_missing_or_invalid_coordinates() -> None:
    assert nearby.find_nearby_vsx(None, 20.0) == []
    assert nearby.find_nearby_vsx(10.0, None) == []
    assert nearby.find_nearby_vsx(400.0, 20.0) == []
    assert nearby.find_nearby_vsx(10.0, -100.0) == []
