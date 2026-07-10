from __future__ import annotations

import pandas as pd

from malca.catalogs.evidence import normalize_catalog_evidence, normalize_catalog_evidence_record
from malca.ltv.review import map_ltv_columns
from malca.products.feature_layers import feature_value_series
from malca.review.store import db_connect, get_candidate_payload, import_candidates


def _neighbors(vsx_type: str, sep_arcsec: float, period: float = 1.25) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "candidate_id": "C1",
                "catalog": "B/vsx/vsx",
                "sep_arcsec": sep_arcsec,
                "Type": vsx_type,
                "Period": period,
            }
        ]
    )


def test_close_definite_vsx_neighbor_promotes_canonical_fields() -> None:
    out = normalize_catalog_evidence(
        pd.DataFrame({"candidate_id": ["C1"]}),
        neighbors_long=_neighbors("ROT", 0.1176, 7.8752),
    )

    assert out.loc[0, "vsx_class"] == "ROT"
    assert float(out.loc[0, "vsx_period"]) == 7.8752
    assert float(out.loc[0, "vsx_sep_arcsec"]) == 0.1176


def test_generic_or_uncertain_vsx_neighbors_are_context_only() -> None:
    generic = normalize_catalog_evidence(
        pd.DataFrame({"candidate_id": ["C1"]}),
        neighbors_long=_neighbors("VAR", 0.1),
    )
    uncertain = normalize_catalog_evidence(
        pd.DataFrame({"candidate_id": ["C1"]}),
        neighbors_long=_neighbors("EA:", 0.1),
    )

    assert pd.isna(generic.loc[0, "vsx_class"])
    assert pd.isna(uncertain.loc[0, "vsx_class"])


def test_vsx_neighbor_beyond_match_radius_does_not_promote() -> None:
    out = normalize_catalog_evidence(
        pd.DataFrame({"candidate_id": ["C1"]}),
        neighbors_long=_neighbors("EA", 3.5),
        vsx_max_sep_arcsec=3.0,
    )

    assert pd.isna(out.loc[0, "vsx_class"])


def test_existing_vsx_class_is_not_overwritten_by_neighbor() -> None:
    out = normalize_catalog_evidence(
        pd.DataFrame({"candidate_id": ["C1"], "vsx_class": ["ROT"], "vsx_period": [2.0]}),
        neighbors_long=_neighbors("EA", 0.1, 1.0),
    )

    assert out.loc[0, "vsx_class"] == "ROT"
    assert float(out.loc[0, "vsx_period"]) == 2.0


def test_ltv_vsx_type_fills_vsx_class_and_keeps_vsx_name_mapping() -> None:
    out = map_ltv_columns(
        pd.DataFrame(
            {
                "asas_sn_id": ["123"],
                "lc_path": ["lc.dat"],
                "ra": [10.0],
                "dec": [-5.0],
                "failed_any": [False],
                "vsx_type": ["EA"],
                "vsx_name": ["VSX Example"],
            }
        )
    )

    assert feature_value_series(out, "external_stats.vsx_class").iloc[0] == "EA"
    assert feature_value_series(out, "external_stats.ltv_vsx_name").iloc[0] == "VSX Example"


def test_record_normalization_maps_vsx_type_to_vsx_class() -> None:
    out = normalize_catalog_evidence_record({"vsx_type": "ROT"})

    assert out["vsx_class"] == "ROT"


def test_review_import_marks_normalized_vsx_type_as_known(tmp_path) -> None:
    db_path = tmp_path / "review.db"
    with db_connect(db_path) as conn:
        import_candidates(
            conn,
            pd.DataFrame(
                [
                    {
                        "candidate_id": "C1",
                        "asas_sn_id": "1",
                        "vsx_type": "ROT",
                    }
                ]
            ),
            source_path=str(tmp_path),
            characterize_before_import=False,
            vet_before_import=False,
        )
        payload = get_candidate_payload(conn, "C1")

    assert payload["vsx_class"] == "ROT"
    assert payload["vetting_likely_known"] is True
