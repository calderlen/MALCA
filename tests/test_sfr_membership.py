from __future__ import annotations

import json

import numpy as np
import pandas as pd

from malca.enrichment.sfr_membership import (
    aggregate_banyan_sfr_probabilities,
    append_sfr_membership_evidence,
    load_sfr_association_crosswalk,
    load_sfr_catalog_members,
)


def _crosswalk() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sfr_name": "Taurus",
                "banyan_assoc": "TAU",
                "relation": "active_cloud_population",
                "include_in_sfr_probability": True,
                "source": "test",
                "notes": "",
            },
            {
                "sfr_name": "Perseus",
                "banyan_assoc": "IC348",
                "relation": "embedded_cluster",
                "include_in_sfr_probability": True,
                "source": "test",
                "notes": "",
            },
            {
                "sfr_name": "Perseus",
                "banyan_assoc": "NGC1333",
                "relation": "embedded_cluster",
                "include_in_sfr_probability": True,
                "source": "test",
                "notes": "",
            },
        ]
    )


def _empty_catalog() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "gaia_id",
            "association_name",
            "sfr_name",
            "catalog_name",
            "catalog_reference",
        ]
    )


def _candidate(**updates: object) -> dict[str, object]:
    row: dict[str, object] = {
        "candidate_id": "C1",
        "gaia_id": "123",
        "sfr_matches": "Taurus",
        "parallax": 7.0,
        "parallax_error": 0.1,
        "pmra": 5.0,
        "pmra_error": 0.1,
        "pmdec": -2.0,
        "pmdec_error": 0.1,
        "banyan_status": "ok",
        "banyan_input_mode": "pm+plx",
        "banyan_ya_prob": 0.99,
        "banyan_probabilities_json": json.dumps({"TAU": 0.91, "OTHER": 0.08}),
    }
    row.update(updates)
    return row


def test_shipped_crosswalk_and_empty_catalog_validate() -> None:
    crosswalk = load_sfr_association_crosswalk()
    catalog = load_sfr_catalog_members()

    assert {"Taurus", "Perseus", "Ophiuchus", "Lupus"}.issubset(
        set(crosswalk["sfr_name"])
    )
    assert catalog.empty


def test_mapped_probability_uses_components_not_global_probability() -> None:
    mapped = aggregate_banyan_sfr_probabilities(
        {"TAU": 0.11, "IC348": 0.3, "NGC1333": 0.25, "UNMAPPED": 0.34},
        _crosswalk(),
    )

    assert np.isclose(mapped["Taurus"]["probability"], 0.11)
    assert np.isclose(mapped["Perseus"]["probability"], 0.55)
    assert mapped["Perseus"]["best_assoc"] == "IC348"


def test_environment_and_mapped_banyan_agreement_is_kinematic_member() -> None:
    result = append_sfr_membership_evidence(
        pd.DataFrame([_candidate()]),
        crosswalk=_crosswalk(),
        catalog_members=_empty_catalog(),
    ).iloc[0]

    assert np.isclose(result["banyan_sfr_prob"], 0.91)
    assert result["banyan_sfr_name"] == "Taurus"
    assert bool(result["banyan_sfr_agrees"])
    assert result["sfr_membership_class"] == "kinematically_consistent_member"
    assert result["sfr_membership_evidence"] == "banyan_mapped_sfr"


def test_global_banyan_probability_cannot_promote_weak_mapped_probability() -> None:
    result = append_sfr_membership_evidence(
        pd.DataFrame(
            [
                _candidate(
                    banyan_ya_prob=0.999,
                    banyan_probabilities_json=json.dumps(
                        {"TAU": 0.2, "UNMAPPED": 0.799}
                    ),
                )
            ]
        ),
        crosswalk=_crosswalk(),
        catalog_members=_empty_catalog(),
    ).iloc[0]

    assert np.isclose(result["banyan_sfr_prob"], 0.2)
    assert not bool(result["banyan_sfr_agrees"])
    assert result["sfr_membership_class"] == "environmental_candidate"
    assert result["sfr_membership_status"] == "weak_kinematics"


def test_exact_catalog_match_has_precedence_and_preserves_provenance() -> None:
    gaia_id = "4093982141060659968"
    catalog = pd.DataFrame(
        [
            {
                "gaia_id": gaia_id,
                "association_name": "Taurus",
                "sfr_name": "Taurus",
                "catalog_name": "Published Taurus members",
                "catalog_reference": "Example et al. 2026",
                "catalog_membership_prob": 0.98,
                "accepted_member": True,
            }
        ]
    )
    result = append_sfr_membership_evidence(
        pd.DataFrame(
            [
                _candidate(
                    gaia_id=f"{gaia_id}.0",
                    banyan_probabilities_json=json.dumps({"TAU": 0.1}),
                    banyan_ya_prob=0.99,
                )
            ]
        ),
        crosswalk=_crosswalk(),
        catalog_members=catalog,
    ).iloc[0]

    assert bool(result["sfr_catalog_member"])
    assert result["sfr_catalog_match_status"] == "exact_gaia_id_match"
    assert result["sfr_catalog_name"] == "Published Taurus members"
    assert result["sfr_catalog_reference"] == "Example et al. 2026"
    assert result["sfr_membership_class"] == "catalog_confirmed_member"
    assert result["sfr_membership_evidence"] == "catalog"


def test_source_id_is_used_when_gaia_id_is_missing() -> None:
    gaia_id = "4093982141060659968"
    catalog = pd.DataFrame(
        [
            {
                "gaia_id": gaia_id,
                "association_name": "Taurus",
                "sfr_name": "Taurus",
                "catalog_name": "Published Taurus members",
                "catalog_reference": "Example et al. 2026",
                "accepted_member": True,
            }
        ]
    )
    result = append_sfr_membership_evidence(
        pd.DataFrame([_candidate(gaia_id=pd.NA, source_id=gaia_id)]),
        crosswalk=_crosswalk(),
        catalog_members=catalog,
    ).iloc[0]

    assert result["sfr_catalog_match_status"] == "exact_gaia_id_match"
    assert result["sfr_membership_class"] == "catalog_confirmed_member"


def test_strong_association_without_cloud_overlap_is_dispersed_member() -> None:
    result = append_sfr_membership_evidence(
        pd.DataFrame([_candidate(sfr_matches="")]),
        crosswalk=_crosswalk(),
        catalog_members=_empty_catalog(),
    ).iloc[0]

    assert not bool(result["sfr_environment_consistent"])
    assert result["sfr_membership_class"] == "dispersed_association_member"
    assert result["sfr_membership_name"] == "Taurus"


def test_local_catalog_model_supplies_proper_motion_consistency() -> None:
    catalog_rows = []
    offsets = np.linspace(-0.3, 0.3, 12)
    for index, offset in enumerate(offsets):
        catalog_rows.append(
            {
                "gaia_id": str(1000 + index),
                "association_name": "IC 348",
                "sfr_name": "Perseus",
                "catalog_name": "IC 348 member sample",
                "catalog_reference": "Example et al. 2026",
                "subcluster": "core",
                "accepted_member": True,
                "parallax": 3.15 + 0.08 * offset,
                "parallax_error": 0.03,
                "pmra": 4.5 + offset,
                "pmra_error": 0.05,
                "pmdec": -5.7 - 0.7 * offset + 0.05 * np.sin(index),
                "pmdec_error": 0.05,
            }
        )
    candidate = _candidate(
        gaia_id="999999",
        sfr_matches="Perseus",
        parallax=3.15,
        parallax_error=0.04,
        pmra=4.5,
        pmra_error=0.06,
        pmdec=-5.7,
        pmdec_error=0.06,
        banyan_status="missing_proper_motion_error",
        banyan_input_mode="",
        banyan_probabilities_json="{}",
    )

    result = append_sfr_membership_evidence(
        pd.DataFrame([candidate]),
        crosswalk=_crosswalk(),
        catalog_members=pd.DataFrame(catalog_rows),
    ).iloc[0]

    assert result["sfr_kinematic_method"] == "catalog_mahalanobis"
    assert bool(result["sfr_kinematic_consistent"])
    assert result["sfr_kinematic_name"] == "Perseus"
    assert result["sfr_kinematic_n_members"] == 12
    assert result["sfr_membership_class"] == "kinematically_consistent_member"


def test_missing_banyan_inputs_are_unknown_without_other_evidence() -> None:
    result = append_sfr_membership_evidence(
        pd.DataFrame(
            [
                _candidate(
                    sfr_matches="",
                    banyan_status="missing_proper_motion",
                    banyan_input_mode="",
                    banyan_probabilities_json="{}",
                )
            ]
        ),
        crosswalk=_crosswalk(),
        catalog_members=_empty_catalog(),
    ).iloc[0]

    assert result["sfr_membership_class"] == "unknown"
    assert result["sfr_membership_status"] == "missing_proper_motion"
