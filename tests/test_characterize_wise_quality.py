from __future__ import annotations

import pandas as pd
from astropy.table import Table

from malca.enrichment import characterize


def test_crossmatch_allwise_retains_photometry_and_quality_metadata(monkeypatch) -> None:
    result = Table(
        rows=[
            (
                0, 0.35, "J000000.00+000000.0", 10.1, 0.02, 9.9, 0.03, 9.1, 0.08,
                8.2, 0.18, "AABC", "000h", 0, 1, 0, "1234", 42.0, 31.0, 12.0,
                4.0, 1.1, 1.2, 1.4, 2.1, 0.0, 0.0, 0.0, 0.02, 18, 17, 13, 6,
                20, 20, 15, 8,
            )
        ],
        names=(
            "_idx", "angDist", "AllWISE", "W1mag", "e_W1mag", "W2mag", "e_W2mag",
            "W3mag", "e_W3mag", "W4mag", "e_W4mag", "qph", "ccf", "ex", "nb",
            "na", "var", "snr1", "snr2", "snr3", "snr4", "chi2W1", "chi2W2",
            "chi2W3", "chi2W4", "sat1", "sat2", "sat3", "sat4", "nW1", "nW2",
            "nW3", "nW4", "mW1", "mW2", "mW3", "mW4",
        ),
    )
    monkeypatch.setattr(characterize.XMatch, "query", lambda **_kwargs: result)

    matched = characterize.crossmatch_allwise(
        pd.DataFrame([{"candidate_id": "c", "ra": 0.0, "dec": 0.0}])
    ).iloc[0]

    assert matched["w3"] == 9.1
    assert matched["allwise_id"] == "J000000.00+000000.0"
    assert matched["allwise_ph_qual"] == "AABC"
    assert matched["allwise_cc_flags"] == "000h"
    assert matched["allwise_ext_flg"] == 0
    assert matched["allwise_nb"] == 1
    assert matched["allwise_na"] == 0
    assert matched["allwise_w3_snr"] == 12.0
    assert matched["allwise_w4_rchi2"] == 2.1
    assert matched["allwise_w4_sat"] == 0.02
    assert matched["allwise_w4_ndet"] == 6
    assert matched["allwise_w4_nframe"] == 8
    assert matched["allwise_sep_arcsec"] == 0.35
