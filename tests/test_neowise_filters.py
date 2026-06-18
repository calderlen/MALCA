import pandas as pd

from malca.neowise_filters import filter_neowise_single_exposure_lc


def test_filter_neowise_keeps_good_qual_frame_and_drops_poor() -> None:
    lc = pd.DataFrame(
        {
            "qual_frame": [0, 5, 10, 5],
            "cc_flags": ["0000", "0000", "0000", "1000"],
            "w1snr": [50.0, 50.0, 50.0, 50.0],
            "w2snr": [50.0, 50.0, 50.0, 50.0],
            "qi_fact": [0.0, 0.5, 1.0, 0.5],
        }
    )
    out = filter_neowise_single_exposure_lc(lc)
    assert list(out["qual_frame"]) == [5, 10]


def test_filter_neowise_does_not_apply_qi_fact_cut() -> None:
    lc = pd.DataFrame(
        {
            "qual_frame": [5, 5],
            "cc_flags": ["0000", "0000"],
            "w1snr": [20.0, 20.0],
            "w2snr": [20.0, 20.0],
            "qi_fact": [0.0, 0.5],
        }
    )
    out = filter_neowise_single_exposure_lc(lc)
    assert len(out) == 2


def test_filter_neowise_decodes_bytes_cc_flags() -> None:
    lc = pd.DataFrame(
        {
            "qual_frame": [10],
            "cc_flags": [b"0000"],
            "w1snr": [20.0],
            "w2snr": [20.0],
        }
    )
    out = filter_neowise_single_exposure_lc(lc)
    assert len(out) == 1
