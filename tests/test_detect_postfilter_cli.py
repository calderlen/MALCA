from __future__ import annotations

import argparse
from pathlib import Path

from malca.detect import _build_post_filter_kwargs


def _base_args() -> argparse.Namespace:
    return argparse.Namespace(
        skip_evidence_strength=False,
        min_bayes_factor=10.0,
        allow_infinite_local_bf=False,
        skip_run_robustness=False,
        min_run_count=1,
        post_filter_min_run_points=2,
        post_filter_min_run_cameras=2,
        apply_morphology=False,
        dip_morphology="gaussian",
        jump_morphology="paczynski",
        min_delta_bic=10.0,
        skip_score_filter=False,
        min_score=0.0,
        apply_periodicity_validation=False,
        periodicity_n_bootstrap=1000,
        periodicity_significance=0.01,
        periodicity_no_exclude_aliases=False,
        periodicity_reject=False,
        periodicity_workers=4,
        periodicity_checkpoint_dir=None,
        skip_gaia_ruwe_validation=False,
        gaia_max_ruwe=1.4,
        gaia_reject=False,
        skip_periodic_catalog_validation=False,
        periodic_catalog_max_sep=3.0,
        periodic_catalog_reject=False,
        phase_plot_max_sig=0.01,
        phase_plot_min_power=0.3,
        phase_plot_allow_alias=False,
        verbose=False,
    )


def test_build_post_filter_kwargs_defaults_match_pipeline_behavior() -> None:
    kwargs = _build_post_filter_kwargs(_base_args())

    assert kwargs["apply_evidence_strength"] is True
    assert kwargs["apply_run_robustness"] is True
    assert kwargs["apply_score"] is True
    assert kwargs["apply_gaia_ruwe_validation"] is True
    assert kwargs["apply_periodic_catalog_validation"] is True

    assert kwargs["apply_morphology"] is False
    assert kwargs["apply_periodicity_validation"] is False

    assert kwargs["gaia_flag_only"] is True
    assert kwargs["periodic_catalog_flag_only"] is True
    assert kwargs["periodicity_flag_only"] is True

    assert kwargs["phase_plot_max_sig"] == 0.01
    assert kwargs["phase_plot_min_power"] == 0.3
    assert kwargs["phase_plot_allow_alias"] is False


def test_build_post_filter_kwargs_respects_cli_overrides() -> None:
    args = _base_args()
    args.skip_score_filter = True
    args.apply_morphology = True
    args.dip_morphology = "paczynski"
    args.jump_morphology = "gaussian"
    args.min_delta_bic = 7.5
    args.apply_periodicity_validation = True
    args.periodicity_n_bootstrap = 250
    args.periodicity_significance = 0.02
    args.periodicity_no_exclude_aliases = True
    args.periodicity_reject = True
    args.periodicity_workers = 2
    args.periodicity_checkpoint_dir = Path("output/checkpoints")
    args.phase_plot_max_sig = 0.05
    args.phase_plot_min_power = 0.5
    args.phase_plot_allow_alias = True
    args.skip_gaia_ruwe_validation = True
    args.gaia_reject = True
    args.skip_periodic_catalog_validation = True
    args.periodic_catalog_reject = True

    kwargs = _build_post_filter_kwargs(args)

    assert kwargs["apply_score"] is False
    assert kwargs["apply_morphology"] is True
    assert kwargs["dip_morphology"] == "paczynski"
    assert kwargs["jump_morphology"] == "gaussian"
    assert kwargs["min_delta_bic"] == 7.5

    assert kwargs["apply_periodicity_validation"] is True
    assert kwargs["periodicity_n_bootstrap"] == 250
    assert kwargs["periodicity_significance"] == 0.02
    assert kwargs["periodicity_exclude_aliases"] is False
    assert kwargs["periodicity_flag_only"] is False
    assert kwargs["periodicity_workers"] == 2
    assert kwargs["periodicity_checkpoint_dir"] == Path("output/checkpoints")

    assert kwargs["phase_plot_max_sig"] == 0.05
    assert kwargs["phase_plot_min_power"] == 0.5
    assert kwargs["phase_plot_allow_alias"] is True

    assert kwargs["apply_gaia_ruwe_validation"] is False
    assert kwargs["gaia_flag_only"] is False

    assert kwargs["apply_periodic_catalog_validation"] is False
    assert kwargs["periodic_catalog_flag_only"] is False
