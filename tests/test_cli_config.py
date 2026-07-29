from __future__ import annotations

import argparse
import json

from malca.cli_config import add_config_args, parse_args_with_config


def test_cli_values_override_config_for_every_arg(tmp_path) -> None:
    config = tmp_path / "pipeline.json"
    config.write_text(json.dumps({"pipeline": {"workers": 2, "mode": "config"}}))

    parser = argparse.ArgumentParser()
    add_config_args(parser)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--mode", default="default")

    args = parse_args_with_config(
        parser,
        command="pipeline",
        argv=["--config", str(config), "--workers", "5"],
    )

    assert args.workers == 5
    assert args.mode == "config"


def test_profile_values_are_defaults_not_cli_overrides(tmp_path) -> None:
    config = tmp_path / "pipeline.json"
    config.write_text(json.dumps({
        "profiles": {"fast": {"pipeline": {"workers": 2, "mode": "profile"}}}
    }))
    parser = argparse.ArgumentParser()
    add_config_args(parser)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--mode", default="default")

    args = parse_args_with_config(
        parser,
        command="pipeline",
        argv=["--config", str(config), "--profile", "fast", "--mode", "cli"],
    )

    assert args.workers == 2
    assert args.mode == "cli"
