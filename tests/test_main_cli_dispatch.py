from __future__ import annotations

import argparse
import sys

import pytest

import malca.__main__ as cli
from malca.ltv import pipeline as ltv_pipeline


def test_stv_pipeline_dispatches_to_pipeline_orchestrator(monkeypatch) -> None:
    calls: list[tuple[str, list[str]]] = []

    def fake_run_module_main(module_name: str, remaining_args: list[str]) -> None:
        calls.append((module_name, remaining_args))

    monkeypatch.setattr(cli, "_run_module_main", fake_run_module_main)
    monkeypatch.setattr(sys, "argv", ["malca", "stv-pipeline", "--stage", "home"])

    assert cli.main() == 0
    assert calls == [("malca.stv.pipeline", ["--stage", "home"])]


@pytest.mark.parametrize(
    ("command", "module_name"),
    [
        ("stv-events", "malca.stv.events"),
        ("stv-filter", "malca.stv.filter"),
        ("stv-tag", "malca.stv.tag"),
        ("stv-plot", "malca.stv.plot"),
    ],
)
def test_stv_subcommands_dispatch(monkeypatch, command: str, module_name: str) -> None:
    calls: list[tuple[str, list[str]]] = []

    def fake_run_module_main(module_name_arg: str, remaining_args: list[str]) -> None:
        calls.append((module_name_arg, remaining_args))

    monkeypatch.setattr(cli, "_run_module_main", fake_run_module_main)
    monkeypatch.setattr(sys, "argv", ["malca", command, "--help"])

    assert cli.main() == 0
    assert calls == [(module_name, ["--help"])]


def test_audit_dispatches_to_audit_module(monkeypatch) -> None:
    calls: list[tuple[str, list[str]]] = []

    def fake_run_module_main(module_name: str, remaining_args: list[str]) -> None:
        calls.append((module_name, remaining_args))

    monkeypatch.setattr(cli, "_run_module_main", fake_run_module_main)
    monkeypatch.setattr(sys, "argv", ["malca", "audit", "ltv-status"])

    assert cli.main() == 0
    assert calls == [("malca.audit", ["ltv-status"])]


def test_ltv_pipeline_dispatches_to_pipeline_module(monkeypatch) -> None:
    calls: list[tuple[str, list[str]]] = []

    def fake_run_module_main(module_name: str, remaining_args: list[str]) -> None:
        calls.append((module_name, remaining_args))

    monkeypatch.setattr(cli, "_run_module_main", fake_run_module_main)
    monkeypatch.setattr(sys, "argv", ["malca", "ltv-pipeline", "--stage", "home"])

    assert cli.main() == 0
    assert calls == [("malca.ltv.pipeline", ["--stage", "home"])]


def test_ltv_pipeline_stage_choices_include_full_extended() -> None:
    args = ltv_pipeline.add_ltv_pipeline_args(argparse.ArgumentParser()).parse_args(
        ["--stage", "full-extended"]
    )
    assert args.stage == "full-extended"


@pytest.mark.parametrize("command", ["ltv-core", "ltv-build", "ltv-ingest", "ltv-bundle"])
def test_single_purpose_ltv_commands_are_not_public(monkeypatch, command: str) -> None:
    monkeypatch.setattr(sys, "argv", ["malca", command, "--help"])

    with pytest.raises(SystemExit) as excinfo:
        cli.main()

    assert excinfo.value.code == 2


@pytest.mark.parametrize("command", ["pipeline", "detect", "events", "filter", "tag", "plot"])
def test_legacy_stv_commands_are_not_public(monkeypatch, command: str) -> None:
    monkeypatch.setattr(sys, "argv", ["malca", command, "--help"])

    with pytest.raises(SystemExit) as excinfo:
        cli.main()

    assert excinfo.value.code == 2


def test_unknown_command_exits_nonzero(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["malca", "does-not-exist"])

    with pytest.raises(SystemExit) as excinfo:
        cli.main()

    assert excinfo.value.code == 2
