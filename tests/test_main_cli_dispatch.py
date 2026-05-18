from __future__ import annotations

import sys

import pytest

import malca.__main__ as cli


def test_detect_alias_dispatches_to_pipeline_orchestrator(monkeypatch) -> None:
    calls: list[tuple[str, list[str]]] = []

    def fake_run_module_main(module_name: str, remaining_args: list[str]) -> None:
        calls.append((module_name, remaining_args))

    monkeypatch.setattr(cli, "_run_module_main", fake_run_module_main)
    monkeypatch.setattr(sys, "argv", ["malca", "detect", "--stage", "home"])

    assert cli.main() == 0
    assert calls == [("malca.detect", ["--stage", "home"])]


def test_audit_dispatches_to_audit_module(monkeypatch) -> None:
    calls: list[tuple[str, list[str]]] = []

    def fake_run_module_main(module_name: str, remaining_args: list[str]) -> None:
        calls.append((module_name, remaining_args))

    monkeypatch.setattr(cli, "_run_module_main", fake_run_module_main)
    monkeypatch.setattr(sys, "argv", ["malca", "audit", "ltv-status"])

    assert cli.main() == 0
    assert calls == [("malca.audit", ["ltv-status"])]


def test_unknown_command_exits_nonzero(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["malca", "does-not-exist"])

    with pytest.raises(SystemExit) as excinfo:
        cli.main()

    assert excinfo.value.code == 2
