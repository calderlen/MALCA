from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


def add_config_args(parser: argparse.ArgumentParser) -> None:
    """Add standard config/profile options to commands with advanced settings."""
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="JSON/TOML config file for advanced options",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="Named profile inside --config",
    )


def load_config(path: Path | str | None) -> dict[str, Any]:
    if path is None:
        return {}
    config_path = Path(path).expanduser()
    if not config_path.exists():
        raise SystemExit(f"Config file not found: {config_path}")
    suffix = config_path.suffix.lower()
    if suffix == ".json":
        with config_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    elif suffix in {".toml", ".tml"}:
        try:
            import tomllib  # type: ignore[import-not-found]
        except ModuleNotFoundError:  # pragma: no cover - only used on Python 3.9/3.10 with tomli installed
            try:
                import tomli as tomllib  # type: ignore[no-redef]
            except ModuleNotFoundError as exc:
                raise SystemExit(
                    "TOML config requires Python 3.11+ or the optional 'tomli' package. "
                    "Use JSON config instead."
                ) from exc
        with config_path.open("rb") as f:
            data = tomllib.load(f)
    else:
        raise SystemExit("Config files must use .json or .toml")
    if not isinstance(data, dict):
        raise SystemExit("Config file must contain an object/table at the top level")
    return data


def apply_config(
    args: argparse.Namespace,
    *,
    command: str,
    valid_keys: set[str] | None = None,
    path_keys: set[str] | None = None,
) -> argparse.Namespace:
    """Merge config values into an argparse namespace.

    Supported layout:

      defaults: {workers: 8}
      pipeline: {trigger_mode: "logbf"}
      profiles:
        fast:
          defaults: {workers: 4}
          pipeline: {run_vetting: false}

    Keys use argparse dest names, not CLI spellings.
    """
    data = load_config(getattr(args, "config", None))
    if not data:
        return args

    profile = getattr(args, "profile", None)
    merged: dict[str, Any] = {}

    for key in ("defaults", "global"):
        section = data.get(key)
        if isinstance(section, Mapping):
            merged.update(section)

    for key in _command_keys(command):
        section = data.get(key)
        if isinstance(section, Mapping):
            merged.update(section)

    if profile:
        profiles = data.get("profiles")
        if not isinstance(profiles, Mapping) or profile not in profiles:
            raise SystemExit(f"Profile '{profile}' not found in config")
        profile_data = profiles[profile]
        if not isinstance(profile_data, Mapping):
            raise SystemExit(f"Profile '{profile}' must be an object/table")
        for key in ("defaults", "global"):
            section = profile_data.get(key)
            if isinstance(section, Mapping):
                merged.update(section)
        for key in _command_keys(command):
            section = profile_data.get(key)
            if isinstance(section, Mapping):
                merged.update(section)
        flat_keys = {
            k: v
            for k, v in profile_data.items()
            if k not in {"defaults", "global", "profiles"}
            and k not in _command_keys(command)
            and not isinstance(v, Mapping)
        }
        if flat_keys:
            merged.update(flat_keys)

    if valid_keys is not None:
        unknown = sorted(set(merged) - valid_keys)
        if unknown:
            raise SystemExit(
                f"Unknown config option(s) for {command}: {', '.join(unknown)}"
            )

    for key, value in merged.items():
        if path_keys is not None and key in path_keys and value is not None:
            setattr(args, key, Path(value).expanduser())
            continue
        setattr(args, key, _coerce_like(value, getattr(args, key, None)))
    return args


def namespace_keys(parser: argparse.ArgumentParser, extra: Mapping[str, Any] | None = None) -> set[str]:
    keys = {action.dest for action in parser._actions if action.dest != argparse.SUPPRESS}
    if extra:
        keys.update(extra)
    return keys


def _command_keys(command: str) -> tuple[str, ...]:
    return (command, command.replace("-", "_"))


def _coerce_like(value: Any, current: Any) -> Any:
    if current is None:
        return value
    if isinstance(current, Path):
        return Path(value).expanduser() if isinstance(value, str) else value
    if isinstance(current, tuple):
        return tuple(value) if isinstance(value, list) else value
    return value
