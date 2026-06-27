from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
from typing import Any

from malca.vsx.filter import tokenize_classes as tokenize_vsx_classes


_DATA_PATH = (
    Path(__file__).resolve().parent
    / "data"
    / "classification_labels"
    / "catalog_class_labels.json"
)


@dataclass(frozen=True)
class CatalogClassResolution:
    """Resolved display metadata for one catalog classification value."""

    column: str
    value: str
    source: str
    label: str
    tokens: tuple[str, ...]
    descriptions: tuple[str, ...]
    matched: bool
    uncertain: bool
    strategy: str


def _clean_text(value: object) -> str:
    return str(value or "").strip()


@lru_cache(maxsize=1)
def _label_data() -> dict[str, Any]:
    with _DATA_PATH.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Classification label data must be a JSON object: {_DATA_PATH}")
    return data


def _column_config(column: str) -> dict[str, Any] | None:
    columns = _label_data().get("columns", {})
    if not isinstance(columns, dict):
        return None
    config = columns.get(str(column or "").strip())
    return config if isinstance(config, dict) else None


def _map_for(name: str | None) -> dict[str, str]:
    if not name:
        return {}
    maps = _label_data().get("maps", {})
    if not isinstance(maps, dict):
        return {}
    mapping = maps.get(str(name), {})
    if not isinstance(mapping, dict):
        return {}
    return {str(key): str(value) for key, value in mapping.items()}


def _lookup_description(map_name: str | None, token: str) -> str | None:
    mapping = _map_for(map_name)
    if not mapping:
        return None
    candidates = [token, token.upper(), token.lower()]
    for candidate in candidates:
        if candidate in mapping:
            return mapping[candidate]
    return None


def _lookup_any_description(config: dict[str, Any], token: str) -> tuple[str | None, str | None]:
    map_names = [config.get("map")]
    map_names.extend(config.get("fallback_maps") or [])
    for map_name in map_names:
        if not map_name:
            continue
        description = _lookup_description(str(map_name), token)
        if description:
            return description, str(map_name)
    return None, None


def _is_uncertain(value: str, tokenizer: str) -> bool:
    if tokenizer in {"vsx", "simbad"}:
        return "?" in value or ":" in value
    return "?" in value


def _simbad_candidate_tokens(value: str) -> list[str]:
    tokens = [value]
    if "?" in value:
        tokens.extend(
            candidate
            for candidate in (
                value.replace("?", "*"),
                value.replace("?", ""),
                value[:-1] + "*" if value.endswith("?") else "",
            )
            if candidate
        )
    return list(dict.fromkeys(tokens))


def _tokens_for_value(config: dict[str, Any], value: str) -> tuple[str, ...]:
    tokenizer = str(config.get("tokenizer") or "exact")
    if tokenizer == "vsx":
        return tuple(tokenize_vsx_classes(value))
    if tokenizer == "simbad":
        return (value,)
    if tokenizer == "exact_bar_fallback":
        description, _map_name = _lookup_any_description(config, value)
        if description:
            return (value,)
        if "|" in value:
            return tuple(part.strip() for part in value.split("|") if part.strip())
    return (value,)


def _descriptions_for_tokens(
    config: dict[str, Any],
    tokens: tuple[str, ...],
    *,
    value: str,
) -> tuple[list[str], bool]:
    descriptions: list[str] = []
    matched = False
    tokenizer = str(config.get("tokenizer") or "exact")

    if tokenizer == "simbad":
        for candidate in _simbad_candidate_tokens(value):
            description, _map_name = _lookup_any_description(config, candidate)
            if description:
                return [description], True
        return [], False

    for token in tokens:
        description, _map_name = _lookup_any_description(config, token)
        if description:
            descriptions.append(description)
            matched = True
        elif token:
            descriptions.append(token)
    if not matched:
        return [], False
    return descriptions, matched


def _description_text(value: str, tokens: tuple[str, ...], descriptions: tuple[str, ...]) -> str:
    if not descriptions:
        return ""
    if len(tokens) <= 1 and len(descriptions) == 1:
        return descriptions[0]
    pairs = []
    for token, description in zip(tokens, descriptions):
        pairs.append(f"{token}: {description}")
    return "; ".join(pairs)


def resolve_catalog_class(column: str, value: object) -> CatalogClassResolution:
    """Resolve a source-specific classification value into display metadata."""
    raw_value = _clean_text(value)
    col = str(column or "").strip()
    config = _column_config(col)
    if not raw_value or not config:
        return CatalogClassResolution(
            column=col,
            value=raw_value,
            source="",
            label=raw_value,
            tokens=tuple([raw_value]) if raw_value else tuple(),
            descriptions=tuple(),
            matched=False,
            uncertain=False,
            strategy="raw",
        )

    source = str(config.get("source_label") or col)
    tokenizer = str(config.get("tokenizer") or "exact")
    tokens = _tokens_for_value(config, raw_value)
    descriptions_list, matched = _descriptions_for_tokens(config, tokens, value=raw_value)
    descriptions = tuple(descriptions_list)
    uncertain = _is_uncertain(raw_value, tokenizer)
    description = _description_text(raw_value, tokens, descriptions)
    if description:
        if uncertain and matched:
            description = f"candidate/uncertain {description}"
        label = f"{raw_value} - {description} [{source}]"
    else:
        label = f"{raw_value} [{source}]"

    return CatalogClassResolution(
        column=col,
        value=raw_value,
        source=source,
        label=label,
        tokens=tokens,
        descriptions=descriptions,
        matched=matched,
        uncertain=uncertain,
        strategy=tokenizer,
    )


def format_catalog_class_label(column: str, value: object) -> str:
    """Return the sidebar/dropdown display label for a catalog class value."""
    return resolve_catalog_class(column, value).label


def classification_tokens(column: str, value: object) -> list[str]:
    """Return source-aware tokens used to explain a catalog class value."""
    return list(resolve_catalog_class(column, value).tokens)
