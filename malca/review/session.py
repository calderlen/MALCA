"""Session state management for Dash review app."""
import hashlib

from malca.review.filter_schema import SPECIAL_FILTERS
from malca.review.store import count_queue, query_queue


_QUEUE_SCOPE_KEYS = {
    "source_path",
    "source_paths",
    "source_path_like",
    "source_path_like_any",
}
_QUEUE_SORT_KEYS = {
    "sort_col",
    "sort_cols",
    "sort_desc",
}


class SessionState:
    """Manages cached queue and navigation state."""

    def __init__(self):
        self.queue_df = None
        self.queue_hash = None
        self.candidate_ids = []
        self.payloads = {}
        self.current_idx = 0

    def needs_refresh(self, filter_dict):
        """Check if queue needs to be refreshed based on filter changes."""
        new_hash = self._compute_hash(filter_dict)
        if new_hash != self.queue_hash:
            self.queue_hash = new_hash
            return True
        return False

    def _compute_hash(self, filter_dict):
        """Compute hash of filter parameters."""
        # Convert dict to sorted tuple for consistent hashing
        items = sorted(filter_dict.items())
        hash_str = str(items)
        return hashlib.md5(hash_str.encode()).hexdigest()

    def load_queue(self, df, payloads):
        """Load queue DataFrame and payloads into cache."""
        self.queue_df = df
        self.candidate_ids = df['candidate_id'].tolist() if not df.empty else []
        self.payloads = payloads
        self.current_idx = 0

    def get_candidate(self, idx):
        """Get candidate at index (O(1) lookup)."""
        if not self.candidate_ids or idx < 0 or idx >= len(self.candidate_ids):
            return None, None

        candidate_id = self.candidate_ids[idx]
        payload = self.payloads.get(candidate_id, {})
        return candidate_id, payload

    def get_queue_size(self):
        """Get total queue size."""
        return len(self.candidate_ids)

    def get_progress_text(self):
        """Get progress text for display."""
        if not self.candidate_ids:
            return "[0/0]"
        return f"[{self.current_idx + 1}/{len(self.candidate_ids)}]"


def _format_filter_value(value: object) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _preview_filter_values(values: list[object], *, max_items: int = 3) -> str:
    rendered = [_format_filter_value(v) for v in values if str(v).strip()]
    if len(rendered) <= max_items:
        return ", ".join(rendered)
    extra = len(rendered) - max_items
    return ", ".join(rendered[:max_items]) + f", +{extra} more"


def _scope_filters(filter_params: dict[str, object]) -> dict[str, object]:
    scope: dict[str, object] = {}
    for key in _QUEUE_SCOPE_KEYS:
        value = filter_params.get(key)
        if value in (None, "", []):
            continue
        scope[key] = value
    return scope


def _active_filter_specs(filter_params: dict[str, object]) -> list[dict[str, object]]:
    specs: list[dict[str, object]] = []
    seen_numeric: set[str] = set()
    select_filter_mode = str(filter_params.get("select_filter_mode") or "exclude").strip().lower()
    if select_filter_mode not in {"include", "exclude"}:
        select_filter_mode = "exclude"

    for key, value in filter_params.items():
        if key in _QUEUE_SCOPE_KEYS or key in _QUEUE_SORT_KEYS:
            continue
        if key == "select_filter_mode":
            continue

        if key in SPECIAL_FILTERS:
            if value:
                specs.append({
                    "key": key,
                    "label": SPECIAL_FILTERS[key],
                    "params": {key: value},
                })
            continue

        if key.startswith("min_") or key.startswith("max_"):
            col = key[4:]
            if col in seen_numeric:
                continue
            seen_numeric.add(col)

            min_key = f"min_{col}"
            max_key = f"max_{col}"
            min_value = filter_params.get(min_key)
            max_value = filter_params.get(max_key)
            if min_value is None and max_value is None:
                continue

            params: dict[str, object] = {}
            if min_value is not None:
                params[min_key] = min_value
            if max_value is not None:
                params[max_key] = max_value

            if min_value is not None and max_value is not None:
                label = f"{col} in [{_format_filter_value(min_value)}, {_format_filter_value(max_value)}]"
            elif min_value is not None:
                label = f"{col} >= {_format_filter_value(min_value)}"
            else:
                label = f"{col} <= {_format_filter_value(max_value)}"

            specs.append({
                "key": col,
                "label": label,
                "params": params,
            })
            continue

        if key.endswith("_mode"):
            mode = str(value or "Any")
            if mode == "Any":
                continue
            col = key[:-5]
            if mode == "Unset":
                label = f"{col} is unset"
            else:
                label = f"{col} = {mode}"
            specs.append({
                "key": key,
                "label": label,
                "params": {key: mode},
            })
            continue

        if key.startswith("exclude_"):
            values = list(value) if isinstance(value, list) else ([] if value in (None, "") else [value])
            if not values:
                continue
            col = key.replace("exclude_", "", 1)
            specs.append({
                "key": key,
                "label": f"{select_filter_mode} {col}: {_preview_filter_values(values)}",
                "params": {key: values, "select_filter_mode": select_filter_mode},
            })
            continue

        if value in (None, "", [], False):
            continue

        specs.append({
            "key": key,
            "label": f"{key} = {_format_filter_value(value)}",
            "params": {key: value},
        })

    return specs


def create_queue_data_dict(conn, filter_params):
    """
    Query queue and create data dict for Dash Store.

    Args:
        conn: Database connection
        filter_params: Dict of filter parameters

    Returns:
        dict: {
            'candidate_ids': list,
            'queue_size': int,
            'filter_hash': str
        }
    """


    filter_params = dict(filter_params or {})

    # Query queue with filters (pass the whole dict through)
    df = query_queue(conn, filters=filter_params, ids_only=True)

    # Extract candidate IDs
    candidate_ids = df['candidate_id'].tolist() if not df.empty else []

    # Compute filter hash
    filter_hash = hashlib.md5(str(sorted(filter_params.items())).encode()).hexdigest()

    scope_filters = _scope_filters(filter_params)
    scope_size = count_queue(conn, filters=scope_filters)

    filter_provenance = []
    for spec in _active_filter_specs(filter_params):
        scoped_filter = dict(scope_filters)
        scoped_filter.update(spec["params"])
        remaining_count = count_queue(conn, filters=scoped_filter)
        filter_provenance.append({
            "key": spec["key"],
            "label": spec["label"],
            "remaining_count": remaining_count,
            "filtered_count": max(scope_size - remaining_count, 0),
        })

    return {
        'candidate_ids': candidate_ids,
        'queue_size': len(candidate_ids),
        'filter_hash': filter_hash,
        'scope_size': scope_size,
        'filtered_out_count': max(scope_size - len(candidate_ids), 0),
        'filter_provenance': filter_provenance,
    }
