# This file was mechanically split from malca.review.app; preserve behavior when editing.
_REVIEW_DB_ENV = "MALCA_REVIEW_DB_PATH"
_REVIEW_PLOT_ENV = "MALCA_REVIEW_PLOT_DIR"


def _env_path_or_none(name: str) -> str | None:
    """Return a stripped env path value, or None when unset/empty."""
    value = os.environ.get(name)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


# Global variables
DB_PATH = _env_path_or_none(_REVIEW_DB_ENV) or str(DEFAULT_DB_PATH)
PLOT_DIR = _env_path_or_none(_REVIEW_PLOT_ENV)
INITIAL_CANDIDATE_QUERY: str | None = None

def _review_persistence_token() -> str:
    """Return a persistence scope tied to the active review DB."""
    try:
        return str(Path(DB_PATH).expanduser().resolve())
    except Exception:
        return str(DB_PATH)
