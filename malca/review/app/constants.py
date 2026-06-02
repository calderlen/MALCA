# This file was mechanically split from malca.review.app; preserve behavior when editing.
_PLOT_STATIC_EXTENSIONS = {".png", ".jpg", ".jpeg", ".pdf", ".svg", ".webp"}
DEFAULT_THEME = "black"
FETCH_BACKEND_OPTIONS = [
    {"label": "SkyPatrol2 API", "value": "skypatrol2"},
    {"label": "SkyPatrol1 Web", "value": "skypatrol1"},
]
DEFAULT_FETCH_BACKEND = str(os.environ.get("MALCA_FETCH_BACKEND", "skypatrol2")).strip().lower()
if DEFAULT_FETCH_BACKEND not in {"skypatrol2", "skypatrol1"}:
    DEFAULT_FETCH_BACKEND = "skypatrol2"
DEFAULT_RESIDUAL_FRACTION = REVIEW_RESIDUAL_FRACTION
DEFAULT_EXTERNAL_SOURCE_VIEW = "asassn"
_REVIEW_PERF_ENV = "MALCA_REVIEW_PERF"


def _details_open(value: object) -> bool:
    """Return whether a lazy details panel should render."""
    return bool(value)


def _review_perf_enabled() -> bool:
    return str(os.environ.get(_REVIEW_PERF_ENV, "")).strip().lower() in {"1", "true", "yes", "on"}


def _review_perf_log(label: str, elapsed: float, **fields) -> None:
    """Print optional review UI timing without adding noise by default."""
    if not _review_perf_enabled():
        return
    suffix = " ".join(f"{key}={value}" for key, value in fields.items() if value not in (None, ""))
    print(f"[review-perf] {label} {elapsed:.4f}s {suffix}".rstrip())


def _review_perf_wrapped(label: str, func):
    def wrapper(*args, **kwargs):
        if not _review_perf_enabled():
            return func(*args, **kwargs)
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            _review_perf_log(label, time.perf_counter() - start)

    return wrapper


EXTERNAL_SOURCE_VIEW_OPTIONS = [
    {"label": "All", "value": "all"},
    {"label": "ASAS-SN Only", "value": "asassn"},
    {"label": "ATLAS", "value": "atlas"},
    {"label": "ZTF", "value": "ztf"},
    {"label": "Gaia Epoch", "value": "gaia_epoch"},
    {"label": "TESS", "value": "tess"},
    {"label": "PS1", "value": "ps1"},
    {"label": "CRTS", "value": "crts"},
]


PLOT_PRESETS = {
    'Fast Review': {
        'overlays': ['raw', 'markers', 'residuals', 'filter_bad_cameras'],
        'camera_mode': 'all',
    },
    'Clean': {
        'overlays': ['raw', 'markers', 'residuals', 'phase', 'filter_bad_cameras'],
        'camera_mode': 'all',
    },
    'Diagnostics': {
        'overlays': ['raw', 'markers', 'residuals', 'phase', 'filter_bad_cameras', 'diagnostics'],
        'camera_mode': 'all',
    },
    'Full': {
        'overlays': ['raw', 'markers', 'residuals', 'phase', 'filter_bad_cameras', 'diagnostics', 'confidence'],
        'camera_mode': 'all',
    },
}

NATIVE_BAND_OPTIONS = [
    {'label': ' g', 'value': 'g'},
    {'label': ' V', 'value': 'V'},
]

