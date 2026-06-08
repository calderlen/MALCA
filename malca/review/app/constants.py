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
DEFAULT_EXTERNAL_SOURCE_VALUES = ["asassn"]
DEFAULT_EXTERNAL_SOURCE_VIEW = "asassn"
DEFAULT_EXTERNAL_SOURCE_LAYOUT = "overlay"
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
    {"label": "ASAS-SN", "value": "asassn"},
    {"label": "ATLAS", "value": "atlas"},
    {"label": "ZTF", "value": "ztf"},
    {"label": "Gaia Epoch", "value": "gaia_epoch"},
    {"label": "TESS", "value": "tess"},
    {"label": "NEOWISE W1/W2", "value": "neowise"},
    {"label": "Kepler/K2", "value": "kepler"},
    {"label": "AAVSO", "value": "aavso"},
    {"label": "OGLE I/V", "value": "ogle"},
    {"label": "SDSS Stripe 82", "value": "stripe82"},
    {"label": "AllWISE MEP", "value": "allwise_mep"},
    {"label": "VVVX/VIRAC2", "value": "vvvx_virac"},
    {"label": "PS1", "value": "ps1"},
    {"label": "CRTS", "value": "crts"},
]
EXTERNAL_SOURCE_VALUES = tuple(str(option["value"]) for option in EXTERNAL_SOURCE_VIEW_OPTIONS)
EXTERNAL_SOURCE_VALUE_SET = set(EXTERNAL_SOURCE_VALUES)
EXTERNAL_SOURCE_LAYOUT_OPTIONS = [
    {"label": "Overlay", "value": "overlay"},
    {"label": "Split", "value": "split"},
]


def normalize_external_source_values(raw_value: object, *, default: list[str] | None = None) -> list[str]:
    """Normalize legacy single-select or new checklist source values."""
    if default is None:
        default = list(DEFAULT_EXTERNAL_SOURCE_VALUES)
    if raw_value is None:
        raw_values = list(default)
    elif isinstance(raw_value, str):
        raw_values = [raw_value]
    elif isinstance(raw_value, (list, tuple, set, np.ndarray, pd.Series)):
        raw_values = list(raw_value)
    else:
        raw_values = [raw_value]

    out: list[str] = []
    seen: set[str] = set()
    for value in raw_values:
        text = str(value).strip().lower()
        if not text:
            continue
        if text == "all":
            return list(EXTERNAL_SOURCE_VALUES)
        if text in {"wise", "w1", "w2", "wise_w1_w2"}:
            text = "neowise"
        elif text in {"k2", "kepler_k2"}:
            text = "kepler"
        elif text in {"sdss_s82", "s82", "stripe_82", "sdss_stripe82"}:
            text = "stripe82"
        elif text in {"allwise", "allwise_multiepoch", "wise_mep"}:
            text = "allwise_mep"
        elif text in {"vvv", "vvvx", "virac", "virac2", "vvvx_virac2"}:
            text = "vvvx_virac"
        if text not in EXTERNAL_SOURCE_VALUE_SET or text in seen:
            continue
        out.append(text)
        seen.add(text)

    # Legacy single external choices always rendered over the native ASAS-SN LC.
    if isinstance(raw_value, str) and out and out[0] != "asassn":
        out.insert(0, "asassn")
    return out


def normalize_external_source_layout(raw_value: object) -> str:
    text = str(raw_value).strip().lower() if raw_value is not None else ""
    return text if text in {"overlay", "split"} else DEFAULT_EXTERNAL_SOURCE_LAYOUT


def legacy_external_source_view(source_values: object) -> str:
    values = normalize_external_source_values(source_values, default=[])
    if not values:
        return ""
    if values == ["asassn"]:
        return "asassn"
    if set(values) == EXTERNAL_SOURCE_VALUE_SET:
        return "all"
    external = [value for value in values if value != "asassn"]
    if "asassn" in values and len(external) == 1:
        return external[0]
    return ",".join(values)


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
