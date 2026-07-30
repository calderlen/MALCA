"""Dependency-light external-photometry choices for the Review TUI."""

from __future__ import annotations

from collections.abc import Iterable


# These sources have magnitude-valued light curves that the TUI can
# median-align onto the ASAS-SN magnitude panel.  Flux-only TESS/Kepler light
# curves are intentionally absent because they require a separate display
# contract rather than an arbitrary magnitude conversion.
TUI_EXTERNAL_PHOTOMETRY_SOURCE_CHOICES: tuple[str, ...] = (
    "atlas",
    "ztf",
    "gaia_epoch",
    "neowise",
    "allwise_mep",
    "aavso",
    "ogle",
    "stripe82",
    "vvvx_virac",
    "ps1",
    "superwasp",
    "kelt",
    "nsvs",
    "asas3",
    "crts",
    "dasch",
)

# Availability filtering is broader than overlay rendering: TESS and Kepler
# flux light curves are valid coverage gates even though they are not
# arbitrarily converted into magnitudes on the ASAS-SN panel.
TUI_EXTERNAL_PHOTOMETRY_AVAILABILITY_SOURCE_CHOICES: tuple[str, ...] = (
    "atlas",
    "ztf",
    "gaia_epoch",
    "tess",
    "kepler",
    "neowise",
    "allwise_mep",
    "aavso",
    "ogle",
    "stripe82",
    "vvvx_virac",
    "ps1",
    "superwasp",
    "kelt",
    "nsvs",
    "asas3",
    "crts",
    "dasch",
)

TUI_EXTERNAL_PHOTOMETRY_SOURCE_LABELS: dict[str, str] = {
    "atlas": "ATLAS",
    "ztf": "ZTF",
    "gaia_epoch": "Gaia epoch",
    "tess": "TESS",
    "kepler": "Kepler",
    "neowise": "NEOWISE W1/W2",
    "allwise_mep": "AllWISE multiepoch",
    "aavso": "AAVSO",
    "ogle": "OGLE",
    "stripe82": "SDSS Stripe 82",
    "vvvx_virac": "VVVX/VIRAC2",
    "ps1": "Pan-STARRS1",
    "superwasp": "SuperWASP",
    "kelt": "KELT",
    "nsvs": "NSVS",
    "asas3": "ASAS-3",
    "crts": "CRTS",
    "dasch": "DASCH",
}

# Preserve the plot content that existed before per-source controls were
# introduced.  The remaining supported sources are opt-in.
DEFAULT_TUI_EXTERNAL_PHOTOMETRY_SOURCES: tuple[str, ...] = (
    "atlas",
    "ztf",
    "neowise",
    "asas3",
    "crts",
    "dasch",
)


def normalize_tui_external_photometry_sources(
    values: object,
) -> tuple[str, ...]:
    """Return unique supported source keys in canonical display order."""

    if values is None:
        requested: Iterable[object] = DEFAULT_TUI_EXTERNAL_PHOTOMETRY_SOURCES
    elif isinstance(values, str):
        requested = (values,)
    elif isinstance(values, Iterable):
        requested = values
    else:
        requested = (values,)

    normalized = {
        str(value or "").strip().lower()
        for value in requested
        if str(value or "").strip()
    }
    normalized.discard("asassn")
    return tuple(
        source
        for source in TUI_EXTERNAL_PHOTOMETRY_SOURCE_CHOICES
        if source in normalized
    )


def normalize_tui_external_photometry_availability_sources(
    values: object,
) -> tuple[str, ...]:
    """Normalize sources used only as candidate-availability gates."""

    if values is None:
        requested: Iterable[object] = ()
    elif isinstance(values, str):
        requested = (values,)
    elif isinstance(values, Iterable):
        requested = values
    else:
        requested = (values,)
    normalized = {
        str(value or "").strip().lower()
        for value in requested
        if str(value or "").strip()
    }
    normalized.discard("asassn")
    return tuple(
        source
        for source in TUI_EXTERNAL_PHOTOMETRY_AVAILABILITY_SOURCE_CHOICES
        if source in normalized
    )


def tui_external_photometry_source_label(source: object) -> str:
    """Return the compact human-readable label for one source key."""

    key = str(source or "").strip().lower()
    return TUI_EXTERNAL_PHOTOMETRY_SOURCE_LABELS.get(
        key,
        key.upper() if key else "External",
    )


__all__ = [
    "DEFAULT_TUI_EXTERNAL_PHOTOMETRY_SOURCES",
    "TUI_EXTERNAL_PHOTOMETRY_AVAILABILITY_SOURCE_CHOICES",
    "TUI_EXTERNAL_PHOTOMETRY_SOURCE_CHOICES",
    "TUI_EXTERNAL_PHOTOMETRY_SOURCE_LABELS",
    "normalize_tui_external_photometry_availability_sources",
    "normalize_tui_external_photometry_sources",
    "tui_external_photometry_source_label",
]
