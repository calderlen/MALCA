from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


QueryMode = Literal["xmatch", "tap"]


@dataclass(frozen=True)
class SpectraCatalogSpec:
  """Configuration for one spectroscopic survey crossmatch."""

  vizier_id: str
  mode: QueryMode = "xmatch"
  query_group: str | None = None
  filter_col: str | None = None
  filter_values: tuple[str, ...] | None = None
  filter_contains: str | None = None
  filter_not_null: bool = False
  tap_table: str | None = None
  tap_select: str = "*"
  ra_col: str = "RAJ2000"
  dec_col: str = "DEJ2000"
  enabled_by_default: bool = True


# SDSS DR16 on VizieR is the best unified optical spec table available via CDS XMatch.
_SDSS_SPEC = "V/154/sdss16"
_SDSS_GROUP = "sdss_spec"


DEFAULT_SPECTRA_CATALOG_SPECS: dict[str, SpectraCatalogSpec] = {
    # --- Tier A: wide-area surveys with direct flux archive access ---
    "desi_dr1": SpectraCatalogSpec(
        vizier_id="V/161/zcatdr1",
        mode="tap",
        tap_table='"V/161/zcatdr1"',
        ra_col="RA",
        dec_col="DEC",
    ),
    "sdss_dr16_spec": SpectraCatalogSpec(
        vizier_id=_SDSS_SPEC,
        query_group=_SDSS_GROUP,
        filter_col="zsp",
        filter_not_null=True
    ),
    "sdss_boss": SpectraCatalogSpec(
        vizier_id=_SDSS_SPEC,
        query_group=_SDSS_GROUP,
        filter_col="survey",
        filter_values=("boss",),
    ),
    "sdss_eboss": SpectraCatalogSpec(
        vizier_id=_SDSS_SPEC,
        query_group=_SDSS_GROUP,
        filter_col="survey",
        filter_values=("eboss",),
    ),
    "sdss_legacy": SpectraCatalogSpec(
        vizier_id=_SDSS_SPEC,
        query_group=_SDSS_GROUP,
        filter_col="survey",
        filter_values=("sdss",),
    ),
    "sdss_segue": SpectraCatalogSpec(
        vizier_id=_SDSS_SPEC,
        query_group=_SDSS_GROUP,
        filter_col="survey",
        filter_values=("segue1", "segue2"),
    ),
    "sdss_spiders": SpectraCatalogSpec(
        vizier_id=_SDSS_SPEC,
        query_group=_SDSS_GROUP,
        filter_col="programname",
        filter_contains="spider",
    ),
    "sdss_tdss": SpectraCatalogSpec(
        vizier_id=_SDSS_SPEC,
        query_group=_SDSS_GROUP,
        filter_col="programname",
        filter_contains="tdss",
    ),
    "apogee_dr16": SpectraCatalogSpec(vizier_id="III/284/allstars"),
    "manga_dr17": SpectraCatalogSpec(vizier_id="J/ApJS/262/36/catalog"),
    "lamost_dr7": SpectraCatalogSpec(vizier_id="V/156/dr7lrs"),
    "galah_dr3": SpectraCatalogSpec(vizier_id="J/MNRAS/506/150/stars"),
    "rave_dr6": SpectraCatalogSpec(vizier_id="III/283/ravedr6"),
    "sixdf_gs": SpectraCatalogSpec(vizier_id="VII/259/6dfgs"),
    "2dfgrs": SpectraCatalogSpec(vizier_id="VII/250/2dfgrs"),
    "milliquas": SpectraCatalogSpec(vizier_id="VII/294/catalog"),
    "sdss2_sn": SpectraCatalogSpec(vizier_id=_SDSS_SPEC, query_group=_SDSS_GROUP),
    "cks": SpectraCatalogSpec(vizier_id="J/AJ/154/107/stars"),
    # --- Tier B: metadata-rich; flux via astroquery or best-effort ---
    "gaia_rvs": SpectraCatalogSpec(
        vizier_id="I/355/rvsmean",
        mode="tap",
        tap_table='"I/355/rvsmean"',
    ),
    "gaia_xp": SpectraCatalogSpec(
        vizier_id="I/355/xpsample",
        mode="tap",
        tap_table='"I/355/xpsample"',
    ),
    "gaia_eso": SpectraCatalogSpec(vizier_id="J/A+A/676/A129/catalog"),
    "vvds": SpectraCatalogSpec(vizier_id="III/250/vvds_dp"),
    "deep2": SpectraCatalogSpec(vizier_id="J/MNRAS/452/525/table3"),
    "zcosmos": SpectraCatalogSpec(vizier_id="J/A+A/559/A2/tables"),
    "vandels": SpectraCatalogSpec(vizier_id="V/151/cdfs"),
    "vipers": SpectraCatalogSpec(vizier_id="J/A+A/609/A84/vipersw1"),
    "wigglez": SpectraCatalogSpec(vizier_id="J/MNRAS/401/1429/wigglez1"),
    "primus": SpectraCatalogSpec(vizier_id="J/ApJ/767/118/primus", enabled_by_default=False),
    "ozdes": SpectraCatalogSpec(vizier_id="J/MNRAS/472/273/ozdesdr1"),
    "gama": SpectraCatalogSpec(vizier_id="VII/291/gladep"),
    "sdss_v": SpectraCatalogSpec(vizier_id=_SDSS_SPEC, query_group=_SDSS_GROUP, enabled_by_default=False),
    "3d_hst": SpectraCatalogSpec(vizier_id="J/ApJS/265/40/catalog"),
    "s5": SpectraCatalogSpec(vizier_id="J/MNRAS/490/3508/s5dr1"),
    "efeds_agn": SpectraCatalogSpec(
        vizier_id="J/A+A/661/A3/ctpmain", ra_col="RAc", dec_col="DEc"
    ),
    "2dflens": SpectraCatalogSpec(vizier_id="J/MNRAS/462/4240/2dflens", enabled_by_default=False),
    "class": SpectraCatalogSpec(vizier_id="VIII/66/class", enabled_by_default=False),
}

# Backward-compatible simple mapping (survey_key -> vizier_id) for callers/tests.
DEFAULT_SPECTRA_CATALOGS: dict[str, str] = {
    key: spec.vizier_id for key, spec in DEFAULT_SPECTRA_CATALOG_SPECS.items() if spec.enabled_by_default
}

# Legacy aliases kept for older notebooks / run configs.
LEGACY_SPECTRA_CATALOG_ALIASES: dict[str, str] = {
    "sdss_dr17_spec": "sdss_dr16_spec",
    "lamost_dr8": "lamost_dr7",
    "lamost_dr9": "lamost_dr7",
    "lamost_dr10": "lamost_dr7",
    "lamost_kepler": "lamost_dr7",
    "lamost_k2": "lamost_dr7",
    "rave_dr5": "rave_dr6",
    "galah_dr4": "galah_dr3",
    "apogee_dr17": "apogee_dr16",
    "fmos_cosmos": "primus",
    "skymapper_spec": "gama",
    "hsc_spectra": "deep2",
}


def resolve_spectra_catalogs(
    catalogs: dict[str, str] | dict[str, SpectraCatalogSpec] | None = None,
) -> dict[str, SpectraCatalogSpec]:
    """Normalize caller catalog overrides into SpectraCatalogSpec entries."""
    if catalogs is None:
        return {
            key: spec
            for key, spec in DEFAULT_SPECTRA_CATALOG_SPECS.items()
            if spec.enabled_by_default
        }

    resolved: dict[str, SpectraCatalogSpec] = {}
    for key, value in catalogs.items():
        canon_key = LEGACY_SPECTRA_CATALOG_ALIASES.get(key, key)
        if isinstance(value, SpectraCatalogSpec):
            resolved[canon_key] = value
            continue
        if canon_key in DEFAULT_SPECTRA_CATALOG_SPECS:
            base = DEFAULT_SPECTRA_CATALOG_SPECS[canon_key]
            resolved[canon_key] = SpectraCatalogSpec(
                vizier_id=str(value),
                mode=base.mode,
                query_group=base.query_group,
                filter_col=base.filter_col,
                filter_values=base.filter_values,
                filter_contains=base.filter_contains,
                tap_table=base.tap_table,
                tap_select=base.tap_select,
                ra_col=base.ra_col,
                dec_col=base.dec_col,
                enabled_by_default=base.enabled_by_default,
            )
        else:
            resolved[canon_key] = SpectraCatalogSpec(vizier_id=str(value))
    return resolved


def grouped_catalog_queries(catalogs: dict[str, SpectraCatalogSpec]) -> dict[str, tuple[str, SpectraCatalogSpec]]:
    """Deduplicate VizieR queries by query_group or vizier_id+mode."""
    groups: dict[str, tuple[str, SpectraCatalogSpec]] = {}
    for survey_key, spec in catalogs.items():
        if spec.query_group:
            group_id = f"{spec.mode}:{spec.query_group}:{spec.vizier_id}"
        else:
            group_id = f"{spec.mode}:{spec.vizier_id}"
        if group_id not in groups:
            groups[group_id] = (survey_key, spec)
    return groups
    for survey_key, spec in catalogs.items():
        if spec.query_group:
            group_id = f"{spec.mode}:{spec.query_group}:{spec.vizier_id}"
        else:
            group_id = f"{spec.mode}:{spec.vizier_id}"
        if group_id not in groups:
            groups[group_id] = (survey_key, spec)
    return groups
