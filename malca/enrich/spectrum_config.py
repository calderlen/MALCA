from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SpectrumFetchConfig:
    """Credentials and paths for spectrum archive access."""
    eso_username: str | None = None
    eso_password: str | None = None
    tns_api_key: str | None = None
    pessto_csv_path: Path | None = None
    ztf_bts_csv_path: Path | None = None
    clagn_catalog_path: Path | None = None


def load_spectrum_fetch_config(
    *,
    eso_username: str | None = None,
    eso_password: str | None = None,
    tns_api_key: str | None = None,
    pessto_csv_path: str | Path | None = None,
    ztf_bts_csv_path: str | Path | None = None,
    clagn_catalog_path: str | Path | None = None,
) -> SpectrumFetchConfig:
    """Build config from explicit args, falling back to environment variables."""
    return SpectrumFetchConfig(
        eso_username=eso_username or os.environ.get("ESO_USERNAME"),
        eso_password=eso_password or os.environ.get("ESO_PASSWORD"),
        tns_api_key=tns_api_key or os.environ.get("TNS_API_KEY"),
        pessto_csv_path=_resolve_path(pessto_csv_path, "PESSTO_CSV_PATH"),
        ztf_bts_csv_path=_resolve_path(ztf_bts_csv_path, "ZTF_BTS_CSV_PATH"),
        clagn_catalog_path=_resolve_path(clagn_catalog_path, "CLAGN_CATALOG_PATH"),
    )


def _resolve_path(explicit: str | Path | None, env_key: str) -> Path | None:
    value = str(explicit) if explicit else os.environ.get(env_key)
    if not value:
        return None
    p = Path(value).expanduser()
    return p if p.exists() else None
