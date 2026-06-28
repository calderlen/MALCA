from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from malca import config


def test_raw_lc_roots_are_env_backed_without_local_fallbacks(monkeypatch) -> None:
    with monkeypatch.context() as m:
        m.delenv("MALCA_LCV2_ROOT", raising=False)
        m.delenv("MALCA_LCV2_MASKED_ROOT", raising=False)
        reloaded = importlib.reload(config)

        assert reloaded.LCV2_ROOT is None
        assert reloaded.LCV2_MASKED_ROOT is None
        with pytest.raises(SystemExit, match="MALCA_LCV2_ROOT"):
            reloaded.require_lcv2_root()
        with pytest.raises(SystemExit, match="MALCA_LCV2_MASKED_ROOT"):
            reloaded.require_lcv2_masked_root()

        m.setenv("MALCA_LCV2_ROOT", "/portable/lcsv2")
        m.setenv("MALCA_LCV2_MASKED_ROOT", "/portable/lcsv2_masked")
        assert reloaded.require_lcv2_root() == Path("/portable/lcsv2")
        assert reloaded.require_lcv2_masked_root() == Path("/portable/lcsv2_masked")

    importlib.reload(config)
