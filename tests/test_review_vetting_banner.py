from __future__ import annotations

import importlib.util
import sys
import types


def _install_review_app_import_stubs() -> None:
    if "celerite2" not in sys.modules and importlib.util.find_spec("celerite2") is None:
        fake_celerite2 = types.ModuleType("celerite2")
        fake_terms = types.ModuleType("celerite2.terms")

        class _FakeGaussianProcess:
            def __init__(self, *args, **kwargs):
                pass

        class _FakeTerm:
            def __init__(self, *args, **kwargs):
                pass

            def __add__(self, other):
                return self

        fake_terms.SHOTerm = _FakeTerm
        fake_terms.RealTerm = _FakeTerm
        fake_celerite2.GaussianProcess = _FakeGaussianProcess
        fake_celerite2.terms = fake_terms
        sys.modules["celerite2"] = fake_celerite2
        sys.modules["celerite2.terms"] = fake_terms

    if "multiprocess" not in sys.modules and importlib.util.find_spec("multiprocess") is None:
        fake_multiprocess = types.ModuleType("multiprocess")
        fake_multiprocess.get_all_start_methods = lambda: ["spawn"]
        fake_multiprocess.set_start_method = lambda *args, **kwargs: None
        sys.modules["multiprocess"] = fake_multiprocess


_install_review_app_import_stubs()

from malca.review.app import _render_vetting_banner


def _component_text(node) -> list[str]:
    if node is None:
        return []
    if isinstance(node, (str, int, float)):
        return [str(node)]
    if isinstance(node, (list, tuple)):
        texts: list[str] = []
        for child in node:
            texts.extend(_component_text(child))
        return texts
    return _component_text(getattr(node, "children", None))


def _component_classes(node) -> list[str]:
    if node is None or isinstance(node, (str, int, float)):
        return []
    if isinstance(node, (list, tuple)):
        classes: list[str] = []
        for child in node:
            classes.extend(_component_classes(child))
        return classes
    class_name = getattr(node, "className", None)
    classes = [str(class_name)] if class_name else []
    classes.extend(_component_classes(getattr(node, "children", None)))
    return classes


def test_vetting_banner_does_not_mark_gaia_flag_only_as_known() -> None:
    banner = _render_vetting_banner(
        {
            "vetting_likely_known": False,
            "gaia_var_flag": True,
            "gaia_var_class": "",
        }
    )

    text = " ".join(_component_text(banner))

    assert "POTENTIALLY NEW" in text
    assert "Gaia DR3" not in text


def test_vetting_banner_marks_known_catalog_cells_as_red_hit_boxes() -> None:
    banner = _render_vetting_banner(
        {
            "vetting_likely_known": True,
            "asassn_var_type": "EA",
            "ztf_var_type": "EA",
        }
    )

    classes = _component_classes(banner)

    assert "vetting-banner-shell known" in classes
    assert classes.count("vetting-banner-cell hit known") == 2


def test_vetting_banner_displays_vsx_class() -> None:
    banner = _render_vetting_banner(
        {
            "vetting_likely_known": True,
            "vsx_class": "GCAS",
        }
    )

    text = " ".join(_component_text(banner))

    assert "VSX" in text
    assert "GCAS" in text


def test_vetting_banner_marks_gaia_class_as_known_without_summary_flag() -> None:
    banner = _render_vetting_banner(
        {
            "gaia_var_flag": False,
            "gaia_var_class": "LPV",
            "gaia_var_score": 0.91,
        }
    )

    text = " ".join(_component_text(banner))

    assert "KNOWN VARIABLE" in text
    assert "Gaia DR3" in text
    assert "LPV" in text
