from __future__ import annotations

import importlib.util
import sys
import types
from urllib.parse import parse_qs, urlparse


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

from malca.review.metadata import build_external_lookup_links
from malca.review.app import _render_vetting_banner
from malca.vsx.nearby import VsxNeighbor


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


def _component_hrefs(node) -> list[str]:
    if node is None or isinstance(node, (str, int, float)):
        return []
    if isinstance(node, (list, tuple)):
        hrefs: list[str] = []
        for child in node:
            hrefs.extend(_component_hrefs(child))
        return hrefs
    href = getattr(node, "href", None)
    hrefs = [str(href)] if href else []
    hrefs.extend(_component_hrefs(getattr(node, "children", None)))
    return hrefs


def test_external_lookup_vsx_link_uses_arcsec_search_with_distance_order() -> None:
    default_url = dict(build_external_lookup_links({"ra": 10.0, "dec": -20.0}))["VSX"]
    assert parse_qs(urlparse(default_url).query)["fieldsize"] == ["30.0"]

    links = build_external_lookup_links({"ra": 10.0, "dec": -20.0}, radius_arcsec=12.5)
    vsx_url = dict(links)["VSX"]
    parsed = urlparse(vsx_url)
    query = parse_qs(parsed.query)

    assert parsed.scheme == "https"
    assert parsed.netloc == "vsx.aavso.org"
    assert parsed.path == "/index.php"
    assert query["view"] == ["results.submit1"]
    assert query["targetcenter"] == ["10.0 -20.0"]
    assert query["format"] == ["d"]
    assert query["fieldsize"] == ["12.5"]
    assert query["fieldunit"] == ["3"]
    assert query["geometry"] == ["r"]
    assert query["order"] == ["9"]
    assert query["filter[]"] == ["0", "1", "2", "3"]


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


def test_vetting_banner_renders_missing_likely_known_with_context_as_new() -> None:
    banner = _render_vetting_banner(
        {
            "candidate_id": "LIKELY-NEW",
            "char_status_yso": "ok",
            "yso_class": "Main Sequence",
        }
    )

    text = " ".join(_component_text(banner))
    classes = _component_classes(banner)

    assert "POTENTIALLY NEW" in text
    assert "Not vetted" not in text
    assert "vetting-banner-shell new" in classes


def test_vetting_banner_keeps_missing_likely_known_without_context_unvetted() -> None:
    banner = _render_vetting_banner({"candidate_id": "RAW"})

    text = " ".join(_component_text(banner))
    classes = _component_classes(banner)

    assert "Not vetted" in text
    assert "POTENTIALLY NEW" not in text
    assert "vetting-banner-empty" in classes


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


def test_vetting_banner_renders_nearby_vsx_as_informational_panel(monkeypatch) -> None:
    import malca.review.app as review_app

    def fake_nearby(ra, dec, *, limit, radius_arcsec):
        assert ra == 10.0
        assert dec == 20.0
        assert limit == 3
        assert radius_arcsec == 12.0
        return [
            VsxNeighbor(
                sep_arcsec=1.23,
                oid="101",
                name="VSX Near",
                ra_deg=10.0,
                dec_deg=20.0,
                vsx_type="ROT",
                type_label="ROT - Rotational variable",
                period_days=2.99288,
                url="https://vsx.aavso.org/index.php?view=detail.top&oid=101",
            ),
            VsxNeighbor(
                sep_arcsec=12.5,
                oid="202",
                name="VSX Mid",
                ra_deg=10.0,
                dec_deg=20.0,
                vsx_type="EA",
                type_label="EA - Algol-type eclipsing binary",
                period_days=None,
                url="https://vsx.aavso.org/index.php?view=detail.top&oid=202",
            ),
        ]

    monkeypatch.setattr(review_app, "find_nearby_vsx", fake_nearby)

    banner = _render_vetting_banner(
        {
            "vetting_likely_known": False,
            "ra": 10.0,
            "dec": 20.0,
        },
        radius_arcsec=12.0,
    )

    text = " ".join(_component_text(banner))
    classes = _component_classes(banner)
    hrefs = _component_hrefs(banner)

    assert "POTENTIALLY NEW" in text
    assert "Nearby VSX" in text
    assert "1.2\"" in text
    assert "VSX Near" in text
    assert "ROT - Rotational variable" in text
    assert "P=2.9929 d" in text
    assert "12.5\"" in text
    assert "VSX Mid" in text
    assert "KNOWN VARIABLE" not in text
    assert "vetting-banner-shell new" in classes
    assert "vetting-banner-cell hit new" not in classes
    assert "https://vsx.aavso.org/index.php?view=detail.top&oid=101" in hrefs
    assert "https://vsx.aavso.org/index.php?view=detail.top&oid=202" in hrefs


def test_vetting_banner_treats_definite_vsx_class_as_known_even_when_summary_is_stale() -> None:
    banner = _render_vetting_banner(
        {
            "vetting_likely_known": False,
            "vsx_class": "EA",
            "vsx_period": 1.7292,
        }
    )

    text = " ".join(_component_text(banner))
    classes = _component_classes(banner)

    assert "KNOWN VARIABLE" in text
    assert "POTENTIALLY NEW" not in text
    assert "VSX" in text
    assert "EA" in text
    assert "vetting-banner-shell known" in classes


def test_vetting_banner_renders_known_for_definite_vsx_class_without_summary_flag() -> None:
    banner = _render_vetting_banner({"vsx_class": "EA"})

    text = " ".join(_component_text(banner))

    assert "KNOWN VARIABLE" in text
    assert "VSX" in text


def test_vetting_banner_keeps_generic_simbad_match_as_context_not_known_hit() -> None:
    banner = _render_vetting_banner(
        {
            "vetting_likely_known": False,
            "simbad_main_id": "IR Source",
            "simbad_otype": "IR",
            "simbad_nbref": 3,
        }
    )

    text = " ".join(_component_text(banner))
    classes = _component_classes(banner)

    assert "POTENTIALLY NEW" in text
    assert "SIMBAD" in text
    assert "IR" in text
    assert "vetting-banner-shell new" in classes
    assert "vetting-banner-cell hit new" not in classes
    assert "vetting-banner-cell hit known" not in classes


def test_vetting_banner_translates_simbad_object_type() -> None:
    banner = _render_vetting_banner(
        {
            "vetting_likely_known": False,
            "simbad_main_id": "WRAY 15-1177",
            "simbad_otype": "Em*",
            "simbad_nbref": 1.0,
        }
    )

    text = " ".join(_component_text(banner))
    classes = _component_classes(banner)

    assert "POTENTIALLY NEW" in text
    assert "Em* - Emission-line star" in text
    assert "WRAY 15-1177" in text
    assert "vetting-banner-shell new" in classes
    assert "vetting-banner-cell hit new" not in classes


def test_vetting_banner_marks_variable_simbad_type_as_known_hit_without_summary_flag() -> None:
    banner = _render_vetting_banner(
        {
            "simbad_main_id": "RR Lyrae",
            "simbad_otype": "RR*",
        }
    )

    text = " ".join(_component_text(banner))
    classes = _component_classes(banner)

    assert "KNOWN VARIABLE" in text
    assert "SIMBAD" in text
    assert "RR*" in text
    assert "vetting-banner-shell known" in classes
    assert "vetting-banner-cell hit known" in classes


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
