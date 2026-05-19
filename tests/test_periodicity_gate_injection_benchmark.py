from malca.evaluation.periodicity_gate_injection_benchmark import discover_control_paths


def test_discover_control_paths_falls_back_to_dat2(tmp_path):
    dat2_path = tmp_path / "source-1.dat2"
    dat2_path.write_text("", encoding="ascii")
    (tmp_path / "source-1.raw2").write_text("", encoding="ascii")

    assert discover_control_paths(tmp_path) == [dat2_path]


def test_discover_control_paths_respects_explicit_file_ext(tmp_path):
    dat2_path = tmp_path / "source-1.dat2"
    dat3_path = tmp_path / "source-2.dat3"
    dat2_path.write_text("", encoding="ascii")
    dat3_path.write_text("", encoding="ascii")

    assert discover_control_paths(tmp_path, file_ext="dat2") == [dat2_path]

