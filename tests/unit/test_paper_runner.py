from app.paper_tools.runner import build_run_matrix


def test_main_matrix_has_expected_run_count() -> None:
    matrix = build_run_matrix("main")
    assert len(matrix) == 15
    assert matrix[0].name.startswith("paper-main-")


def test_sensor_ablation_matrix_has_expected_run_count() -> None:
    matrix = build_run_matrix("sensors")
    assert len(matrix) == 30
    names = {config.name for config in matrix}
    assert any(name.endswith("sensor5") for name in names)
    assert any(name.endswith("sensor7") for name in names)


def test_track_ablation_matrix_has_expected_run_count() -> None:
    matrix = build_run_matrix("tracks")
    assert len(matrix) == 30
    names = {config.name for config in matrix}
    assert any(name.endswith("track-standard") for name in names)
    assert any(name.endswith("track-long") for name in names)
