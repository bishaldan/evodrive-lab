import numpy as np

from app.simulator.track import generate_track


def test_track_generation_is_deterministic() -> None:
    first = generate_track(123, segments=12, segment_length=4.0, track_width=6.0)
    second = generate_track(123, segments=12, segment_length=4.0, track_width=6.0)
    assert np.allclose(first.centerline, second.centerline)
    assert np.allclose(first.left_wall, second.left_wall)
    assert np.allclose(first.right_wall, second.right_wall)

