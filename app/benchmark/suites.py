from __future__ import annotations

TRACK_SUITES: dict[str, list[int]] = {
    "train": [11, 17, 23, 29],
    "validation": [101, 109],
    "test": [211, 223, 227],
}


def get_track_suite(name: str) -> list[int]:
    if name not in TRACK_SUITES:
        raise KeyError(f"Unknown track suite: {name}")
    return TRACK_SUITES[name]

