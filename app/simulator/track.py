from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.simulator.geometry import unit_vector


@dataclass(slots=True)
class Track:
    seed: int
    centerline: np.ndarray
    left_wall: np.ndarray
    right_wall: np.ndarray

    @property
    def checkpoints(self) -> np.ndarray:
        return self.centerline


def _compute_normals(points: np.ndarray) -> np.ndarray:
    tangents = np.zeros_like(points)
    tangents[0] = points[1] - points[0]
    tangents[-1] = points[-1] - points[-2]
    tangents[1:-1] = points[2:] - points[:-2]
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    tangents = tangents / np.clip(norms, 1e-9, None)
    return np.column_stack((-tangents[:, 1], tangents[:, 0]))


def generate_track(
    seed: int,
    *,
    segments: int,
    segment_length: float,
    track_width: float,
) -> Track:
    rng = np.random.default_rng(seed)
    points: list[np.ndarray] = [np.array([0.0, 0.0], dtype=float)]
    heading = 0.0

    for index in range(1, segments):
        bend = rng.normal(0.0, 0.28)
        if index > 4:
            bend += 0.08 * np.sin(index / 3.0)
        heading += float(np.clip(bend, -0.45, 0.45))
        next_point = points[-1] + (unit_vector(heading) * segment_length)
        points.append(next_point.astype(float))

    centerline = np.asarray(points, dtype=float)
    normals = _compute_normals(centerline)
    offset = normals * (track_width / 2.0)
    left_wall = centerline + offset
    right_wall = centerline - offset
    return Track(seed=seed, centerline=centerline, left_wall=left_wall, right_wall=right_wall)

