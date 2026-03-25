from __future__ import annotations

import math

import numpy as np


def unit_vector(angle: float) -> np.ndarray:
    return np.array([math.cos(angle), math.sin(angle)], dtype=float)


def angle_wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _cross(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def ray_segment_intersection(
    origin: np.ndarray,
    direction: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    max_distance: float,
) -> float | None:
    segment = end - start
    denom = _cross(direction, segment)
    if abs(denom) < 1e-9:
        return None
    offset = start - origin
    ray_t = _cross(offset, segment) / denom
    seg_u = _cross(offset, direction) / denom
    if 0.0 <= ray_t <= max_distance and 0.0 <= seg_u <= 1.0:
        return float(ray_t)
    return None


def point_to_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    segment = end - start
    length_sq = float(np.dot(segment, segment))
    if length_sq <= 1e-9:
        return float(np.linalg.norm(point - start))
    t = max(0.0, min(1.0, float(np.dot(point - start, segment) / length_sq)))
    projection = start + (segment * t)
    return float(np.linalg.norm(point - projection))


def project_point_to_polyline(
    point: np.ndarray,
    polyline: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    best_distance = float("inf")
    best_progress = 0.0
    best_projection = polyline[0]
    best_tangent = polyline[1] - polyline[0]
    total_length = 0.0
    length_so_far = 0.0

    for index in range(len(polyline) - 1):
        total_length += float(np.linalg.norm(polyline[index + 1] - polyline[index]))

    for index in range(len(polyline) - 1):
        start = polyline[index]
        end = polyline[index + 1]
        segment = end - start
        segment_length = float(np.linalg.norm(segment))
        if segment_length <= 1e-9:
            continue
        t = max(0.0, min(1.0, float(np.dot(point - start, segment) / (segment_length**2))))
        projection = start + (segment * t)
        distance = float(np.linalg.norm(point - projection))
        if distance < best_distance:
            best_distance = distance
            best_projection = projection
            best_tangent = segment / segment_length
            best_progress = (length_so_far + (segment_length * t)) / max(total_length, 1e-9)
        length_so_far += segment_length

    return best_progress, best_projection, best_tangent


def polyline_to_segments(polyline: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    return [(polyline[i], polyline[i + 1]) for i in range(len(polyline) - 1)]

