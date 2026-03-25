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

    @property
    def start_gate(self) -> np.ndarray:
        return np.asarray([self.left_wall[0], self.right_wall[0]], dtype=float)

    @property
    def finish_gate(self) -> np.ndarray:
        return np.asarray([self.left_wall[-1], self.right_wall[-1]], dtype=float)

    @property
    def checkpoint_indices(self) -> list[int]:
        if len(self.centerline) <= 4:
            return []
        gate_count = min(5, max(3, len(self.centerline) // 6))
        indices = np.linspace(2, len(self.centerline) - 3, num=gate_count, dtype=int)
        return sorted({int(index) for index in indices.tolist()})

    @property
    def checkpoint_progress(self) -> list[float]:
        if len(self.centerline) <= 1:
            return []
        max_index = max(len(self.centerline) - 1, 1)
        return [index / max_index for index in self.checkpoint_indices]

    @property
    def checkpoint_gates(self) -> np.ndarray:
        gates = [
            np.asarray([self.left_wall[index], self.right_wall[index]], dtype=float)
            for index in self.checkpoint_indices
        ]
        return np.asarray(gates, dtype=float) if gates else np.empty((0, 2, 2), dtype=float)

    @property
    def profile(self) -> dict[str, float | int | str]:
        segments = np.diff(self.centerline, axis=0)
        lengths = np.linalg.norm(segments, axis=1)
        headings = np.unwrap(np.arctan2(segments[:, 1], segments[:, 0]))
        heading_changes = np.abs(np.diff(headings))
        corner_count = int(np.sum(heading_changes > 0.2))
        mean_turn = float(np.mean(heading_changes)) if len(heading_changes) else 0.0
        max_turn = float(np.max(heading_changes)) if len(heading_changes) else 0.0
        track_length = float(np.sum(lengths))
        longest_straight = float(np.max(lengths)) if len(lengths) else 0.0
        difficulty_score = (corner_count * 0.65) + (mean_turn * 16.0) + (max_turn * 6.0)
        if difficulty_score < 3.0:
            difficulty = "Rookie"
        elif difficulty_score < 5.5:
            difficulty = "Sport"
        else:
            difficulty = "Expert"
        return {
            "track_length": round(track_length, 2),
            "corner_count": corner_count,
            "checkpoint_count": len(self.checkpoint_indices),
            "mean_turn": round(mean_turn, 4),
            "max_turn": round(max_turn, 4),
            "longest_straight": round(longest_straight, 2),
            "difficulty": difficulty,
        }


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
