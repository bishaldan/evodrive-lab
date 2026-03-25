from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.patches import Polygon


def _car_outline(position: list[float], heading: float, scale: float) -> np.ndarray:
    px, py = position
    forward = np.array([math.cos(heading), math.sin(heading)], dtype=float)
    left = np.array([-math.sin(heading), math.cos(heading)], dtype=float)
    points = [
        np.array([px, py]) + (forward * scale * 1.65),
        np.array([px, py]) + (forward * scale * 0.65) + (left * scale * 0.8),
        np.array([px, py]) - (forward * scale * 0.25) + (left * scale * 0.88),
        np.array([px, py]) - (forward * scale * 1.2) + (left * scale * 0.56),
        np.array([px, py]) - (forward * scale * 1.2) - (left * scale * 0.56),
        np.array([px, py]) - (forward * scale * 0.25) - (left * scale * 0.88),
        np.array([px, py]) + (forward * scale * 0.65) - (left * scale * 0.8),
    ]
    return np.vstack([*points, points[0]])


def _road_polygon(payload: dict[str, object]) -> tuple[np.ndarray, np.ndarray]:
    left = np.asarray(payload["left_wall"], dtype=float)
    right = np.asarray(payload["right_wall"], dtype=float)
    polygon = np.vstack([left, right[::-1], left[:1]])
    return polygon[:, 0], polygon[:, 1]


def _draw_course(ax, payload: dict[str, object]) -> None:
    left = np.asarray(payload["left_wall"], dtype=float)
    right = np.asarray(payload["right_wall"], dtype=float)
    centerline = np.asarray(payload.get("centerline", []), dtype=float)
    road_xs, road_ys = _road_polygon(payload)

    ax.fill(road_xs, road_ys, color="#dad7cd", alpha=0.95, zorder=0)
    ax.plot(left[:, 0], left[:, 1], color="#1f2933", linewidth=2.2, zorder=2)
    ax.plot(right[:, 0], right[:, 1], color="#1f2933", linewidth=2.2, zorder=2)
    if len(centerline):
        ax.plot(centerline[:, 0], centerline[:, 1], color="#f4f1de", linewidth=1.2, linestyle="--", zorder=1)

    for key, color, width in [("checkpoint_gates", "#f4a261", 1.4), ("start_gate", "#2a9d8f", 2.2), ("finish_gate", "#e63946", 2.2)]:
        gates = payload.get(key)
        if not gates:
            continue
        if key in {"start_gate", "finish_gate"}:
            gates = [gates]
        for gate in gates:  # type: ignore[assignment]
            gate_array = np.asarray(gate, dtype=float)
            ax.plot(gate_array[:, 0], gate_array[:, 1], color=color, linewidth=width, zorder=3)


def _set_camera(ax, payload: dict[str, object]) -> None:
    left = np.asarray(payload["left_wall"], dtype=float)
    right = np.asarray(payload["right_wall"], dtype=float)
    points = np.vstack([left, right])
    track_width = float(payload.get("physics", {}).get("track_width", 6.0))
    padding = max(track_width * 1.1, 5.0)
    ax.set_xlim(points[:, 0].min() - padding, points[:, 0].max() + padding)
    ax.set_ylim(points[:, 1].min() - padding, points[:, 1].max() + padding)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("#f8f5ee")


def export_replay_media(
    replay_path: str | Path,
    reports_dir: Path,
    *,
    prefix: str | None = None,
    stride: int = 4,
    max_frames: int = 60,
) -> list[tuple[str, str, dict[str, object]]]:
    replay_file = Path(replay_path)
    if not replay_file.exists():
        return []
    payload = json.loads(replay_file.read_text(encoding="utf-8"))
    frames = payload.get("frames", [])
    if not frames:
        return []

    reports_dir.mkdir(parents=True, exist_ok=True)
    stem = prefix or replay_file.stem
    still_path = reports_dir / f"{stem}_course.png"
    gif_path = reports_dir / f"{stem}_replay.gif"

    car_scale = float(payload.get("physics", {}).get("car_radius", 0.8))
    sampled_frames = frames[:: max(stride, 1)]
    if len(sampled_frames) > max_frames:
        sample_indices = np.linspace(0, len(sampled_frames) - 1, num=max_frames, dtype=int)
        sampled_frames = [sampled_frames[index] for index in sample_indices.tolist()]

    final_frame = frames[-1]
    trajectory = np.asarray([frame["position"] for frame in frames], dtype=float)
    final_outline = _car_outline(final_frame["position"], float(final_frame["heading"]), car_scale)

    figure, axis = plt.subplots(figsize=(8.8, 8.8))
    _draw_course(axis, payload)
    _set_camera(axis, payload)
    axis.plot(trajectory[:, 0], trajectory[:, 1], color="#118ab2", linewidth=2.2, alpha=0.95, zorder=4)
    axis.add_patch(Polygon(final_outline, closed=True, facecolor="#ef476f", edgecolor="#8d1d3f", linewidth=1.6, zorder=5))
    profile = payload.get("track_profile", {})
    axis.set_title(
        f"Track {payload['track_seed']} | {profile.get('difficulty', 'Course')} | "
        f"{profile.get('track_length', '--')} units",
        fontsize=12,
    )
    figure.tight_layout()
    figure.savefig(still_path, dpi=170)
    plt.close(figure)

    anim_figure, anim_axis = plt.subplots(figsize=(8.8, 8.8))
    _draw_course(anim_axis, payload)
    _set_camera(anim_axis, payload)
    trajectory_line, = anim_axis.plot([], [], color="#118ab2", linewidth=2.0, alpha=0.9, zorder=4)
    car_patch = Polygon(_car_outline(sampled_frames[0]["position"], float(sampled_frames[0]["heading"]), car_scale), closed=True)
    car_patch.set_facecolor("#ef476f")
    car_patch.set_edgecolor("#8d1d3f")
    car_patch.set_linewidth(1.5)
    anim_axis.add_patch(car_patch)
    crash_marker = anim_axis.scatter([], [], color="#d62828", s=48, zorder=6)

    def update(frame_index: int):
        frame = sampled_frames[frame_index]
        path = np.asarray([item["position"] for item in sampled_frames[: frame_index + 1]], dtype=float)
        trajectory_line.set_data(path[:, 0], path[:, 1])
        car_patch.set_xy(_car_outline(frame["position"], float(frame["heading"]), car_scale))
        if frame.get("crashed"):
            crash_marker.set_offsets(np.asarray(frame["position"], dtype=float))
        else:
            crash_marker.set_offsets(np.empty((0, 2)))
        anim_axis.set_title(
            f"Step {frame['step']} | Progress {float(frame.get('progress', 0.0)):.2f} | "
            f"Sector {int(frame.get('sector', 0))}",
            fontsize=12,
        )
        return trajectory_line, car_patch, crash_marker

    replay_animation = animation.FuncAnimation(
        anim_figure,
        update,
        frames=len(sampled_frames),
        interval=90,
        blit=False,
    )
    replay_animation.save(gif_path, writer=animation.PillowWriter(fps=12))
    plt.close(anim_figure)

    return [
        ("media", str(still_path), {"source": str(replay_file), "format": "png", "kind": "course_overview"}),
        ("media", str(gif_path), {"source": str(replay_file), "format": "gif", "kind": "animated_replay"}),
    ]
