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


def _reference_lap_frames(centerline: np.ndarray, *, subdivisions: int = 3) -> list[dict[str, object]]:
    frames: list[dict[str, object]] = []
    if len(centerline) < 2:
        return frames
    step = 1
    for index in range(len(centerline) - 1):
        start = centerline[index]
        end = centerline[index + 1]
        for ratio in np.linspace(0.0, 1.0, subdivisions, endpoint=False):
            position = start + ((end - start) * ratio)
            direction = end - start
            heading = math.atan2(direction[1], direction[0])
            progress = (index + ratio) / max(len(centerline) - 1, 1)
            frames.append(
                {
                    "step": step,
                    "position": position.tolist(),
                    "heading": heading,
                    "progress": progress,
                    "sector": 0,
                    "crashed": False,
                }
            )
            step += 1
    last_direction = centerline[-1] - centerline[-2]
    frames.append(
        {
            "step": step,
            "position": centerline[-1].tolist(),
            "heading": math.atan2(last_direction[1], last_direction[0]),
            "progress": 1.0,
            "sector": 0,
            "crashed": False,
        }
    )
    return frames


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
    reference_gif_path = reports_dir / f"{stem}_reference_lap.gif"

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

    reference_frames = _reference_lap_frames(np.asarray(payload["centerline"], dtype=float), subdivisions=4)
    ref_figure, ref_axis = plt.subplots(figsize=(8.8, 8.8))
    _draw_course(ref_axis, payload)
    _set_camera(ref_axis, payload)
    ref_line, = ref_axis.plot([], [], color="#2a9d8f", linewidth=2.2, alpha=0.95, zorder=4)
    ref_patch = Polygon(
        _car_outline(reference_frames[0]["position"], float(reference_frames[0]["heading"]), car_scale),
        closed=True,
    )
    ref_patch.set_facecolor("#2a9d8f")
    ref_patch.set_edgecolor("#155e63")
    ref_patch.set_linewidth(1.5)
    ref_axis.add_patch(ref_patch)

    def update_reference(frame_index: int):
        frame = reference_frames[frame_index]
        path = np.asarray([item["position"] for item in reference_frames[: frame_index + 1]], dtype=float)
        ref_line.set_data(path[:, 0], path[:, 1])
        ref_patch.set_xy(_car_outline(frame["position"], float(frame["heading"]), car_scale))
        ref_axis.set_title(f"Reference lap | Progress {float(frame['progress']):.2f}", fontsize=12)
        return ref_line, ref_patch

    reference_animation = animation.FuncAnimation(
        ref_figure,
        update_reference,
        frames=len(reference_frames),
        interval=85,
        blit=False,
    )
    reference_animation.save(reference_gif_path, writer=animation.PillowWriter(fps=12))
    plt.close(ref_figure)

    return [
        ("media", str(still_path), {"source": str(replay_file), "format": "png", "kind": "course_overview"}),
        ("media", str(gif_path), {"source": str(replay_file), "format": "gif", "kind": "animated_replay"}),
        ("media", str(reference_gif_path), {"source": str(replay_file), "format": "gif", "kind": "reference_lap"}),
    ]


def export_population_media(
    live_path: str | Path,
    reports_dir: Path,
    *,
    prefix: str | None = None,
    stride: int = 3,
    max_frames: int = 84,
) -> list[tuple[str, str, dict[str, object]]]:
    live_file = Path(live_path)
    if not live_file.exists():
        return []
    payload = json.loads(live_file.read_text(encoding="utf-8"))
    cars = payload.get("cars", [])
    if not cars:
        return []

    reports_dir.mkdir(parents=True, exist_ok=True)
    stem = prefix or live_file.stem
    gif_path = reports_dir / f"{stem}_population.gif"

    max_step = max((len(car.get("frames", [])) for car in cars), default=0)
    if max_step == 0:
        return []
    base_step_values = list(range(1, max_step + 1, max(stride, 1)))
    pause_frames = 5
    core_frame_budget = max(max_frames - (pause_frames * 2), 12)
    if len(base_step_values) > core_frame_budget:
        sample_indices = np.linspace(0, len(base_step_values) - 1, num=core_frame_budget, dtype=int)
        base_step_values = [base_step_values[index] for index in sample_indices.tolist()]
    elif base_step_values[-1] != max_step:
        base_step_values.append(max_step)
    step_values = ([base_step_values[0]] * pause_frames) + base_step_values + ([base_step_values[-1]] * pause_frames)

    car_scale = float(payload.get("physics", {}).get("car_radius", 0.8))

    figure, axis = plt.subplots(figsize=(9.4, 9.4))
    _draw_course(axis, payload)
    _set_camera(axis, payload)

    trajectory_lines = []
    car_patches = []
    car_labels = []
    for car in cars:
        color = car.get("color", "#118ab2")
        (line,) = axis.plot([], [], color=color, linewidth=2.1, alpha=0.85, zorder=4)
        patch = Polygon(_car_outline(car["frames"][0]["position"], float(car["frames"][0]["heading"]), car_scale), closed=True)
        patch.set_facecolor(color)
        patch.set_edgecolor("#23313f")
        patch.set_linewidth(1.1)
        axis.add_patch(patch)
        first_position = np.asarray(car["frames"][0]["position"], dtype=float)
        label = axis.text(
            first_position[0],
            first_position[1],
            str(car.get("rank", len(car_labels) + 1)),
            color="#ffffff",
            fontsize=7.5,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=6,
            bbox={
                "boxstyle": "circle,pad=0.28",
                "facecolor": color,
                "edgecolor": "#23313f",
                "linewidth": 0.9,
                "alpha": 0.96,
            },
        )
        trajectory_lines.append(line)
        car_patches.append(patch)
        car_labels.append(label)

    total_cars = len(cars)

    def update_population(frame_index: int):
        step = step_values[frame_index]
        for idx, car in enumerate(cars):
            frames = car.get("frames", [])
            if not frames:
                continue
            position_index = min(max(step, 1), len(frames)) - 1
            current = frames[position_index]
            path = np.asarray([frame["position"] for frame in frames[: position_index + 1]], dtype=float)
            trajectory_lines[idx].set_data(path[:, 0], path[:, 1])
            body_color = "#d62828" if current.get("crashed") else car.get("color", "#118ab2")
            trajectory_lines[idx].set_color(body_color)
            car_patches[idx].set_facecolor(body_color)
            heading = float(current["heading"])
            position = np.asarray(current["position"], dtype=float)
            car_patches[idx].set_xy(_car_outline(current["position"], heading, car_scale))
            label_offset = np.array([-math.sin(heading), math.cos(heading)], dtype=float) * car_scale * 0.45
            car_labels[idx].set_position((position[0] + label_offset[0], position[1] + label_offset[1]))
        axis.set_title(
            f"{total_cars}-car population replay | Generation {payload.get('generation', 0)} | Step {step}",
            fontsize=12,
        )
        return [*trajectory_lines, *car_patches, *car_labels]

    population_animation = animation.FuncAnimation(
        figure,
        update_population,
        frames=len(step_values),
        interval=95,
        blit=False,
    )
    population_animation.save(gif_path, writer=animation.PillowWriter(fps=10))
    plt.close(figure)

    return [
        (
            "media",
            str(gif_path),
            {"source": str(live_file), "format": "gif", "kind": "population_replay"},
        )
    ]
