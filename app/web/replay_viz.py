from __future__ import annotations

import math
from typing import Any

import plotly.graph_objects as go


def _car_outline(position: list[float], heading: float, scale: float) -> tuple[list[float], list[float]]:
    px, py = position
    forward = (math.cos(heading), math.sin(heading))
    left = (-math.sin(heading), math.cos(heading))
    nose = (px + (forward[0] * scale * 1.65), py + (forward[1] * scale * 1.65))
    front_left = (
        px + (forward[0] * scale * 0.65) + (left[0] * scale * 0.8),
        py + (forward[1] * scale * 0.65) + (left[1] * scale * 0.8),
    )
    mid_left = (
        px - (forward[0] * scale * 0.25) + (left[0] * scale * 0.88),
        py - (forward[1] * scale * 0.25) + (left[1] * scale * 0.88),
    )
    rear_left = (
        px - (forward[0] * scale * 1.2) + (left[0] * scale * 0.56),
        py - (forward[1] * scale * 1.2) + (left[1] * scale * 0.56),
    )
    rear_right = (
        px - (forward[0] * scale * 1.2) - (left[0] * scale * 0.56),
        py - (forward[1] * scale * 1.2) - (left[1] * scale * 0.56),
    )
    mid_right = (
        px - (forward[0] * scale * 0.25) - (left[0] * scale * 0.88),
        py - (forward[1] * scale * 0.25) - (left[1] * scale * 0.88),
    )
    front_right = (
        px + (forward[0] * scale * 0.65) - (left[0] * scale * 0.8),
        py + (forward[1] * scale * 0.65) - (left[1] * scale * 0.8),
    )
    xs = [nose[0], front_left[0], mid_left[0], rear_left[0], rear_right[0], mid_right[0], front_right[0], nose[0]]
    ys = [nose[1], front_left[1], mid_left[1], rear_left[1], rear_right[1], mid_right[1], front_right[1], nose[1]]
    return xs, ys


def _sensor_traces(frame: dict[str, Any], physics: dict[str, Any]) -> tuple[list[float], list[float]]:
    distances = frame.get("sensor_distances")
    if not distances:
        return [], []
    sensor_range = float(physics.get("sensor_range", 1.0))
    spread = math.radians(float(physics.get("ray_spread_deg", 90.0)))
    count = max(len(distances), 1)
    angles = [(-spread / 2.0) + (spread * idx / max(count - 1, 1)) for idx in range(count)]
    px, py = frame["position"]
    xs: list[float] = []
    ys: list[float] = []
    for offset, normalized in zip(angles, distances, strict=True):
        ray_length = float(normalized) * sensor_range
        theta = float(frame["heading"]) + offset
        xs.extend([px, px + (math.cos(theta) * ray_length), None])
        ys.extend([py, py + (math.sin(theta) * ray_length), None])
    return xs, ys


def _with_alpha(hex_color: str, alpha: float) -> str:
    color = hex_color.lstrip("#")
    red = int(color[0:2], 16)
    green = int(color[2:4], 16)
    blue = int(color[4:6], 16)
    return f"rgba({red}, {green}, {blue}, {alpha})"


def _current_frame(car: dict[str, Any], step: int) -> dict[str, Any] | None:
    frames = car.get("frames", [])
    if not frames:
        return None
    index = min(max(step, 1), len(frames)) - 1
    return frames[index]


def _trajectory_until(car: dict[str, Any], step: int) -> tuple[list[float], list[float]]:
    frames = car.get("frames", [])
    if not frames:
        return [], []
    index = min(max(step, 1), len(frames))
    trajectory = frames[:index]
    return [frame["position"][0] for frame in trajectory], [frame["position"][1] for frame in trajectory]


def _road_surface(left_wall: list[list[float]], right_wall: list[list[float]]) -> tuple[list[float], list[float]]:
    xs = [point[0] for point in left_wall] + [point[0] for point in reversed(right_wall)] + [left_wall[0][0]]
    ys = [point[1] for point in left_wall] + [point[1] for point in reversed(right_wall)] + [left_wall[0][1]]
    return xs, ys


def _gate_trace(
    gates: list[list[list[float]]] | list[list[float]] | None,
    *,
    color: str,
    name: str,
    width: int = 3,
    dash: str | None = None,
) -> go.Scatter | None:
    if not gates:
        return None
    if len(gates) == 2 and isinstance(gates[0][0], (int, float)):
        gates = [gates]  # type: ignore[list-item]
    xs: list[float | None] = []
    ys: list[float | None] = []
    for gate in gates:  # type: ignore[assignment]
        xs.extend([gate[0][0], gate[1][0], None])
        ys.extend([gate[0][1], gate[1][1], None])
    return go.Scatter(
        x=xs,
        y=ys,
        mode="lines",
        line={"color": color, "width": width, "dash": dash},
        name=name,
    )


def build_replay_figure(payload: dict[str, Any], frame_stride: int = 6) -> go.Figure:
    left_wall = payload["left_wall"]
    right_wall = payload["right_wall"]
    centerline = payload.get("centerline", [])
    start_gate = payload.get("start_gate")
    finish_gate = payload.get("finish_gate")
    checkpoint_gates = payload.get("checkpoint_gates", [])
    track_profile = payload.get("track_profile", {})
    frames = payload["frames"][:: max(frame_stride, 1)]
    physics = payload.get("physics", {})
    car_scale = float(physics.get("car_radius", 0.8))

    figure = go.Figure()
    road_xs, road_ys = _road_surface(left_wall, right_wall)
    figure.add_trace(
        go.Scatter(
            x=road_xs,
            y=road_ys,
            mode="lines",
            fill="toself",
            fillcolor="rgba(90, 90, 90, 0.15)",
            line={"color": "rgba(90, 90, 90, 0.15)", "width": 1},
            name="Road",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[point[0] for point in left_wall],
            y=[point[1] for point in left_wall],
            mode="lines",
            line={"color": "#263238", "width": 4},
            name="Left wall",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[point[0] for point in right_wall],
            y=[point[1] for point in right_wall],
            mode="lines",
            line={"color": "#263238", "width": 4},
            name="Right wall",
        )
    )
    if centerline:
        figure.add_trace(
            go.Scatter(
                x=[point[0] for point in centerline],
                y=[point[1] for point in centerline],
                mode="lines",
                line={"color": "#90A4AE", "width": 2, "dash": "dot"},
                name="Centerline",
            )
        )
    checkpoint_trace = _gate_trace(checkpoint_gates, color="#F4A261", name="Checkpoints", width=2, dash="dot")
    if checkpoint_trace is not None:
        figure.add_trace(checkpoint_trace)
    start_trace = _gate_trace(start_gate, color="#2A9D8F", name="Start", width=4)
    if start_trace is not None:
        figure.add_trace(start_trace)
    finish_trace = _gate_trace(finish_gate, color="#E63946", name="Finish", width=4, dash="dash")
    if finish_trace is not None:
        figure.add_trace(finish_trace)

    initial = frames[0]
    car_xs, car_ys = _car_outline(initial["position"], float(initial["heading"]), car_scale)
    sensor_xs, sensor_ys = _sensor_traces(initial, physics)
    trajectory_xs = [frame["position"][0] for frame in frames[:1]]
    trajectory_ys = [frame["position"][1] for frame in frames[:1]]
    dynamic_trace_offset = len(figure.data)

    figure.add_trace(
        go.Scatter(
            x=trajectory_xs,
            y=trajectory_ys,
            mode="lines",
            line={"color": "#00A8E8", "width": 3},
            name="Trajectory",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=sensor_xs,
            y=sensor_ys,
            mode="lines",
            line={"color": "#FFB703", "width": 2},
            name="Sensors",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=car_xs,
            y=car_ys,
            mode="lines",
            fill="toself",
            fillcolor="rgba(239, 71, 111, 0.75)",
            line={"color": "#D90429", "width": 2},
            name="Car",
        )
    )

    animation_frames: list[go.Frame] = []
    for index, frame in enumerate(frames):
        car_xs, car_ys = _car_outline(frame["position"], float(frame["heading"]), car_scale)
        sensor_xs, sensor_ys = _sensor_traces(frame, physics)
        trajectory_xs = [item["position"][0] for item in frames[: index + 1]]
        trajectory_ys = [item["position"][1] for item in frames[: index + 1]]
        animation_frames.append(
            go.Frame(
                name=str(frame["step"]),
                data=[
                    go.Scatter(x=trajectory_xs, y=trajectory_ys),
                    go.Scatter(x=sensor_xs, y=sensor_ys),
                    go.Scatter(x=car_xs, y=car_ys),
                ],
                traces=[dynamic_trace_offset, dynamic_trace_offset + 1, dynamic_trace_offset + 2],
            )
        )

    all_x = [point[0] for point in left_wall] + [point[0] for point in right_wall]
    all_y = [point[1] for point in left_wall] + [point[1] for point in right_wall]
    padding = max(float(physics.get("track_width", 6.0)), 4.0)

    figure.frames = animation_frames
    figure.update_layout(
        title=(
            f"Replay on track seed {payload['track_seed']}"
            f" | {track_profile.get('difficulty', 'Course')}"
            f" | {track_profile.get('corner_count', 0)} corners"
        ),
        template="plotly_white",
        height=680,
        xaxis={
            "range": [min(all_x) - padding, max(all_x) + padding],
            "showgrid": False,
            "zeroline": False,
        },
        yaxis={
            "range": [min(all_y) - padding, max(all_y) + padding],
            "showgrid": False,
            "zeroline": False,
            "scaleanchor": "x",
            "scaleratio": 1,
        },
        legend={"orientation": "h", "y": 1.04},
        updatemenus=[
            {
                "type": "buttons",
                "showactive": True,
                "x": 0.0,
                "y": 1.12,
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 80, "redraw": True},
                                "transition": {"duration": 0},
                                "fromcurrent": True,
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "currentvalue": {"prefix": "Step: "},
                "pad": {"t": 36},
                "steps": [
                    {
                        "label": str(frame["step"]),
                        "method": "animate",
                        "args": [
                            [str(frame["step"])],
                            {
                                "mode": "immediate",
                                "frame": {"duration": 0, "redraw": True},
                                "transition": {"duration": 0},
                            },
                        ],
                    }
                    for frame in frames
                ],
            }
        ],
    )
    return figure


def build_population_figure(payload: dict[str, Any], frame_stride: int = 4) -> go.Figure:
    left_wall = payload["left_wall"]
    right_wall = payload["right_wall"]
    centerline = payload.get("centerline", [])
    start_gate = payload.get("start_gate")
    finish_gate = payload.get("finish_gate")
    checkpoint_gates = payload.get("checkpoint_gates", [])
    track_profile = payload.get("track_profile", {})
    cars = payload.get("cars", [])
    physics = payload.get("physics", {})
    if not cars:
        return go.Figure()

    max_step = max((len(car.get("frames", [])) for car in cars), default=0)
    if max_step == 0:
        return go.Figure()
    step_values = list(range(1, max_step + 1, max(frame_stride, 1)))
    if step_values[-1] != max_step:
        step_values.append(max_step)
    car_scale = float(physics.get("car_radius", 0.8))

    figure = go.Figure()
    road_xs, road_ys = _road_surface(left_wall, right_wall)
    figure.add_trace(
        go.Scatter(
            x=road_xs,
            y=road_ys,
            mode="lines",
            fill="toself",
            fillcolor="rgba(90, 90, 90, 0.15)",
            line={"color": "rgba(90, 90, 90, 0.15)", "width": 1},
            name="Road",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[point[0] for point in left_wall],
            y=[point[1] for point in left_wall],
            mode="lines",
            line={"color": "#263238", "width": 4},
            name="Left wall",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[point[0] for point in right_wall],
            y=[point[1] for point in right_wall],
            mode="lines",
            line={"color": "#263238", "width": 4},
            name="Right wall",
        )
    )
    if centerline:
        figure.add_trace(
            go.Scatter(
                x=[point[0] for point in centerline],
                y=[point[1] for point in centerline],
                mode="lines",
                line={"color": "#90A4AE", "width": 2, "dash": "dot"},
                name="Centerline",
            )
        )
    checkpoint_trace = _gate_trace(checkpoint_gates, color="#F4A261", name="Checkpoints", width=2, dash="dot")
    if checkpoint_trace is not None:
        figure.add_trace(checkpoint_trace)
    start_trace = _gate_trace(start_gate, color="#2A9D8F", name="Start", width=4)
    if start_trace is not None:
        figure.add_trace(start_trace)
    finish_trace = _gate_trace(finish_gate, color="#E63946", name="Finish", width=4, dash="dash")
    if finish_trace is not None:
        figure.add_trace(finish_trace)

    trace_indexes: list[int] = []
    initial_step = step_values[0]
    for car in cars:
        color = car.get("color", "#00A8E8")
        current = _current_frame(car, initial_step)
        trajectory_xs, trajectory_ys = _trajectory_until(car, initial_step)
        if current is None:
            car_xs, car_ys = [], []
        else:
            car_xs, car_ys = _car_outline(current["position"], float(current["heading"]), car_scale)
        trace_indexes.extend([len(figure.data), len(figure.data) + 1])
        figure.add_trace(
            go.Scatter(
                x=trajectory_xs,
                y=trajectory_ys,
                mode="lines",
                line={"color": color, "width": 2},
                opacity=0.55,
                name=f"Car {car['rank']} path",
                showlegend=False,
            )
        )
        figure.add_trace(
            go.Scatter(
                x=car_xs,
                y=car_ys,
                mode="lines",
                fill="toself",
                fillcolor=_with_alpha(color, 0.8),
                line={"color": color, "width": 2},
                name=f"Car {car['rank']}",
            )
        )

    animation_frames: list[go.Frame] = []
    for step in step_values:
        frame_data: list[go.Scatter] = []
        for car in cars:
            color = car.get("color", "#00A8E8")
            current = _current_frame(car, step)
            trajectory_xs, trajectory_ys = _trajectory_until(car, step)
            crashed = bool(current.get("crashed")) if current else False
            if current is None:
                car_xs, car_ys = [], []
            else:
                car_xs, car_ys = _car_outline(current["position"], float(current["heading"]), car_scale)
            body_color = "#D90429" if crashed else color
            frame_data.append(
                go.Scatter(
                    x=trajectory_xs,
                    y=trajectory_ys,
                    mode="lines",
                    line={"color": body_color, "width": 2},
                    opacity=0.55,
                )
            )
            frame_data.append(
                go.Scatter(
                    x=car_xs,
                    y=car_ys,
                    mode="lines",
                    fill="toself",
                    fillcolor=_with_alpha(body_color, 0.8),
                    line={"color": body_color, "width": 2},
                )
            )
        animation_frames.append(go.Frame(name=str(step), data=frame_data, traces=trace_indexes))

    all_x = [point[0] for point in left_wall] + [point[0] for point in right_wall]
    all_y = [point[1] for point in left_wall] + [point[1] for point in right_wall]
    padding = max(float(physics.get("track_width", 6.0)), 4.0)

    figure.frames = animation_frames
    figure.update_layout(
        title=(
            f"Generation {payload.get('generation', 0)} population replay"
            f" | {track_profile.get('difficulty', 'Course')}"
            f" | {track_profile.get('checkpoint_count', 0)} checkpoints"
        ),
        template="plotly_white",
        height=720,
        xaxis={
            "range": [min(all_x) - padding, max(all_x) + padding],
            "showgrid": False,
            "zeroline": False,
        },
        yaxis={
            "range": [min(all_y) - padding, max(all_y) + padding],
            "showgrid": False,
            "zeroline": False,
            "scaleanchor": "x",
            "scaleratio": 1,
        },
        legend={"orientation": "h", "y": 1.04},
        updatemenus=[
            {
                "type": "buttons",
                "showactive": True,
                "x": 0.0,
                "y": 1.12,
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 80, "redraw": True},
                                "transition": {"duration": 0},
                                "fromcurrent": True,
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "currentvalue": {"prefix": "Step: "},
                "pad": {"t": 36},
                "steps": [
                    {
                        "label": str(step),
                        "method": "animate",
                        "args": [
                            [str(step)],
                            {
                                "mode": "immediate",
                                "frame": {"duration": 0, "redraw": True},
                                "transition": {"duration": 0},
                            },
                        ],
                    }
                    for step in step_values
                ],
            }
        ],
    )
    return figure
