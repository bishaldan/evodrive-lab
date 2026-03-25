from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from app.config.models import EnvironmentConfig
from app.simulator.geometry import (
    angle_wrap,
    point_to_segment_distance,
    polyline_to_segments,
    project_point_to_polyline,
    ray_segment_intersection,
    unit_vector,
)
from app.simulator.track import Track, generate_track

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:  # pragma: no cover - import guard
    gym = None
    spaces = None

try:
    from Box2D import b2World
except Exception:  # pragma: no cover - import guard
    b2World = None


BaseEnv = gym.Env if gym is not None else object


class DrivingEnv(BaseEnv):
    metadata = {"render_modes": ["json"]}

    def __init__(self, config: EnvironmentConfig, *, track_seed: int = 0) -> None:
        self.config = config
        self.physics = config.physics
        self.reward = config.reward
        self.track_seed = track_seed
        self.track: Track | None = None
        self.wall_segments: list[tuple[np.ndarray, np.ndarray]] = []
        self.current_step = 0
        self.progress = 0.0
        self.trajectory: list[dict[str, Any]] = []
        self.position = np.zeros(2, dtype=float)
        self.heading = 0.0
        self.speed = 0.0
        self.angular_velocity = 0.0
        self.last_action = np.zeros(2, dtype=float)
        self.box2d_world = None
        self.box2d_body = None

        obs_size = self.physics.num_rays + 4
        if spaces is not None:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
            self.observation_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(obs_size,),
                dtype=np.float32,
            )

    def _build_track(self, track_seed: int) -> None:
        self.track = generate_track(
            track_seed,
            segments=self.physics.max_track_segments,
            segment_length=self.physics.segment_length,
            track_width=self.physics.track_width,
        )
        self.wall_segments = polyline_to_segments(self.track.left_wall)
        self.wall_segments.extend(polyline_to_segments(self.track.right_wall))

    def _init_box2d(self) -> None:
        if not self.physics.use_box2d or b2World is None or self.track is None:
            self.box2d_world = None
            self.box2d_body = None
            return
        self.box2d_world = b2World(gravity=(0, 0), doSleep=True)
        for start, end in self.wall_segments:
            wall = self.box2d_world.CreateStaticBody()
            wall.CreateEdgeFixture(vertices=[tuple(start), tuple(end)], density=0.0, friction=0.3)
        self.box2d_body = self.box2d_world.CreateDynamicBody(
            position=tuple(self.position),
            angle=self.heading,
            linearDamping=0.15,
            angularDamping=0.25,
        )
        self.box2d_body.CreateCircleFixture(radius=self.physics.car_radius * 0.8, density=1.0, friction=0.2)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        if options and "track_seed" in options:
            self.track_seed = int(options["track_seed"])
        elif seed is not None:
            self.track_seed = int(seed)
        self._build_track(self.track_seed)
        assert self.track is not None
        self.position = self.track.centerline[0].copy()
        direction = self.track.centerline[1] - self.track.centerline[0]
        self.heading = float(math.atan2(direction[1], direction[0]))
        self.speed = 0.0
        self.angular_velocity = 0.0
        self.progress = 0.0
        self.current_step = 0
        self.last_action = np.zeros(2, dtype=float)
        self.trajectory = []
        self._init_box2d()
        observation = self._get_observation()
        return observation, {"track_seed": self.track_seed}

    def _step_with_fallback(self, throttle: float, steering: float) -> None:
        self.speed += throttle * self.physics.acceleration * self.physics.dt
        self.speed *= self.physics.linear_damping
        self.speed = float(np.clip(self.speed, -self.physics.max_speed * 0.25, self.physics.max_speed))
        self.angular_velocity = steering * self.physics.turn_rate * (0.2 + abs(self.speed / self.physics.max_speed))
        self.heading = angle_wrap(self.heading + (self.angular_velocity * self.physics.dt))
        self.position = self.position + (unit_vector(self.heading) * self.speed * self.physics.dt)

    def _step_with_box2d(self, throttle: float, steering: float) -> None:
        if self.box2d_world is None or self.box2d_body is None:
            self._step_with_fallback(throttle, steering)
            return
        body = self.box2d_body
        forward = unit_vector(body.angle)
        force = forward * throttle * self.physics.acceleration * 15.0
        body.ApplyForceToCenter((float(force[0]), float(force[1])), wake=True)
        body.ApplyTorque(float(steering * self.physics.turn_rate * 12.0), wake=True)
        self.box2d_world.Step(self.physics.dt, 6, 2)
        velocity = np.array([body.linearVelocity[0], body.linearVelocity[1]], dtype=float)
        speed = float(np.linalg.norm(velocity))
        if speed > self.physics.max_speed:
            velocity = (velocity / speed) * self.physics.max_speed
            body.linearVelocity = (float(velocity[0]), float(velocity[1]))
            speed = self.physics.max_speed
        self.position = np.array([body.position[0], body.position[1]], dtype=float)
        self.heading = float(body.angle)
        self.speed = speed
        self.angular_velocity = float(body.angularVelocity)

    def _wall_distance(self) -> float:
        if not self.wall_segments:
            return float("inf")
        return min(
            point_to_segment_distance(self.position, start, end) for start, end in self.wall_segments
        )

    def _sensor_distances(self) -> np.ndarray:
        if self.track is None:
            return np.ones(self.physics.num_rays, dtype=float)
        distances = np.full(self.physics.num_rays, self.physics.sensor_range, dtype=float)
        offsets = np.linspace(
            -math.radians(self.physics.ray_spread_deg) / 2.0,
            math.radians(self.physics.ray_spread_deg) / 2.0,
            self.physics.num_rays,
        )
        for ray_index, offset in enumerate(offsets):
            direction = unit_vector(self.heading + float(offset))
            for start, end in self.wall_segments:
                hit = ray_segment_intersection(
                    self.position,
                    direction,
                    start,
                    end,
                    self.physics.sensor_range,
                )
                if hit is not None:
                    distances[ray_index] = min(distances[ray_index], hit)
        return distances / self.physics.sensor_range

    def _get_observation(self) -> np.ndarray:
        assert self.track is not None
        sensors = self._sensor_distances()
        progress, _, tangent = project_point_to_polyline(self.position, self.track.centerline)
        heading_error = angle_wrap(float(math.atan2(tangent[1], tangent[0])) - self.heading)
        return np.asarray(
            [
                *sensors.tolist(),
                np.clip(self.speed / self.physics.max_speed, -1.0, 1.0),
                math.sin(heading_error),
                math.cos(heading_error),
                np.clip(self.angular_velocity / max(self.physics.turn_rate, 1e-6), -1.0, 1.0),
            ],
            dtype=np.float32,
        )

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        assert self.track is not None
        action = np.asarray(action, dtype=float).reshape(2)
        throttle = float(np.clip(action[0], -1.0, 1.0))
        steering = float(np.clip(action[1], -1.0, 1.0))
        previous_progress = self.progress
        self.last_action = action
        self.current_step += 1

        if self.box2d_world is not None:
            self._step_with_box2d(throttle, steering)
        else:
            self._step_with_fallback(throttle, steering)

        sensor_distances = self._sensor_distances()
        progress, projection, tangent = project_point_to_polyline(self.position, self.track.centerline)
        self.progress = float(np.clip(progress, 0.0, 1.0))
        crash_distance = self._wall_distance()
        crashed = crash_distance <= self.physics.car_radius
        terminated = crashed or self.progress >= 0.995
        truncated = self.current_step >= self.physics.max_steps
        lateral_error = float(np.linalg.norm(self.position - projection))
        sector = int(np.searchsorted(self.track.checkpoint_progress, self.progress, side="right"))

        reward = (self.progress - previous_progress) * self.reward.progress_weight
        reward -= self.reward.time_penalty
        reward -= abs(steering) * self.reward.smoothness_penalty
        if self.progress < previous_progress:
            reward -= self.reward.reverse_penalty
        if crashed:
            reward -= self.reward.crash_penalty
        if self.progress >= 0.995:
            reward += self.reward.completion_bonus

        heading_target = float(math.atan2(tangent[1], tangent[0]))
        self.trajectory.append(
            {
                "step": self.current_step,
                "position": self.position.round(5).tolist(),
                "heading": round(self.heading, 5),
                "speed": round(self.speed, 5),
                "action": [round(throttle, 5), round(steering, 5)],
                "progress": round(self.progress, 5),
                "heading_target": round(heading_target, 5),
                "sensor_distances": sensor_distances.round(5).tolist(),
                "lateral_error": round(lateral_error, 5),
                "sector": sector,
                "finished": self.progress >= 0.995,
                "crashed": crashed,
            }
        )

        observation = np.asarray(
            [
                *sensor_distances.tolist(),
                np.clip(self.speed / self.physics.max_speed, -1.0, 1.0),
                math.sin(angle_wrap(heading_target - self.heading)),
                math.cos(angle_wrap(heading_target - self.heading)),
                np.clip(self.angular_velocity / max(self.physics.turn_rate, 1e-6), -1.0, 1.0),
            ],
            dtype=np.float32,
        )
        info = {
            "progress": self.progress,
            "completion": self.progress,
            "crashed": crashed,
            "track_seed": self.track_seed,
            "steps": self.current_step,
            "sector": sector,
        }
        return observation, float(reward), terminated, truncated, info

    def save_replay(self, path: str | Path) -> str:
        assert self.track is not None
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "track_seed": self.track.seed,
            "centerline": self.track.centerline.round(4).tolist(),
            "left_wall": self.track.left_wall.round(4).tolist(),
            "right_wall": self.track.right_wall.round(4).tolist(),
            "start_gate": self.track.start_gate.round(4).tolist(),
            "finish_gate": self.track.finish_gate.round(4).tolist(),
            "checkpoint_gates": self.track.checkpoint_gates.round(4).tolist(),
            "track_profile": self.track.profile,
            "physics": {
                "sensor_range": self.physics.sensor_range,
                "num_rays": self.physics.num_rays,
                "ray_spread_deg": self.physics.ray_spread_deg,
                "car_radius": self.physics.car_radius,
                "track_width": self.physics.track_width,
            },
            "frames": self.trajectory,
        }
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return str(output_path)
