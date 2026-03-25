from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class AlgorithmName(str, Enum):
    GA = "ga"
    NEAT = "neat"
    PPO = "ppo"


class RunMode(str, Enum):
    TRAIN = "train"
    BENCHMARK = "benchmark"


class RunStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class RewardConfig(BaseModel):
    progress_weight: float = 8.0
    time_penalty: float = 0.02
    crash_penalty: float = 3.0
    completion_bonus: float = 12.0
    smoothness_penalty: float = 0.01
    reverse_penalty: float = 0.5


class PhysicsConfig(BaseModel):
    dt: float = 1.0 / 30.0
    max_steps: int = 360
    sensor_range: float = 12.0
    num_rays: int = 5
    ray_spread_deg: float = 90.0
    track_width: float = 6.0
    car_radius: float = 0.8
    max_speed: float = 12.0
    acceleration: float = 18.0
    turn_rate: float = 2.75
    linear_damping: float = 0.985
    max_track_segments: int = 28
    segment_length: float = 6.0
    use_box2d: bool = True


class EnvironmentConfig(BaseModel):
    train_suite: str = "train"
    validation_suite: str = "validation"
    test_suite: str = "test"
    physics: PhysicsConfig = Field(default_factory=PhysicsConfig)
    reward: RewardConfig = Field(default_factory=RewardConfig)


class RunConfig(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    name: str | None = None
    algorithm: AlgorithmName
    mode: RunMode = RunMode.TRAIN
    seed: int = 42
    checkpoint_every: int = 5
    total_iterations: int = 10
    env: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    algorithm_params: dict[str, Any] = Field(default_factory=dict)


class MetricPoint(BaseModel):
    step: int
    phase: str
    reward: float
    completion: float
    crash_rate: float
    wall_time: float
    extras: dict[str, Any] = Field(default_factory=dict)


class ReplayArtifact(BaseModel):
    path: str
    artifact_type: str = "replay"
    metadata: dict[str, Any] = Field(default_factory=dict)


class RunSummary(BaseModel):
    id: str
    name: str
    algorithm: AlgorithmName
    mode: RunMode
    status: RunStatus
    seed: int
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    error_message: str | None = None
    summary: dict[str, Any] = Field(default_factory=dict)


class BenchmarkRequest(BaseModel):
    algorithms: list[AlgorithmName] = Field(
        default_factory=lambda: [AlgorithmName.GA, AlgorithmName.NEAT, AlgorithmName.PPO]
    )
    seed: int = 42
    total_iterations: int = 8
    checkpoint_every: int = 4
    env: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    algorithm_params: dict[str, dict[str, Any]] = Field(default_factory=dict)

