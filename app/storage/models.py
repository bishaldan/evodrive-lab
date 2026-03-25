from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlmodel import Field, SQLModel


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class RunRecord(SQLModel, table=True):
    id: str = Field(primary_key=True)
    name: str
    algorithm: str
    mode: str
    status: str
    seed: int
    config_json: str
    summary_json: str = "{}"
    created_at: datetime = Field(default_factory=utc_now)
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    error_message: Optional[str] = None


class MetricRecord(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    run_id: str = Field(index=True)
    step: int
    phase: str
    reward: float
    completion: float
    crash_rate: float
    wall_time: float
    extras_json: str = "{}"


class ArtifactRecord(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    run_id: str = Field(index=True)
    artifact_type: str
    path: str
    metadata_json: str = "{}"

