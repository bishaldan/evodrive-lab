from __future__ import annotations

import json
from datetime import datetime, timezone
from uuid import uuid4

from sqlmodel import select

from app.config.models import MetricPoint, RunConfig, RunMode, RunStatus, RunSummary
from app.storage.db import get_session
from app.storage.models import ArtifactRecord, MetricRecord, RunRecord


def _now() -> datetime:
    return datetime.now(timezone.utc)


def create_run(config: RunConfig) -> RunSummary:
    run_id = uuid4().hex[:12]
    algorithm = str(config.algorithm)
    mode = str(config.mode)
    name = config.name or f"{algorithm}-{run_id}"
    record = RunRecord(
        id=run_id,
        name=name,
        algorithm=algorithm,
        mode=mode,
        status=RunStatus.QUEUED.value,
        seed=config.seed,
        config_json=config.model_dump_json(),
    )
    with get_session() as session:
        session.add(record)
        session.commit()
    return get_run(run_id)


def list_runs() -> list[RunSummary]:
    with get_session() as session:
        records = session.exec(select(RunRecord).order_by(RunRecord.created_at.desc())).all()
    return [_to_summary(record) for record in records]


def get_run(run_id: str) -> RunSummary:
    with get_session() as session:
        record = session.get(RunRecord, run_id)
        if record is None:
            raise KeyError(run_id)
    return _to_summary(record)


def get_run_record(run_id: str) -> RunRecord:
    with get_session() as session:
        record = session.get(RunRecord, run_id)
        if record is None:
            raise KeyError(run_id)
        session.expunge(record)
        return record


def mark_run_running(run_id: str) -> None:
    with get_session() as session:
        record = session.get(RunRecord, run_id)
        if record is None:
            raise KeyError(run_id)
        record.status = RunStatus.RUNNING.value
        record.started_at = _now()
        session.add(record)
        session.commit()


def finish_run(run_id: str, summary: dict[str, object]) -> None:
    with get_session() as session:
        record = session.get(RunRecord, run_id)
        if record is None:
            raise KeyError(run_id)
        record.status = RunStatus.COMPLETED.value
        record.finished_at = _now()
        record.summary_json = json.dumps(summary)
        session.add(record)
        session.commit()


def fail_run(run_id: str, message: str) -> None:
    with get_session() as session:
        record = session.get(RunRecord, run_id)
        if record is None:
            raise KeyError(run_id)
        record.status = RunStatus.FAILED.value
        record.finished_at = _now()
        record.error_message = message
        session.add(record)
        session.commit()


def add_metric(run_id: str, point: MetricPoint) -> None:
    with get_session() as session:
        record = MetricRecord(
            run_id=run_id,
            step=point.step,
            phase=point.phase,
            reward=point.reward,
            completion=point.completion,
            crash_rate=point.crash_rate,
            wall_time=point.wall_time,
            extras_json=json.dumps(point.extras),
        )
        session.add(record)
        session.commit()


def list_metrics(run_id: str) -> list[dict[str, object]]:
    with get_session() as session:
        records = session.exec(
            select(MetricRecord)
            .where(MetricRecord.run_id == run_id)
            .order_by(MetricRecord.step.asc(), MetricRecord.id.asc())
        ).all()
    return [
        {
            "step": record.step,
            "phase": record.phase,
            "reward": record.reward,
            "completion": record.completion,
            "crash_rate": record.crash_rate,
            "wall_time": record.wall_time,
            "extras": json.loads(record.extras_json),
        }
        for record in records
    ]


def add_artifact(run_id: str, artifact_type: str, path: str, metadata: dict[str, object] | None = None) -> None:
    with get_session() as session:
        record = ArtifactRecord(
            run_id=run_id,
            artifact_type=artifact_type,
            path=path,
            metadata_json=json.dumps(metadata or {}),
        )
        session.add(record)
        session.commit()


def list_artifacts(run_id: str) -> list[dict[str, object]]:
    with get_session() as session:
        records = session.exec(
            select(ArtifactRecord)
            .where(ArtifactRecord.run_id == run_id)
            .order_by(ArtifactRecord.id.asc())
        ).all()
    return [
        {
            "artifact_type": record.artifact_type,
            "path": record.path,
            "metadata": json.loads(record.metadata_json),
        }
        for record in records
    ]


def next_queued_run() -> RunRecord | None:
    with get_session() as session:
        record = session.exec(
            select(RunRecord)
            .where(RunRecord.status == RunStatus.QUEUED.value)
            .order_by(RunRecord.created_at.asc())
        ).first()
        if record is None:
            return None
        session.expunge(record)
        return record


def _to_summary(record: RunRecord) -> RunSummary:
    return RunSummary(
        id=record.id,
        name=record.name,
        algorithm=record.algorithm,
        mode=record.mode,
        status=record.status,
        seed=record.seed,
        created_at=record.created_at.isoformat(),
        started_at=record.started_at.isoformat() if record.started_at else None,
        finished_at=record.finished_at.isoformat() if record.finished_at else None,
        error_message=record.error_message,
        summary=json.loads(record.summary_json),
    )
