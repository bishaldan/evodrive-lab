from __future__ import annotations

from pathlib import Path

from app.algorithms.ga import run_ga
from app.algorithms.neat_runner import run_neat
from app.algorithms.ppo_runner import run_ppo
from app.export.media import export_replay_media
from app.config.models import MetricPoint, RunConfig
from app.export.reporting import export_run_report
from app.storage.models import RunRecord
from app.storage.repository import add_artifact, add_metric, fail_run, finish_run, mark_run_running


def execute_run(record: RunRecord, runs_dir: Path, reports_dir: Path) -> None:
    mark_run_running(record.id)
    config = RunConfig.model_validate_json(record.config_json)
    run_dir = runs_dir / record.id
    run_dir.mkdir(parents=True, exist_ok=True)

    def metric_callback(point: MetricPoint) -> None:
        add_metric(record.id, point)

    try:
        algorithm = str(config.algorithm)
        if algorithm == "ga":
            result = run_ga(config, run_dir, metric_callback)
        elif algorithm == "neat":
            result = run_neat(config, run_dir, metric_callback)
        elif algorithm == "ppo":
            result = run_ppo(config, run_dir, metric_callback)
        else:  # pragma: no cover - defensive fallback
            raise ValueError(f"Unsupported algorithm: {config.algorithm}")

        report_paths = export_run_report(record.id, reports_dir)
        for artifact_type, path, metadata in result.artifacts:
            add_artifact(record.id, artifact_type, path, metadata)
            if artifact_type == "replay" and path:
                try:
                    media_artifacts = export_replay_media(path, reports_dir, prefix=record.id)
                except Exception:
                    media_artifacts = []
                for media_type, media_path, media_metadata in media_artifacts:
                    add_artifact(record.id, media_type, media_path, media_metadata)
        for report_path in report_paths:
            add_artifact(record.id, "report", report_path, {})
        finish_run(record.id, result.summary)
    except Exception as exc:  # pragma: no cover - runtime error path
        fail_run(record.id, str(exc))
        raise
