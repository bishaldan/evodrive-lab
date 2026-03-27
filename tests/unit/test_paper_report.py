from types import SimpleNamespace

import pandas as pd

from app.paper_tools.report import (
    _select_canonical_runs,
    build_status_summary,
    select_qualitative_cases,
    summarize_runs,
)


def test_summarize_runs_groups_by_algorithm() -> None:
    dataframe = pd.DataFrame(
        [
            {
                "algorithm": "ga",
                "condition": "base",
                "status": "completed",
                "train_reward": 1.0,
                "train_completion": 0.3,
                "train_crash_rate": 0.1,
                "validation_reward": 2.0,
                "validation_completion": 0.5,
                "validation_crash_rate": 0.0,
                "validation_steps": 100.0,
                "test_reward": 1.8,
                "test_completion": 0.4,
                "test_crash_rate": 0.1,
                "test_steps": 110.0,
                "generalization_gap_reward": 0.2,
                "generalization_gap_completion": 0.1,
                "duration_seconds": 12.0,
            },
            {
                "algorithm": "ga",
                "condition": "base",
                "status": "completed",
                "train_reward": 2.0,
                "train_completion": 0.4,
                "train_crash_rate": 0.0,
                "validation_reward": 3.0,
                "validation_completion": 0.6,
                "validation_crash_rate": 0.0,
                "validation_steps": 120.0,
                "test_reward": 2.4,
                "test_completion": 0.5,
                "test_crash_rate": 0.0,
                "test_steps": 130.0,
                "generalization_gap_reward": 0.6,
                "generalization_gap_completion": 0.1,
                "duration_seconds": 13.0,
            },
        ]
    )
    summary = summarize_runs(dataframe, ["algorithm"])
    assert summary.loc[0, "algorithm"] == "ga"
    assert summary.loc[0, "n"] == 2
    assert round(float(summary.loc[0, "test_completion_mean"]), 3) == 0.45


def test_select_qualitative_cases_returns_three_cases_per_algorithm() -> None:
    dataframe = pd.DataFrame(
        [
            {"family": "main", "status": "completed", "algorithm": "ga", "run_id": "1", "seed": 7, "test_completion": 0.1, "test_reward": 1.0, "replay_path": "a"},
            {"family": "main", "status": "completed", "algorithm": "ga", "run_id": "2", "seed": 11, "test_completion": 0.3, "test_reward": 2.0, "replay_path": "b"},
            {"family": "main", "status": "completed", "algorithm": "ga", "run_id": "3", "seed": 23, "test_completion": 0.8, "test_reward": 3.0, "replay_path": "c"},
        ]
    )
    cases = select_qualitative_cases(dataframe)
    assert set(cases["case"].tolist()) == {"best", "median", "failure"}


def test_build_status_summary_reports_incomplete_matrices() -> None:
    manifest = pd.DataFrame(
        [
            {"family": "main", "status": "completed"},
            {"family": "main", "status": "failed"},
            {"family": "ablation-sensors", "status": "queued"},
            {"family": "ablation-tracks", "status": "running"},
        ]
    )
    text, warnings = build_status_summary(manifest)
    assert "Main comparison" in text
    assert warnings
    assert any("incomplete" in warning for warning in warnings)


def test_select_canonical_runs_prefers_completed_record() -> None:
    runs = [
        SimpleNamespace(
            name="paper-main-ga-seed007-base",
            status="running",
            created_at="2026-03-27T04:00:00",
            started_at="2026-03-27T04:00:01",
            finished_at=None,
        ),
        SimpleNamespace(
            name="paper-main-ga-seed007-base",
            status="completed",
            created_at="2026-03-27T04:02:00",
            started_at="2026-03-27T04:02:01",
            finished_at="2026-03-27T04:05:00",
        ),
        SimpleNamespace(
            name="paper-main-neat-seed007-base",
            status="queued",
            created_at="2026-03-27T04:03:00",
            started_at=None,
            finished_at=None,
        ),
    ]
    canonical, duplicates = _select_canonical_runs(runs)
    canonical_names = {run.name: run.status for run in canonical}
    assert canonical_names["paper-main-ga-seed007-base"] == "completed"
    assert duplicates["paper-main-ga-seed007-base"] == 2
