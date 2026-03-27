from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from app.paper_tools.common import PAPER_RESULTS_DIR, ensure_dirs, markdown_table, parse_run_name
from app.paper_tools.runner import build_run_matrix
from app.storage.repository import list_artifacts, list_metrics, list_runs


MAIN_RAW_CSV = "paper_main_runs.csv"
MAIN_SUMMARY_CSV = "paper_main_summary.csv"
SENSOR_RAW_CSV = "paper_ablation_sensors_runs.csv"
SENSOR_SUMMARY_CSV = "paper_ablation_sensors_summary.csv"
TRACK_RAW_CSV = "paper_ablation_tracks_runs.csv"
TRACK_SUMMARY_CSV = "paper_ablation_tracks_summary.csv"
RUN_MANIFEST_CSV = "paper_run_manifest.csv"
QUALITATIVE_CSV = "paper_qualitative_cases.csv"
QUALITATIVE_MD = "paper_qualitative_cases.md"

MAIN_SUMMARY_MD = "paper_main_summary.md"
SENSOR_SUMMARY_MD = "paper_ablation_sensors_summary.md"
TRACK_SUMMARY_MD = "paper_ablation_tracks_summary.md"
STATUS_SUMMARY_MD = "paper_status.md"

METRIC_COLUMNS = [
    "train_reward",
    "train_completion",
    "train_crash_rate",
    "validation_reward",
    "validation_completion",
    "validation_crash_rate",
    "validation_steps",
    "test_reward",
    "test_completion",
    "test_crash_rate",
    "test_steps",
    "generalization_gap_reward",
    "generalization_gap_completion",
    "duration_seconds",
]


def _duration_seconds(started_at: str | None, finished_at: str | None) -> float | None:
    if not started_at or not finished_at:
        return None
    started = datetime.fromisoformat(started_at)
    finished = datetime.fromisoformat(finished_at)
    return round((finished - started).total_seconds(), 3)


def _last_phase_metrics(metrics: list[dict[str, Any]], phase: str) -> dict[str, Any] | None:
    filtered = [metric for metric in metrics if metric["phase"] == phase]
    if not filtered:
        return None
    return filtered[-1]


def _first_artifact_path(artifacts: list[dict[str, Any]], artifact_type: str) -> str | None:
    for artifact in artifacts:
        if artifact["artifact_type"] == artifact_type:
            return artifact["path"]
    return None


def _status_rank(status: str) -> int:
    ranking = {
        "completed": 4,
        "running": 3,
        "queued": 2,
        "failed": 1,
    }
    return ranking.get(status, 0)


def _canonical_sort_key(run: Any) -> tuple[int, str, str, str]:
    status = str(run.status.value if hasattr(run.status, "value") else run.status)
    finished_at = run.finished_at or ""
    started_at = run.started_at or ""
    created_at = run.created_at or ""
    return (_status_rank(status), finished_at, started_at, created_at)


def _select_canonical_runs(all_runs: list[Any]) -> tuple[list[Any], dict[str, int]]:
    grouped: dict[str, list[Any]] = {}
    for run in all_runs:
        meta = parse_run_name(run.name)
        if meta is None:
            continue
        grouped.setdefault(run.name, []).append(run)

    canonical_runs: list[Any] = []
    duplicate_counts: dict[str, int] = {}
    for name, runs in grouped.items():
        ordered = sorted(runs, key=_canonical_sort_key, reverse=True)
        canonical_runs.append(ordered[0])
        duplicate_counts[name] = len(runs)
    return canonical_runs, duplicate_counts


def collect_paper_rows() -> tuple[pd.DataFrame, pd.DataFrame]:
    run_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    canonical_runs, duplicate_counts = _select_canonical_runs(list_runs())

    for run in canonical_runs:
        meta = parse_run_name(run.name)
        if meta is None:
            continue
        artifacts = list_artifacts(run.id)
        metrics = list_metrics(run.id)
        train_metrics = _last_phase_metrics(metrics, "train")
        validation = dict(run.summary.get("validation", {}))
        test = dict(run.summary.get("test", {}))
        duration = _duration_seconds(run.started_at, run.finished_at)
        replay_path = test.get("replay_path") or _first_artifact_path(artifacts, "replay")

        row = {
            "run_id": run.id,
            "name": run.name,
            "family": meta["family"],
            "algorithm": meta["algorithm"],
            "seed": meta["seed"],
            "condition": meta["condition"],
            "status": run.status,
            "error_message": run.error_message,
            "duplicate_count": duplicate_counts.get(run.name, 1),
            "duration_seconds": duration,
            "train_reward": train_metrics["reward"] if train_metrics else None,
            "train_completion": train_metrics["completion"] if train_metrics else None,
            "train_crash_rate": train_metrics["crash_rate"] if train_metrics else None,
            "validation_reward": validation.get("mean_reward"),
            "validation_completion": validation.get("mean_completion"),
            "validation_crash_rate": validation.get("crash_rate"),
            "validation_steps": validation.get("mean_steps"),
            "test_reward": test.get("mean_reward"),
            "test_completion": test.get("mean_completion"),
            "test_crash_rate": test.get("crash_rate"),
            "test_steps": test.get("mean_steps"),
            "generalization_gap_reward": None,
            "generalization_gap_completion": None,
            "replay_path": replay_path,
        }
        if row["validation_reward"] is not None and row["test_reward"] is not None:
            row["generalization_gap_reward"] = row["validation_reward"] - row["test_reward"]
        if row["validation_completion"] is not None and row["test_completion"] is not None:
            row["generalization_gap_completion"] = row["validation_completion"] - row["test_completion"]

        run_rows.append(row)
        manifest_rows.append(
            {
                "run_id": run.id,
                "name": run.name,
                "family": meta["family"],
                "algorithm": meta["algorithm"],
                "seed": meta["seed"],
                "condition": meta["condition"],
                "status": run.status,
                "error_message": run.error_message or "",
                "duplicate_count": duplicate_counts.get(run.name, 1),
            }
        )

    return pd.DataFrame(run_rows), pd.DataFrame(manifest_rows)


def summarize_runs(dataframe: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    completed = dataframe[dataframe["status"] == "completed"].copy()
    if completed.empty:
        return pd.DataFrame(columns=[*group_columns, "n"])

    aggregation_spec: dict[str, list[str]] = {column: ["mean", "std"] for column in METRIC_COLUMNS}
    summary = completed.groupby(group_columns, dropna=False).agg(aggregation_spec)
    summary.columns = [f"{column}_{stat}" for column, stat in summary.columns]
    summary = summary.reset_index()
    summary["n"] = completed.groupby(group_columns, dropna=False).size().values
    summary = summary.fillna(0.0)
    return summary


def build_status_summary(manifest_df: pd.DataFrame) -> tuple[str, list[str]]:
    expected_counts = {
        "main": len(build_run_matrix("main")),
        "ablation-sensors": len(build_run_matrix("sensors")),
        "ablation-tracks": len(build_run_matrix("tracks")),
    }
    family_labels = {
        "main": "Main comparison",
        "ablation-sensors": "Sensor ablation",
        "ablation-tracks": "Track ablation",
    }
    lines = ["# Paper Benchmark Status", ""]
    warnings: list[str] = []

    if manifest_df.empty:
        warning = "No paper runs were found in the registry."
        warnings.append(warning)
        lines.append(f"- Warning: {warning}")
        return "\n".join(lines) + "\n", warnings

    for family, expected in expected_counts.items():
        subset = manifest_df[manifest_df["family"] == family].copy()
        completed = int((subset["status"] == "completed").sum())
        queued = int((subset["status"] == "queued").sum())
        running = int((subset["status"] == "running").sum())
        failed = int((subset["status"] == "failed").sum())
        total = int(len(subset))
        lines.append(
            f"- {family_labels[family]}: {completed}/{expected} completed, "
            f"{queued} queued, {running} running, {failed} failed, {total} total recorded"
        )
        if completed < expected:
            warnings.append(
                f"{family_labels[family]} is incomplete: expected {expected} completed runs but found {completed}."
            )
        if failed:
            warnings.append(f"{family_labels[family]} includes {failed} failed run(s).")

    if "duplicate_count" in manifest_df.columns:
        duplicate_rows = manifest_df[manifest_df["duplicate_count"] > 1]
    else:
        duplicate_rows = manifest_df.iloc[0:0]
    if not duplicate_rows.empty:
        warnings.append(
            f"Duplicate run names were detected for {len(duplicate_rows)} canonical run(s); reporting uses the most advanced record per name."
        )

    if warnings:
        lines.append("")
        lines.append("## Warnings")
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("")
        lines.append("All paper matrices are complete.")

    return "\n".join(lines) + "\n", warnings


def select_qualitative_cases(dataframe: pd.DataFrame) -> pd.DataFrame:
    completed = dataframe[
        (dataframe["status"] == "completed") & (dataframe["family"] == "main")
    ].copy()
    if completed.empty:
        return pd.DataFrame(
            columns=["algorithm", "case", "run_id", "seed", "test_completion", "test_reward", "replay_path"]
        )

    rows: list[dict[str, Any]] = []
    for algorithm, subset in completed.groupby("algorithm"):
        ordered = subset.sort_values("test_completion", ascending=True).reset_index(drop=True)
        cases = {
            "failure": ordered.iloc[0],
            "median": ordered.iloc[len(ordered) // 2],
            "best": ordered.iloc[-1],
        }
        for case_name, record in cases.items():
            rows.append(
                {
                    "algorithm": algorithm,
                    "case": case_name,
                    "run_id": record["run_id"],
                    "seed": int(record["seed"]),
                    "test_completion": record["test_completion"],
                    "test_reward": record["test_reward"],
                    "replay_path": record["replay_path"] or "",
                }
            )
    return pd.DataFrame(rows)


def export_markdown_tables(results_dir: Path) -> list[Path]:
    outputs: list[Path] = []
    export_pairs = [
        (results_dir / MAIN_SUMMARY_CSV, results_dir / MAIN_SUMMARY_MD),
        (results_dir / SENSOR_SUMMARY_CSV, results_dir / SENSOR_SUMMARY_MD),
        (results_dir / TRACK_SUMMARY_CSV, results_dir / TRACK_SUMMARY_MD),
        (results_dir / QUALITATIVE_CSV, results_dir / QUALITATIVE_MD),
    ]
    for csv_path, markdown_path in export_pairs:
        if not csv_path.exists():
            continue
        dataframe = pd.read_csv(csv_path)
        headers = dataframe.columns.tolist()
        rows = dataframe.values.tolist()
        markdown_path.write_text(markdown_table(headers, rows), encoding="utf-8")
        outputs.append(markdown_path)
    return outputs


def aggregate_all(results_dir: Path) -> dict[str, Path]:
    ensure_dirs(results_dir, results_dir / "figures")
    runs_df, manifest_df = collect_paper_rows()

    outputs: dict[str, Path] = {}
    manifest_path = results_dir / RUN_MANIFEST_CSV
    manifest_df.to_csv(manifest_path, index=False)
    outputs["manifest"] = manifest_path
    status_summary, warnings = build_status_summary(manifest_df)
    status_path = results_dir / STATUS_SUMMARY_MD
    status_path.write_text(status_summary, encoding="utf-8")
    outputs["status"] = status_path

    main_df = runs_df[runs_df["family"] == "main"].copy()
    main_raw_path = results_dir / MAIN_RAW_CSV
    main_df.to_csv(main_raw_path, index=False)
    outputs["main_raw"] = main_raw_path

    main_summary = summarize_runs(main_df, ["algorithm"])
    main_summary_path = results_dir / MAIN_SUMMARY_CSV
    main_summary.to_csv(main_summary_path, index=False)
    outputs["main_summary"] = main_summary_path

    sensor_df = runs_df[runs_df["family"] == "ablation-sensors"].copy()
    sensor_raw_path = results_dir / SENSOR_RAW_CSV
    sensor_df.to_csv(sensor_raw_path, index=False)
    outputs["sensor_raw"] = sensor_raw_path

    sensor_summary = summarize_runs(sensor_df, ["algorithm", "condition"])
    sensor_summary_path = results_dir / SENSOR_SUMMARY_CSV
    sensor_summary.to_csv(sensor_summary_path, index=False)
    outputs["sensor_summary"] = sensor_summary_path

    track_df = runs_df[runs_df["family"] == "ablation-tracks"].copy()
    track_raw_path = results_dir / TRACK_RAW_CSV
    track_df.to_csv(track_raw_path, index=False)
    outputs["track_raw"] = track_raw_path

    track_summary = summarize_runs(track_df, ["algorithm", "condition"])
    track_summary_path = results_dir / TRACK_SUMMARY_CSV
    track_summary.to_csv(track_summary_path, index=False)
    outputs["track_summary"] = track_summary_path

    qualitative = select_qualitative_cases(runs_df)
    qualitative_path = results_dir / QUALITATIVE_CSV
    qualitative.to_csv(qualitative_path, index=False)
    outputs["qualitative"] = qualitative_path
    export_markdown_tables(results_dir)
    if warnings:
        print({"warnings": warnings})
    return outputs


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate paper benchmark results.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name in ["aggregate-main", "aggregate-ablation", "export-tables", "aggregate-all"]:
        subparser = subparsers.add_parser(name)
        subparser.add_argument("--results-dir", default=str(PAPER_RESULTS_DIR))
        if name == "aggregate-ablation":
            subparser.add_argument("--family", choices=["sensors", "tracks"], required=True)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    ensure_dirs(results_dir, results_dir / "figures")

    if args.command == "aggregate-all":
        print(aggregate_all(results_dir))
        return

    runs_df, manifest_df = collect_paper_rows()
    manifest_df.to_csv(results_dir / RUN_MANIFEST_CSV, index=False)

    if args.command == "aggregate-main":
        main_df = runs_df[runs_df["family"] == "main"].copy()
        main_df.to_csv(results_dir / MAIN_RAW_CSV, index=False)
        summarize_runs(main_df, ["algorithm"]).to_csv(results_dir / MAIN_SUMMARY_CSV, index=False)
    elif args.command == "aggregate-ablation":
        family = "ablation-sensors" if args.family == "sensors" else "ablation-tracks"
        runs_name = SENSOR_RAW_CSV if args.family == "sensors" else TRACK_RAW_CSV
        summary_name = SENSOR_SUMMARY_CSV if args.family == "sensors" else TRACK_SUMMARY_CSV
        subset = runs_df[runs_df["family"] == family].copy()
        subset.to_csv(results_dir / runs_name, index=False)
        summarize_runs(subset, ["algorithm", "condition"]).to_csv(results_dir / summary_name, index=False)
    else:
        print([str(path) for path in export_markdown_tables(results_dir)])


if __name__ == "__main__":  # pragma: no cover
    main()
