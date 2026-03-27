from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from app.config.models import RunConfig
from app.paper_tools.common import PAPER_CONFIG_DIR, build_run_name, deep_merge, load_json
from app.storage.repository import create_run, get_run_record, list_runs
from app.worker.runner import execute_run


RUNS_DIR = Path(os.getenv("EVODRIVE_RUNS_DIR", "runs"))
REPORTS_DIR = Path(os.getenv("EVODRIVE_REPORTS_DIR", "reports"))

CONFIG_MAP = {
    "main": PAPER_CONFIG_DIR / "main.json",
    "sensors": PAPER_CONFIG_DIR / "ablation_sensors.json",
    "tracks": PAPER_CONFIG_DIR / "ablation_tracks.json",
}


def _build_run_config(
    experiment_family: str,
    algorithm: str,
    seed: int,
    condition: dict[str, Any],
    base_spec: dict[str, Any],
) -> RunConfig:
    per_algorithm = base_spec["per_algorithm"][algorithm]
    env_config = deep_merge(base_spec.get("env", {}), condition.get("env", {}))
    algorithm_params = deep_merge(
        per_algorithm.get("algorithm_params", {}),
        condition.get("algorithm_params", {}),
    )
    payload = {
        "name": build_run_name(experiment_family, algorithm, seed, condition["label"]),
        "algorithm": algorithm,
        "mode": "benchmark",
        "seed": seed,
        "checkpoint_every": per_algorithm.get("checkpoint_every", 4),
        "total_iterations": per_algorithm.get("total_iterations", 1),
        "env": env_config,
        "algorithm_params": algorithm_params,
    }
    return RunConfig.model_validate(payload)


def load_matrix_configs(kind: str) -> list[dict[str, Any]]:
    if kind == "full":
        return [load_json(path) for path in CONFIG_MAP.values()]
    if kind not in CONFIG_MAP:
        raise KeyError(f"Unknown matrix config: {kind}")
    return [load_json(CONFIG_MAP[kind])]


def build_run_matrix(kind: str) -> list[RunConfig]:
    configs = load_matrix_configs(kind)
    matrix: list[RunConfig] = []
    for spec in configs:
        family = spec["family"]
        algorithms = spec["algorithms"]
        seeds = spec["seeds"]
        conditions = spec["conditions"]
        for algorithm in algorithms:
            for seed in seeds:
                for condition in conditions:
                    matrix.append(_build_run_config(family, algorithm, seed, condition, spec))
    return matrix


def queue_matrix(kind: str, *, execute_sync: bool = False, allow_duplicates: bool = False) -> dict[str, Any]:
    run_matrix = build_run_matrix(kind)
    existing_names = {run.name for run in list_runs()}
    created: list[str] = []
    skipped: list[str] = []

    for config in run_matrix:
        if not allow_duplicates and config.name in existing_names:
            skipped.append(config.name or "<unnamed>")
            continue
        summary = create_run(config)
        created.append(summary.name)
        if execute_sync:
            record = get_run_record(summary.id)
            execute_run(record, RUNS_DIR, REPORTS_DIR)

    return {
        "kind": kind,
        "requested": len(run_matrix),
        "created": len(created),
        "skipped": len(skipped),
        "created_names": created,
        "skipped_names": skipped,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Queue or execute the paper experiment matrix.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    queue_main = subparsers.add_parser("queue-main")
    queue_main.add_argument("--sync", action="store_true")
    queue_main.add_argument("--allow-duplicates", action="store_true")

    queue_ablation = subparsers.add_parser("queue-ablation")
    queue_ablation.add_argument("--family", choices=["sensors", "tracks"], required=True)
    queue_ablation.add_argument("--sync", action="store_true")
    queue_ablation.add_argument("--allow-duplicates", action="store_true")

    queue_full = subparsers.add_parser("queue-full")
    queue_full.add_argument("--sync", action="store_true")
    queue_full.add_argument("--allow-duplicates", action="store_true")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "queue-main":
        result = queue_matrix("main", execute_sync=args.sync, allow_duplicates=args.allow_duplicates)
    elif args.command == "queue-ablation":
        result = queue_matrix(args.family, execute_sync=args.sync, allow_duplicates=args.allow_duplicates)
    else:
        result = queue_matrix("full", execute_sync=args.sync, allow_duplicates=args.allow_duplicates)

    print(result)


if __name__ == "__main__":  # pragma: no cover
    main()
