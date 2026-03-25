from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

from app.storage.repository import get_run, list_metrics


def export_run_report(run_id: str, reports_dir: Path) -> list[str]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    run = get_run(run_id)
    metrics = list_metrics(run_id)
    if not metrics:
        return []

    csv_path = reports_dir / f"{run_id}_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as file_handle:
        writer = csv.DictWriter(
            file_handle,
            fieldnames=["step", "phase", "reward", "completion", "crash_rate", "wall_time"],
        )
        writer.writeheader()
        for metric in metrics:
            writer.writerow(
                {
                    "step": metric["step"],
                    "phase": metric["phase"],
                    "reward": metric["reward"],
                    "completion": metric["completion"],
                    "crash_rate": metric["crash_rate"],
                    "wall_time": metric["wall_time"],
                }
            )

    plot_path = reports_dir / f"{run_id}_reward.png"
    plt.figure(figsize=(8, 4))
    plt.plot([point["step"] for point in metrics], [point["reward"] for point in metrics], label="reward")
    plt.plot(
        [point["step"] for point in metrics],
        [point["completion"] * 100.0 for point in metrics],
        label="completion x100",
    )
    plt.title(f"{run.name} ({run.algorithm})")
    plt.xlabel("step")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    return [str(csv_path), str(plot_path)]
