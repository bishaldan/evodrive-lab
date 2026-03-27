from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from app.paper_tools.common import ALGORITHM_COLORS, PAPER_FIGURES_DIR, PAPER_RESULTS_DIR, ensure_dirs

EXPECTED_MAIN_SEEDS = 5
EXPECTED_ABLATION_SEEDS = 5


def _save_figure(figure: plt.Figure, stem: str, figures_dir: Path) -> list[Path]:
    outputs = [figures_dir / f"{stem}.png", figures_dir / f"{stem}.pdf"]
    for output in outputs:
        figure.savefig(output, bbox_inches="tight", dpi=180)
    plt.close(figure)
    return outputs


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    dataframe = pd.read_csv(path)
    if dataframe.empty:
        raise ValueError(
            f"No completed benchmark rows found in {path}. "
            "Run the paper experiments first, then regenerate the summaries."
        )
    return dataframe


def _require_columns(dataframe: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in dataframe.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {', '.join(missing)}")


def _require_minimum_counts(
    dataframe: pd.DataFrame, expected: int, group_columns: list[str], label: str
) -> None:
    if "n" not in dataframe.columns:
        raise ValueError(f"{label} summary is missing the 'n' column.")
    incomplete = dataframe[dataframe["n"] < expected]
    if not incomplete.empty:
        groups = incomplete[group_columns + ["n"]].to_dict(orient="records")
        raise ValueError(
            f"{label} summary is incomplete. Expected at least {expected} completed seeds per group, "
            f"but found: {groups}"
        )


def plot_main_comparison(results_dir: Path, figures_dir: Path) -> list[Path]:
    summary = _load_csv(results_dir / "paper_main_summary.csv")
    _require_columns(
        summary,
        [
            "algorithm",
            "n",
            "validation_completion_mean",
            "validation_completion_std",
            "test_completion_mean",
            "test_completion_std",
            "test_reward_mean",
            "test_reward_std",
            "test_crash_rate_mean",
            "test_crash_rate_std",
        ],
        "Main comparison",
    )
    _require_minimum_counts(summary, EXPECTED_MAIN_SEEDS, ["algorithm"], "Main comparison")
    algorithms = summary["algorithm"].tolist()
    positions = np.arange(len(algorithms))

    completion_fig, completion_ax = plt.subplots(figsize=(8, 4.8))
    width = 0.35
    completion_ax.bar(
        positions - (width / 2),
        summary["validation_completion_mean"],
        width=width,
        yerr=summary["validation_completion_std"],
        label="Validation",
        color="#6c8ebf",
        capsize=4,
    )
    completion_ax.bar(
        positions + (width / 2),
        summary["test_completion_mean"],
        width=width,
        yerr=summary["test_completion_std"],
        label="Test",
        color="#2f4858",
        capsize=4,
    )
    completion_ax.set_xticks(positions)
    completion_ax.set_xticklabels([algorithm.upper() for algorithm in algorithms])
    completion_ax.set_ylabel("Mean completion")
    completion_ax.set_title("Validation vs test completion")
    completion_ax.legend()

    reward_fig, reward_ax = plt.subplots(figsize=(8, 4.8))
    reward_ax.bar(
        positions,
        summary["test_reward_mean"],
        yerr=summary["test_reward_std"],
        color=[ALGORITHM_COLORS[algorithm] for algorithm in algorithms],
        capsize=4,
    )
    reward_ax.set_xticks(positions)
    reward_ax.set_xticklabels([algorithm.upper() for algorithm in algorithms])
    reward_ax.set_ylabel("Mean test reward")
    reward_ax.set_title("Test reward by algorithm")

    crash_fig, crash_ax = plt.subplots(figsize=(8, 4.8))
    crash_ax.bar(
        positions,
        summary["test_crash_rate_mean"],
        yerr=summary["test_crash_rate_std"],
        color=[ALGORITHM_COLORS[algorithm] for algorithm in algorithms],
        capsize=4,
    )
    crash_ax.set_xticks(positions)
    crash_ax.set_xticklabels([algorithm.upper() for algorithm in algorithms])
    crash_ax.set_ylabel("Mean test crash rate")
    crash_ax.set_title("Test crash rate by algorithm")

    outputs: list[Path] = []
    outputs.extend(_save_figure(completion_fig, "main_validation_vs_test_completion", figures_dir))
    outputs.extend(_save_figure(reward_fig, "main_test_reward", figures_dir))
    outputs.extend(_save_figure(crash_fig, "main_test_crash_rate", figures_dir))
    return outputs


def plot_ablation(summary_path: Path, figures_dir: Path, stem: str, title: str) -> list[Path]:
    summary = _load_csv(summary_path)
    _require_columns(
        summary,
        ["algorithm", "condition", "n", "test_completion_mean", "test_completion_std"],
        title,
    )
    _require_minimum_counts(summary, EXPECTED_ABLATION_SEEDS, ["algorithm", "condition"], title)
    algorithms = sorted(summary["algorithm"].unique().tolist())
    conditions = sorted(summary["condition"].unique().tolist())
    positions = np.arange(len(conditions))
    width = 0.24

    figure, axis = plt.subplots(figsize=(9, 5))
    for index, algorithm in enumerate(algorithms):
        subset = summary[summary["algorithm"] == algorithm].sort_values("condition")
        axis.bar(
            positions + ((index - 1) * width),
            subset["test_completion_mean"],
            width=width,
            yerr=subset["test_completion_std"],
            label=algorithm.upper(),
            color=ALGORITHM_COLORS[algorithm],
            capsize=4,
        )
    axis.set_xticks(positions)
    axis.set_xticklabels(conditions)
    axis.set_ylabel("Mean test completion")
    axis.set_title(title)
    axis.legend()
    return _save_figure(figure, stem, figures_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper-ready plots from benchmark summaries.")
    parser.add_argument(
        "command",
        choices=["build-all", "main", "ablations"],
        help="Which plot set to build.",
    )
    parser.add_argument("--results-dir", default=str(PAPER_RESULTS_DIR))
    parser.add_argument("--figures-dir", default=str(PAPER_FIGURES_DIR))
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir)
    ensure_dirs(results_dir, figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    outputs: list[Path] = []
    if args.command in {"build-all", "main"}:
        outputs.extend(plot_main_comparison(results_dir, figures_dir))
    if args.command in {"build-all", "ablations"}:
        outputs.extend(
            plot_ablation(
                results_dir / "paper_ablation_sensors_summary.csv",
                figures_dir,
                "ablation_sensor_completion",
                "Sensor-count ablation: test completion",
            )
        )
        outputs.extend(
            plot_ablation(
                results_dir / "paper_ablation_tracks_summary.csv",
                figures_dir,
                "ablation_track_completion",
                "Track-length ablation: test completion",
            )
        )
    print([str(output) for output in outputs])


if __name__ == "__main__":  # pragma: no cover
    main()
