from pathlib import Path

import pandas as pd

from app.paper_tools.plots import plot_ablation, plot_main_comparison


def test_plot_main_comparison_writes_png_and_pdf(tmp_path: Path) -> None:
    summary = pd.DataFrame(
        [
            {
                "algorithm": "ga",
                "n": 5,
                "validation_completion_mean": 0.6,
                "validation_completion_std": 0.1,
                "test_completion_mean": 0.5,
                "test_completion_std": 0.1,
                "test_reward_mean": 3.0,
                "test_reward_std": 0.2,
                "test_crash_rate_mean": 0.2,
                "test_crash_rate_std": 0.05,
            },
            {
                "algorithm": "neat",
                "n": 5,
                "validation_completion_mean": 0.5,
                "validation_completion_std": 0.1,
                "test_completion_mean": 0.4,
                "test_completion_std": 0.1,
                "test_reward_mean": 2.5,
                "test_reward_std": 0.2,
                "test_crash_rate_mean": 0.3,
                "test_crash_rate_std": 0.04,
            },
            {
                "algorithm": "ppo",
                "n": 5,
                "validation_completion_mean": 0.4,
                "validation_completion_std": 0.1,
                "test_completion_mean": 0.3,
                "test_completion_std": 0.1,
                "test_reward_mean": 2.0,
                "test_reward_std": 0.3,
                "test_crash_rate_mean": 0.4,
                "test_crash_rate_std": 0.06,
            },
        ]
    )
    results_dir = tmp_path / "results"
    figures_dir = tmp_path / "figures"
    results_dir.mkdir()
    figures_dir.mkdir()
    summary.to_csv(results_dir / "paper_main_summary.csv", index=False)

    outputs = plot_main_comparison(results_dir, figures_dir)
    assert any(path.suffix == ".png" for path in outputs)
    assert any(path.suffix == ".pdf" for path in outputs)


def test_plot_ablation_writes_png_and_pdf(tmp_path: Path) -> None:
    summary = pd.DataFrame(
        [
            {"algorithm": "ga", "condition": "sensor5", "n": 5, "test_completion_mean": 0.4, "test_completion_std": 0.05},
            {"algorithm": "ga", "condition": "sensor7", "n": 5, "test_completion_mean": 0.5, "test_completion_std": 0.04},
            {"algorithm": "neat", "condition": "sensor5", "n": 5, "test_completion_mean": 0.35, "test_completion_std": 0.03},
            {"algorithm": "neat", "condition": "sensor7", "n": 5, "test_completion_mean": 0.45, "test_completion_std": 0.02},
            {"algorithm": "ppo", "condition": "sensor5", "n": 5, "test_completion_mean": 0.3, "test_completion_std": 0.04},
            {"algorithm": "ppo", "condition": "sensor7", "n": 5, "test_completion_mean": 0.33, "test_completion_std": 0.04},
        ]
    )
    summary_path = tmp_path / "paper_ablation_sensors_summary.csv"
    figures_dir = tmp_path / "figures"
    figures_dir.mkdir()
    summary.to_csv(summary_path, index=False)

    outputs = plot_ablation(summary_path, figures_dir, "sensor_completion", "Sensor ablation")
    assert any(path.suffix == ".png" for path in outputs)
    assert any(path.suffix == ".pdf" for path in outputs)
