from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[2]
PAPER_DIR = ROOT_DIR / "paper"
PAPER_CONFIG_DIR = PAPER_DIR / "configs"
PAPER_RESULTS_DIR = PAPER_DIR / "results"
PAPER_FIGURES_DIR = PAPER_RESULTS_DIR / "figures"
PAPER_LATEX_DIR = PAPER_DIR / "latex"

ALGORITHM_COLORS: dict[str, str] = {
    "ga": "#ff595e",
    "neat": "#1982c4",
    "ppo": "#8ac926",
}

RUN_NAME_RE = re.compile(
    r"^paper-(?P<family>main|ablation-sensors|ablation-tracks)-"
    r"(?P<algorithm>ga|neat|ppo)-seed(?P<seed>\d+)-(?P<condition>[a-z0-9-]+)$"
)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_dirs(results_dir: Path | None = None, figures_dir: Path | None = None) -> None:
    resolved_results = results_dir or PAPER_RESULTS_DIR
    resolved_figures = figures_dir or PAPER_FIGURES_DIR
    resolved_results.mkdir(parents=True, exist_ok=True)
    resolved_figures.mkdir(parents=True, exist_ok=True)


def build_run_name(family: str, algorithm: str, seed: int, condition: str) -> str:
    return f"paper-{family}-{algorithm}-seed{seed:03d}-{condition}"


def parse_run_name(name: str) -> dict[str, Any] | None:
    match = RUN_NAME_RE.match(name)
    if match is None:
        return None
    payload = match.groupdict()
    payload["seed"] = int(payload["seed"])
    return payload


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines) + "\n"
