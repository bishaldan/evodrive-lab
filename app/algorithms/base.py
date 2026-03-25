from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from app.config.models import MetricPoint


MetricCallback = Callable[[MetricPoint], None]


@dataclass(slots=True)
class AlgorithmResult:
    summary: dict[str, object]
    artifacts: list[tuple[str, str, dict[str, object]]] = field(default_factory=list)

    def add_artifact(self, artifact_type: str, path: str | Path, metadata: dict[str, object] | None = None) -> None:
        self.artifacts.append((artifact_type, str(path), metadata or {}))

