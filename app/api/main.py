from __future__ import annotations

import json
import os
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
import uvicorn

from app.config.models import BenchmarkRequest, RunConfig
from app.storage.db import init_db
from app.storage.repository import (
    create_run,
    get_run,
    list_artifacts,
    list_metrics,
    list_runs,
)

RUNS_DIR = Path(os.getenv("EVODRIVE_RUNS_DIR", "runs"))

@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    yield


app = FastAPI(title="EvoDrive Lab API", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/runs")
def get_runs() -> list[dict[str, object]]:
    return [run.model_dump() for run in list_runs()]


@app.post("/runs")
def post_run(config: RunConfig) -> dict[str, object]:
    run = create_run(config)
    return run.model_dump()


@app.get("/runs/{run_id}")
def get_run_details(run_id: str) -> dict[str, object]:
    try:
        return get_run(run_id).model_dump()
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Run not found") from exc


@app.get("/runs/{run_id}/metrics")
def get_run_metrics(run_id: str) -> list[dict[str, object]]:
    return list_metrics(run_id)


@app.get("/runs/{run_id}/replays")
def get_run_replays(run_id: str) -> list[dict[str, object]]:
    return [artifact for artifact in list_artifacts(run_id) if artifact["artifact_type"] == "replay"]


@app.get("/runs/{run_id}/live")
def get_run_live_state(run_id: str) -> dict[str, object]:
    live_path = RUNS_DIR / run_id / "live" / "latest.json"
    if not live_path.exists():
        raise HTTPException(status_code=404, detail="Live state not available")
    return json.loads(live_path.read_text(encoding="utf-8"))


@app.post("/benchmarks")
def post_benchmark(request: BenchmarkRequest) -> dict[str, object]:
    created_runs = []
    for algorithm in request.algorithms:
        config = RunConfig(
            name=f"benchmark-{algorithm.value}",
            algorithm=algorithm,
            mode="benchmark",
            seed=request.seed,
            checkpoint_every=request.checkpoint_every,
            total_iterations=request.total_iterations,
            env=request.env,
            algorithm_params=request.algorithm_params.get(algorithm.value, {}),
        )
        created_runs.append(create_run(config).model_dump())
    return {"runs": created_runs}


def run() -> None:
    uvicorn.run("app.api.main:app", host="0.0.0.0", port=8000, reload=False)
