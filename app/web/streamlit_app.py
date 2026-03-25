from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

from app.web.replay_viz import build_population_figure, build_replay_figure

API_URL = os.getenv("EVODRIVE_API_URL", "http://localhost:8000")


def api_get(path: str):
    response = requests.get(f"{API_URL}{path}", timeout=15)
    response.raise_for_status()
    return response.json()


def api_post(path: str, payload: dict):
    response = requests.post(f"{API_URL}{path}", json=payload, timeout=30)
    response.raise_for_status()
    return response.json()


def api_get_optional(path: str):
    response = requests.get(f"{API_URL}{path}", timeout=15)
    if response.status_code == 404:
        return None
    response.raise_for_status()
    return response.json()


def sorted_runs(runs: list[dict[str, object]]) -> list[dict[str, object]]:
    ordered = sorted(runs, key=lambda run: str(run["created_at"]), reverse=True)
    status_order = {"running": 0, "queued": 1, "completed": 2, "failed": 3}
    return sorted(ordered, key=lambda run: status_order.get(str(run["status"]), 99))


st.set_page_config(page_title="EvoDrive Lab", layout="wide")
st.title("EvoDrive Lab")
st.caption("Docker-first driving benchmark for GA, NEAT, and PPO.")

simulation_tab, train_tab, replay_tab, benchmark_tab, runs_tab = st.tabs(
    ["Simulation", "Train", "Replay", "Benchmark", "Runs"]
)

with simulation_tab:
    st.subheader("Live population simulation")
    st.caption("Start a GA generation loop and watch multiple cars attempt the same track together.")
    config_col, selected_col = st.columns([1.2, 1.8])
    with config_col:
        simulation_seed = st.number_input("Simulation seed", min_value=0, max_value=999999, value=42, key="sim-seed")
        generation_count = st.number_input(
            "Generations",
            min_value=1,
            max_value=40,
            value=8,
            key="sim-generations",
        )
        population_size = st.slider("Cars per generation", min_value=3, max_value=12, value=6, key="sim-population")
        frame_stride_live = st.slider("Playback speed / sampling", min_value=1, max_value=8, value=3, key="sim-stride")
        auto_refresh = st.checkbox("Auto refresh while training", value=True)
        if st.button("Start Simulation", use_container_width=True):
            payload = {
                "name": f"live-ga-{int(simulation_seed)}",
                "algorithm": "ga",
                "mode": "train",
                "seed": int(simulation_seed),
                "total_iterations": int(generation_count),
                "algorithm_params": {
                    "population_size": int(population_size),
                    "elite_count": max(2, int(population_size) // 3),
                    "live_display_count": int(population_size),
                },
            }
            result = api_post("/runs", payload)
            st.session_state["live_run_id"] = result["id"]
            st.success(f"Simulation run {result['id']} queued.")

    all_runs = sorted_runs(api_get("/runs"))
    simulation_runs = [run for run in all_runs if run["algorithm"] == "ga"]
    with selected_col:
        if not simulation_runs:
            st.info("No GA simulation runs yet. Start one from the left panel.")
        else:
            run_labels = {f"{run['name']} [{run['status']}]": run["id"] for run in simulation_runs}
            preferred_run_id = st.session_state.get("live_run_id")
            run_keys = list(run_labels.keys())
            default_index = 0
            if preferred_run_id is not None:
                for index, key in enumerate(run_keys):
                    if run_labels[key] == preferred_run_id:
                        default_index = index
                        break
            selected_live = st.selectbox("Simulation run", run_keys, index=default_index)
            selected_live_id = run_labels[selected_live]
            details = api_get(f"/runs/{selected_live_id}")
            st.session_state["live_run_id"] = selected_live_id

            live_state = api_get_optional(f"/runs/{selected_live_id}/live")
            if live_state is None:
                st.info("The simulation is queued or still preparing its first generation replay.")
            else:
                summary_col, stat_col = st.columns([2.3, 1])
                with summary_col:
                    figure = build_population_figure(live_state, frame_stride=frame_stride_live)
                    st.plotly_chart(figure, use_container_width=True)
                with stat_col:
                    st.json(
                        {
                            "status": details["status"],
                            "generation": live_state["generation"],
                            "track_seed": live_state["track_seed"],
                            "cars_shown": live_state["display_count"],
                            "best_score": live_state["best_score"],
                            "mean_score": live_state["mean_score"],
                        }
                    )
                    leaderboard_df = pd.DataFrame(live_state.get("leaderboard", []))
                    if not leaderboard_df.empty:
                        st.dataframe(leaderboard_df, use_container_width=True, hide_index=True)
            simulation_metrics = api_get(f"/runs/{selected_live_id}/metrics")
            if simulation_metrics:
                metric_df = pd.DataFrame(simulation_metrics)
                history_fig = px.line(
                    metric_df,
                    x="step",
                    y=["reward", "completion"],
                    title="Generation history",
                )
                st.plotly_chart(history_fig, use_container_width=True)
            if auto_refresh and details["status"] in {"queued", "running"}:
                time.sleep(2)
                st.rerun()

with train_tab:
    st.subheader("Create a training run")
    algorithm = st.selectbox("Algorithm", ["ga", "neat", "ppo"])
    iterations = st.number_input("Total iterations", min_value=1, max_value=50, value=8)
    seed = st.number_input("Seed", min_value=0, max_value=999999, value=42)
    if st.button("Queue run", use_container_width=True):
        payload = {
            "algorithm": algorithm,
            "mode": "train",
            "seed": int(seed),
            "total_iterations": int(iterations),
        }
        result = api_post("/runs", payload)
        st.success(f"Queued run {result['id']} ({result['algorithm']}).")

with replay_tab:
    st.subheader("Replay artifacts")
    st.caption("Select a completed run, then press Play in the chart to watch the car drive the track.")
    runs = sorted_runs(api_get("/runs"))
    run_options = {f"{run['name']} [{run['status']}]": run["id"] for run in runs}
    selected = st.selectbox("Run", list(run_options.keys())) if run_options else None
    if selected:
        run_id = run_options[selected]
        replays = api_get(f"/runs/{run_id}/replays")
        if not replays:
            st.info("No replay exported yet.")
        else:
            replay = replays[0]
            replay_path = Path(replay["path"])
            st.write(f"Replay file: `{replay_path}`")
            if replay_path.exists():
                payload = json.loads(replay_path.read_text(encoding="utf-8"))
                if not payload.get("frames"):
                    st.warning("Replay was exported, but it does not contain any frames yet.")
                    st.stop()
                frame_stride = st.slider("Replay speed / sampling", min_value=1, max_value=12, value=6)
                figure = build_replay_figure(payload, frame_stride=frame_stride)
                st.plotly_chart(figure, use_container_width=True)

                frames = pd.DataFrame(payload["frames"])
                latest = frames.iloc[-1].to_dict() if not frames.empty else {}
                left_col, right_col = st.columns(2)
                with left_col:
                    st.json(
                        {
                            "track_seed": payload["track_seed"],
                            "frame_count": len(payload["frames"]),
                            "final_progress": latest.get("progress"),
                            "final_speed": latest.get("speed"),
                            "crashed": latest.get("crashed"),
                        }
                    )
                with right_col:
                    if not frames.empty:
                        telemetry = frames[["step", "speed", "progress"]].copy()
                        telemetry_fig = px.line(
                            telemetry,
                            x="step",
                            y=["speed", "progress"],
                            title="Replay telemetry",
                        )
                        st.plotly_chart(telemetry_fig, use_container_width=True)
            else:
                st.warning("Replay path was registered, but the file is not present on disk.")

with benchmark_tab:
    st.subheader("Queue benchmark comparison")
    selected_algorithms = st.multiselect(
        "Algorithms",
        options=["ga", "neat", "ppo"],
        default=["ga", "neat", "ppo"],
    )
    benchmark_iterations = st.number_input("Benchmark iterations", min_value=1, max_value=20, value=6)
    if st.button("Queue benchmark", use_container_width=True):
        result = api_post(
            "/benchmarks",
            {"algorithms": selected_algorithms, "total_iterations": int(benchmark_iterations)},
        )
        st.success(f"Queued {len(result['runs'])} benchmark runs.")
        st.dataframe(pd.DataFrame(result["runs"]))

with runs_tab:
    st.subheader("Run browser")
    runs = api_get("/runs")
    if not runs:
        st.info("No runs yet.")
    else:
        run_df = pd.DataFrame(runs)
        st.dataframe(run_df, use_container_width=True)
        selected_run_id = st.selectbox("Run details", run_df["id"].tolist())
        metrics = api_get(f"/runs/{selected_run_id}/metrics")
        if metrics:
            metric_df = pd.DataFrame(metrics)
            reward_fig = px.line(metric_df, x="step", y="reward", color="phase", title="Reward")
            completion_fig = px.line(metric_df, x="step", y="completion", color="phase", title="Completion")
            st.plotly_chart(reward_fig, use_container_width=True)
            st.plotly_chart(completion_fig, use_container_width=True)
        details = api_get(f"/runs/{selected_run_id}")
        st.json(details)
