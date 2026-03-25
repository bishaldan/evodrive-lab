from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from app.algorithms.base import AlgorithmResult, MetricCallback
from app.benchmark.evaluate import evaluate_policy
from app.benchmark.suites import get_track_suite
from app.config.defaults import with_algorithm_defaults
from app.config.models import MetricPoint, RunConfig
from app.policies.network import FeedForwardPolicy, mlp_parameter_count
from app.simulator.environment import DrivingEnv


LIVE_COLORS = [
    "#ff595e",
    "#1982c4",
    "#8ac926",
    "#ffca3a",
    "#6a4c93",
    "#ff924c",
    "#00bfb3",
    "#c1121f",
]


def _uniform_crossover(rng: np.random.Generator, first: np.ndarray, second: np.ndarray) -> np.ndarray:
    mask = rng.random(len(first)) < 0.5
    return np.where(mask, first, second)


def _mutate(rng: np.random.Generator, genome: np.ndarray, mutation_rate: float, mutation_std: float) -> np.ndarray:
    mask = rng.random(len(genome)) < mutation_rate
    noise = rng.normal(0.0, mutation_std, size=len(genome))
    mutated = genome.copy()
    mutated[mask] += noise[mask]
    return mutated


def _simulate_policy_trace(
    policy: FeedForwardPolicy,
    env_config,
    track_seed: int,
) -> tuple[DrivingEnv, float, dict[str, float | bool | int]]:
    env = DrivingEnv(env_config, track_seed=track_seed)
    observation, _ = env.reset(options={"track_seed": track_seed})
    total_reward = 0.0
    terminated = False
    truncated = False
    info: dict[str, float | bool | int] = {"completion": 0.0, "crashed": False, "steps": 0}
    while not (terminated or truncated):
        action = policy.act(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
    return env, total_reward, info


def _write_live_population_snapshot(
    run_dir: Path,
    generation: int,
    env_config,
    track_seed: int,
    layer_sizes: list[int],
    scored_population: list[tuple[float, np.ndarray, dict[str, float | str]]],
    display_count: int,
) -> Path:
    live_dir = run_dir / "live"
    live_dir.mkdir(parents=True, exist_ok=True)
    displayed = scored_population[: max(3, display_count)]
    cars: list[dict[str, object]] = []
    leaderboard: list[dict[str, object]] = []
    track_payload: dict[str, object] | None = None

    for rank, (score, genome, train_summary) in enumerate(displayed, start=1):
        policy = FeedForwardPolicy(layer_sizes, genome)
        env, reward, info = _simulate_policy_trace(policy, env_config, track_seed)
        assert env.track is not None
        if track_payload is None:
            track_payload = {
                "track_seed": track_seed,
                "centerline": env.track.centerline.round(4).tolist(),
                "left_wall": env.track.left_wall.round(4).tolist(),
                "right_wall": env.track.right_wall.round(4).tolist(),
                "start_gate": env.track.start_gate.round(4).tolist(),
                "finish_gate": env.track.finish_gate.round(4).tolist(),
                "checkpoint_gates": env.track.checkpoint_gates.round(4).tolist(),
                "track_profile": env.track.profile,
                "physics": {
                    "sensor_range": env.physics.sensor_range,
                    "num_rays": env.physics.num_rays,
                    "ray_spread_deg": env.physics.ray_spread_deg,
                    "car_radius": env.physics.car_radius,
                    "track_width": env.physics.track_width,
                },
            }
        completion = float(info["completion"])
        crashed = bool(info["crashed"])
        steps = int(info["steps"])
        leaderboard.append(
            {
                "rank": rank,
                "score": round(score, 5),
                "reward": round(reward, 5),
                "completion": round(completion, 5),
                "crashed": crashed,
                "steps": steps,
                "max_sector": max((int(frame.get("sector", 0)) for frame in env.trajectory), default=0),
                "mean_reward": train_summary["mean_reward"],
                "mean_completion": train_summary["mean_completion"],
            }
        )
        cars.append(
            {
                "car_id": f"g{generation + 1}-r{rank}",
                "rank": rank,
                "score": round(score, 5),
                "reward": round(reward, 5),
                "completion": round(completion, 5),
                "crashed": crashed,
                "steps": steps,
                "color": LIVE_COLORS[(rank - 1) % len(LIVE_COLORS)],
                "max_sector": max((int(frame.get("sector", 0)) for frame in env.trajectory), default=0),
                "frames": env.trajectory,
            }
        )

    assert track_payload is not None
    payload = {
        "type": "population_replay",
        "algorithm": "ga",
        "generation": generation + 1,
        "best_score": round(scored_population[0][0], 5),
        "mean_score": round(float(np.mean([item[0] for item in scored_population])), 5),
        "population_size": len(scored_population),
        "display_count": len(cars),
        **track_payload,
        "cars": cars,
        "leaderboard": leaderboard,
    }
    generation_path = live_dir / f"generation_{generation + 1:03d}.json"
    latest_path = live_dir / "latest.json"
    payload_json = json.dumps(payload, indent=2)
    generation_path.write_text(payload_json, encoding="utf-8")
    latest_path.write_text(payload_json, encoding="utf-8")
    return latest_path


def run_ga(config: RunConfig, run_dir: Path, metric_callback: MetricCallback) -> AlgorithmResult:
    config = with_algorithm_defaults(config)
    params = config.algorithm_params
    rng = np.random.default_rng(config.seed)
    train_seeds = get_track_suite(config.env.train_suite)
    validation_seeds = get_track_suite(config.env.validation_suite)
    test_seeds = get_track_suite(config.env.test_suite)

    probe_env = DrivingEnv(config.env, track_seed=train_seeds[0])
    observation, _ = probe_env.reset(seed=config.seed, options={"track_seed": train_seeds[0]})
    hidden_sizes = list(params["hidden_sizes"])
    layer_sizes = [len(observation), *hidden_sizes, 2]
    parameter_count = mlp_parameter_count(layer_sizes[0], layer_sizes[1:-1], layer_sizes[-1])

    population_size = int(params["population_size"])
    elite_count = int(params["elite_count"])
    mutation_rate = float(params["mutation_rate"])
    mutation_std = float(params["mutation_std"])
    live_display_count = min(max(int(params.get("live_display_count", 6)), 3), population_size)

    population = rng.normal(0.0, 0.55, size=(population_size, parameter_count))
    best_genome = population[0].copy()
    best_score = float("-inf")
    start = time.time()
    live_snapshot_path: Path | None = None

    for generation in range(config.total_iterations):
        scored_population: list[tuple[float, np.ndarray, dict[str, float | str]]] = []
        for genome in population:
            policy = FeedForwardPolicy(layer_sizes, genome)
            evaluation = evaluate_policy(policy.act, train_seeds, config.env)
            score = evaluation.mean_reward + (evaluation.mean_completion * 100.0)
            scored_population.append((score, genome.copy(), evaluation.to_summary()))
        scored_population.sort(key=lambda item: item[0], reverse=True)
        generation_score, generation_best, train_summary = scored_population[0]
        if generation_score > best_score:
            best_score = generation_score
            best_genome = generation_best.copy()
        metric_callback(
            MetricPoint(
                step=generation,
                phase="train",
                reward=float(train_summary["mean_reward"]),
                completion=float(train_summary["mean_completion"]),
                crash_rate=float(train_summary["crash_rate"]),
                wall_time=time.time() - start,
                extras={"score": generation_score},
            )
        )
        live_snapshot_path = _write_live_population_snapshot(
            run_dir,
            generation,
            config.env,
            train_seeds[0],
            layer_sizes,
            scored_population,
            live_display_count,
        )
        elites = [entry[1] for entry in scored_population[:elite_count]]
        new_population = elites.copy()
        while len(new_population) < population_size:
            parent_a = elites[rng.integers(0, len(elites))]
            parent_b = elites[rng.integers(0, len(elites))]
            child = _uniform_crossover(rng, parent_a, parent_b)
            child = _mutate(rng, child, mutation_rate, mutation_std)
            new_population.append(child)
        population = np.asarray(new_population, dtype=float)

    weights_path = run_dir / "checkpoints" / "ga_best_weights.npz"
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(weights_path, weights=best_genome)

    best_policy = FeedForwardPolicy(layer_sizes, best_genome)
    validation = evaluate_policy(best_policy.act, validation_seeds, config.env)
    test_replay_path = run_dir / "replays" / "ga_test_replay.json"
    test_replay_path.parent.mkdir(parents=True, exist_ok=True)
    test = evaluate_policy(best_policy.act, test_seeds, config.env, replay_path=test_replay_path)

    result = AlgorithmResult(
        summary={
            "best_score": round(best_score, 5),
            "validation": validation.to_summary(),
            "test": test.to_summary(),
            "parameter_count": parameter_count,
        }
    )
    result.add_artifact("checkpoint", weights_path, {"algorithm": "ga"})
    result.add_artifact("replay", test.replay_path or "", {"suite": "test", "algorithm": "ga"})
    if live_snapshot_path is not None:
        result.add_artifact("live", live_snapshot_path, {"algorithm": "ga", "kind": "population_replay"})
    return result
