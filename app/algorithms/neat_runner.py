from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np

from app.algorithms.base import AlgorithmResult, MetricCallback
from app.benchmark.evaluate import evaluate_policy
from app.benchmark.suites import get_track_suite
from app.config.defaults import with_algorithm_defaults
from app.config.models import MetricPoint, RunConfig
from app.simulator.environment import DrivingEnv

try:
    import neat
except Exception:  # pragma: no cover - import guard
    neat = None


class _MetricsReporter(neat.reporting.BaseReporter if neat is not None else object):  # type: ignore[misc]
    def __init__(self, metric_callback: MetricCallback, started_at: float) -> None:
        self.metric_callback = metric_callback
        self.started_at = started_at
        self.generation = 0

    def start_generation(self, generation: int) -> None:
        self.generation = generation

    def post_evaluate(self, config, population, species, best_genome) -> None:  # pragma: no cover - exercised via runtime
        fitness_values = [float(genome.fitness or 0.0) for genome in population.values()]
        completion = float(getattr(best_genome, "completion", 0.0))
        crash_rate = float(getattr(best_genome, "crash_rate", 0.0))
        self.metric_callback(
            MetricPoint(
                step=self.generation,
                phase="train",
                reward=max(fitness_values) if fitness_values else 0.0,
                completion=completion,
                crash_rate=crash_rate,
                wall_time=time.time() - self.started_at,
            )
        )


def _write_neat_config(path: Path, inputs: int, outputs: int, population_size: int, fitness_threshold: float) -> None:
    config_text = f"""
[NEAT]
fitness_criterion     = max
fitness_threshold     = {fitness_threshold}
pop_size              = {population_size}
reset_on_extinction   = False

[DefaultGenome]
activation_default      = tanh
activation_mutate_rate  = 0.0
activation_options      = tanh
aggregation_default     = sum
aggregation_mutate_rate = 0.0
aggregation_options     = sum
bias_init_mean          = 0.0
bias_init_stdev         = 1.0
bias_max_value          = 30.0
bias_min_value          = -30.0
bias_mutate_power       = 0.5
bias_mutate_rate        = 0.7
bias_replace_rate       = 0.1
compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient   = 0.5
conn_add_prob           = 0.5
conn_delete_prob        = 0.2
enabled_default         = True
enabled_mutate_rate     = 0.01
feed_forward            = True
initial_connection      = full
node_add_prob           = 0.2
node_delete_prob        = 0.1
num_hidden              = 0
num_inputs              = {inputs}
num_outputs             = {outputs}
response_init_mean      = 1.0
response_init_stdev     = 0.0
response_max_value      = 30.0
response_min_value      = -30.0
response_mutate_power   = 0.0
response_mutate_rate    = 0.0
response_replace_rate   = 0.0
weight_init_mean        = 0.0
weight_init_stdev       = 1.0
weight_max_value        = 30
weight_min_value        = -30
weight_mutate_power     = 0.5
weight_mutate_rate      = 0.8
weight_replace_rate     = 0.1

[DefaultSpeciesSet]
compatibility_threshold = 3.0

[DefaultStagnation]
species_fitness_func = max
max_stagnation       = 10
species_elitism      = 2

[DefaultReproduction]
elitism            = 2
survival_threshold = 0.2
"""
    path.write_text(config_text.strip(), encoding="utf-8")


def run_neat(config: RunConfig, run_dir: Path, metric_callback: MetricCallback) -> AlgorithmResult:
    if neat is None:  # pragma: no cover - import guard
        raise RuntimeError("neat-python is required for NEAT runs.")

    config = with_algorithm_defaults(config)
    params = config.algorithm_params
    train_seeds = get_track_suite(config.env.train_suite)
    validation_seeds = get_track_suite(config.env.validation_suite)
    test_seeds = get_track_suite(config.env.test_suite)
    probe_env = DrivingEnv(config.env, track_seed=train_seeds[0])
    observation, _ = probe_env.reset(seed=config.seed, options={"track_seed": train_seeds[0]})

    neat_dir = run_dir / "checkpoints"
    neat_dir.mkdir(parents=True, exist_ok=True)
    config_path = neat_dir / "neat-config.ini"
    _write_neat_config(
        config_path,
        inputs=len(observation),
        outputs=2,
        population_size=int(params["population_size"]),
        fitness_threshold=float(params["fitness_threshold"]),
    )
    neat_config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(config_path),
    )

    def evaluate_genomes(genomes, config_obj) -> None:  # pragma: no cover - exercised via runtime
        for _, genome in genomes:
            network = neat.nn.FeedForwardNetwork.create(genome, config_obj)
            result = evaluate_policy(
                lambda obs: np.asarray(network.activate(obs), dtype=np.float32),
                train_seeds,
                config.env,
            )
            genome.fitness = result.mean_reward + (result.mean_completion * 100.0)
            genome.completion = result.mean_completion
            genome.crash_rate = result.crash_rate

    started_at = time.time()
    population = neat.Population(neat_config)
    population.add_reporter(_MetricsReporter(metric_callback, started_at))
    winner = population.run(evaluate_genomes, config.total_iterations)
    winner_network = neat.nn.FeedForwardNetwork.create(winner, neat_config)

    validation = evaluate_policy(
        lambda obs: np.asarray(winner_network.activate(obs), dtype=np.float32),
        validation_seeds,
        config.env,
    )
    replay_dir = run_dir / "replays"
    replay_dir.mkdir(parents=True, exist_ok=True)
    test = evaluate_policy(
        lambda obs: np.asarray(winner_network.activate(obs), dtype=np.float32),
        test_seeds,
        config.env,
        replay_path=replay_dir / "neat_test_replay.json",
    )

    winner_path = neat_dir / "neat_winner.pkl"
    with winner_path.open("wb") as file_handle:
        pickle.dump({"genome": winner, "config_path": str(config_path)}, file_handle)

    result = AlgorithmResult(
        summary={
            "winner_fitness": round(float(winner.fitness or 0.0), 5),
            "validation": validation.to_summary(),
            "test": test.to_summary(),
        }
    )
    result.add_artifact("checkpoint", winner_path, {"algorithm": "neat"})
    result.add_artifact("config", config_path, {"algorithm": "neat"})
    if test.replay_path:
        result.add_artifact("replay", test.replay_path, {"suite": "test", "algorithm": "neat"})
    return result

