from __future__ import annotations

from app.config.models import AlgorithmName, RunConfig


DEFAULT_ALGORITHM_PARAMS: dict[AlgorithmName, dict[str, object]] = {
    AlgorithmName.GA: {
        "population_size": 24,
        "elite_count": 4,
        "hidden_sizes": [16, 16],
        "mutation_std": 0.15,
        "mutation_rate": 0.2,
        "crossover_rate": 0.5,
        "live_display_count": 6,
    },
    AlgorithmName.NEAT: {
        "population_size": 32,
        "fitness_threshold": 120.0,
        "episodes_per_seed": 1,
    },
    AlgorithmName.PPO: {
        "policy_hidden_sizes": [32, 32],
        "total_timesteps": 4_096,
        "learning_rate": 3e-4,
        "rollout_steps": 512,
        "ppo_epochs": 6,
        "clip_epsilon": 0.2,
    },
}


def with_algorithm_defaults(config: RunConfig) -> RunConfig:
    merged = dict(DEFAULT_ALGORITHM_PARAMS[config.algorithm])
    merged.update(config.algorithm_params)
    return config.model_copy(update={"algorithm_params": merged})
