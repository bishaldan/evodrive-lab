from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.algorithms.base import AlgorithmResult, MetricCallback
from app.benchmark.evaluate import evaluate_policy
from app.benchmark.suites import get_track_suite
from app.config.defaults import with_algorithm_defaults
from app.config.models import MetricPoint, RunConfig
from app.simulator.environment import DrivingEnv


@dataclass(slots=True)
class _TrajectoryBatch:
    observations: np.ndarray
    actions: np.ndarray
    old_log_probs: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray


class _Adam:
    def __init__(self, parameters: list[np.ndarray], learning_rate: float) -> None:
        self.parameters = parameters
        self.learning_rate = learning_rate
        self.first_moment = [np.zeros_like(parameter) for parameter in parameters]
        self.second_moment = [np.zeros_like(parameter) for parameter in parameters]
        self.timestep = 0

    def step(self, gradients: list[np.ndarray]) -> None:
        self.timestep += 1
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8
        for index, (parameter, gradient) in enumerate(zip(self.parameters, gradients, strict=True)):
            self.first_moment[index] = (beta1 * self.first_moment[index]) + ((1 - beta1) * gradient)
            self.second_moment[index] = (beta2 * self.second_moment[index]) + ((1 - beta2) * (gradient**2))
            first_hat = self.first_moment[index] / (1 - (beta1**self.timestep))
            second_hat = self.second_moment[index] / (1 - (beta2**self.timestep))
            parameter -= self.learning_rate * first_hat / (np.sqrt(second_hat) + epsilon)


class _MLP:
    def __init__(self, layer_sizes: list[int], seed: int, *, output_tanh: bool) -> None:
        rng = np.random.default_rng(seed)
        self.layer_sizes = layer_sizes
        self.output_tanh = output_tanh
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:]):
            scale = math.sqrt(2.0 / max(in_size, 1))
            self.weights.append(rng.normal(0.0, scale, size=(in_size, out_size)).astype(np.float64))
            self.biases.append(np.zeros(out_size, dtype=np.float64))
        self.parameters = [*self.weights, *self.biases]

    def forward(self, inputs: np.ndarray) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
        activations = [inputs]
        pre_activations: list[np.ndarray] = []
        values = inputs
        for layer_index, (weights, biases) in enumerate(zip(self.weights, self.biases, strict=True)):
            linear = values @ weights + biases
            pre_activations.append(linear)
            is_last = layer_index == len(self.weights) - 1
            if is_last:
                values = np.tanh(linear) if self.output_tanh else linear
            else:
                values = np.tanh(linear)
            activations.append(values)
        return values, activations, pre_activations

    def backward(
        self,
        activations: list[np.ndarray],
        pre_activations: list[np.ndarray],
        grad_output: np.ndarray,
    ) -> list[np.ndarray]:
        grad_weights: list[np.ndarray] = [np.zeros_like(weight) for weight in self.weights]
        grad_biases: list[np.ndarray] = [np.zeros_like(bias) for bias in self.biases]
        delta = grad_output
        if self.output_tanh:
            delta = delta * (1.0 - np.tanh(pre_activations[-1]) ** 2)
        for index in reversed(range(len(self.weights))):
            grad_weights[index] = activations[index].T @ delta / max(len(activations[index]), 1)
            grad_biases[index] = delta.mean(axis=0)
            if index == 0:
                continue
            delta = (delta @ self.weights[index].T) * (1.0 - np.tanh(pre_activations[index - 1]) ** 2)
        return [*grad_weights, *grad_biases]


class _PPOPolicy:
    def __init__(self, observation_size: int, hidden_sizes: list[int], seed: int, learning_rate: float) -> None:
        self.network = _MLP([observation_size, *hidden_sizes, 2], seed, output_tanh=False)
        self.log_std = np.full(2, -0.7, dtype=np.float64)
        self.optimizer = _Adam([*self.network.parameters, self.log_std], learning_rate)

    def act(self, observation: np.ndarray, rng: np.random.Generator, deterministic: bool = False) -> tuple[np.ndarray, float]:
        observation_batch = observation.reshape(1, -1).astype(np.float64)
        mean, _, _ = self.network.forward(observation_batch)
        mean = mean[0]
        std = np.exp(self.log_std)
        if deterministic:
            action = np.tanh(mean)
        else:
            raw_action = mean + (std * rng.normal(size=mean.shape))
            action = np.clip(raw_action, -1.0, 1.0)
        log_prob = self._log_prob(action, mean, std)
        return action.astype(np.float32), float(log_prob)

    def _log_prob(self, actions: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        variance = std**2
        return -0.5 * np.sum((((actions - mean) ** 2) / variance) + (2 * self.log_std) + np.log(2 * np.pi), axis=-1)

    def update(self, batch: _TrajectoryBatch, clip_epsilon: float, epochs: int) -> None:
        observations = batch.observations.astype(np.float64)
        actions = batch.actions.astype(np.float64)
        advantages = batch.advantages.astype(np.float64)
        returns = batch.returns.astype(np.float64)  # used for scaling only
        old_log_probs = batch.old_log_probs.astype(np.float64)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        scale = max(float(np.abs(returns).mean()), 1.0)

        for _ in range(epochs):
            mean, activations, pre_activations = self.network.forward(observations)
            std = np.exp(self.log_std)
            log_probs = self._log_prob(actions, mean, std)
            ratios = np.exp(log_probs - old_log_probs)

            active = np.ones_like(ratios, dtype=bool)
            active[(advantages > 0) & (ratios > 1.0 + clip_epsilon)] = False
            active[(advantages < 0) & (ratios < 1.0 - clip_epsilon)] = False

            coeff = np.zeros_like(ratios)
            coeff[active] = -(advantages[active] * ratios[active]) / len(ratios)

            grad_mean = coeff[:, None] * ((mean - actions) / (std**2))
            grad_log_std = np.sum(
                coeff[:, None] * (((((actions - mean) ** 2) / (std**2)) - 1.0)),
                axis=0,
            )

            network_grads = self.network.backward(activations, pre_activations, grad_mean / scale)
            self.optimizer.step([*network_grads, grad_log_std / scale])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, np.ndarray] = {"log_std": self.log_std}
        for index, weights in enumerate(self.network.weights):
            payload[f"w{index}"] = weights
        for index, biases in enumerate(self.network.biases):
            payload[f"b{index}"] = biases
        np.savez(path, **payload)


class _ValueModel:
    def __init__(self, observation_size: int, hidden_sizes: list[int], seed: int, learning_rate: float) -> None:
        self.network = _MLP([observation_size, *hidden_sizes, 1], seed + 101, output_tanh=False)
        self.optimizer = _Adam(self.network.parameters, learning_rate)

    def predict(self, observations: np.ndarray) -> np.ndarray:
        values, _, _ = self.network.forward(observations.astype(np.float64))
        return values[:, 0]

    def update(self, observations: np.ndarray, returns: np.ndarray, epochs: int) -> None:
        observations = observations.astype(np.float64)
        targets = returns.astype(np.float64).reshape(-1, 1)
        for _ in range(epochs):
            predictions, activations, pre_activations = self.network.forward(observations)
            grad_output = (2.0 * (predictions - targets)) / max(len(observations), 1)
            gradients = self.network.backward(activations, pre_activations, grad_output)
            self.optimizer.step(gradients)


def _collect_rollouts(
    policy: _PPOPolicy,
    value_model: _ValueModel,
    config: RunConfig,
    track_seeds: list[int],
    rollout_steps: int,
    rng: np.random.Generator,
) -> tuple[_TrajectoryBatch, float, float, float]:
    env = DrivingEnv(config.env, track_seed=track_seeds[0])
    observation, _ = env.reset(options={"track_seed": int(rng.choice(track_seeds))})
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    dones: list[float] = []
    log_probs: list[float] = []
    values: list[float] = []
    completions: list[float] = []
    crashes: list[float] = []

    for _ in range(rollout_steps):
        action, log_prob = policy.act(observation, rng, deterministic=False)
        value = float(value_model.predict(observation.reshape(1, -1))[0])
        next_observation, reward, terminated, truncated, info = env.step(action)
        observations.append(observation.copy())
        actions.append(action.copy())
        rewards.append(float(reward))
        dones.append(1.0 if (terminated or truncated) else 0.0)
        log_probs.append(float(log_prob))
        values.append(value)
        completions.append(float(info["completion"]))
        crashes.append(float(info["crashed"]))
        observation = next_observation
        if terminated or truncated:
            observation, _ = env.reset(options={"track_seed": int(rng.choice(track_seeds))})

    gamma = 0.99
    gae_lambda = 0.95
    observations_array = np.asarray(observations, dtype=np.float64)
    values_array = np.asarray(values, dtype=np.float64)
    rewards_array = np.asarray(rewards, dtype=np.float64)
    dones_array = np.asarray(dones, dtype=np.float64)
    next_value = float(value_model.predict(observation.reshape(1, -1))[0])

    advantages = np.zeros_like(rewards_array)
    last_advantage = 0.0
    for index in reversed(range(len(rewards_array))):
        mask = 1.0 - dones_array[index]
        next_val = next_value if index == len(rewards_array) - 1 else values_array[index + 1]
        delta = rewards_array[index] + (gamma * next_val * mask) - values_array[index]
        last_advantage = delta + (gamma * gae_lambda * mask * last_advantage)
        advantages[index] = last_advantage
    returns = advantages + values_array

    batch = _TrajectoryBatch(
        observations=observations_array,
        actions=np.asarray(actions, dtype=np.float64),
        old_log_probs=np.asarray(log_probs, dtype=np.float64),
        advantages=advantages,
        returns=returns,
    )
    return (
        batch,
        float(np.mean(rewards_array)),
        float(np.mean(completions)),
        float(np.mean(crashes)),
    )


def run_ppo(config: RunConfig, run_dir: Path, metric_callback: MetricCallback) -> AlgorithmResult:
    config = with_algorithm_defaults(config)
    params = config.algorithm_params
    train_seeds = get_track_suite(config.env.train_suite)
    validation_seeds = get_track_suite(config.env.validation_suite)
    test_seeds = get_track_suite(config.env.test_suite)
    rng = np.random.default_rng(config.seed)

    probe_env = DrivingEnv(config.env, track_seed=train_seeds[0])
    observation, _ = probe_env.reset(options={"track_seed": train_seeds[0]})
    hidden_sizes = list(params["policy_hidden_sizes"])
    learning_rate = float(params["learning_rate"])
    rollout_steps = int(params.get("rollout_steps", 512))
    ppo_epochs = int(params.get("ppo_epochs", 6))
    clip_epsilon = float(params.get("clip_epsilon", 0.2))
    total_timesteps = int(params["total_timesteps"])
    updates = max(1, math.ceil(total_timesteps / rollout_steps))

    policy = _PPOPolicy(len(observation), hidden_sizes, config.seed, learning_rate)
    value_model = _ValueModel(len(observation), hidden_sizes, config.seed, learning_rate)
    started_at = time.time()

    for update in range(updates):
        batch, mean_reward, mean_completion, crash_rate = _collect_rollouts(
            policy,
            value_model,
            config,
            train_seeds,
            rollout_steps,
            rng,
        )
        value_model.update(batch.observations, batch.returns, epochs=ppo_epochs)
        policy.update(batch, clip_epsilon=clip_epsilon, epochs=ppo_epochs)

        metric_callback(
            MetricPoint(
                step=(update + 1) * rollout_steps,
                phase="train",
                reward=mean_reward,
                completion=mean_completion,
                crash_rate=crash_rate,
                wall_time=time.time() - started_at,
            )
        )

        if (update + 1) % max(config.checkpoint_every, 1) == 0:
            validation = evaluate_policy(
                lambda obs: policy.act(obs, rng, deterministic=True)[0],
                validation_seeds,
                config.env,
            )
            metric_callback(
                MetricPoint(
                    step=(update + 1) * rollout_steps,
                    phase="validation",
                    reward=validation.mean_reward,
                    completion=validation.mean_completion,
                    crash_rate=validation.crash_rate,
                    wall_time=time.time() - started_at,
                )
            )

    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model_path = checkpoint_dir / "ppo_lite_weights.npz"
    policy.save(model_path)

    validation = evaluate_policy(
        lambda obs: policy.act(obs, rng, deterministic=True)[0],
        validation_seeds,
        config.env,
    )
    replay_dir = run_dir / "replays"
    replay_dir.mkdir(parents=True, exist_ok=True)
    test = evaluate_policy(
        lambda obs: policy.act(obs, rng, deterministic=True)[0],
        test_seeds,
        config.env,
        replay_path=replay_dir / "ppo_test_replay.json",
    )

    result = AlgorithmResult(
        summary={
            "timesteps": updates * rollout_steps,
            "implementation": "numpy_ppo_lite",
            "validation": validation.to_summary(),
            "test": test.to_summary(),
        }
    )
    result.add_artifact("checkpoint", model_path, {"algorithm": "ppo", "implementation": "numpy_ppo_lite"})
    if test.replay_path:
        result.add_artifact("replay", test.replay_path, {"suite": "test", "algorithm": "ppo"})
    return result
