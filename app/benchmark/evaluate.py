from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from app.config.models import EnvironmentConfig
from app.simulator.environment import DrivingEnv


PolicyFn = Callable[[np.ndarray], np.ndarray]


@dataclass(slots=True)
class EvaluationResult:
    mean_reward: float
    mean_completion: float
    crash_rate: float
    mean_steps: float
    replay_path: str | None = None

    def to_summary(self) -> dict[str, float | str]:
        summary: dict[str, float | str] = {
            "mean_reward": round(self.mean_reward, 5),
            "mean_completion": round(self.mean_completion, 5),
            "crash_rate": round(self.crash_rate, 5),
            "mean_steps": round(self.mean_steps, 2),
        }
        if self.replay_path:
            summary["replay_path"] = self.replay_path
        return summary


def run_episode(
    env: DrivingEnv,
    policy_fn: PolicyFn,
) -> tuple[float, float, bool, int]:
    observation, _ = env.reset(options={"track_seed": env.track_seed})
    total_reward = 0.0
    done = False
    truncated = False
    info: dict[str, float | bool | int] = {"completion": 0.0, "crashed": False, "steps": 0}
    while not (done or truncated):
        action = policy_fn(observation)
        observation, reward, done, truncated, info = env.step(action)
        total_reward += reward
    return total_reward, float(info["completion"]), bool(info["crashed"]), int(info["steps"])


def evaluate_policy(
    policy_fn: PolicyFn,
    track_seeds: list[int],
    env_config: EnvironmentConfig,
    replay_path: str | Path | None = None,
) -> EvaluationResult:
    rewards: list[float] = []
    completions: list[float] = []
    crashes: list[bool] = []
    steps: list[int] = []
    saved_replay: str | None = None

    for index, track_seed in enumerate(track_seeds):
        env = DrivingEnv(env_config, track_seed=track_seed)
        reward, completion, crashed, episode_steps = run_episode(env, policy_fn)
        rewards.append(reward)
        completions.append(completion)
        crashes.append(crashed)
        steps.append(episode_steps)
        if replay_path is not None and index == 0:
            saved_replay = env.save_replay(replay_path)

    return EvaluationResult(
        mean_reward=float(np.mean(rewards)) if rewards else 0.0,
        mean_completion=float(np.mean(completions)) if completions else 0.0,
        crash_rate=float(np.mean(crashes)) if crashes else 0.0,
        mean_steps=float(np.mean(steps)) if steps else 0.0,
        replay_path=saved_replay,
    )

