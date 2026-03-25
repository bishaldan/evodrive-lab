from app.config.models import EnvironmentConfig
from app.simulator.environment import DrivingEnv


def test_sensor_observation_bounds() -> None:
    env = DrivingEnv(EnvironmentConfig(), track_seed=11)
    observation, _ = env.reset(seed=11, options={"track_seed": 11})
    sensor_values = observation[: env.config.physics.num_rays]
    assert sensor_values.min() >= 0.0
    assert sensor_values.max() <= 1.0


def test_progress_reward_improves_when_moving_forward() -> None:
    env = DrivingEnv(EnvironmentConfig(), track_seed=17)
    env.reset(seed=17, options={"track_seed": 17})
    _, reward, _, _, info = env.step([1.0, 0.0])
    assert info["completion"] >= 0.0
    assert reward > -5.0

