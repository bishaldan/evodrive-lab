import numpy as np

from app.algorithms.ga import _mutate, _uniform_crossover


def test_ga_operations_preserve_shapes() -> None:
    rng = np.random.default_rng(0)
    first = rng.normal(size=16)
    second = rng.normal(size=16)
    crossed = _uniform_crossover(rng, first, second)
    mutated = _mutate(rng, crossed, mutation_rate=0.5, mutation_std=0.1)
    assert crossed.shape == first.shape
    assert mutated.shape == first.shape

