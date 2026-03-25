from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def mlp_parameter_count(input_size: int, hidden_sizes: list[int], output_size: int) -> int:
    total = 0
    previous = input_size
    for size in [*hidden_sizes, output_size]:
        total += (previous * size) + size
        previous = size
    return total


@dataclass(slots=True)
class FeedForwardPolicy:
    layer_sizes: list[int]
    flat_weights: np.ndarray
    layers: list[tuple[np.ndarray, np.ndarray]] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        self.flat_weights = np.asarray(self.flat_weights, dtype=float)
        expected = mlp_parameter_count(self.layer_sizes[0], self.layer_sizes[1:-1], self.layer_sizes[-1])
        if len(self.flat_weights) != expected:
            raise ValueError(f"Expected {expected} weights, got {len(self.flat_weights)}.")
        self.layers = []
        index = 0
        for in_size, out_size in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            weight_count = in_size * out_size
            weights = self.flat_weights[index : index + weight_count].reshape(in_size, out_size)
            index += weight_count
            bias = self.flat_weights[index : index + out_size]
            index += out_size
            self.layers.append((weights, bias))

    def act(self, observation: np.ndarray) -> np.ndarray:
        values = np.asarray(observation, dtype=float)
        for layer_index, (weights, bias) in enumerate(self.layers):
            values = values @ weights + bias
            if layer_index < len(self.layers) - 1:
                values = np.tanh(values)
        return np.tanh(values)
