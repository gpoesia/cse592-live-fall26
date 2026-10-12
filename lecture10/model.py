"""
A linear classifier and a simple ReLU neural net
using our scalar autograd engine.
"""

import math
import random

from parameter import Matrix, Vector


class Layer:
    def __init__(self, n_inputs, n_outputs, rng):
        scale = math.sqrt(6 / n_inputs)
        self.weights = Matrix([
            [rng.uniform(-scale, scale) for _ in range(n_inputs)]
            for _ in range(n_outputs)
        ])
        self.bias = Vector([0.0] * n_outputs)

    def __call__(self, x):
        return self.weights * x + self.bias

    def parameters(self):
        return self.weights.parameters() + self.bias.parameters()


class LinearClassifier:
    def __init__(self, seed=592):
        self.layers = [Layer(2, 1, random.Random(seed))]

    def __call__(self, x):
        return self.layers[0](Vector(x)).values[0]

    def parameters(self):
        return self.layers[0].parameters()


class NeuralNet:
    """Three linear layers with ReLU activations:
        2 (input size) -> width -> width -> 1 (output).
    """

    def __init__(self, width=16, seed=592):
        rng = random.Random(seed)
        self.layers = [Layer(2, width, rng), Layer(width, width, rng),
                       Layer(width, 1, rng)]

    def __call__(self, x):
        x = Vector(x)
        for layer in self.layers[:-1]:
            x = layer(x).relu()
        return self.layers[-1](x).values[0]

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
