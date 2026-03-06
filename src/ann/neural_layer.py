"""
Neural Layer Implementation
Handles weight initialization, forward pass, and gradient computation
"""

import numpy as np

from .activations import get_activation, get_activation_derivative


class NeuralLayer:
    """
    Fully connected layer with optional activation.
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        activation=None,
        weight_init="xavier",
        rng=None,
    ):
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.activation_name = activation if activation is not None else "linear"
        self.activation_fn = get_activation(self.activation_name)
        self.activation_grad_fn = get_activation_derivative(self.activation_name)
        self.rng = np.random.default_rng() if rng is None else rng

        self.W, self.b = self._initialize_parameters(weight_init)

        self.input_cache = None
        self.z_cache = None
        self.a_cache = None

        # Exposed for autograder checks
        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.b)

    def _initialize_parameters(self, weight_init):
        init_name = weight_init.lower()

        if init_name == "xavier":
            limit = np.sqrt(6.0 / (self.input_dim + self.output_dim))
            W = self.rng.uniform(-limit, limit, size=(self.input_dim, self.output_dim))
            b = np.zeros((1, self.output_dim), dtype=np.float64)
            return W.astype(np.float64), b

        if init_name == "random":
            W = self.rng.normal(0.0, 0.01, size=(self.input_dim, self.output_dim))
            b = np.zeros((1, self.output_dim), dtype=np.float64)
            return W.astype(np.float64), b

        if init_name == "zeros":
            W = np.zeros((self.input_dim, self.output_dim), dtype=np.float64)
            b = np.zeros((1, self.output_dim), dtype=np.float64)
            return W, b

        raise ValueError(f"Unsupported weight initialization: {weight_init}")

    def forward(self, X):
        self.input_cache = X
        self.z_cache = X @ self.W + self.b
        self.a_cache = self.activation_fn(self.z_cache)
        return self.a_cache

    def backward(self, grad_output):
        # grad_output is dL/dA for this layer output
        grad_activation = self.activation_grad_fn(self.z_cache, self.a_cache)
        grad_z = grad_output * grad_activation

        self.grad_W = self.input_cache.T @ grad_z
        self.grad_b = np.sum(grad_z, axis=0, keepdims=True)

        grad_input = grad_z @ self.W.T
        return grad_input
