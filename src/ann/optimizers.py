"""
Optimization Algorithms
Implements: SGD, Momentum, Adam, Nadam, etc.
"""

import numpy as np


class BaseOptimizer:
    def __init__(self, learning_rate=1e-3, weight_decay=0.0):
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.initialized = False

    def initialize(self, layers):
        self.initialized = True

    def _regularize(self, grad, weight):
        if self.weight_decay <= 0.0:
            return grad
        return grad + self.weight_decay * weight

    def step(self, layers):
        raise NotImplementedError


class SGDOptimizer(BaseOptimizer):
    def step(self, layers):
        for layer in layers:
            grad_w = self._regularize(layer.grad_W, layer.W)
            layer.W -= self.learning_rate * grad_w
            layer.b -= self.learning_rate * layer.grad_b


class MomentumOptimizer(BaseOptimizer):
    def __init__(self, learning_rate=1e-3, weight_decay=0.0, beta=0.9):
        super().__init__(learning_rate, weight_decay)
        self.beta = float(beta)
        self.v_w = []
        self.v_b = []

    def initialize(self, layers):
        self.v_w = [np.zeros_like(layer.W) for layer in layers]
        self.v_b = [np.zeros_like(layer.b) for layer in layers]
        self.initialized = True

    def step(self, layers):
        if not self.initialized:
            self.initialize(layers)

        for i, layer in enumerate(layers):
            grad_w = self._regularize(layer.grad_W, layer.W)
            self.v_w[i] = self.beta * self.v_w[i] + (1.0 - self.beta) * grad_w
            self.v_b[i] = self.beta * self.v_b[i] + (1.0 - self.beta) * layer.grad_b

            layer.W -= self.learning_rate * self.v_w[i]
            layer.b -= self.learning_rate * self.v_b[i]


class NAGOptimizer(BaseOptimizer):
    def __init__(self, learning_rate=1e-3, weight_decay=0.0, beta=0.9):
        super().__init__(learning_rate, weight_decay)
        self.beta = float(beta)
        self.v_w = []
        self.v_b = []

    def initialize(self, layers):
        self.v_w = [np.zeros_like(layer.W) for layer in layers]
        self.v_b = [np.zeros_like(layer.b) for layer in layers]
        self.initialized = True

    def step(self, layers):
        if not self.initialized:
            self.initialize(layers)

        for i, layer in enumerate(layers):
            grad_w = self._regularize(layer.grad_W, layer.W)

            v_w_prev = self.v_w[i].copy()
            v_b_prev = self.v_b[i].copy()

            self.v_w[i] = self.beta * self.v_w[i] - self.learning_rate * grad_w
            self.v_b[i] = self.beta * self.v_b[i] - self.learning_rate * layer.grad_b

            layer.W += -self.beta * v_w_prev + (1.0 + self.beta) * self.v_w[i]
            layer.b += -self.beta * v_b_prev + (1.0 + self.beta) * self.v_b[i]


class RMSPropOptimizer(BaseOptimizer):
    def __init__(self, learning_rate=1e-3, weight_decay=0.0, beta=0.9, epsilon=1e-8):
        super().__init__(learning_rate, weight_decay)
        self.beta = float(beta)
        self.epsilon = float(epsilon)
        self.s_w = []
        self.s_b = []

    def initialize(self, layers):
        self.s_w = [np.zeros_like(layer.W) for layer in layers]
        self.s_b = [np.zeros_like(layer.b) for layer in layers]
        self.initialized = True

    def step(self, layers):
        if not self.initialized:
            self.initialize(layers)

        for i, layer in enumerate(layers):
            grad_w = self._regularize(layer.grad_W, layer.W)

            self.s_w[i] = self.beta * self.s_w[i] + (1.0 - self.beta) * np.square(grad_w)
            self.s_b[i] = self.beta * self.s_b[i] + (1.0 - self.beta) * np.square(layer.grad_b)

            layer.W -= self.learning_rate * grad_w / (np.sqrt(self.s_w[i]) + self.epsilon)
            layer.b -= self.learning_rate * layer.grad_b / (np.sqrt(self.s_b[i]) + self.epsilon)


class AdamOptimizer(BaseOptimizer):
    def __init__(
        self,
        learning_rate=1e-3,
        weight_decay=0.0,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
    ):
        super().__init__(learning_rate, weight_decay)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.epsilon = float(epsilon)
        self.m_w = []
        self.m_b = []
        self.v_w = []
        self.v_b = []
        self.t = 0

    def initialize(self, layers):
        self.m_w = [np.zeros_like(layer.W) for layer in layers]
        self.m_b = [np.zeros_like(layer.b) for layer in layers]
        self.v_w = [np.zeros_like(layer.W) for layer in layers]
        self.v_b = [np.zeros_like(layer.b) for layer in layers]
        self.t = 0
        self.initialized = True

    def step(self, layers):
        if not self.initialized:
            self.initialize(layers)

        self.t += 1

        for i, layer in enumerate(layers):
            grad_w = self._regularize(layer.grad_W, layer.W)
            grad_b = layer.grad_b

            self.m_w[i] = self.beta1 * self.m_w[i] + (1.0 - self.beta1) * grad_w
            self.m_b[i] = self.beta1 * self.m_b[i] + (1.0 - self.beta1) * grad_b

            self.v_w[i] = self.beta2 * self.v_w[i] + (1.0 - self.beta2) * np.square(grad_w)
            self.v_b[i] = self.beta2 * self.v_b[i] + (1.0 - self.beta2) * np.square(grad_b)

            m_w_hat = self.m_w[i] / (1.0 - self.beta1 ** self.t)
            m_b_hat = self.m_b[i] / (1.0 - self.beta1 ** self.t)
            v_w_hat = self.v_w[i] / (1.0 - self.beta2 ** self.t)
            v_b_hat = self.v_b[i] / (1.0 - self.beta2 ** self.t)

            layer.W -= self.learning_rate * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)
            layer.b -= self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)


class NadamOptimizer(AdamOptimizer):
    def step(self, layers):
        if not self.initialized:
            self.initialize(layers)

        self.t += 1

        for i, layer in enumerate(layers):
            grad_w = self._regularize(layer.grad_W, layer.W)
            grad_b = layer.grad_b

            self.m_w[i] = self.beta1 * self.m_w[i] + (1.0 - self.beta1) * grad_w
            self.m_b[i] = self.beta1 * self.m_b[i] + (1.0 - self.beta1) * grad_b

            self.v_w[i] = self.beta2 * self.v_w[i] + (1.0 - self.beta2) * np.square(grad_w)
            self.v_b[i] = self.beta2 * self.v_b[i] + (1.0 - self.beta2) * np.square(grad_b)

            m_w_hat = self.m_w[i] / (1.0 - self.beta1 ** self.t)
            m_b_hat = self.m_b[i] / (1.0 - self.beta1 ** self.t)
            v_w_hat = self.v_w[i] / (1.0 - self.beta2 ** self.t)
            v_b_hat = self.v_b[i] / (1.0 - self.beta2 ** self.t)

            m_w_nesterov = (
                self.beta1 * m_w_hat
                + ((1.0 - self.beta1) * grad_w) / (1.0 - self.beta1 ** self.t)
            )
            m_b_nesterov = (
                self.beta1 * m_b_hat
                + ((1.0 - self.beta1) * grad_b) / (1.0 - self.beta1 ** self.t)
            )

            layer.W -= self.learning_rate * m_w_nesterov / (np.sqrt(v_w_hat) + self.epsilon)
            layer.b -= self.learning_rate * m_b_nesterov / (np.sqrt(v_b_hat) + self.epsilon)


def get_optimizer(name, learning_rate=1e-3, weight_decay=0.0):
    name = name.lower()

    if name == "sgd":
        return SGDOptimizer(learning_rate=learning_rate, weight_decay=weight_decay)
    if name == "momentum":
        return MomentumOptimizer(learning_rate=learning_rate, weight_decay=weight_decay)
    if name == "nag":
        return NAGOptimizer(learning_rate=learning_rate, weight_decay=weight_decay)
    if name == "rmsprop":
        return RMSPropOptimizer(learning_rate=learning_rate, weight_decay=weight_decay)
    if name == "adam":
        return AdamOptimizer(learning_rate=learning_rate, weight_decay=weight_decay)
    if name == "nadam":
        return NadamOptimizer(learning_rate=learning_rate, weight_decay=weight_decay)

    raise ValueError(f"Unsupported optimizer: {name}")
