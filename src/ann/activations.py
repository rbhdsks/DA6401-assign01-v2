"""
Activation Functions and Their Derivatives
Implements: ReLU, Sigmoid, Tanh, Softmax
"""

import numpy as np


def relu(x):
    return np.maximum(0.0, x)


def relu_derivative(x):
    return (x > 0.0).astype(x.dtype)


def sigmoid(x):
    clipped = np.clip(x, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def sigmoid_derivative_from_activation(a):
    return a * (1.0 - a)


def tanh(x):
    return np.tanh(x)


def tanh_derivative_from_activation(a):
    return 1.0 - np.square(a)


def softmax(x):
    shifted = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(shifted)
    denom = np.sum(exp_x, axis=1, keepdims=True)
    return exp_x / denom


def get_activation(name):
    name = name.lower()
    if name == "relu":
        return relu
    if name == "sigmoid":
        return sigmoid
    if name == "tanh":
        return tanh
    if name in {"linear", "identity", "none"}:
        return lambda x: x
    raise ValueError(f"Unsupported activation: {name}")


def get_activation_derivative(name):
    name = name.lower()
    if name == "relu":
        return lambda z, a: relu_derivative(z)
    if name == "sigmoid":
        return lambda z, a: sigmoid_derivative_from_activation(a)
    if name == "tanh":
        return lambda z, a: tanh_derivative_from_activation(a)
    if name in {"linear", "identity", "none"}:
        return lambda z, a: np.ones_like(z)
    raise ValueError(f"Unsupported activation: {name}")
