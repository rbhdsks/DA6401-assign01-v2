"""
Loss/Objective Functions and Their Derivatives
Implements: Cross-Entropy, Mean Squared Error (MSE)
"""

import numpy as np

from .activations import softmax

EPS = 1e-12


def one_hot_encode(y, num_classes):
    if y.ndim == 2:
        return y
    y = y.astype(int)
    out = np.zeros((y.shape[0], num_classes), dtype=np.float64)
    out[np.arange(y.shape[0]), y] = 1.0
    return out


def cross_entropy_loss(probs, y_true_one_hot):
    clipped = np.clip(probs, EPS, 1.0 - EPS)
    return -np.sum(y_true_one_hot * np.log(clipped)) / y_true_one_hot.shape[0]


def mse_loss(predictions, y_true_one_hot):
    diff = predictions - y_true_one_hot
    return np.mean(np.square(diff))


def cross_entropy_loss_and_grad(logits, y_true, num_classes):
    y_one_hot = one_hot_encode(y_true, num_classes)
    probs = softmax(logits)
    loss = cross_entropy_loss(probs, y_one_hot)
    grad_logits = (probs - y_one_hot) / y_one_hot.shape[0]
    return loss, probs, grad_logits


def mse_loss_and_grad(logits, y_true, num_classes):
    y_one_hot = one_hot_encode(y_true, num_classes)
    probs = softmax(logits)

    loss = mse_loss(probs, y_one_hot)

    # dL/dS where S is softmax output
    grad_softmax = (2.0 / y_one_hot.size) * (probs - y_one_hot)
    # dS/dZ Jacobian-vector product for each sample:
    # J_softmax(v) = s * (v - <v, s>)
    dot = np.sum(grad_softmax * probs, axis=1, keepdims=True)
    grad_logits = probs * (grad_softmax - dot)

    return loss, probs, grad_logits


def loss_and_gradient(logits, y_true, num_classes, loss_name):
    loss_name = loss_name.lower()
    if loss_name == "cross_entropy":
        return cross_entropy_loss_and_grad(logits, y_true, num_classes)
    if loss_name == "mse":
        return mse_loss_and_grad(logits, y_true, num_classes)
    raise ValueError(f"Unsupported loss: {loss_name}")
