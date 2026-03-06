"""
Inference Script
Evaluate trained models on test sets
"""

import argparse
import json
import os

import numpy as np

from ann.neural_network import NeuralNetwork
from utils.data_loader import load_data


def parse_arguments():
    """
    Parse command-line arguments for inference.
    """
    parser = argparse.ArgumentParser(description="Run inference on test set")

    parser.add_argument(
        "-mp",
        "--model_path",
        type=str,
        required=True,
        help="Path to saved model weights (relative path)",
    )
    parser.add_argument("-d", "--dataset", type=str, default="mnist", choices=["mnist", "fashion_mnist"])
    parser.add_argument("-b", "--batch_size", type=int, default=256)

    parser.add_argument("-nhl", "--num_layers", type=int, default=None)
    parser.add_argument("-sz", "--hidden_size", type=int, nargs="+", default=None)
    parser.add_argument("-a", "--activation", type=str, default="relu", choices=["relu", "sigmoid", "tanh"])

    parser.add_argument("-l", "--loss", type=str, default="cross_entropy", choices=["cross_entropy", "mse"])
    parser.add_argument("-wi", "--weight_init", type=str, default="xavier", choices=["random", "xavier", "zeros"])

    parser.add_argument("--config_path", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def _load_weight_dict(model_path):
    payload = np.load(model_path, allow_pickle=True)

    if isinstance(payload, np.ndarray) and payload.dtype == object:
        payload = payload.item()

    if not isinstance(payload, dict):
        raise ValueError("Model file must contain a serialized dict with keys W0, b0, ...")

    return payload


def _infer_hidden_sizes(weight_dict):
    W_keys = sorted((k for k in weight_dict if k.startswith("W")), key=lambda x: int(x[1:]))
    if not W_keys:
        raise ValueError("No weight matrices found in model file")

    hidden_sizes = []
    for wk in W_keys[:-1]:
        hidden_sizes.append(int(weight_dict[wk].shape[1]))

    input_dim = int(weight_dict[W_keys[0]].shape[0])
    num_classes = int(weight_dict[W_keys[-1]].shape[1])
    return input_dim, num_classes, hidden_sizes


def load_model(model_path):
    """
    Load trained model from disk.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    return _load_weight_dict(model_path)


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model on test data.

    Returns dictionary with logits, loss, accuracy, f1, precision, recall
    """
    result = model.evaluate(X_test, y_test)
    return {
        "logits": result["logits"],
        "loss": result["loss"],
        "accuracy": result["accuracy"],
        "f1": result["f1"],
        "precision": result["precision"],
        "recall": result["recall"],
    }


def _load_config_if_present(config_path):
    if not config_path:
        return {}
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def main():
    """
    Main inference function.

    Returns dictionary - logits, loss, accuracy, f1, precision, recall
    """
    args = parse_arguments()

    weight_dict = load_model(args.model_path)
    inferred_input_dim, inferred_num_classes, inferred_hidden_sizes = _infer_hidden_sizes(weight_dict)

    cfg = _load_config_if_present(args.config_path)

    hidden_sizes = (
        args.hidden_size
        if args.hidden_size is not None
        else cfg.get("hidden_size", inferred_hidden_sizes)
    )
    num_layers = (
        args.num_layers
        if args.num_layers is not None
        else cfg.get("num_layers", len(hidden_sizes))
    )

    if len(hidden_sizes) == 1 and num_layers > 1:
        hidden_sizes = hidden_sizes * num_layers
    elif len(hidden_sizes) != num_layers:
        hidden_sizes = inferred_hidden_sizes
        num_layers = len(hidden_sizes)

    _, _, _, _, X_test, y_test = load_data(
        dataset=args.dataset,
        validation_split=0.1,
        seed=args.seed,
    )

    model_args = {
        "input_dim": inferred_input_dim,
        "num_classes": inferred_num_classes,
        "hidden_size": hidden_sizes,
        "num_layers": num_layers,
        "activation": cfg.get("activation", args.activation),
        "loss": cfg.get("loss", args.loss),
        "optimizer": "sgd",
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "weight_init": cfg.get("weight_init", args.weight_init),
        "seed": args.seed,
    }

    model = NeuralNetwork(model_args)
    model.set_weights(weight_dict)

    metrics = evaluate_model(model, X_test, y_test)

    print("Evaluation complete!")
    print(
        f"loss={metrics['loss']:.4f} "
        f"accuracy={metrics['accuracy']:.4f} "
        f"precision={metrics['precision']:.4f} "
        f"recall={metrics['recall']:.4f} "
        f"f1={metrics['f1']:.4f}"
    )

    return metrics


if __name__ == "__main__":
    main()
