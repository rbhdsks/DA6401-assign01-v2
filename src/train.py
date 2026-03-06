"""
Main Training Script
Entry point for training neural networks with command-line arguments
"""

import argparse
import json
import os

import numpy as np
import wandb

from ann.neural_network import NeuralNetwork
from utils.data_loader import load_data


def _str2bool(value):
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "on"}:
        return True
    if text in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _resolve_hidden_sizes(num_layers, hidden_size):
    if len(hidden_size) == 1 and num_layers > 1:
        return hidden_size * num_layers
    if len(hidden_size) != num_layers:
        raise ValueError(
            "--hidden_size must provide either one value (repeated) or exactly --num_layers values"
        )
    return hidden_size


def _serialize_yaml_scalar(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    if text.startswith("${") and text.endswith("}"):
        return f'"{text}"'
    if any(ch in text for ch in [":", "#", "{", "}", "[", "]", ",", " "]):
        return f'"{text}"'
    return text


def _write_default_sweep_yaml(args):
    """
    Generate a W&B sweep config aligned with assignment requirements:
    - varies all key training hyperparameters
    - supports >=100 runs (run_cap set to 100)
    """
    sweep_path = args.sweep_yaml_path
    _ensure_parent_dir(sweep_path)

    # Keep sweep runtime bounded: fix all sweep trials to 10 epochs.
    epoch_vals = [10]
    batch_vals = sorted(
        {
            max(32, args.batch_size // 2),
            args.batch_size,
            min(512, max(32, args.batch_size * 2)),
        }
    )
    layer_vals = sorted(
        {
            max(2, args.num_layers - 1),
            args.num_layers,
            min(6, args.num_layers + 1),
        }
    )
    base_hidden = int(args.hidden_size[0]) if args.hidden_size else 128
    hidden_vals = sorted({64, 96, 128, max(32, min(256, base_hidden))})

    lines = [
        "program: src/train.py",
        "project: DA6401-DL-ASSIGNMENT01",
        "method: random",
        "run_cap: 100",
        "metric:",
        "  name: val/accuracy",
        "  goal: maximize",
        "parameters:",
        "  dataset:",
        f"    value: {args.dataset}",
        "  epochs:",
        f"    values: [{', '.join(str(v) for v in epoch_vals)}]",
        "  batch_size:",
        f"    values: [{', '.join(str(v) for v in batch_vals)}]",
        "  loss:",
        "    values: [cross_entropy, mse]",
        "  optimizer:",
        "    values: [sgd, momentum, nag, rmsprop, adam, nadam]",
        "  learning_rate:",
        "    distribution: log_uniform_values",
        "    min: 0.0001",
        "    max: 0.005",
        "  weight_decay:",
        "    distribution: log_uniform_values",
        "    min: 0.000001",
        "    max: 0.001",
        "  num_layers:",
        f"    values: [{', '.join(str(v) for v in layer_vals)}]",
        "  hidden_size:",
        f"    values: [{', '.join(str(v) for v in hidden_vals)}]",
        "  activation:",
        "    values: [sigmoid, tanh, relu]",
        "  weight_init:",
        "    values: [random, xavier]",
        "  seed:",
        "    values: [7, 21, 42]",
        "  model_save_path:",
        "    value: models/sweep_model.npy",
        "  config_save_path:",
        "    value: models/sweep_config.json",
        "  no_sweep_yaml:",
        "    value: true",
    ]

    if args.wandb_entity:
        lines.insert(2, f"entity: {_serialize_yaml_scalar(args.wandb_entity)}")
        lines.extend(["  wandb_entity:", f"    value: {_serialize_yaml_scalar(args.wandb_entity)}"])

    with open(sweep_path, "w", encoding="utf-8") as fp:
        fp.write("\n".join(lines) + "\n")

    print(f"Generated sweep config at: {sweep_path}")


def parse_arguments():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Train a neural network")

    parser.add_argument("-d", "--dataset", type=str, default="mnist", choices=["mnist", "fashion_mnist"])
    parser.add_argument("-e", "--epochs", type=int, default=10)
    parser.add_argument("-b", "--batch_size", type=int, default=64)
    parser.add_argument("-l", "--loss", type=str, default="cross_entropy", choices=["cross_entropy", "mse"])
    parser.add_argument(
        "-o",
        "--optimizer",
        type=str,
        default="adam",
        choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"],
    )
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-3)
    parser.add_argument("-wd", "--weight_decay", type=float, default=0.0)
    parser.add_argument("-nhl", "--num_layers", type=int, default=2)
    parser.add_argument(
        "-sz",
        "--hidden_size",
        type=int,
        nargs="+",
        default=[128],
        help="Hidden sizes; pass one value to repeat across all hidden layers",
    )
    parser.add_argument("-a", "--activation", type=str, default="relu", choices=["sigmoid", "tanh", "relu"])
    parser.add_argument(
        "-wi",
        "--weight_init",
        type=str,
        default="xavier",
        choices=["random", "xavier", "zeros"],
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation_split", type=float, default=0.1)

    parser.add_argument("--wandb_entity", type=str, default="")
    parser.add_argument("--wandb_run_name", type=str, default="")
    parser.add_argument(
        "--no_wandb",
        nargs="?",
        const=True,
        default=False,
        type=_str2bool,
        help="Disable Weights & Biases logging (supports true/false)",
    )
    parser.add_argument(
        "--sweep_yaml_path",
        type=str,
        default="sweep.yml",
        help="Path to auto-generate W&B sweep config YAML",
    )
    parser.add_argument(
        "--no_sweep_yaml",
        nargs="?",
        const=True,
        default=False,
        type=_str2bool,
        help="Disable auto-generation of sweep YAML after parsing args (supports true/false)",
    )

    parser.add_argument(
        "--model_save_path",
        type=str,
        default="src/best_model.npy",
        help="Path to save trained model weights (relative path)",
    )
    parser.add_argument(
        "--config_save_path",
        type=str,
        default="src/best_config.json",
        help="Path to save best config (relative path)",
    )

    return parser.parse_args()


def _init_wandb(args):
    if args.no_wandb:
        return None

    run = wandb.init(
        project="DA6401-DL-ASSIGNMENT01",
        entity=args.wandb_entity or None,
        name=args.wandb_run_name or None,
        config=args,
    )
    return run


def _ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def main():
    """
    Main training function.
    """
    args = parse_arguments()
    args.hidden_size = _resolve_hidden_sizes(args.num_layers, args.hidden_size)
    if not args.no_sweep_yaml:
        _write_default_sweep_yaml(args)

    np.random.seed(args.seed)

    X_train, y_train, X_val, y_val, X_test, y_test = load_data(
        dataset=args.dataset,
        validation_split=args.validation_split,
        seed=args.seed,
    )

    model_args = {
        "input_dim": X_train.shape[1],
        "num_classes": len(np.unique(y_train)),
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "activation": args.activation,
        "loss": args.loss,
        "optimizer": args.optimizer,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "weight_init": args.weight_init,
        "seed": args.seed,
    }

    model = NeuralNetwork(model_args)
    wandb_run = _init_wandb(args)

    history = model.train(
        X_train,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        X_val=X_val,
        y_val=y_val,
        shuffle=True,
        verbose=True,
        wandb_run=wandb_run,
    )

    # Restore best validation-F1 model before final test evaluation
    model.set_weights(history["best_weights"])

    train_metrics = model.evaluate(X_train, y_train)
    val_metrics = model.evaluate(X_val, y_val)
    test_metrics = model.evaluate(X_test, y_test)

    _ensure_parent_dir(args.model_save_path)
    np.save(args.model_save_path, model.get_weights(), allow_pickle=True)

    config_payload = {
        "dataset": args.dataset,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "loss": args.loss,
        "optimizer": args.optimizer,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "num_layers": args.num_layers,
        "hidden_size": args.hidden_size,
        "activation": args.activation,
        "weight_init": args.weight_init,
        "seed": args.seed,
        "model_path": args.model_save_path,
        "metrics": {
            "train": {
                "loss": train_metrics["loss"],
                "accuracy": train_metrics["accuracy"],
                "precision": train_metrics["precision"],
                "recall": train_metrics["recall"],
                "f1": train_metrics["f1"],
            },
            "val": {
                "loss": val_metrics["loss"],
                "accuracy": val_metrics["accuracy"],
                "precision": val_metrics["precision"],
                "recall": val_metrics["recall"],
                "f1": val_metrics["f1"],
            },
            "test": {
                "loss": test_metrics["loss"],
                "accuracy": test_metrics["accuracy"],
                "precision": test_metrics["precision"],
                "recall": test_metrics["recall"],
                "f1": test_metrics["f1"],
            },
        },
    }

    _ensure_parent_dir(args.config_save_path)
    with open(args.config_save_path, "w", encoding="utf-8") as fp:
        json.dump(config_payload, fp, indent=2)

    if wandb_run is not None:
        wandb_run.log(
            {
                "final/train_accuracy": train_metrics["accuracy"],
                "final/val_accuracy": val_metrics["accuracy"],
                "final/test_accuracy": test_metrics["accuracy"],
                "final/train_f1": train_metrics["f1"],
                "final/val_f1": val_metrics["f1"],
                "final/test_f1": test_metrics["f1"],
            }
        )
        wandb_run.finish()

    print("Training complete!")
    print(f"Saved model weights to: {args.model_save_path}")
    print(f"Saved best config to: {args.config_save_path}")
    print(
        "Test metrics | "
        f"loss={test_metrics['loss']:.4f} "
        f"accuracy={test_metrics['accuracy']:.4f} "
        f"precision={test_metrics['precision']:.4f} "
        f"recall={test_metrics['recall']:.4f} "
        f"f1={test_metrics['f1']:.4f}"
    )


if __name__ == "__main__":
    main()
