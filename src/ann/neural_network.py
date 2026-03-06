"""
Main Neural Network Model class
Handles forward and backward propagation loops
"""

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from .activations import softmax
from .neural_layer import NeuralLayer
from .objective_functions import loss_and_gradient
from .optimizers import get_optimizer


class NeuralNetwork:
    """
    Main model class that orchestrates the neural network training and inference.
    """

    def __init__(self, cli_args):
        args = vars(cli_args) if hasattr(cli_args, "__dict__") else dict(cli_args)

        self.input_dim = int(args.get("input_dim", 784))
        self.num_classes = int(args.get("num_classes", 10))

        self.loss_name = args.get("loss", "cross_entropy")
        self.activation = args.get("activation", "relu")
        self.weight_init = args.get("weight_init", "xavier")

        self.learning_rate = float(args.get("learning_rate", 1e-3))
        self.weight_decay = float(args.get("weight_decay", 0.0))
        self.optimizer_name = args.get("optimizer", "sgd")

        self.seed = int(args.get("seed", 42))
        self.rng = np.random.default_rng(self.seed)

        hidden_sizes = self._resolve_hidden_sizes(args)

        layer_dims = [self.input_dim] + hidden_sizes + [self.num_classes]
        self.layers = []

        for idx in range(len(layer_dims) - 1):
            is_last = idx == len(layer_dims) - 2
            layer_activation = "linear" if is_last else self.activation
            layer = NeuralLayer(
                input_dim=layer_dims[idx],
                output_dim=layer_dims[idx + 1],
                activation=layer_activation,
                weight_init=self.weight_init,
                rng=self.rng,
            )
            self.layers.append(layer)

        self.optimizer = get_optimizer(
            self.optimizer_name,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        self.optimizer.initialize(self.layers)

        self.grad_W = np.empty(0, dtype=object)
        self.grad_b = np.empty(0, dtype=object)

    @staticmethod
    def _resolve_hidden_sizes(args):
        hidden_size = args.get("hidden_size", [128])
        num_layers = int(args.get("num_layers", len(hidden_size)))

        if isinstance(hidden_size, int):
            hidden_size = [hidden_size]

        if len(hidden_size) == 1 and num_layers > 1:
            return hidden_size * num_layers

        if len(hidden_size) != num_layers:
            raise ValueError(
                "Mismatch between --num_layers and --hidden_size. "
                "Provide one size (to repeat) or exactly one per hidden layer."
            )

        return [int(x) for x in hidden_size]

    def forward(self, X):
        """
        Forward propagation through all layers.
        Returns logits (no softmax applied)
        X is shape (b, D_in) and output is shape (b, D_out).
        b is batch size, D_in is input dimension, D_out is output dimension.
        """
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def _compute_loss_and_grad(self, y_true, logits):
        return loss_and_gradient(
            logits=logits,
            y_true=y_true,
            num_classes=self.num_classes,
            loss_name=self.loss_name,
        )

    def backward(self, y_true, y_pred):
        """
        Backward propagation to compute gradients.
        Returns two numpy arrays: grad_Ws, grad_bs.
        - `grad_Ws[0]` is gradient for the last (output) layer weights,
          `grad_bs[0]` is gradient for the last layer biases, and so on.
        """
        _, _, grad = self._compute_loss_and_grad(y_true, y_pred)

        grad_W_list = []
        grad_b_list = []

        # Backprop through layers in reverse; collect grads so index 0 = last layer
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
            grad_W_list.append(layer.grad_W)
            grad_b_list.append(layer.grad_b)

        # create explicit object arrays to avoid numpy trying to broadcast shapes
        self.grad_W = np.empty(len(grad_W_list), dtype=object)
        self.grad_b = np.empty(len(grad_b_list), dtype=object)
        for i, (gw, gb) in enumerate(zip(grad_W_list, grad_b_list)):
            self.grad_W[i] = gw
            self.grad_b[i] = gb

        return self.grad_W, self.grad_b

    def update_weights(self):
        self.optimizer.step(self.layers)

    def train(
        self,
        X_train,
        y_train,
        epochs=1,
        batch_size=32,
        X_val=None,
        y_val=None,
        shuffle=True,
        verbose=True,
        wandb_run=None,
    ):
        history = {
            "train_loss": [],
            "train_accuracy": [],
            "train_precision": [],
            "train_recall": [],
            "train_f1": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_precision": [],
            "val_recall": [],
            "val_f1": [],
        }

        best_val_f1 = -np.inf
        best_weights = None

        n_samples = X_train.shape[0]
        indices = np.arange(n_samples)

        for epoch in range(epochs):
            if shuffle:
                self.rng.shuffle(indices)

            X_epoch = X_train[indices]
            y_epoch = y_train[indices]

            epoch_loss_sum = 0.0

            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                xb = X_epoch[start:end]
                yb = y_epoch[start:end]

                logits = self.forward(xb)
                batch_loss, _, _ = self._compute_loss_and_grad(yb, logits)
                self.backward(yb, logits)
                self.update_weights()

                epoch_loss_sum += batch_loss * (end - start)

            train_metrics = self.evaluate(X_train, y_train)
            train_metrics["loss"] = epoch_loss_sum / n_samples

            history["train_loss"].append(train_metrics["loss"])
            history["train_accuracy"].append(train_metrics["accuracy"])
            history["train_precision"].append(train_metrics["precision"])
            history["train_recall"].append(train_metrics["recall"])
            history["train_f1"].append(train_metrics["f1"])

            log_payload = {
                "epoch": epoch + 1,
                "train/loss": train_metrics["loss"],
                "train/accuracy": train_metrics["accuracy"],
                "train/precision": train_metrics["precision"],
                "train/recall": train_metrics["recall"],
                "train/f1": train_metrics["f1"],
            }

            if X_val is not None and y_val is not None:
                val_metrics = self.evaluate(X_val, y_val)

                history["val_loss"].append(val_metrics["loss"])
                history["val_accuracy"].append(val_metrics["accuracy"])
                history["val_precision"].append(val_metrics["precision"])
                history["val_recall"].append(val_metrics["recall"])
                history["val_f1"].append(val_metrics["f1"])

                log_payload.update(
                    {
                        "val/loss": val_metrics["loss"],
                        "val/accuracy": val_metrics["accuracy"],
                        "val/precision": val_metrics["precision"],
                        "val/recall": val_metrics["recall"],
                        "val/f1": val_metrics["f1"],
                    }
                )

                if val_metrics["f1"] > best_val_f1:
                    best_val_f1 = val_metrics["f1"]
                    best_weights = self.get_weights()

                if verbose:
                    print(
                        f"Epoch {epoch + 1:03d}/{epochs} | "
                        f"train_loss={train_metrics['loss']:.4f} "
                        f"train_acc={train_metrics['accuracy']:.4f} | "
                        f"val_loss={val_metrics['loss']:.4f} "
                        f"val_acc={val_metrics['accuracy']:.4f} "
                        f"val_f1={val_metrics['f1']:.4f}"
                    )
            else:
                if verbose:
                    print(
                        f"Epoch {epoch + 1:03d}/{epochs} | "
                        f"train_loss={train_metrics['loss']:.4f} "
                        f"train_acc={train_metrics['accuracy']:.4f}"
                    )

            if wandb_run is not None:
                wandb_run.log(log_payload)

        if best_weights is None:
            best_weights = self.get_weights()

        history["best_weights"] = best_weights
        history["best_val_f1"] = best_val_f1
        return history

    def evaluate(self, X, y, return_logits=True):
        logits = self.forward(X)
        loss, probs, _ = self._compute_loss_and_grad(y, logits)
        preds = np.argmax(probs, axis=1)

        accuracy = np.mean(preds == y)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y,
            preds,
            average="macro",
            zero_division=0,
        )

        result = {
            "loss": float(loss),
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "predictions": preds,
            "probabilities": probs,
        }

        if return_logits:
            result["logits"] = logits

        return result

    def predict(self, X):
        logits = self.forward(X)
        probs = softmax(logits)
        return np.argmax(probs, axis=1)

    def predict_proba(self, X):
        logits = self.forward(X)
        return softmax(logits)

    def get_weights(self):
        d = {}
        for i, layer in enumerate(self.layers):
            d[f"W{i}"] = layer.W.copy()
            d[f"b{i}"] = layer.b.copy()
        return d

    def set_weights(self, weight_dict):
        for i, layer in enumerate(self.layers):
            w_key = f"W{i}"
            b_key = f"b{i}"
            if w_key in weight_dict:
                layer.W = weight_dict[w_key].copy()
            if b_key in weight_dict:
                layer.b = weight_dict[b_key].copy()
