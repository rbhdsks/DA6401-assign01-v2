# DA6401 Assignment 1: NumPy MLP for MNIST/Fashion-MNIST

This repository implements a configurable Multi-Layer Perceptron (MLP) for image classification using only NumPy for model math.

## Features Implemented

- Fully connected neural network with configurable hidden depth/width
- Activations: `relu`, `sigmoid`, `tanh` (+ stable `softmax` output)
- Losses: `cross_entropy`, `mse`
- Optimizers: `sgd`, `momentum`, `nag`, `rmsprop`, `adam`, `nadam`
- L2 regularization via `--weight_decay`
- Weight initialization: `random`, `xavier`, `zeros`
- End-to-end `train.py` and `inference.py` CLI
- Metrics: Accuracy, Precision, Recall, F1 (macro)
- Model serialization to `.npy` and config/metrics dump to `.json`
- Optional Weights & Biases logging

## Project Structure

```text
src/
  ann/
    activations.py
    neural_layer.py
    neural_network.py
    objective_functions.py
    optimizers.py
  utils/
    data_loader.py
  train.py
  inference.py
```

## Setup

```bash
pip install -r requirements.txt
```

## Training

Example command:

```bash
PYTHONPATH=src python src/train.py \
  -d mnist \
  -e 20 \
  -b 128 \
  -l cross_entropy \
  -o adam \
  -lr 0.001 \
  -wd 0.0005 \
  -nhl 3 \
  -sz 128 128 64 \
  -a relu \
  -wi xavier \
  --model_save_path models/best_model.npy \
  --config_save_path models/best_config.json
```

### CLI Arguments (`train.py`)

- `-d, --dataset`: `mnist` or `fashion_mnist`
- `-e, --epochs`: number of epochs
- `-b, --batch_size`: mini-batch size
- `-l, --loss`: `cross_entropy` or `mse`
- `-o, --optimizer`: `sgd|momentum|nag|rmsprop|adam|nadam`
- `-lr, --learning_rate`: learning rate
- `-wd, --weight_decay`: L2 weight decay
- `-nhl, --num_layers`: number of hidden layers
- `-sz, --hidden_size`: one size (repeated) or one per hidden layer
- `-a, --activation`: `sigmoid|tanh|relu`
- `-wi, --weight_init`: `random|xavier|zeros`
- `--model_save_path`: output `.npy` path
- `--config_save_path`: output `.json` path
- `--wandb_entity`, `--wandb_run_name`, `--no_wandb`
- `--sweep_yaml_path`, `--no_sweep_yaml`

## Inference

```bash
PYTHONPATH=src python src/inference.py \
  -mp models/best_model.npy \
  -d mnist \
  --config_path models/best_config.json
```

Outputs:

- `loss`
- `accuracy`
- `precision`
- `recall`
- `f1`

## Model Artifacts

After training:

- `models/best_model.npy`: serialized dict of weights (`W0`, `b0`, ...)
- `models/best_config.json`: best configuration + train/val/test metrics

## Notes on Data Loading

`src/utils/data_loader.py` uses layered fallbacks for robustness:

1. Local Keras cache files (if available)
2. `keras.datasets`
3. Direct download from `storage.googleapis.com/tensorflow/tf-keras-datasets/` (no TensorFlow import required)

This keeps the pipeline usable even when one source is unavailable.
