"""
Data Loading and Preprocessing
Handles MNIST and Fashion-MNIST datasets
"""

import gzip
import os
from urllib.error import URLError
from urllib.request import urlretrieve

import numpy as np
from sklearn.model_selection import train_test_split


def _keras_cache_path(filename):
    return os.path.join(os.path.expanduser("~"), ".keras", "datasets", filename)


def _load_mnist_from_local_cache():
    path = _keras_cache_path("mnist.npz")
    if not os.path.exists(path):
        return None

    with np.load(path, allow_pickle=False) as data:
        X_train = data["x_train"]
        y_train = data["y_train"]
        X_test = data["x_test"]
        y_test = data["y_test"]
    return (X_train, y_train), (X_test, y_test)


def _read_idx_images_gz(path):
    with gzip.open(path, "rb") as f:
        magic = int.from_bytes(f.read(4), byteorder="big")
        if magic != 2051:
            raise ValueError(f"Invalid IDX image file: {path}")
        num = int.from_bytes(f.read(4), byteorder="big")
        rows = int.from_bytes(f.read(4), byteorder="big")
        cols = int.from_bytes(f.read(4), byteorder="big")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(num, rows, cols)


def _read_idx_labels_gz(path):
    with gzip.open(path, "rb") as f:
        magic = int.from_bytes(f.read(4), byteorder="big")
        if magic != 2049:
            raise ValueError(f"Invalid IDX label file: {path}")
        num = int.from_bytes(f.read(4), byteorder="big")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(num)


def _load_fashion_from_local_cache():
    train_images = _keras_cache_path("train-images-idx3-ubyte.gz")
    train_labels = _keras_cache_path("train-labels-idx1-ubyte.gz")
    test_images = _keras_cache_path("t10k-images-idx3-ubyte.gz")
    test_labels = _keras_cache_path("t10k-labels-idx1-ubyte.gz")

    paths = [train_images, train_labels, test_images, test_labels]
    if not all(os.path.exists(p) for p in paths):
        return None

    X_train = _read_idx_images_gz(train_images)
    y_train = _read_idx_labels_gz(train_labels)
    X_test = _read_idx_images_gz(test_images)
    y_test = _read_idx_labels_gz(test_labels)
    return (X_train, y_train), (X_test, y_test)


def _load_with_keras(dataset):
    try:
        if dataset == "mnist":
            from keras.datasets import mnist as ds
        else:
            from keras.datasets import fashion_mnist as ds
        return ds.load_data()
    except Exception:
        return None


def _download_file(url, dest_path):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    urlretrieve(url, dest_path)


def _load_mnist_via_direct_download():
    """
    Download canonical MNIST .npz used by tf-keras datasets and load it.
    Avoids importing tensorflow/keras in broken environments.
    """
    path = _keras_cache_path("mnist.npz")
    if not os.path.exists(path):
        url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
        _download_file(url, path)
    return _load_mnist_from_local_cache()


def _load_fashion_via_direct_download():
    """
    Download Fashion-MNIST IDX gzip files used by tf-keras datasets and load them.
    """
    files = {
        "train-images-idx3-ubyte.gz": "https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz": "https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz": "https://storage.googleapis.com/tensorflow/tf-keras-datasets/t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz": "https://storage.googleapis.com/tensorflow/tf-keras-datasets/t10k-labels-idx1-ubyte.gz",
    }

    for fname, url in files.items():
        path = _keras_cache_path(fname)
        if not os.path.exists(path):
            _download_file(url, path)
    return _load_fashion_from_local_cache()


def _load_raw_dataset(dataset):
    dataset = dataset.lower()

    if dataset == "mnist":
        loaders = [
            _load_mnist_from_local_cache,
            lambda: _load_with_keras("mnist"),
            _load_mnist_via_direct_download,
        ]
    elif dataset == "fashion_mnist":
        loaders = [
            _load_fashion_from_local_cache,
            lambda: _load_with_keras("fashion_mnist"),
            _load_fashion_via_direct_download,
        ]
    else:
        raise ValueError("dataset must be one of: 'mnist', 'fashion_mnist'")

    for loader in loaders:
        try:
            data = loader()
            if data is not None:
                return data
        except URLError:
            continue
        except OSError:
            continue

    raise RuntimeError(
        "Failed to load dataset. Ensure internet access for first-time download "
        "or place dataset files under ~/.keras/datasets."
    )


def load_data(dataset="mnist", validation_split=0.1, seed=42, normalize=True, flatten=True):
    """
    Load dataset and return train/val/test split.

    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test
    """
    (X_train_full, y_train_full), (X_test, y_test) = _load_raw_dataset(dataset)

    if normalize:
        X_train_full = X_train_full.astype("float64") / 255.0
        X_test = X_test.astype("float64") / 255.0

    if flatten:
        X_train_full = X_train_full.reshape(X_train_full.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=validation_split,
        random_state=seed,
        stratify=y_train_full,
    )

    return X_train, y_train, X_val, y_val, X_test, y_test


def load_dataset(*args, **kwargs):
    """Alias for compatibility."""
    return load_data(*args, **kwargs)
