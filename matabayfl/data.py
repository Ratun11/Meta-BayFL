from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Literal, Optional
import numpy as np
import tensorflow as tf

DatasetName = Literal["mnist", "cifar10"]

@dataclass
class ClientDataset:
    x: np.ndarray
    y: np.ndarray

def _normalize_images(x: np.ndarray) -> np.ndarray:
    x = x.astype("float32") / 255.0
    return x

def load_dataset(name: DatasetName):
    if name == "mnist":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train = _normalize_images(x_train)[..., None]  # (N,28,28,1)
        x_test = _normalize_images(x_test)[..., None]
        return (x_train, y_train), (x_test, y_test)
    if name == "cifar10":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        y_train = y_train.squeeze().astype("int64")
        y_test = y_test.squeeze().astype("int64")
        x_train = _normalize_images(x_train)
        x_test = _normalize_images(x_test)
        return (x_train, y_train), (x_test, y_test)
    raise ValueError(f"Unknown dataset: {name}")

def split_clients_dirichlet(
    x: np.ndarray,
    y: np.ndarray,
    num_clients: int,
    alpha: float,
    min_size: int = 10,
    seed: int = 42,
) -> List[ClientDataset]:
    """Non-IID partitioning using a Dirichlet distribution over classes.

    Returns a list of ClientDataset. Ensures each client has at least min_size samples.
    """
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    num_classes = len(classes)

    # indices per class
    idx_by_class = [np.where(y == c)[0] for c in classes]
    for c in range(num_classes):
        rng.shuffle(idx_by_class[c])

    while True:
        client_indices = [[] for _ in range(num_clients)]
        for c in range(num_classes):
            proportions = rng.dirichlet(alpha=np.repeat(alpha, num_clients))
            # split class indices by proportions
            class_idx = idx_by_class[c]
            split_points = (np.cumsum(proportions) * len(class_idx)).astype(int)[:-1]
            splits = np.split(class_idx, split_points)
            for i in range(num_clients):
                client_indices[i].extend(splits[i].tolist())
        sizes = [len(ci) for ci in client_indices]
        if min(sizes) >= min_size:
            break

    clients = []
    for i in range(num_clients):
        ci = np.array(client_indices[i], dtype=int)
        rng.shuffle(ci)
        clients.append(ClientDataset(x=x[ci], y=y[ci]))
    return clients

def make_tf_dataset(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = True):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(x), 10000), reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds
