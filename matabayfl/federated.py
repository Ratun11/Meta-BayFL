from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Callable, Tuple
import numpy as np
import tensorflow as tf

from .data import ClientDataset, make_tf_dataset
from .models import negative_log_likelihood

@dataclass
class RoundMetrics:
    round: int
    lr_selected: List[float]
    train_loss: float
    train_acc: float
    test_loss: float
    test_acc: float

def _get_weights(model: tf.keras.Model):
    return [w.numpy() for w in model.get_weights()]

def _set_weights(model: tf.keras.Model, weights):
    model.set_weights(weights)

def fedavg(weights_list: List[List[np.ndarray]]) -> List[np.ndarray]:
    """Simple FedAvg with equal weighting."""
    avg = []
    for params in zip(*weights_list):
        stacked = np.stack(params, axis=0)
        avg.append(np.mean(stacked, axis=0))
    return avg

def evaluate(model: tf.keras.Model, ds: tf.data.Dataset) -> Tuple[float, float]:
    loss, acc = model.evaluate(ds, verbose=0)
    return float(loss), float(acc)

def train_one_client(
    base_weights: List[np.ndarray],
    make_model: Callable[[], tf.keras.Model],
    client: ClientDataset,
    batch_size: int,
    local_epochs: int,
    candidate_lrs: List[float],
    val_split: float,
    seed: int,
) -> Tuple[List[np.ndarray], float]:
    """Meta step: pick best lr on a small validation split, then train with it."""
    rng = np.random.default_rng(seed)
    n = len(client.x)
    idx = np.arange(n)
    rng.shuffle(idx)
    val_n = max(1, int(val_split * n))
    val_idx, tr_idx = idx[:val_n], idx[val_n:]

    x_tr, y_tr = client.x[tr_idx], client.y[tr_idx]
    x_val, y_val = client.x[val_idx], client.y[val_idx]

    tr_ds = make_tf_dataset(x_tr, y_tr, batch_size=batch_size, shuffle=True)
    val_ds = make_tf_dataset(x_val, y_val, batch_size=batch_size, shuffle=False)

    best_lr = candidate_lrs[0]
    best_val_loss = float("inf")

    # Evaluate each lr with a short warmup (1 epoch) for selection.
    for lr in candidate_lrs:
        model = make_model()
        _set_weights(model, base_weights)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss=negative_log_likelihood,
            metrics=["accuracy"],
        )
        model.fit(tr_ds, epochs=1, verbose=0)
        val_loss, _ = evaluate(model, val_ds)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_lr = lr

    # Final local training with selected lr
    model = make_model()
    _set_weights(model, base_weights)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=best_lr),
        loss=negative_log_likelihood,
        metrics=["accuracy"],
    )
    model.fit(tr_ds, epochs=local_epochs, verbose=0)

    return _get_weights(model), float(best_lr)

def run_metabayfl(
    clients: List[ClientDataset],
    make_model: Callable[[], tf.keras.Model],
    test_ds: tf.data.Dataset,
    rounds: int,
    batch_size: int,
    local_epochs: int,
    candidate_lrs: List[float],
    val_split: float = 0.2,
    seed: int = 42,
) -> List[RoundMetrics]:
    # Initialize
    global_model = make_model()
    global_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=candidate_lrs[0]),
        loss=negative_log_likelihood,
        metrics=["accuracy"],
    )
    global_weights = _get_weights(global_model)

    history: List[RoundMetrics] = []

    # For tracking: evaluate on an aggregate train proxy (optional)
    # We'll just use test for simplicity and reproducibility.
    for r in range(1, rounds + 1):
        local_weights = []
        selected_lrs = []

        for i, client in enumerate(clients):
            w_i, lr_i = train_one_client(
                base_weights=global_weights,
                make_model=make_model,
                client=client,
                batch_size=batch_size,
                local_epochs=local_epochs,
                candidate_lrs=candidate_lrs,
                val_split=val_split,
                seed=seed + 1000 * r + i,
            )
            local_weights.append(w_i)
            selected_lrs.append(lr_i)

        global_weights = fedavg(local_weights)

        # Evaluate
        global_model = make_model()
        _set_weights(global_model, global_weights)
        global_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=candidate_lrs[0]),
            loss=negative_log_likelihood,
            metrics=["accuracy"],
        )
        test_loss, test_acc = evaluate(global_model, test_ds)

        history.append(RoundMetrics(
            round=r,
            lr_selected=selected_lrs,
            train_loss=float("nan"),
            train_acc=float("nan"),
            test_loss=test_loss,
            test_acc=test_acc,
        ))
    return history
