"""Reproducible runner for Meta-BayFL (supplementary code).

Example:
  python run_metabayfl.py --dataset mnist --num_clients 5 --alpha 0.5 --rounds 10
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import tensorflow as tf

from metabayfl.utils import set_global_determinism
from metabayfl.data import load_dataset, split_clients_dirichlet, make_tf_dataset
from metabayfl.models import make_bnn_classifier, negative_log_likelihood
from metabayfl.federated import run_metabayfl

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["mnist", "cifar10"], default="mnist")
    p.add_argument("--num_clients", type=int, default=5)
    p.add_argument("--alpha", type=float, default=0.5, help="Dirichlet alpha (smaller => more non-IID)")
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--local_epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--candidate_lrs", type=float, nargs="+", default=[1e-2, 1e-3, 1e-4])
    p.add_argument("--val_split", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outdir", type=str, default="runs")
    return p.parse_args()

def main():
    args = parse_args()
    set_global_determinism(args.seed)

    (x_train, y_train), (x_test, y_test) = load_dataset(args.dataset)
    num_classes = int(np.max(y_train)) + 1
    input_shape = tuple(x_train.shape[1:])

    clients = split_clients_dirichlet(
        x_train, y_train,
        num_clients=args.num_clients,
        alpha=args.alpha,
        min_size=50 if args.dataset == "cifar10" else 10,
        seed=args.seed,
    )

    test_ds = make_tf_dataset(x_test, y_test, batch_size=args.batch_size, shuffle=False)

    def make_model():
        model = make_bnn_classifier(input_shape=input_shape, num_classes=num_classes)
        # KL weight can be scaled by dataset size; keep simple for reproducibility
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.candidate_lrs[0]),
            loss=negative_log_likelihood,
            metrics=["accuracy"],
        )
        return model

    history = run_metabayfl(
        clients=clients,
        make_model=make_model,
        test_ds=test_ds,
        rounds=args.rounds,
        batch_size=args.batch_size,
        local_epochs=args.local_epochs,
        candidate_lrs=args.candidate_lrs,
        val_split=args.val_split,
        seed=args.seed,
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    # Save metrics as jsonl
    with (outdir / "metrics.jsonl").open("w", encoding="utf-8") as f:
        for m in history:
            f.write(json.dumps({
                "round": m.round,
                "test_loss": m.test_loss,
                "test_acc": m.test_acc,
                "lr_selected": m.lr_selected,
            }) + "\n")

    print(f"Saved run to: {outdir.resolve()}")
    if history:
        print(f"Final test_acc={history[-1].test_acc:.4f} test_loss={history[-1].test_loss:.4f}")

if __name__ == "__main__":
    main()
