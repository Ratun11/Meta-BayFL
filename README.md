# Meta-BayFL (Supplementary Code)

This folder provides a **reproducible** reference implementation for the paper:
**"Probabilistic Federated Learning on Uncertain and Heterogeneous Data with Model Personalization"**.

## Quick start

1. Create an environment and install requirements:

```bash
pip install -r requirements.txt
```

2. Run Meta-BayFL on MNIST (Dirichlet non-IID partition):

```bash
python run_metabayfl.py --dataset mnist --num_clients 5 --alpha 0.5 --rounds 10 --local_epochs 5
```

3. Run Meta-BayFL on CIFAR-10:

```bash
python run_metabayfl.py --dataset cifar10 --num_clients 5 --alpha 0.5 --rounds 50 --local_epochs 5
```

Outputs are saved under `runs/` (config + per-round metrics).

## Notes on reproducibility

- The runner sets random seeds for Python/NumPy/TensorFlow.
- Exact determinism can still vary across GPU drivers and TF builds.

## Folder structure

- `metabayfl/`: minimal library (data split, BNN model, Meta-BayFL training loop)
- `run_metabayfl.py`: main runnable script (recommended entry point)
- `DFL/`: original scripts provided by the authors (kept for reference)

## Citation

If you use this code, please cite the paper.
