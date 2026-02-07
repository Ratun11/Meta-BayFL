from __future__ import annotations
import os
import random
import numpy as np
import tensorflow as tf

def set_global_determinism(seed: int = 42) -> None:
    """Best-effort reproducibility across TF/NumPy/Python.

    Note: Full determinism may depend on CUDA/cuDNN and TF build.
    """
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    # Optional TF determinism flags (no-op on some installs)
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
