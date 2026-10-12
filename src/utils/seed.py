"""src/utils/seed.py — Global deterministic seed manager."""

import random
import numpy as np


def set_global_seed(seed: int = 42) -> None:
    """
    Set deterministic seeds for Python random, NumPy, and (optionally)
    any frameworks that respect environment-level seeding.
    """
    random.seed(seed)
    np.random.seed(seed)

    # Attempt to seed torch if present in environment
    try:
        import torch  # type: ignore
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
