"""Shared reproducibility controls for the Temporal-Testing workflows."""

from __future__ import annotations

import os
import random


DEFAULT_SEED = 42


def seed_everything(seed: int = DEFAULT_SEED, *, deterministic_torch: bool = True) -> int:
    """Seed supported local RNGs and enable deterministic PyTorch behavior.

    ``PYTHONHASHSEED`` is exported for child processes. Python's hash seed for
    the current interpreter is fixed at interpreter startup, so callers that
    depend on hash iteration should also start Python with the same environment
    variable. The Temporal-Testing retrieval code uses stable hashes and does
    not depend on Python's randomized ``hash()``.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if seed < 0:
        raise ValueError("seed must be non-negative")

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)

    try:
        import numpy as np
    except ImportError:
        pass
    else:
        np.random.seed(seed)

    try:
        import torch
    except ImportError:
        pass
    else:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.use_deterministic_algorithms(True, warn_only=True)
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.allow_tf32 = False
            if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
                torch.backends.cuda.matmul.allow_tf32 = False

    return seed
