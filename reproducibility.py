"""reproducibility.py

Lightweight reproducibility utilities.

Why this exists:
- Our evaluation scripts rely on deterministic request selection + router seeds.
- Top-tier conference expectations generally include *explicit* seed control across
  Python, NumPy, and Torch, and recording environment metadata.

This module is intentionally dependency-light and safe to import even when
Torch is unavailable.
"""

from __future__ import annotations

import os
import platform
import random
import sys
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional


def set_global_seed(seed: int, *, deterministic_torch: bool = False) -> None:
    """Best-effort global seeding.

    Notes:
      - LLM decoding can still be nondeterministic on GPU due to kernel-level
        nondeterminism, unless deterministic modes are enabled.
      - We keep deterministic_torch=False by default because it can reduce
        performance and may raise errors for unsupported ops.
    """

    seed = int(seed)

    # Python
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)

    # NumPy
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except Exception:
        pass

    # Torch (optional)
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        if deterministic_torch:
            # CUBLAS workspace config is required by some deterministic GEMM paths.
            # Safe no-op on CPU.
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                # Older torch or unsupported operations; best-effort only.
                pass

            try:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            except Exception:
                pass
    except Exception:
        # Torch not installed or failed to import.
        pass


@dataclass
class EnvInfo:
    python_version: str
    platform: str
    executable: str
    argv: str
    cuda_available: bool
    torch_version: Optional[str]
    transformers_version: Optional[str]
    numpy_version: Optional[str]


def collect_env_info(argv: Optional[str] = None) -> Dict[str, Any]:
    """Collect minimal environment metadata for run provenance."""

    py = sys.version.replace("\n", " ")
    plat = platform.platform()
    exe = sys.executable
    argv_s = argv if argv is not None else " ".join(sys.argv)

    torch_version: Optional[str] = None
    cuda_available = False
    try:
        import torch  # type: ignore

        torch_version = str(getattr(torch, "__version__", None))
        cuda_available = bool(torch.cuda.is_available())
    except Exception:
        pass

    transformers_version: Optional[str] = None
    try:
        import transformers  # type: ignore

        transformers_version = str(getattr(transformers, "__version__", None))
    except Exception:
        pass

    numpy_version: Optional[str] = None
    try:
        import numpy as np  # type: ignore

        numpy_version = str(getattr(np, "__version__", None))
    except Exception:
        pass

    info = EnvInfo(
        python_version=py,
        platform=plat,
        executable=exe,
        argv=argv_s,
        cuda_available=cuda_available,
        torch_version=torch_version,
        transformers_version=transformers_version,
        numpy_version=numpy_version,
    )
    return asdict(info)
