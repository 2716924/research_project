"""One-time PyTorch / MinAtar setup for CPU-only training on WITS cluster."""

import os

os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")

# Force CPU-only — no GPU probing or CUDA driver warnings.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TORCH_CUDA_ARCH_LIST"] = ""
os.environ["RCT_DEVICE"] = "cpu"

import torch
import minatar  # noqa: F401
from minatar.gym import register_envs

register_envs()

torch.backends.nnpack.enabled = False

DEVICE = torch.device("cpu")

# Thread configuration for CPU parallelism.
# RCT_TORCH_THREADS lets several follow-up runs share one node.
_num_threads = int(os.environ.get("RCT_TORCH_THREADS")
                   or os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
torch.set_num_threads(_num_threads)
# Inter-op threads handle parallel independent ops; keep low to avoid contention.
torch.set_num_interop_threads(min(2, _num_threads))
