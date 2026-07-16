"""One-time PyTorch / MinAtar setup for training scripts."""

import os

os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")

import torch
import minatar  # noqa: F401
from minatar.gym import register_envs

register_envs()

torch.backends.nnpack.enabled = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_num_threads = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
torch.set_num_threads(_num_threads)
