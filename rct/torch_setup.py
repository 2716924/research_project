"""One-time PyTorch / MinAtar setup for training scripts."""

import os

os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")

# Hide CUDA before importing torch on CPU-only cluster nodes. Importing torch
# first can probe the GPU driver and emit warnings even when we only want CPU.
_requested_device = os.environ.get("RCT_DEVICE", "").strip().lower()
if _requested_device == "cpu" or os.environ.get("CUDA_VISIBLE_DEVICES") == "":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["TORCH_CUDA_ARCH_LIST"] = ""
    _force_cpu = True
else:
    _force_cpu = False

import torch
import minatar  # noqa: F401
from minatar.gym import register_envs

register_envs()

torch.backends.nnpack.enabled = False

if _force_cpu or _requested_device == "cpu":
    DEVICE = torch.device("cpu")
elif _requested_device == "cuda":
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    DEVICE = torch.device("cuda")
else:
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")

_num_threads = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
torch.set_num_threads(_num_threads)
