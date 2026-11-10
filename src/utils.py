"""Shared helpers: config loading, seeding, device selection, memory and environment reporting."""
from __future__ import annotations

import os
import platform
import random
from contextlib import nullcontext

import numpy as np
import psutil
import torch
import yaml


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str = "auto") -> str:
    # PyTorch ROCm builds (AMD) also report through torch.cuda.
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def autocast_ctx(device: str, precision: str):
    if precision == "fp32":
        return nullcontext()
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[precision]
    return torch.autocast(device_type="cuda" if device.startswith("cuda") else "cpu", dtype=dtype)


def memory_mb(device: str) -> float:
    """Peak GPU memory on cuda, current process RSS on CPU."""
    if device.startswith("cuda"):
        return torch.cuda.max_memory_allocated() / 2**20
    return psutil.Process(os.getpid()).memory_info().rss / 2**20


def peak_memory_mb(device: str) -> float:
    """Highest memory used so far: GPU memory on cuda, process RAM on CPU."""
    if device.startswith("cuda"):
        return torch.cuda.max_memory_allocated() / 2**20
    info = psutil.Process(os.getpid()).memory_info()
    if hasattr(info, "peak_wset"):  # Windows
        return info.peak_wset / 2**20
    import resource  # Linux / WSL: ru_maxrss is in KiB

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**10


def env_info(device: str) -> dict:
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
        "torch": torch.__version__,
        "device": device,
        "torch_threads": torch.get_num_threads(),
        "cuda_available": torch.cuda.is_available(),
        "hip_version": getattr(torch.version, "hip", None),
        "cuda_version": torch.version.cuda,
    }
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_mem_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
    return info
