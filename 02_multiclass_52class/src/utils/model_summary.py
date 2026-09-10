"""
Model summary utilities for parameter counting and efficiency analysis.

"""

import time
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Count trainable and total parameters.
    
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def model_size_mb(model: nn.Module) -> float:
    """
    Estimate model size in megabytes.

    """
    param_size = 0
    for param in model.parameters():
        param_size += param.numel() * param.element_size()
    return param_size / (1024 * 1024)


def measure_inference_time(
    model: nn.Module,
    input_shape: Tuple[int, int, int, int] = (1, 1, 512, 512),
    device: str = "cpu",
    num_warmup: int = 10,
    num_runs: int = 100,
) -> dict:
    """
    Measure model inference time.

    """
    model.eval()
    model.to(device)

    dummy_input = torch.randn(*input_shape).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy_input)

    # Timed runs
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            if "cuda" in device:
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(dummy_input)
            if "cuda" in device:
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms

    times = np.array(times)
    return {
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
        "fps": float(1000.0 / np.mean(times)),
    }


def get_model_summary(
    model: nn.Module,
    input_shape: Tuple[int, int, int, int] = (1, 1, 512, 512),
    device: str = "cpu",
) -> dict:
    """
    Generate a comprehensive model summary.

    """
    trainable, total = count_parameters(model)
    size_mb = model_size_mb(model)
    inference = measure_inference_time(model, input_shape, device)

    return {
        "trainable_params": trainable,
        "total_params": total,
        "model_size_mb": round(size_mb, 2),
        "inference_time_ms": round(inference["mean_ms"], 2),
        "inference_std_ms": round(inference["std_ms"], 2),
        "fps": round(inference["fps"], 2),
    }
