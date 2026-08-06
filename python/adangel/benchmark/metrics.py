"""Numerically explicit experiment metrics."""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Sequence


def percentile(values: Sequence[float], percent: float) -> float:
    ordered = sorted(float(x) for x in values)
    if not ordered:
        raise ValueError("cannot compute a percentile of an empty sequence")
    position = (len(ordered) - 1) * float(percent) / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize_samples(values: Sequence[float]) -> dict[str, float | int]:
    samples = [float(x) for x in values]
    if not samples or any(not math.isfinite(x) or x < 0 for x in samples):
        raise ValueError("timing samples must be non-empty, finite and non-negative")
    mean = statistics.fmean(samples)
    std = statistics.pstdev(samples)
    p25, median, p75 = percentile(samples, 25), percentile(samples, 50), percentile(samples, 75)
    return {
        "count": len(samples),
        "mean_ms": mean,
        "median_ms": median,
        "p5_ms": percentile(samples, 5),
        "p95_ms": percentile(samples, 95),
        "iqr_ms": p75 - p25,
        "cv_percent": 0.0 if mean == 0.0 and std == 0.0 else (math.inf if mean == 0.0 else std / mean * 100.0),
    }


def mse_fp64(output, reference) -> float:
    import torch

    if output.shape != reference.shape:
        raise ValueError("MSE operands have different shapes")
    if not torch.isfinite(output).all() or not torch.isfinite(reference).all():
        raise ValueError("NaN/Inf found in experiment output")
    delta = output.to(torch.float64) - reference.to(torch.float64)
    return float(torch.mean(delta * delta).item())


def bootstrap_median_ci(values: Sequence[float], samples: int, confidence: float, seed: int) -> tuple[float, float]:
    values = [float(x) for x in values]
    if not values or samples <= 0 or not 0.0 < confidence < 1.0:
        raise ValueError("invalid bootstrap parameters")
    rng = random.Random(seed)
    medians = []
    for _ in range(samples):
        medians.append(statistics.median(rng.choice(values) for _ in values))
    tail = (1.0 - confidence) * 50.0
    return percentile(medians, tail), percentile(medians, 100.0 - tail)
