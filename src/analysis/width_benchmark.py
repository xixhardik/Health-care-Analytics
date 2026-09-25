"""Width feasibility benchmark for the U-Net (Sprint 3 audit, section B).

Measures - rather than estimates - what each candidate width actually costs on
this machine: parameter count, peak resident memory during a training step, and
wall-clock throughput for both training and inference.

Why subprocesses
----------------
Peak memory is measured as the process's resident-set-size (RSS) high-water
mark. PyTorch's CPU allocator caches freed blocks, so measuring several widths
inside one process would attribute the largest width's cache to every later
measurement. Each width is therefore benchmarked in a **fresh subprocess** and
the number reported is that subprocess's own peak.

This module only ever runs a handful of steps. It does not train anything.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

#: Widths under consideration for Sprint 3.
CANDIDATE_WIDTHS = (16, 32, 64)

#: The measurement script, executed in a subprocess per width. Kept as source
#: text so it is self-contained and cannot accidentally import benchmark state.
_WORKER = r'''
import json, os, sys, time
sys.path.insert(0, r"{project_root}")

import psutil
import torch
import torch.nn as nn

from src.models.unet import build_unet
from src.models.losses import DiceCrossEntropyLoss

width = int(sys.argv[1])
batch = int(sys.argv[2])
rows, cols = int(sys.argv[3]), int(sys.argv[4])
n_steps = int(sys.argv[5])

process = psutil.Process(os.getpid())

def memory_snapshot():
    """Commit charge is the reliable metric here.

    Resident-set size only reports pages currently in physical RAM, so under
    memory pressure Windows pages memory out and RSS *understates* what the
    process actually allocated - it can even fall as a model gets larger.
    'pagefile' in psutil on Windows is the commit charge (private bytes), which
    reflects allocation regardless of paging.
    """
    info = process.memory_info()
    return {{
        "rss": info.rss,
        "commit": getattr(info, "pagefile", info.rss),
        "peak_rss": getattr(info, "peak_wset", info.rss),
        "peak_commit": getattr(info, "peak_pagefile", info.rss),
    }}

baseline = memory_snapshot()
baseline_rss = baseline["rss"]

model = build_unet(n_classes=4, base_channels=width, depth=4)
model = model.to(memory_format=torch.channels_last)
params = model.count_parameters()

criterion = DiceCrossEntropyLoss(n_classes=4)
optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)

x = torch.randn(batch, 1, rows, cols).to(memory_format=torch.channels_last)
y = torch.randint(0, 4, (batch, rows, cols))

def train_step():
    out = model(x)
    loss = criterion(out, y)["loss"]
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()

# Warm-up: first step allocates workspaces and picks oneDNN kernels.
model.train()
train_step()

start = time.perf_counter()
for _ in range(n_steps):
    train_step()
train_seconds = (time.perf_counter() - start) / n_steps
after_train = memory_snapshot()

# Inference
model.eval()
with torch.inference_mode():
    model(x)
    start = time.perf_counter()
    for _ in range(n_steps):
        model(x)
    infer_seconds = (time.perf_counter() - start) / n_steps

system = psutil.virtual_memory()
result = {{
    "width": width,
    "batch_size": batch,
    "input_rows": rows,
    "input_cols": cols,
    "params_total": params["total"],
    "params_millions": round(params["total"] / 1e6, 3),
    "baseline_commit_mb": round(baseline["commit"] / 1e6, 1),
    # Primary memory metric: peak commit charge during the training steps.
    "train_peak_commit_mb": round(after_train["peak_commit"] / 1e6, 1),
    "train_commit_delta_mb": round(
        (after_train["peak_commit"] - baseline["commit"]) / 1e6, 1
    ),
    "train_peak_rss_mb": round(after_train["peak_rss"] / 1e6, 1),
    "train_rss_mb": round(after_train["rss"] / 1e6, 1),
    # Paging indicator: commit far above resident means the OS pushed pages out,
    # which is exactly what makes a too-large model collapse in throughput.
    "commit_minus_rss_mb": round(
        (after_train["commit"] - after_train["rss"]) / 1e6, 1
    ),
    "system_ram_available_mb_at_measure": round(system.available / 1e6, 1),
    "train_s_per_step": round(train_seconds, 4),
    "train_img_per_s": round(batch / train_seconds, 3),
    "infer_s_per_step": round(infer_seconds, 4),
    "infer_img_per_s": round(batch / infer_seconds, 3),
}}
print("__RESULT__" + json.dumps(result))
'''


def benchmark_width(
    width: int,
    *,
    project_root: Path,
    batch_size: int = 8,
    size: tuple[int, int] = (352, 256),
    n_steps: int = 3,
    timeout: int = 1800,
) -> dict:
    """Benchmark one width in a fresh subprocess.

    Returns the measurement dict, or a dict with an ``error`` key if the
    subprocess failed (for example, ran out of memory) - which is itself a
    feasibility result worth recording rather than an exception to propagate.
    """
    script = _WORKER.format(project_root=str(project_root))
    command = [
        sys.executable, "-c", script,
        str(width), str(batch_size), str(size[0]), str(size[1]), str(n_steps),
    ]
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout,
            cwd=str(project_root),
        )
    except subprocess.TimeoutExpired:
        return {
            "width": width, "batch_size": batch_size,
            "error": f"timed out after {timeout}s",
            "feasible": False,
        }

    for line in completed.stdout.splitlines():
        if line.startswith("__RESULT__"):
            return json.loads(line[len("__RESULT__"):])

    return {
        "width": width, "batch_size": batch_size,
        "error": (completed.stderr or completed.stdout or "no output")[-600:],
        "feasible": False,
    }


def project_epoch_time(
    measurement: dict, *, n_train_slices: int, n_val_slices: int
) -> dict:
    """Project epoch wall-clock time from measured throughput.

    Training and validation are projected separately because validation is
    forward-only and therefore several times faster per image.
    """
    if "train_img_per_s" not in measurement:
        return {}

    train_seconds = n_train_slices / measurement["train_img_per_s"]
    val_seconds = n_val_slices / measurement["infer_img_per_s"]
    total = train_seconds + val_seconds
    return {
        "n_train_slices": n_train_slices,
        "n_val_slices": n_val_slices,
        "train_minutes": round(train_seconds / 60, 1),
        "val_minutes": round(val_seconds / 60, 1),
        "epoch_minutes": round(total / 60, 1),
        "hours_for_30_epochs": round(total * 30 / 3600, 1),
    }


#: Peak commit charge of the width-16 / batch-8 configuration, which has
#: **demonstrably** completed two multi-hour training runs on this machine
#: (Sprint 2 baseline and Sprint 2 extended). Empirical proof beats any
#: threshold derived from "available RAM" at a single moment in time, so this is
#: used as the memory reference: a configuration needing no more than this is
#: known to work.
PROVEN_COMMIT_MB = 2550.0


def assess_feasibility(
    measurement: dict,
    projection: dict,
    *,
    available_ram_mb: float,
    overnight_budget_hours: float = 10.0,
    proven_commit_mb: float = PROVEN_COMMIT_MB,
) -> dict:
    """Judge a configuration on memory and on time, independently.

    **Memory** is judged against the peak commit charge of a configuration that
    has actually completed long training runs here, not against free RAM at a
    snapshot. Free RAM fluctuates with whatever else is open, whereas
    "width 16 / batch 8 ran for 4 hours twice" is hard evidence. A configuration
    at or below that commit level is treated as memory-proven; moderately above
    it is 'at risk' (the page file absorbs it, at the cost of throughput);
    far above it is infeasible.

    **Time** is whether 30 epochs fit the stated wall-clock budget.

    The two are reported separately because they have different remedies:
    memory pressure is fixed by a smaller batch, time is not.
    """
    if "error" in measurement:
        return {
            "memory_ok": False, "time_ok": False, "verdict": "failed to run",
            "detail": measurement["error"],
        }

    peak = measurement["train_peak_commit_mb"]
    if peak <= proven_commit_mb:
        memory_status = "proven (<= a configuration already run successfully)"
        memory_ok = True
    elif peak <= 1.6 * proven_commit_mb:
        memory_status = "at risk (above the proven level; expect paging)"
        memory_ok = False
    else:
        memory_status = "exceeds practical memory"
        memory_ok = False

    hours = projection.get("hours_for_30_epochs")
    time_ok = hours is not None and hours <= overnight_budget_hours

    if memory_ok and time_ok:
        verdict = "feasible"
    elif memory_ok and not time_ok:
        verdict = "memory-proven but too slow"
    elif not memory_ok and time_ok:
        verdict = "fast enough but memory-risky"
    else:
        verdict = "not feasible"

    return {
        "peak_commit_mb": peak,
        "proven_commit_reference_mb": proven_commit_mb,
        "commit_vs_proven": round(peak / proven_commit_mb, 2),
        "available_ram_mb_at_audit": round(available_ram_mb, 1),
        "memory_status": memory_status,
        "memory_ok": bool(memory_ok),
        "hours_for_30_epochs": hours,
        "overnight_budget_hours": overnight_budget_hours,
        "time_ok": bool(time_ok),
        "verdict": verdict,
    }


def relative_cost(measurements: list[dict]) -> list[dict]:
    """Express each width's cost relative to width 16, the current baseline."""
    reference = next(
        (m for m in measurements if m.get("width") == 16 and "error" not in m), None
    )
    if reference is None:
        return measurements

    out = []
    for measurement in measurements:
        entry = dict(measurement)
        if "error" not in measurement:
            entry["params_vs_w16"] = round(
                measurement["params_total"] / reference["params_total"], 2
            )
            entry["peak_commit_vs_w16"] = round(
                measurement["train_peak_commit_mb"] / reference["train_peak_commit_mb"], 2
            )
            entry["train_time_vs_w16"] = round(
                reference["train_img_per_s"] / measurement["train_img_per_s"], 2
            )
        out.append(entry)
    return out
