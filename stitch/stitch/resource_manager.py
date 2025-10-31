"""
System Resource Management for Parallel Processing

This module provides a centralized utility to determine the optimal number of parallel
workers for CPU-bound or GPU-bound tasks. It inspects the system's hardware
(CPU cores, system RAM, GPU count, and GPU VRAM) to calculate a safe and efficient
number of workers, preventing resource exhaustion and memory-related crashes.

Core Logic:
-----------
The `get_optimal_workers` function calculates the maximum number of workers that can be
supported by each of the following resources:
1.  **CPU Cores**: Respects HPC scheduler environments (like SLURM) and process
    affinity, ensuring the process doesn't try to use more cores than it's
    allocated. A safety buffer is kept to leave cores for the OS and main process.
2.  **System RAM**: Calculates how many worker processes can fit into the available
    system memory, based on estimated RAM usage per worker.
3.  **GPU VRAM**: For GPU-bound tasks, it calculates how many models/data chunks can
    fit onto the available GPU(s). It is conservative and uses the smallest GPU as
    the limiting factor.

The final number of workers is the minimum of these calculated limits, ensuring that
no single resource is over-provisioned.

How to Use:
-----------
Import the `get_optimal_workers` function.

- For GPU-intensive tasks (e.g., running a deep learning model):
  Set `use_gpu=True` and provide an estimate for the model's size in VRAM.

  ```python
  from ops_analysis.processes.resource_manager import get_optimal_workers

  # Get the ideal number of workers for a GPU task with a 4GB model.
  num_workers = get_optimal_workers(use_gpu=True, model_vram_gb=4)

  # Then use this number in your parallel processing backend:
  # from joblib import Parallel, delayed
  # Parallel(n_jobs=num_workers)(delayed(my_func)(i) for i in my_list)
  ```

- For CPU-only tasks (e.g., data processing, image manipulation):
  Set `use_gpu=False`. The VRAM arguments will be ignored.

  ```python
  # Get the ideal number of workers for a CPU-only task.
  num_workers = get_optimal_workers(use_gpu=False)
  ```

- For more advanced use, you can dynamically measure a model's VRAM footprint
  by providing a "factory" function that creates the model object.

  ```python
  from cellpose import models

  # Define a function (e.g., a lambda) that creates and returns the model.
  # Note: we are measuring the size of `model.net`, the underlying torch module.
  model_factory = lambda: models.CellposeModel(gpu=True, model_type='cyto3').net

  # The resource manager will create, measure, and then destroy the model.
  model_size_gb = get_model_vram_gb(model_factory)
  num_workers = get_optimal_workers(use_gpu=True, model_vram_gb=model_size_gb)
  ```
"""

import os
import torch
import psutil
import logging
from typing import Callable, Tuple
import numpy as np
import subprocess
import re

# Set up basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def _measure_vram(fcn, *args, **kwargs):
    """Measure VRAM usage with realistic overhead multiplier.

    Note: PyTorch CUDA memory allocation has significant overhead:
    - Memory fragmentation from PyTorch's caching allocator
    - Reserved but unallocated memory
    - Temporary tensors during forward/backward pass
    - Multiple workers can cause additional fragmentation

    We apply a 2.5x multiplier to account for:
    - PyTorch memory fragmentation during multi-worker execution
    - Additional memory for intermediate tensors
    - Memory allocation spikes during processing
    """
    peak_vram_gb = 4.0  # Default conservative estimate
    overhead_multiplier = 2.5  # Accounts for PyTorch overhead and fragmentation

    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        fcn(*args, **kwargs)

        peak_vram_bytes = torch.cuda.max_memory_allocated()
        measured_vram_gb = peak_vram_bytes / (1024**3)

        # Apply overhead multiplier for realistic estimate
        peak_vram_gb = measured_vram_gb * overhead_multiplier

        print(f"Measured peak VRAM (raw): {measured_vram_gb:.3f} GB")
        print(
            f"Estimated real-world VRAM per worker (with {overhead_multiplier}x overhead): {peak_vram_gb:.3f} GB"
        )
        print("--- End VRAM Measurement ---\n")

    except Exception as e:
        print(f"VRAM measurement failed: {e}. Falling back to default.")
    finally:
        torch.cuda.empty_cache()
        return peak_vram_gb


def _measure_ram(fcn, *args, **kwargs):
    """Measure peak RAM usage of a CPU function in GB."""
    peak_ram_gb = 1.0  # Default conservative estimate
    try:
        import gc

        gc.collect()

        process = psutil.Process()
        baseline_mem = process.memory_info().rss / (1024**3)

        fcn(*args, **kwargs)

        peak_mem = process.memory_info().rss / (1024**3)
        peak_ram_gb = peak_mem - baseline_mem

        print(f"Measured peak RAM for one worker: {peak_ram_gb:.3f} GB")
        print("--- End RAM Measurement ---\n")

    except Exception as e:
        print(f"RAM measurement failed: {e}. Falling back to default.")
    finally:
        return max(peak_ram_gb, 0.1)  # Minimum 100MB


def get_cpu_resources():
    """Returns the total number of logical CPU cores and total system RAM in GB."""
    cores = os.cpu_count() or 1
    memory = psutil.virtual_memory().total / (1024**3)  # Convert bytes to GB
    return cores, memory


def _get_ram_from_scontrol() -> Tuple[float, str]:
    """
    Tries to get job memory allocation by running `scontrol show job`.
    This is often the most reliable method on HPC systems.
    """
    try:
        job_id = os.environ.get("SLURM_JOB_ID")
        if not job_id:
            return None, None

        cmd = ["scontrol", "show", "job", job_id]
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True, timeout=5
        )

        search_pattern = r"TRES=.*mem=(\d+)([KMGTP]?)"
        match = re.search(search_pattern, result.stdout)

        if match:
            val_str, unit = match.groups()
            val = int(val_str)

            unit = unit.upper()
            source_str = f"scontrol (AllocTRES)"

            if unit == "G":
                return float(val), source_str
            elif unit == "M":
                return val / 1024.0, source_str
            elif unit == "K":
                return val / (1024.0**2), source_str
            elif unit == "T":
                return val * 1024.0, source_str
            else:  # Default unit is Megabytes
                return val / 1024.0, source_str

    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        ValueError,
        subprocess.TimeoutExpired,
    ):
        return None, None

    return None, None


def _get_cpu_limit():
    """
    Determines the number of available CPU cores for the process.
    Checks SLURM variables in priority order: CPUS_PER_TASK (GPU jobs) then CPUS_ON_NODE (CPU jobs).
    Returns a tuple of (limit, source_name).
    """
    # Check SLURM_CPUS_PER_TASK first (set for GPU tasks and srun with --cpus-per-task)
    try:
        limit = int(os.environ["SLURM_CPUS_PER_TASK"])
        return limit, "SLURM_CPUS_PER_TASK"
    except (KeyError, ValueError):
        pass

    # Check SLURM_CPUS_ON_NODE (set by submitit for CPU-only tasks)
    try:
        limit = int(os.environ["SLURM_CPUS_ON_NODE"])
        return limit, "SLURM_CPUS_ON_NODE"
    except (KeyError, ValueError):
        pass

    # Fallback: Process affinity (respects cgroup CPU limits)
    try:
        limit = len(os.sched_getaffinity(0))
        return limit, "sched_getaffinity"
    except AttributeError:
        limit = os.cpu_count() or 1
        return limit, "os.cpu_count"


def _get_available_ram_gb() -> Tuple[float, str]:
    """
    Determines the available RAM for the process, respecting SLURM and cgroup limits.
    Returns a tuple of (ram_in_gb, source_name).
    """
    # Try getting memory from `scontrol` first, as it's the most direct method
    ram_gb, source = _get_ram_from_scontrol()
    if ram_gb is not None:
        return ram_gb, source

    # Check SLURM env vars first. They are the most reliable source.
    try:
        # SLURM_MEM_PER_NODE is specified in MB
        mem_mb = int(os.environ["SLURM_MEM_PER_NODE"])
        mem_gb = mem_mb / 1024
        return mem_gb, "SLURM_MEM_PER_NODE"
    except KeyError:
        try:
            # If per-node is not set, try per-cpu
            mem_per_cpu_mb = int(os.environ["SLURM_MEM_PER_CPU"])
            # Default to 1 CPU if not specified, though it's unlikely in this context
            cpus_per_task, _ = _get_cpu_limit()
            mem_gb = (mem_per_cpu_mb * cpus_per_task) / 1024
            return mem_gb, "SLURM_MEM_PER_CPU"
        except KeyError:
            pass  # Continue to cgroup check

    # If not in SLURM, check cgroup limits (v2 and v1, which are the
    # underlying mechanism SLURM uses to enforce memory limits).
    # Cgroup v2 path (most modern systems)
    cgroup_v2_path = "/sys/fs/cgroup/memory.max"
    # Cgroup v1 path (older systems)
    cgroup_v1_path = "/sys/fs/cgroup/memory/memory.limit_in_bytes"

    try:
        # Prefer the modern cgroup v2 path if it exists
        if os.path.exists(cgroup_v2_path):
            with open(cgroup_v2_path, "r") as f:
                limit_str = f.read().strip()
            # If the limit is 'max', there's effectively no container limit.
            if limit_str != "max":
                limit_bytes = int(limit_str)
                mem_gb = limit_bytes / (1024**3)
                return mem_gb, "cgroup v2"

        # Fallback to the older cgroup v1 path
        elif os.path.exists(cgroup_v1_path):
            with open(cgroup_v1_path, "r") as f:
                limit_bytes = int(f.read().strip())

            # This value can be a huge number if no limit is set.
            # We check if it's smaller than the total system RAM as a safeguard.
            total_system_ram_bytes = psutil.virtual_memory().total
            if limit_bytes < total_system_ram_bytes:
                mem_gb = limit_bytes / (1024**3)
                return mem_gb, "cgroup v1"
    except (IOError, ValueError):
        # This can happen if the file is unreadable or contains an unexpected value.
        pass  # Continue to the final fallback

    # Fallback to total system memory if no other limit is found
    mem_gb = psutil.virtual_memory().total / (1024**3)
    return mem_gb, "psutil (system total)"


def get_gpu_resources():
    """
    Returns the number of available GPUs, a list of their names, and a list of their available VRAM in GB.
    Returns (0, [], []) if no GPUs are available.
    """
    if not torch.cuda.is_available():
        return 0, [], []

    gpu_count = torch.cuda.device_count()
    gpu_vram = []
    gpu_names = []
    for i in range(gpu_count):
        properties = torch.cuda.get_device_properties(i)
        vram = properties.total_memory / (1024**3)
        gpu_vram.append(vram)
        gpu_names.append(properties.name)

    return gpu_count, gpu_names, gpu_vram


def get_optimal_workers(
    use_gpu: bool = True,
    model_vram_gb: float = 2.0,
    data_vram_gb: float = 1.0,
    model_ram_gb: float = 2.0,
    data_ram_gb: float = 2.0,
    safety_factor: float = 0.8,
    cpu_safety_buffer: int = 1,
    verbose: bool = False,
):
    """
    Calculates the optimal number of parallel workers based on system resources.

    Args:
        use_gpu (bool): Set to True for GPU-bound tasks, False for CPU-only tasks.
        model_vram_gb (float): For GPU tasks, the estimated VRAM (GB) per model.
        data_vram_gb (float, optional): For GPU tasks, VRAM (GB) for worker's data.
        model_ram_gb (float, optional): Estimated RAM (in GB) required to load one model.
        data_ram_gb (float, optional): Estimated RAM (in GB) for one worker's data.
        safety_factor (float, optional): A factor to avoid using 100% of resources.
        cpu_safety_buffer (int, optional): How many CPU cores to leave free. Defaults to 1.
        verbose (bool, optional): If True, prints a detailed report of resources and limits.
                                  If False, prints only the final worker count. Defaults to True.

    Returns:
        int: The suggested number of workers.
    """
    system_cores, system_ram = get_cpu_resources()
    gpu_count, gpu_names, total_vram_list = get_gpu_resources()

    # Get job-specific limits, which may be different from system-wide totals
    job_cores, core_limit_source = _get_cpu_limit()
    job_ram_gb, ram_limit_source = _get_available_ram_gb()

    if verbose:
        # --- Print System Resource Report ---
        print("\n--- System Resource Report ---")
        print(f"System-wide: {system_cores} CPU Cores, {system_ram:.2f} GB RAM")
        print(
            f"Job-allocated: {job_cores} CPU Cores (source: {core_limit_source}), {job_ram_gb:.2f} GB RAM (source: {ram_limit_source})"
        )

        if gpu_count > 0:
            print(f"Found {gpu_count} GPU(s):")
            for i in range(gpu_count):
                print(f"  - GPU {i}: {gpu_names[i]}, {total_vram_list[i]:.2f} GB VRAM")
        else:
            print("No CUDA-enabled GPU found.")
        print("------------------------------\n")

    # Constraint 1: Number of CPU cores, respecting SLURM/affinity and a buffer
    cpu_limit = max(1, job_cores - cpu_safety_buffer)
    if verbose:
        print(f"Worker limit based on available CPUs (after buffer): {cpu_limit}")

    # Constraint 2: System RAM
    # How many workers can fit in the available RAM?
    ram_per_worker = model_ram_gb + data_ram_gb
    available_ram = job_ram_gb * safety_factor
    ram_limit = (
        int(available_ram / ram_per_worker) if ram_per_worker > 0 else float("inf")
    )
    if verbose:
        print(
            f"Worker limit based on Job RAM ({available_ram:.2f}GB available): {ram_limit}"
        )

    # Constraint 3: GPU VRAM - only applied if use_gpu is True.
    if use_gpu and gpu_count > 0:
        # We check the smallest GPU as the limiting factor
        min_gpu_vram = min(total_vram_list)
        vram_per_worker = model_vram_gb + data_vram_gb
        available_vram = min_gpu_vram * safety_factor

        # This is the limit *per GPU*. Total limit is this * number of GPUs
        workers_per_gpu = (
            int(available_vram / vram_per_worker)
            if vram_per_worker > 0
            else float("inf")
        )
        vram_limit = workers_per_gpu * gpu_count
        if verbose:
            print(
                f"Worker limit based on GPU VRAM ({available_vram:.2f}GB available per GPU): {vram_limit}"
            )
    else:
        # If no GPU, or if task is CPU-only, VRAM is not a constraint.
        vram_limit = float("inf")
        if verbose:
            if not use_gpu:
                print(
                    "CPU-only task specified (use_gpu=False), VRAM is not a constraint."
                )
            else:
                print("No GPU found, VRAM is not a constraint.")

    # The optimal number is the minimum of all constraints
    optimal_workers = max(1, min(cpu_limit, ram_limit, vram_limit))

    if verbose:
        print(f"Final determined optimal number of workers: {optimal_workers}")
    else:
        resource_type = "GPU" if use_gpu and gpu_count > 0 else "CPU"
        print(f"Optimal workers ({resource_type}): {optimal_workers}")

    return optimal_workers
