"""
Utility functions for the stitch module.

This module provides:
- GPU/CPU array abstraction (CuPy/NumPy)
- SLURM-aware worker detection
- Dtype resolution utilities
- Helper functions for stitching operations
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, Tuple

# Try to use CuPy for GPU acceleration, fall back to NumPy
try:
    import cupy as xp
    from cupyx.scipy import ndimage as cundi
    # Check if GPU is actually available at runtime
    try:
        _ = xp.array([1.0])  # Test GPU access
        _USING_CUPY = True
        print("[utils.py] Using CuPy (GPU) for array operations")
    except Exception as e:
        # CuPy imported but no GPU available - fallback to CPU
        print(f"[utils.py] CuPy available but GPU not accessible ({type(e).__name__}), falling back to CPU")
        import numpy as xp
        from scipy import ndimage as cundi
        _USING_CUPY = False
except (ModuleNotFoundError, ImportError):
    import numpy as xp
    from scipy import ndimage as cundi
    _USING_CUPY = False
    print("[utils.py] Using NumPy (CPU) for array operations")


# Cache for EDT-based blending weight maps to avoid recomputation per tile size/exponent
_EDT_WEIGHT_CACHE: Dict[Tuple[int, int, int], xp.ndarray] = {}


def _to_numpy(arr):
    """Convert array to NumPy (transfers from GPU to CPU if needed)."""
    if _USING_CUPY and isinstance(arr, xp.ndarray):
        # Use synchronous transfer for correctness
        return xp.asnumpy(arr)
    return np.asarray(arr)


def _load_to_gpu_async(data):
    """Load data to GPU asynchronously if using CuPy."""
    if _USING_CUPY:
        # Use default stream for async transfer
        return xp.asarray(data)
    return xp.asarray(data)


def _get_optimal_workers(use_gpu: bool = False, cpu_safety_buffer: int = 1, verbose: bool = False) -> int:
    """
    Determine optimal number of workers respecting SLURM allocation and system resources.

    Simplified version of ops_analysis.utils.resource_manager.get_optimal_workers.
    Checks in order: SLURM env vars -> process affinity -> os.cpu_count()

    Args:
        use_gpu: If True, returns 1 (GPU tasks typically don't benefit from CPU parallelism)
        cpu_safety_buffer: Number of cores to leave free for OS/main process
        verbose: If True, print which resource limit was used

    Returns:
        int: Optimal number of workers
    """
    if use_gpu:
        # GPU tasks should use sequential processing to avoid memory contention
        if verbose:
            print("[Workers] GPU mode: using 1 worker")
        return 1

    # Check SLURM environment first (respects HPC allocation)
    for env_var in ["SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"]:
        try:
            limit = int(os.environ[env_var])
            if verbose:
                print(f"[Workers] Using {env_var}: {limit} cores (allocated {limit - cpu_safety_buffer})")
            return max(1, limit - cpu_safety_buffer)
        except (KeyError, ValueError):
            pass

    # Fallback: Process affinity (respects cgroup/container CPU limits)
    try:
        limit = len(os.sched_getaffinity(0))
        if verbose:
            print(f"[Workers] Using sched_getaffinity: {limit} cores (allocated {limit - cpu_safety_buffer})")
        return max(1, limit - cpu_safety_buffer)
    except AttributeError:
        pass

    # Final fallback: os.cpu_count()
    limit = os.cpu_count() or 4
    if verbose:
        print(f"[Workers] Using os.cpu_count: {limit} cores (allocated {limit - cpu_safety_buffer})")
    return max(1, limit - cpu_safety_buffer)


def _discover_positions_fast(store_path: Path):
    """Fast position discovery using filesystem globbing.

    Only works for HCS layout: store/row/col/tile/0/.zarray
    Returns list of position paths like "A/1/000123", or None if not HCS layout
    """
    store_path = Path(store_path)

    # Check for HCS layout by looking for .zgroup and plate metadata
    if not (store_path / ".zgroup").exists():
        return None

    # Check if it's HCS layout by looking for plate metadata
    zattrs_path = store_path / ".zattrs"
    if zattrs_path.exists():
        try:
            with open(zattrs_path, 'r') as f:
                attrs = json.load(f)
                # HCS stores have 'plate' metadata
                if 'plate' not in attrs:
                    return None
        except Exception:
            return None

    # Glob for all .zarray files at the expected depth
    zarray_paths = sorted(store_path.glob("*/*/*/0/.zarray"))

    if not zarray_paths:
        return None

    # Extract position paths (parent's parent's parent relative to store)
    positions = []
    for zarray in zarray_paths:
        # zarray is at: store/row/col/tile/0/.zarray
        # Position is: row/col/tile
        tile_dir = zarray.parent.parent  # Go up from 0/.zarray to tile dir
        rel_path = tile_dir.relative_to(store_path)
        positions.append(str(rel_path))

    return positions


def find_contributing_fovs_yx(
    chunk: Tuple[slice, slice],
    fov_extents: Dict[str, Tuple[int, int, int, int]],
) -> list:
    """Return FOV names whose YX extents overlap the given output chunk.

    Parameters
    ----------
    chunk : Tuple[slice, slice]
        Output block slices in (Y, X) order.
    fov_extents : Dict[str, Tuple[int, int, int, int]]
        Map of tile_name -> (ys, ye, xs, xe) extents in output coordinates.

    Returns
    -------
    list
        Names of FOVs overlapping the chunk.
    """
    ys_chunk, xs_chunk = chunk
    y0, y1 = int(ys_chunk.start), int(ys_chunk.stop)
    x0, x1 = int(xs_chunk.start), int(xs_chunk.stop)
    contributing = []
    for name, (ys, ye, xs, xe) in fov_extents.items():
        if ye <= y0 or ys >= y1:
            continue
        if xe <= x0 or xs >= x1:
            continue
        contributing.append(name)
    return contributing


def _get_optimal_block_size(final_shape, tile_size, default_size=(1024, 1024)):
    """
    Determine optimal block size based on available GPU memory.

    Args:
        final_shape: Full output shape (T, C, Z, Y, X)
        tile_size: Individual tile size (Y, X)
        default_size: Default block size if GPU not available

    Returns:
        Tuple of (block_y, block_x) sizes
    """
    if not _USING_CUPY:
        return default_size

    try:
        # Get GPU memory info
        mempool = xp.get_default_memory_pool()
        device = xp.cuda.Device()
        free_mem, total_mem = xp.cuda.runtime.memGetInfo()

        # Estimate memory needed per block
        # We need: numer + denom buffers, plus tile cache
        dtype_size = 2  # float16
        tcz_size = final_shape[0] * final_shape[1] * final_shape[2]

        # Use 40% of free memory for block processing (conservative)
        target_mem = free_mem * 0.4

        # Calculate block size that fits in memory
        # memory = tcz_size * block_y * block_x * dtype_size * 2 (numer + denom)
        # Add buffer for tile cache
        block_pixels_max = target_mem / (tcz_size * dtype_size * 3)  # 3x for safety margin

        # Start with max tile dimension and find largest power-of-2 block
        max_dim = max(tile_size)
        block_size = min(4096, max_dim)  # Cap at 4096

        while block_size * block_size > block_pixels_max and block_size > 512:
            block_size //= 2

        block_size = max(512, block_size)  # Minimum 512
        block_size = min(block_size, final_shape[-2], final_shape[-1])  # Don't exceed image size

        print(f"[GPU Memory] Free: {free_mem / 1e9:.2f} GB, Total: {total_mem / 1e9:.2f} GB")
        print(f"[GPU Memory] Using block size: {block_size}x{block_size}")

        return (block_size, block_size)

    except Exception as e:
        print(f"[GPU Memory] Could not determine optimal block size: {e}. Using default.")
        return default_size


def _resolve_value_dtype(value_precision_bits: int):
    """Return a numpy dtype for requested float precision (16/32/64)."""
    bits = int(value_precision_bits)
    if bits == 16:
        return xp.float16
    if bits == 64:
        return xp.float64
    return xp.float32


def _resolve_shift_dtype(shifts: dict, tile_size: tuple, base_bits: int = 32):
    """Return an integer dtype for pixel shift indices with safety promotion.

    Starts at base_bits (16/32 supported) and promotes to int64 if min/max
    shifts plus tile_size would overflow the chosen dtype.
    """
    # Choose starting dtype
    if int(base_bits) == 16:
        dtype_idx = xp.int16
    else:
        dtype_idx = xp.int32

    try:
        info = np.iinfo(np.dtype(dtype_idx))
        min_allowed = int(info.min)
        max_allowed = int(info.max)
        if shifts:
            y_shifts = [int(s[0]) for s in shifts.values()]
            x_shifts = [int(s[1]) for s in shifts.values()]
            min_y_shift = min(y_shifts)
            min_x_shift = min(x_shifts)
            max_y_shift = max(y_shifts)
            max_x_shift = max(x_shifts)
        else:
            min_y_shift = min_x_shift = 0
            max_y_shift = max_x_shift = 0
        end_y = max_y_shift + int(tile_size[0])
        end_x = max_x_shift + int(tile_size[1])
        needs_promo = (
            min_y_shift < min_allowed
            or min_x_shift < min_allowed
            or end_y > max_allowed
            or end_x > max_allowed
        )
        if needs_promo:
            # Promote stepwise to int32 then int64 depending on start
            if dtype_idx == xp.int16:
                print(
                    "WARNING: shift index range exceeds int16; promoting shifts to int32."
                )
                dtype_idx = xp.int32
                info2 = np.iinfo(np.int32)
                if (
                    min_y_shift < int(info2.min)
                    or min_x_shift < int(info2.min)
                    or end_y > int(info2.max)
                    or end_x > int(info2.max)
                ):
                    print(
                        "WARNING: shift index range exceeds int32; promoting shifts to int64."
                    )
                    dtype_idx = xp.int64
            else:
                print(
                    "WARNING: shift index range exceeds int32; promoting shifts to int64."
                )
                dtype_idx = xp.int64
    except Exception:
        # Best-effort: fall back to int64 when any check fails
        dtype_idx = xp.int64

    return dtype_idx


def get_output_shape(shifts: dict, tile_size: tuple) -> tuple:
    """Get the output shape of the stitched image from the raw shifts"""

    x_shifts = [shift[0] for shift in shifts.values()]
    y_shifts = [shift[1] for shift in shifts.values()]
    max_x = int(xp.max(xp.asarray(x_shifts)))
    max_y = int(xp.max(xp.asarray(y_shifts)))

    return max_x + tile_size[0], max_y + tile_size[1]
