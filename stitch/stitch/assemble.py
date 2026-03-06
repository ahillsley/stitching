from iohub.ngff import open_ome_zarr
import os
import math
import json
import zarr
import dask.array as da
from tqdm import tqdm
from stitch.stitch.tile import augment_tile, pairwise_shifts, optimal_positions
from stitch.connect import read_shifts_biahub
from collections import defaultdict, deque
from iohub.ngff import TransformationMeta
from typing import Any, Callable, Dict, Literal, Optional, Tuple, Type, Union
import itertools
from numpy.typing import ArrayLike
from itertools import product
import yaml
from pathlib import Path
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
from joblib import Parallel, delayed

try:
    from dask.distributed import LocalCluster, Client
    _DASK_DISTRIBUTED_AVAILABLE = True
except ImportError:
    _DASK_DISTRIBUTED_AVAILABLE = False

# Import GPU/CPU abstraction and helper functions from utils
from stitch.stitch.utils import (
    xp,
    cundi,
    _USING_CUPY,
    _EDT_WEIGHT_CACHE,
    _to_numpy,
    _load_to_gpu_async,
    _get_optimal_workers,
    _discover_positions_fast,
    find_contributing_fovs_yx,
    _get_optimal_block_size,
    _resolve_value_dtype,
    _resolve_shift_dtype,
    get_output_shape,
)


# ── Module-level per-process write pipeline ──────────────────────────────────
# These persist across Dask tasks within the same worker process, enabling
# write pipelining: band N's writes run in background while band N+1 loads+GPU.
_band_write_pool = None
_band_pending_futures = []
_band_pending_data = []  # prevent GC of numpy arrays being written
_band_write_submit_time = None  # wall-clock start of write submission
_blosc_benchmark_done = False


# ── CPU / IO monitoring ──────────────────────────────────────────────────────
from ops_utils.profiling.proc_monitor import start_monitor as _start_cpu_monitor


def _init_band_write_pool(max_workers=8):
    """Get or create the persistent write thread pool for this worker process."""
    global _band_write_pool
    if _band_write_pool is None:
        _band_write_pool = ThreadPoolExecutor(max_workers=max_workers)
    return _band_write_pool


def _wait_band_writes():
    """Wait for all pending writes from the previous band, collect stats, free buffers."""
    global _band_pending_futures, _band_pending_data, _band_write_submit_time
    if not _band_pending_futures:
        return
    total_bytes = 0
    block_times = []
    for fut in _band_pending_futures:
        data_bytes, elapsed = fut.result()
        total_bytes += data_bytes
        block_times.append(elapsed)
    wall_time = time.time() - _band_write_submit_time if _band_write_submit_time else 0
    n = len(block_times)
    avg_t = sum(block_times) / n if n else 0
    max_t = max(block_times) if block_times else 0
    min_t = min(block_times) if block_times else 0
    throughput = total_bytes / wall_time / 1e6 if wall_time > 0 else 0
    print(f"[Write Stats] {n} blocks, {total_bytes/1e9:.2f}GB, "
          f"wall={wall_time:.1f}s ({throughput:.0f}MB/s), "
          f"per_block: min={min_t:.1f}s avg={avg_t:.1f}s max={max_t:.1f}s")
    _band_pending_futures.clear()
    _band_pending_data.clear()
    _band_write_submit_time = None


def _run_blosc_benchmark(sample_chunk):
    """One-time blosc compression benchmark to separate compression vs I/O costs."""
    global _blosc_benchmark_done
    if _blosc_benchmark_done:
        return
    _blosc_benchmark_done = True
    import numcodecs
    codec = numcodecs.Blosc(cname='lz4', clevel=1, shuffle=numcodecs.Blosc.BITSHUFFLE)
    chunk_bytes = sample_chunk.nbytes
    # Benchmark compression
    t0 = time.time()
    n_iters = 5
    for _ in range(n_iters):
        compressed = codec.encode(sample_chunk)
    compress_time = (time.time() - t0) / n_iters
    ratio = chunk_bytes / len(compressed)
    # Estimate for full band: ~825 chunks, 8 writer threads
    est_total = 825 * compress_time / 8
    print(f"[Blosc Bench] chunk={chunk_bytes/1e6:.1f}MB, compress={compress_time*1000:.1f}ms, "
          f"ratio={ratio:.1f}x, compressed={len(compressed)/1e6:.1f}MB, "
          f"est_825_chunks_8threads={est_total:.1f}s (PID={os.getpid()})")


def _compute_edt_weights(tile_size, blending_exponent, dtype_val):
    """Compute EDT blending weights for a tile of the given size."""
    ty, tx = int(tile_size[0]), int(tile_size[1])
    _mask = xp.zeros((ty, tx), dtype=bool)
    if ty > 2 and tx > 2:
        _mask[1:-1, 1:-1] = True
    _dist = cundi.distance_transform_edt(_mask).astype(xp.float32)
    _dist += 1e-6
    _weights = xp.where(_dist > 0, _dist ** float(blending_exponent), 0.0)
    return xp.asarray(_weights, dtype=dtype_val)


def _process_x_block_gpu(x0, x1, y0, y1, y_tiles, tile_cache, final_shape, dtype_val, use_edt, tile_weights, stream=None, profile=False):
    """
    Process a single X block on GPU, optionally using a specific CUDA stream.

    Returns:
        If profile=False: norm_cpu array
        If profile=True:  (norm_cpu, profile_dict) where profile_dict has timing breakdown
    """
    timings = {} if profile else None

    # Create buffers on the specified stream (or default)
    with xp.cuda.Stream(stream) if stream is not None else xp.cuda.Stream.null:
        if profile:
            xp.cuda.Device().synchronize()
            t0 = time.time()

        numer = xp.zeros(
            (final_shape[0], final_shape[1], final_shape[2], y1 - y0, x1 - x0),
            dtype=dtype_val,
        )
        denom = xp.zeros_like(numer, dtype=dtype_val)

        if profile:
            xp.cuda.Device().synchronize()
            timings['alloc'] = time.time() - t0
            t_accum_start = time.time()
            n_tiles_hit = 0
            t_slice_total = 0.0
            t_gpu_total = 0.0

        for tile_name, t_end, c_end, z_end, ys, ye, xs, xe in y_tiles:
            # X intersection within this block
            ix0 = max(x0, xs)
            ix1 = min(x1, xe)
            if ix0 >= ix1:
                continue
            iy0 = max(y0, ys)
            iy1 = min(y1, ye)

            if profile:
                t_sl = time.time()

            tile_full, t_end, c_end, z_end, ys, ye, xs, xe = tile_cache[tile_name]
            out_sl = (
                slice(0, t_end),
                slice(0, c_end),
                slice(0, z_end),
                slice(iy0 - y0, iy1 - y0),
                slice(ix0 - x0, ix1 - x0),
            )
            tile_sl = (
                slice(0, t_end),
                slice(0, c_end),
                slice(0, z_end),
                slice(iy0 - ys, iy1 - ys),
                slice(ix0 - xs, ix1 - xs),
            )
            # Slice on CPU first (cheap numpy indexing), then transfer to GPU
            block = xp.asarray(tile_full[tile_sl], dtype=dtype_val)

            if profile:
                t_slice_total += time.time() - t_sl
                t_gpu_op = time.time()

            if use_edt:
                wloc = tile_weights[
                    (iy0 - ys) : (iy1 - ys), (ix0 - xs) : (ix1 - xs)
                ]
                wloc = xp.asarray(wloc, dtype=dtype_val)
                nz = block != 0
                wloc = wloc * nz
                numer[out_sl] += block * wloc
                denom[out_sl] += wloc
            else:
                numer[out_sl] += block
                denom[out_sl] += (block != 0).astype(dtype_val)

            if profile:
                xp.cuda.Device().synchronize()
                t_gpu_total += time.time() - t_gpu_op
                n_tiles_hit += 1

        if profile:
            xp.cuda.Device().synchronize()
            timings['accum_total'] = time.time() - t_accum_start
            timings['accum_slice'] = t_slice_total
            timings['accum_gpu'] = t_gpu_total
            timings['accum_python'] = timings['accum_total'] - t_slice_total - t_gpu_total
            timings['n_tiles_hit'] = n_tiles_hit
            t_norm = time.time()

        # Normalize
        norm = xp.nan_to_num(numer / xp.maximum(denom, 1e-12))

        if profile:
            xp.cuda.Device().synchronize()
            timings['normalize'] = time.time() - t_norm
            t_d2h = time.time()

        # Transfer to CPU for writing (synchronous on this stream)
        norm_cpu = _to_numpy(norm)

        if profile:
            timings['d2h'] = time.time() - t_d2h

        # Clean up
        del numer, denom, norm

        if profile:
            return norm_cpu, timings
        return norm_cpu


def _process_y_band_gpu(y0, y1, total_x, y_tiles, tile_cache, final_shape,
                        dtype_val, use_edt, tile_weights, profile=False,
                        return_gpu=False):
    """Process entire Y-band as a single GPU operation.

    Instead of 52 separate X-block operations with individual alloc/normalize/D2H
    cycles, allocates one buffer spanning the full band width and processes all
    tiles in a single pass.

    If return_gpu=True, returns the GPU array without D2H transfer (caller handles it).

    Caller should wrap this in a per-well CUDA stream context to avoid
    default-stream serialization across wells.
    """
    timings = {} if profile else None

    if profile:
        xp.cuda.Device().synchronize()
        t0 = time.time()

    band_height = y1 - y0
    numer = xp.zeros(
        (final_shape[0], final_shape[1], final_shape[2], band_height, total_x),
        dtype=dtype_val,
    )
    denom = xp.zeros_like(numer, dtype=dtype_val)

    if profile:
        xp.cuda.Device().synchronize()
        timings['alloc'] = time.time() - t0
        mempool = xp.get_default_memory_pool()
        timings['alloc_gpu_mb'] = mempool.used_bytes() / 1e6
        t_accum_start = time.time()
        n_tiles_hit = 0
        t_slice_cpu_total = 0.0
        t_h2d_total = 0.0
        t_gpu_kernel_total = 0.0

    for tile_name, t_end, c_end, z_end, ys, ye, xs, xe in y_tiles:
        iy0 = max(y0, ys)
        iy1 = min(y1, ye)
        ix0 = max(0, xs)
        ix1 = min(total_x, xe)
        if ix0 >= ix1 or iy0 >= iy1:
            continue

        tile_full, t_end, c_end, z_end, ys, ye, xs, xe = tile_cache[tile_name]
        if profile:
            t_sl = time.time()
        out_sl = (
            slice(0, t_end),
            slice(0, c_end),
            slice(0, z_end),
            slice(iy0 - y0, iy1 - y0),
            slice(ix0, ix1),
        )
        tile_sl = (
            slice(0, t_end),
            slice(0, c_end),
            slice(0, z_end),
            slice(iy0 - ys, iy1 - ys),
            slice(ix0 - xs, ix1 - xs),
        )
        cpu_slice = tile_full[tile_sl]
        if profile:
            t_slice_cpu_total += time.time() - t_sl
            t_h = time.time()

        # Slice on CPU first (cheap numpy indexing), then transfer to GPU
        block = xp.asarray(cpu_slice, dtype=dtype_val)

        if use_edt:
            wloc = tile_weights[
                (iy0 - ys) : (iy1 - ys), (ix0 - xs) : (ix1 - xs)
            ]
            wloc = xp.asarray(wloc, dtype=dtype_val)
            if profile:
                t_h2d_total += time.time() - t_h
                t_k = time.time()
            nz = block != 0
            wloc = wloc * nz
            numer[out_sl] += block * wloc
            denom[out_sl] += wloc
        else:
            if profile:
                t_h2d_total += time.time() - t_h
                t_k = time.time()
            numer[out_sl] += block
            denom[out_sl] += (block != 0).astype(dtype_val)

        if profile:
            t_gpu_kernel_total += time.time() - t_k
            n_tiles_hit += 1

    if profile:
        xp.cuda.Device().synchronize()
        timings['accum'] = time.time() - t_accum_start
        timings['n_tiles_hit'] = n_tiles_hit
        timings['slice_cpu'] = t_slice_cpu_total
        timings['h2d'] = t_h2d_total
        timings['gpu_kernel'] = t_gpu_kernel_total
        t_norm = time.time()

    # Normalize in-place to minimize GPU memory (avoids ~20GB of temporaries)
    xp.maximum(denom, 1e-12, out=denom)
    xp.divide(numer, denom, out=numer)
    del denom
    # Release CuPy memory pool's hold on denom + tile temporaries.
    # Reduces GPU footprint from ~15GB to ~7GB before D2H, giving headroom for other workers.
    xp.get_default_memory_pool().free_all_blocks()
    xp.nan_to_num(numer, copy=False)
    norm = numer  # just rename, no allocation

    if profile:
        xp.cuda.Device().synchronize()
        timings['normalize'] = time.time() - t_norm
        mempool = xp.get_default_memory_pool()
        timings['post_norm_gpu_mb'] = mempool.used_bytes() / 1e6

    if return_gpu:
        # Return GPU array — caller handles D2H (for pipelined transfer)
        if profile:
            return norm, timings
        return norm

    if profile:
        t_d2h = time.time()

    # Transfer entire band to CPU
    norm_cpu = _to_numpy(norm)

    if profile:
        timings['d2h'] = time.time() - t_d2h

    del norm

    if profile:
        return norm_cpu, timings
    return norm_cpu


def _d2h_and_submit_writes(norm_gpu, transfer_stream, arr_out, final_shape,
                           y0, y1, tx, total_x, _write_executor, _write_futures):
    """Transfer GPU result to CPU on a dedicated stream, then submit zarr writes.

    Runs in a background thread so the main thread can start the next band's
    GPU compute while this D2H transfer is in progress.  The transfer_stream
    ensures the DMA engine handles the copy independently of the compute stream.
    """
    with transfer_stream:
        norm_cpu = norm_gpu.get()  # syncs transfer_stream only, then DMA copy
    del norm_gpu
    # Release GPU memory immediately so other parallel wells can use it
    xp.get_default_memory_pool().free_all_blocks()

    # Split into X-block chunks for zarr chunk alignment
    for x0 in range(0, total_x, tx):
        x1 = min(total_x, x0 + tx)
        block_cpu = norm_cpu[:, :, :, :, x0:x1]
        fut = _write_executor.submit(
            _write_zarr_block, arr_out, final_shape, y0, y1, x0, x1, block_cpu
        )
        _write_futures.append(fut)


def _write_zarr_block(arr_out, final_shape, y0, y1, x0, x1, norm_cpu):
    """Write a single normalized block to the output zarr array.

    Called from a background thread to overlap writes with GPU processing.
    Returns (data_bytes, elapsed) for profiling.
    """
    t0 = time.time()
    data_bytes = norm_cpu.nbytes
    arr_out[
        (
            slice(0, final_shape[0]),
            slice(0, final_shape[1]),
            slice(0, final_shape[2]),
            slice(y0, y1),
            slice(x0, x1),
        )
    ] = norm_cpu
    return data_bytes, time.time() - t0


def _load_band_tiles(y_tiles, store_path, flipud, fliplr, rot90):
    """Load all tiles for a Y-band in parallel, returning a tile_cache dict.

    Uses direct zarr array access (bypasses iohub HCS metadata parsing).
    Used for pipelined I/O: loading band N+1 while GPU processes band N.
    """
    tile_cache = {}

    def _load_single(tile_meta):
        tile_name, t_end, c_end, z_end, ys, ye, xs, xe = tile_meta
        tile_full = zarr.open(str(store_path / tile_name / "0"), mode="r")
        # Keep tiles on CPU to avoid GPU memory contention between parallel wells.
        # Tiles are transferred to GPU per-slice during band processing.
        tile_cpu = augment_tile(np.asarray(tile_full), flipud, fliplr, rot90)
        return tile_name, (tile_cpu, t_end, c_end, z_end, ys, ye, xs, xe)

    with ThreadPoolExecutor(max_workers=min(16, len(y_tiles))) as loader:
        futures = [loader.submit(_load_single, m) for m in y_tiles]
        for future in as_completed(futures):
            name, data = future.result()
            tile_cache[name] = data

    return tile_cache




def assemble(
    shifts: dict,
    tile_size: tuple,
    fov_store_path: str,
    flipud: bool = False,
    fliplr: bool = False,
    rot90: int = 0,
    divide_tile_size: Optional[Tuple[int, int]] = (4000, 4000),
    tcz_policy: Literal["max", "min"] = "min",
    blending_method: Literal["average", "edt"] = "edt",
    blending_exponent: float = 1.0,
    value_precision_bits: Literal[16, 32, 64] = 32,
):
    """Assemble the stitched image give the total shift of all tiles
    - Assume that we have paired image / total shift to be applied
    args:
        - shifts: dict of tile_name: (x_shift, y_shift)
        - tile_size: tuple of the tile size, only x and y dims
        - fov_store_path: path to the zarr file with the tiles
        - blending_method: 'average' (legacy) or 'edt' (distance-transform-based weighting)
        - blending_exponent: exponent applied to EDT distances for weighting
        - value_precision_bits: controls accumulation/output dtype (16→float16, 32→float32, 64→float64)
    """
    print(f"Using blending_method: {blending_method}")

    final_shape_xy = get_output_shape(shifts, tile_size)
    fov_store = open_ome_zarr(fov_store_path)

    # Determine target T, C, Z according to policy across all tiles
    tcz_list = []
    for tname in shifts.keys():
        tcz_list.append(fov_store[tname].data.shape[:3])
    tcz_arr = xp.asarray(tcz_list)
    # Warn if the time dimension varies across tiles, and state how it will be handled
    try:
        unique_T, counts_T = xp.unique(tcz_arr[:, 0], return_counts=True)
        if unique_T.size > 1:
            distribution = {
                int(k): int(v) for k, v in zip(unique_T.tolist(), counts_T.tolist())
            }
            target_T = (
                int(unique_T.max()) if tcz_policy == "max" else int(unique_T.min())
            )
            handling = (
                "zero-pad missing frames (leave zeros beyond each tile's T)"
                if tcz_policy == "max"
                else "truncate extra frames to the minimum T across tiles"
            )
            print(
                "WARNING: Inconsistent number of time frames (T) across tiles. "
                f"Distribution T->count: {distribution}. "
                f"Policy: '{tcz_policy}'. Target T={target_T}; will {handling}."
            )
    except Exception:
        # Best-effort warning; do not fail stitching if unique/count fails for any reason
        pass
    if tcz_policy == "min":
        tcz_target = (
            int(tcz_arr[:, 0].min()),
            int(tcz_arr[:, 1].min()),
            int(tcz_arr[:, 2].min()),
        )
    else:  # "max" (default)
        tcz_target = (
            int(tcz_arr[:, 0].max()),
            int(tcz_arr[:, 1].max()),
            int(tcz_arr[:, 2].max()),
        )

    # Resolve dtypes from requested precisions for 16, 32, 64
    dtype_val = _resolve_value_dtype(value_precision_bits)
    # Start at 32-bit indices and promote as needed
    dtype_idx = _resolve_shift_dtype(shifts, tile_size, base_bits=32)

    print(f"Using dtype for output: {dtype_val}")
    print(f"Using dtype for shifts: {dtype_idx}")

    final_shape = tcz_target + final_shape_xy
    # WARNING: allocating the full canvas can exceed memory for large mosaics.
    # This in-memory path is kept for smaller datasets; a streamed path exists below.
    output_image = xp.zeros(final_shape, dtype=dtype_val)
    # For legacy averaging we accumulate a divisor (counts). For EDT we accumulate a float weight sum.
    use_edt = str(blending_method).lower() == "edt"
    if use_edt:
        weight_sum = xp.zeros(final_shape, dtype=dtype_val)
        # Precompute or reuse a tile-local weight map based on distance to edges
        ty, tx = int(tile_size[0]), int(tile_size[1])
        cache_key = (ty, tx, int(float(blending_exponent) * 1e6))
        tile_weights = _EDT_WEIGHT_CACHE.get(cache_key)
        if tile_weights is None:
            # Boolean mask with interior True, edges False
            if ty > 2 and tx > 2:
                _mask = xp.zeros((ty, tx), dtype=bool)
                _mask[1:-1, 1:-1] = True
                _dist = cundi.distance_transform_edt(_mask).astype(xp.float32)
                _dist += 1e-6
                _weights = xp.where(_dist > 0, _dist ** float(blending_exponent), 0.0)
                tile_weights = xp.asarray(_weights, dtype=dtype_val)
            else:
                tile_weights = xp.ones((ty, tx), dtype=dtype_val)
            _EDT_WEIGHT_CACHE[cache_key] = tile_weights
    else:
        divisor = xp.zeros(final_shape, dtype=xp.uint8)

    for tile_name, shift in tqdm(shifts.items()):

        tile = fov_store[tile_name].data  # 5D array OME (T, C, Z, Y, X)
        tile = augment_tile(xp.asarray(tile), flipud, fliplr, rot90)
        # ignore sub-pixel shifts (which biahub does too by order=0 interpolation)
        # Use a wide integer type to avoid overflow for large mosaics (e.g., 51*2048 > 65535)
        shift_array = xp.asarray(shift, dtype=dtype_idx)
        # Future: add rotation / interpolation by first padding, then placing padded block into
        # final output

        # Compute bounds for T, C, Z respecting target and tile sizes
        t_end = min(tile.shape[0], final_shape[0])
        c_end = min(tile.shape[1], final_shape[1])
        z_end = min(tile.shape[2], final_shape[2])

        ys, ye = shift_array[0], shift_array[0] + tile_size[0]
        xs, xe = shift_array[1], shift_array[1] + tile_size[1]

        tile_block = tile[
            0:t_end, 0:c_end, 0:z_end, : tile_size[0], : tile_size[1]
        ].astype(dtype_val, copy=False)

        if use_edt:
            # Weighted accumulation using precomputed tile-local weights (Y,X)
            # Ignore zero-valued padded pixels
            w = tile_weights
            nz_mask = tile_block != 0
            w_masked = w * nz_mask
            # Expand weights to T,C,Z dims by broadcasting
            output_image[
                0:t_end,
                0:c_end,
                0:z_end,
                ys:ye,
                xs:xe,
            ] += (
                tile_block * w_masked
            )
            weight_sum[
                0:t_end,
                0:c_end,
                0:z_end,
                ys:ye,
                xs:xe,
            ] += w_masked
        else:
            # Legacy simple averaging by counts
            output_image[
                0:t_end,
                0:c_end,
                0:z_end,
                ys:ye,
                xs:xe,
            ] += tile_block

            # Only add divisor where the tile is not zero
            divi_tile = (tile_block != 0).astype(xp.uint8)
            divisor[
                0:t_end,
                0:c_end,
                0:z_end,
                ys:ye,
                xs:xe,
            ] += divi_tile

    stitched = xp.zeros_like(output_image, dtype=dtype_val)

    def _divide(a, b):
        return xp.nan_to_num(a / b)

    if use_edt:
        out = divide_tile(
            output_image,
            weight_sum,
            func=_divide,
            out_array=stitched,
            tile=divide_tile_size,
        )
    else:
        out = divide_tile(
            output_image,
            divisor,
            func=_divide,
            out_array=stitched,
            tile=divide_tile_size,
        )
    del out  # free memory

    return stitched


def assemble_streaming(
    shifts: dict,
    tile_size: tuple,
    fov_store_path: str,
    stitched_pos=None,
    arr_out=None,
    flipud: bool = False,
    fliplr: bool = False,
    rot90: int = 0,
    tcz_policy: Literal["max", "min"] = "min",
    blending_method: Literal["average", "edt"] = "edt",
    blending_exponent: float = 1.0,
    value_precision_bits: Literal[16, 32, 64] = 32,
    chunks_size: Optional[Tuple[int, int, int, int, int]] = None,
    scale: Optional[Tuple[float, float, float, float, float]] = None,
    divide_tile_size: Optional[Tuple[int, int]] = (1024, 1024),
    use_adaptive_blocks: bool = True,
    parallel_x_blocks: bool = True,
    profile: bool = False,
    debug_zero_mask: bool = False,
    per_channel_edt: bool = True,
    parallel_y_bands: bool = False,
    n_workers: Optional[int] = None,
):
    """Streamed assembly that avoids saving auxiliary arrays.

    For each output YX block (divide_tile_size), accumulates numerator and denominator in RAM,
    normalizes, and writes directly to the on-disk output image '0'. No auxiliary arrays are saved.

    Args:
        use_adaptive_blocks: If True, automatically determine optimal block size based on GPU memory
        parallel_x_blocks: If True, process X blocks in parallel using CUDA streams (GPU only)
        profile: If True, print detailed timing breakdown per Y-band (adds GPU syncs — slower but accurate)
        debug_zero_mask: If True, print debug info about zero mask statistics
        per_channel_edt: If True, compute EDT weights per-channel based on nonzero mask (data-driven)
        parallel_y_bands: If True, process Y-bands in parallel using joblib (CPU only, ignored for GPU)
        n_workers: Number of workers for parallel Y-band processing. If None, uses _get_optimal_workers()
    """
    # Allow enabling profiling via environment variable (overrides parameter)
    if os.environ.get("STITCH_PROFILE", "").strip() in ("1", "true", "yes"):
        profile = True

    print(f"[assemble.streaming] Using blending_method: {blending_method}")
    print(f"[assemble.streaming] Adaptive blocks: {use_adaptive_blocks}, Parallel X-blocks: {parallel_x_blocks}")
    if profile:
        print(f"[assemble.streaming] PROFILING ENABLED (GPU syncs will slow overall runtime)")

    # Read tile shapes directly from .zarray JSON files (avoids opening full HCS store).
    fov_store_p = Path(fov_store_path)
    tile_shapes = {}
    for tname in shifts.keys():
        with open(fov_store_p / tname / "0" / ".zarray") as f:
            meta = json.load(f)
        tile_shapes[tname] = tuple(meta["shape"])

    # Determine target T,C,Z
    tcz_list = [s[:3] for s in tile_shapes.values()]
    tcz_arr = xp.asarray(tcz_list)
    if tcz_policy == "min":
        tcz_target = (
            int(tcz_arr[:, 0].min()),
            int(tcz_arr[:, 1].min()),
            int(tcz_arr[:, 2].min()),
        )
    else:
        tcz_target = (
            int(tcz_arr[:, 0].max()),
            int(tcz_arr[:, 1].max()),
            int(tcz_arr[:, 2].max()),
        )

    dtype_val = _resolve_value_dtype(value_precision_bits)
    final_shape_xy = get_output_shape(shifts, tile_size)
    final_shape = tcz_target + final_shape_xy

    if any(d == 0 for d in final_shape):
        raise RuntimeError(
            f"[assemble_streaming] Computed output shape has zero dimension: {final_shape}. "
            f"tcz_target={tcz_target}, final_shape_xy={final_shape_xy}, "
            f"num_tiles={len(shifts)}, tcz_policy='{tcz_policy}'"
        )

    # Adaptive block sizing based on GPU memory
    if use_adaptive_blocks:
        divide_tile_size = _get_optimal_block_size(final_shape, tile_size, divide_tile_size)

    if chunks_size is None:
        ty, tx = divide_tile_size if divide_tile_size is not None else (1024, 1024)
        chunks_size = (1, 1, 1, int(ty), int(tx))

    # Create output array on disk only (no auxiliary arrays)
    if arr_out is None:
        try:
            stitched_pos.create_zeros(
                "0",
                shape=final_shape,
                chunks=chunks_size,
                dtype=dtype_val,
                transform=(
                    [TransformationMeta(type="scale", scale=scale)]
                    if scale is not None
                    else None
                ),
            )
        except Exception:
            pass

    use_edt = str(blending_method).lower() == "edt"

    # Precompute EDT weights for one tile
    tile_weights = _compute_edt_weights(tile_size, blending_exponent, dtype_val) if use_edt else None

    if arr_out is None:
        arr_out = stitched_pos["0"]
    dtype_idx = _resolve_shift_dtype(shifts, tile_size, base_bits=32)

    # Precompute tile metadata for fast intersection checks (using cached shapes)
    tile_meta = []
    for tile_name, shift in shifts.items():
        ts = tile_shapes[tile_name]
        shift_array = xp.asarray(shift, dtype=dtype_idx)
        t_end = min(ts[0], final_shape[0])
        c_end = min(ts[1], final_shape[1])
        z_end = min(ts[2], final_shape[2])
        ys, ye = int(shift_array[0]), int(shift_array[0] + tile_size[0])
        xs, xe = int(shift_array[1]), int(shift_array[1] + tile_size[1])
        tile_meta.append((tile_name, t_end, c_end, z_end, ys, ye, xs, xe))

    # Blockwise accumulation over YX
    ty, tx = divide_tile_size if divide_tile_size is not None else (1024, 1024)
    total_y, total_x = final_shape[-2], final_shape[-1]

    # Pre-identify all Y-bands and their tiles for batch loading
    y_bands = []
    for y0 in range(0, total_y, ty):
        y1 = min(total_y, y0 + ty)
        y_tiles = [
            (nm, t_end, c_end, z_end, ys, ye, xs, xe)
            for (nm, t_end, c_end, z_end, ys, ye, xs, xe) in tile_meta
            if not (ye <= y0 or ys >= y1)
        ]
        y_bands.append((y0, y1, y_tiles))

    # Pipeline: prefetch multiple Y-bands ahead so tiles are ready when GPU needs them.
    # With 2 concurrent loaders and 3-band prefetch, effective loading rate (~3.5s/band)
    # roughly matches GPU processing rate (~2-3s/band), eliminating most tile-wait stalls.
    _pipeline_executor = ThreadPoolExecutor(max_workers=2)
    _prefetch_depth = 3

    # Background writer: submit zarr writes to a thread pool so GPU can
    # continue with the next Y-band while chunks are flushed to disk.
    # Zarr chunks are independent files — parallel writes are safe.
    _write_executor = ThreadPoolExecutor(max_workers=14)
    _write_futures = []

    # D2H pipelining: transfer GPU results to CPU on a dedicated CUDA stream
    # in a background thread, so GPU can start the next band's compute immediately.
    _d2h_executor = ThreadPoolExecutor(max_workers=1)
    _transfer_stream = xp.cuda.Stream(non_blocking=True) if _USING_CUPY else None
    _prev_d2h_future = None

    _load_futures = deque()
    for i in range(min(_prefetch_depth, len(y_bands))):
        _load_futures.append(_pipeline_executor.submit(
            _load_band_tiles, y_bands[i][2], fov_store_p, flipud, fliplr, rot90
        ))

    for band_idx, (y0, y1, y_tiles) in enumerate(tqdm(y_bands, desc="Stitching Y")):
        t_band_start = time.time()

        # Wait for pre-loaded tiles (pipelined: loaded during previous bands' processing)
        t_load_start = time.time()
        tile_cache = _load_futures.popleft().result()
        t_load_elapsed = time.time() - t_load_start

        # Keep prefetch pipeline full
        next_prefetch = band_idx + _prefetch_depth
        if next_prefetch < len(y_bands):
            _load_futures.append(_pipeline_executor.submit(
                _load_band_tiles, y_bands[next_prefetch][2], fov_store_p, flipud, fliplr, rot90
            ))

        print(f"  [Y-band {band_idx+1}/{len(y_bands)}] Loaded {len(y_tiles)} tiles (waited {t_load_elapsed:.2f}s)")

        # X-block list for zarr chunk alignment (used by both GPU and CPU paths)
        x_blocks = [(x0, min(total_x, x0 + tx)) for x0 in range(0, total_x, tx)]

        # Full Y-band GPU processing with pipelined D2H
        t_process_start = time.time()
        if _USING_CUPY:
            ret = _process_y_band_gpu(
                y0, y1, total_x, y_tiles, tile_cache, final_shape,
                dtype_val, use_edt, tile_weights, profile=profile,
                return_gpu=True,
            )
            if profile:
                norm_gpu, band_timings = ret
            else:
                norm_gpu = ret

            t_process_elapsed = time.time() - t_process_start

            # Wait for previous band's D2H + writes to complete before submitting new one
            if _prev_d2h_future is not None:
                _prev_d2h_future.result()

            # Submit D2H + write for current band in background thread.
            # GPU is free to process next band while DMA engine handles the transfer.
            _prev_d2h_future = _d2h_executor.submit(
                _d2h_and_submit_writes, norm_gpu, _transfer_stream,
                arr_out, final_shape, y0, y1, tx, total_x,
                _write_executor, _write_futures,
            )

            # Release freed GPU memory back to CUDA so other parallel wells
            # can allocate from it. CuPy's memory pool holds onto freed blocks
            # by default, which causes OOM when multiple wells share one GPU.
            xp.get_default_memory_pool().free_all_blocks()

            if profile:
                print(f"  [Y-band {band_idx+1}/{len(y_bands)}] PROFILE: "
                      f"alloc={band_timings['alloc']*1000:.1f}ms  "
                      f"accum={band_timings['accum']*1000:.1f}ms ({band_timings['n_tiles_hit']} tiles)  "
                      f"norm={band_timings['normalize']*1000:.1f}ms  "
                      f"total={t_process_elapsed:.2f}s (D2H pipelined)")
            else:
                print(f"  [Y-band {band_idx+1}/{len(y_bands)}] GPU processing: {t_process_elapsed:.2f}s (D2H pipelined)")

        else:
            # Sequential CPU processing with per-channel EDT support
            for x0, x1 in x_blocks:
                numer = xp.zeros(
                    (final_shape[0], final_shape[1], final_shape[2], y1 - y0, x1 - x0),
                    dtype=dtype_val,
                )
                denom = xp.zeros_like(numer, dtype=dtype_val)

                for tile_name, t_end, c_end, z_end, ys, ye, xs, xe in y_tiles:
                    ix0 = max(x0, xs)
                    ix1 = min(x1, xe)
                    if ix0 >= ix1:
                        continue
                    iy0 = max(y0, ys)
                    iy1 = min(y1, ye)
                    tile_full, t_end, c_end, z_end, ys, ye, xs, xe = tile_cache[tile_name]
                    tile_sl = (
                        slice(0, t_end),
                        slice(0, c_end),
                        slice(0, z_end),
                        slice(iy0 - ys, iy1 - ys),
                        slice(ix0 - xs, ix1 - xs),
                    )
                    block = tile_full[tile_sl].astype(dtype_val, copy=False)

                    if use_edt and per_channel_edt:
                        for c_idx in range(c_end):
                            ch_out_sl = (
                                slice(0, t_end),
                                slice(c_idx, c_idx + 1),
                                slice(0, z_end),
                                slice(iy0 - y0, iy1 - y0),
                                slice(ix0 - x0, ix1 - x0),
                            )
                            block_ch = block[:, c_idx : c_idx + 1, :, :, :]
                            nz2d = xp.any(block[:, c_idx, :, :, :] != 0, axis=(0, 1))

                            if debug_zero_mask:
                                total_px = nz2d.size
                                nonzero_px = int(xp.count_nonzero(nz2d))
                                zero_px = int(total_px - nonzero_px)
                                print(f"    [debug] tile={tile_name} ch={c_idx}: {nonzero_px}/{total_px} nonzero ({zero_px} zeros)")

                            if not xp.any(nz2d):
                                continue

                            if int(xp.count_nonzero(nz2d)) == nz2d.size:
                                wloc2d = tile_weights[iy0 - ys : iy1 - ys, ix0 - xs : ix1 - xs]
                            else:
                                _dist = cundi.distance_transform_edt(nz2d).astype(xp.float32)
                                _dist += 1e-6
                                wloc2d = xp.where(_dist > 0, _dist ** float(blending_exponent), 0.0)
                                wloc2d = xp.asarray(wloc2d, dtype=dtype_val)

                            wloc5 = wloc2d[None, None, None, :, :]
                            numer[ch_out_sl] += (block_ch.astype(xp.float32) * wloc5).astype(dtype_val)
                            denom[ch_out_sl] += wloc5

                    elif use_edt:
                        out_sl = (
                            slice(0, t_end),
                            slice(0, c_end),
                            slice(0, z_end),
                            slice(iy0 - y0, iy1 - y0),
                            slice(ix0 - x0, ix1 - x0),
                        )
                        wloc = tile_weights[
                            (iy0 - ys) : (iy1 - ys), (ix0 - xs) : (ix1 - xs)
                        ]
                        wloc = xp.asarray(wloc, dtype=dtype_val)
                        nz = block != 0
                        wloc = wloc * nz
                        numer[out_sl] += block * wloc
                        denom[out_sl] += wloc
                    else:
                        out_sl = (
                            slice(0, t_end),
                            slice(0, c_end),
                            slice(0, z_end),
                            slice(iy0 - y0, iy1 - y0),
                            slice(ix0 - x0, ix1 - x0),
                        )
                        numer[out_sl] += block
                        denom[out_sl] += (block != 0).astype(dtype_val)

                norm = xp.nan_to_num(numer / xp.maximum(denom, 1e-12))
                norm_cpu = _to_numpy(norm)
                arr_out[
                    (
                        slice(0, final_shape[0]),
                        slice(0, final_shape[1]),
                        slice(0, final_shape[2]),
                        slice(y0, y1),
                        slice(x0, x1),
                    )
                ] = norm_cpu
                del numer, denom, norm, norm_cpu

            t_process_elapsed = time.time() - t_process_start
            print(f"  [Y-band {band_idx+1}/{len(y_bands)}] Processing: {t_process_elapsed:.2f}s")

        # Free cache for this Y band
        t_cleanup = time.time()
        tile_cache.clear()
        t_cleanup_elapsed = time.time() - t_cleanup

        t_band_elapsed = time.time() - t_band_start
        if profile:
            print(f"  [Y-band {band_idx+1}/{len(y_bands)}] cleanup={t_cleanup_elapsed:.2f}s  "
                  f"band_total={t_band_elapsed:.2f}s ({len(x_blocks)} X-blocks)")
        else:
            print(f"  [Y-band {band_idx+1}/{len(y_bands)}] Total: {t_band_elapsed:.2f}s ({len(x_blocks)} X-blocks)")

    _pipeline_executor.shutdown(wait=False)

    # Wait for last band's D2H + write submission to complete
    if _prev_d2h_future is not None:
        _prev_d2h_future.result()
    _d2h_executor.shutdown(wait=True)

    # Drain all background zarr writes and shut down the write executor
    t_drain_start = time.time()
    n_failed = 0
    for fut in _write_futures:
        try:
            fut.result()
        except Exception as e:
            n_failed += 1
            print(f"  WARNING: background zarr write failed: {e}")
    _write_executor.shutdown(wait=True)
    t_drain_elapsed = time.time() - t_drain_start
    print(f"  [Write drain] Waited {t_drain_elapsed:.2f}s for {len(_write_futures)} background writes"
          f"{f' ({n_failed} failed)' if n_failed else ''}")

    return arr_out


def _process_single_well(well_id, shifts, output_store, input_store_path, tile_shape,
                        flipud, fliplr, rot90, kwargs, blending_method, chunks_size, scale,
                        well_lock, parallel_y_bands=False, n_workers=None):
    """
    Process a single well for parallel stitching.
    Thread-safe helper function to stitch one well.

    Args:
        parallel_y_bands: If True, process Y-bands in parallel (CPU mode)
        n_workers: Number of workers for parallel Y-band processing
    """
    try:
        print(f"[Well Processing] Starting well {well_id}")

        # Thread-safe creation of stitched position
        with well_lock:
            stitched_pos = output_store.create_position("A", well_id, "0")

        # Process well (this is where GPU computation happens)
        assemble_streaming(
            shifts=shifts,
            tile_size=tile_shape[-2:],
            fov_store_path=input_store_path,
            stitched_pos=stitched_pos,
            flipud=flipud,
            fliplr=fliplr,
            rot90=rot90,
            tcz_policy=kwargs.get("tcz_policy", "min"),
            blending_method=blending_method,
            blending_exponent=kwargs.get("blending_exponent", 1.0),
            value_precision_bits=kwargs.get("value_precision_bits", 32),
            chunks_size=chunks_size,
            scale=scale,
            divide_tile_size=kwargs.get("target_chunks_yx", (1024, 1024)),
            profile=kwargs.get("profile", False),
            debug_zero_mask=kwargs.get("debug_zero_mask", False),
            per_channel_edt=kwargs.get("per_channel_edt", True),
            parallel_y_bands=parallel_y_bands,
            n_workers=n_workers,
        )

        del stitched_pos  # free reference
        print(f"[Well Processing] Completed well {well_id}")
        return well_id, True

    except Exception as e:
        print(f"[Well Processing] Failed well {well_id}: {e}")
        return well_id, False


def _stitch_band_dask_worker(well_id, band_idx, y0, y1, y_tiles,
                              output_store_path, input_store_path,
                              final_shape, tile_size, tx,
                              flipud, fliplr, rot90,
                              blending_method, blending_exponent,
                              value_precision_bits, cuda_path="",
                              is_last_band=False):
    """
    Process one Y-band for one well. Self-contained Dask work unit.

    Each worker runs in its own process with its own GIL and CUDA context.
    Uses a persistent write pool to pipeline writes: band N's writes run in
    background while band N+1 does load+GPU, cutting idle time significantly.
    """
    import numcodecs
    numcodecs.blosc.set_nthreads(4)

    # Ensure CUDA is findable for CuPy JIT kernel compilation
    if cuda_path and 'CUDA_PATH' not in os.environ:
        os.environ['CUDA_PATH'] = cuda_path

    try:
        t_start = time.time()

        print(f"[Band Worker] Starting well={well_id} band={band_idx+1} "
              f"y=[{y0}:{y1}] tiles={len(y_tiles)} (PID={os.getpid()})")

        # Open output zarr array directly (instant, no HCS parsing)
        arr_out = zarr.open(os.path.join(output_store_path, "A", well_id, "0", "0"), mode="r+")

        # Compute EDT weights on this worker's GPU
        dtype_val = _resolve_value_dtype(value_precision_bits)
        use_edt = str(blending_method).lower() == "edt"
        tile_weights = _compute_edt_weights(tile_size, blending_exponent, dtype_val) if use_edt else None

        # Load tiles for this band
        t_load = time.time()
        fov_store_p = Path(input_store_path)
        tile_cache = _load_band_tiles(y_tiles, fov_store_p, flipud, fliplr, rot90)
        t_load_elapsed = time.time() - t_load

        # GPU process this Y-band
        t_gpu = time.time()
        total_x = final_shape[-1]
        norm_gpu = _process_y_band_gpu(
            y0, y1, total_x, y_tiles, tile_cache, final_shape,
            dtype_val, use_edt, tile_weights, return_gpu=True,
        )
        t_gpu_elapsed = time.time() - t_gpu

        # D2H transfer
        t_d2h = time.time()
        norm_cpu = norm_gpu.get()
        del norm_gpu, tile_cache
        xp.get_default_memory_pool().free_all_blocks()
        t_d2h_elapsed = time.time() - t_d2h

        # Wait for previous band's writes AFTER load+GPU+D2H completes.
        # This lets writes overlap with load+GPU of the current band.
        # Requires enough CPUs (128) so write threads don't starve load threads.
        t_wait = time.time()
        _wait_band_writes()
        t_wait_elapsed = time.time() - t_wait

        # One-time blosc compression benchmark (first band only)
        _run_blosc_benchmark(
            np.ascontiguousarray(norm_cpu[0, 0, 0, :, :tx])
        )

        # Submit writes to persistent pool (non-blocking).
        global _band_write_submit_time
        _band_write_submit_time = time.time()
        pool = _init_band_write_pool()
        n_blocks = 0
        for x0 in range(0, total_x, tx):
            x1 = min(total_x, x0 + tx)
            _band_pending_futures.append(pool.submit(
                _write_zarr_block, arr_out, final_shape, y0, y1, x0, x1,
                norm_cpu[:, :, :, :, x0:x1]
            ))
            n_blocks += 1
        # Keep norm_cpu alive until writes finish (prevent GC)
        _band_pending_data.append(norm_cpu)
        print(f"[Band Worker] Submitted {n_blocks} write blocks, "
              f"{norm_cpu.nbytes/1e9:.2f}GB ({norm_cpu.dtype})")

        # For the last band of a well, wait for writes to fully complete
        if is_last_band:
            t_flush = time.time()
            _wait_band_writes()
            t_flush_elapsed = time.time() - t_flush
            print(f"[Band Worker] Final flush well={well_id}: {t_flush_elapsed:.1f}s")

        t_total = time.time() - t_start
        print(f"[Band Worker] Done well={well_id} band={band_idx+1} "
              f"load={t_load_elapsed:.1f}s gpu={t_gpu_elapsed:.1f}s "
              f"d2h={t_d2h_elapsed:.1f}s wait_prev={t_wait_elapsed:.1f}s total={t_total:.1f}s")
        return well_id, band_idx, True

    except Exception as e:
        import traceback
        print(f"[Band Worker] Failed well={well_id} band={band_idx+1}: {e}")
        traceback.print_exc()
        return well_id, band_idx, False


# ---------------------------------------------------------------------------
# Prefetch-enabled worker loop: processes a chunk of bands with tile
# prefetching so that band N+1's tiles load while band N is on GPU.
# ---------------------------------------------------------------------------

# Module-level prefetch executor (one per Dask worker process, lazy-init)
_prefetch_executor = None


def _init_prefetch_executor():
    global _prefetch_executor
    if _prefetch_executor is None:
        _prefetch_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="prefetch"
        )
    return _prefetch_executor


def _stitch_bands_loop_worker(
    band_list, output_store_path, input_store_path,
    tile_size, tx, flipud, fliplr, rot90,
    blending_method, blending_exponent,
    value_precision_bits, cuda_path="",
):
    """Process multiple bands in sequence with tile prefetching.

    Each Dask worker is assigned a chunk of bands. While the GPU processes
    band N, a background thread prefetches tiles for band N+1, hiding load
    latency behind GPU computation (which releases the GIL).
    """
    from datetime import datetime

    import numcodecs
    numcodecs.blosc.set_nthreads(4)

    if cuda_path and 'CUDA_PATH' not in os.environ:
        os.environ['CUDA_PATH'] = cuda_path

    profile = os.environ.get("STITCH_PROFILE", "").strip() in ("1", "true", "yes")

    fov_store_p = Path(input_store_path)
    dtype_val = _resolve_value_dtype(value_precision_bits)
    use_edt = str(blending_method).lower() == "edt"
    tile_weights = (
        _compute_edt_weights(tile_size, blending_exponent, dtype_val)
        if use_edt else None
    )

    prefetch_pool = _init_prefetch_executor()
    prefetch_future = None
    results = []

    for i, (well_id, band_idx, y0, y1, y_tiles, final_shape, is_last) in enumerate(band_list):
        gpu_timings = None
        try:
            t_start = time.time()
            ts = datetime.now().strftime("%H:%M:%S.%f")[:12]
            print(
                f"[{ts}] [Band Worker] Starting well={well_id} band={band_idx+1} "
                f"y=[{y0}:{y1}] tiles={len(y_tiles)} (PID={os.getpid()})"
            )

            arr_out = zarr.open(
                os.path.join(output_store_path, "A", well_id, "0", "0"),
                mode="r+",
            )

            # ---- Load tiles: from prefetch or blocking load ----
            t_load = time.time()
            if prefetch_future is not None:
                tile_cache = prefetch_future.result()
                t_load_elapsed = time.time() - t_load
                load_src = "prefetch"
            else:
                tile_cache = _load_band_tiles(
                    y_tiles, fov_store_p, flipud, fliplr, rot90
                )
                t_load_elapsed = time.time() - t_load
                load_src = "cold"

            # ---- Start prefetching next band BEFORE GPU work ----
            # GPU releases GIL → prefetch threads get uncontested GIL access
            prefetch_future = None
            if i + 1 < len(band_list):
                next_y_tiles = band_list[i + 1][4]  # y_tiles field
                prefetch_future = prefetch_pool.submit(
                    _load_band_tiles,
                    next_y_tiles, fov_store_p, flipud, fliplr, rot90,
                )

            # ---- GPU process this Y-band ----
            t_gpu = time.time()
            total_x = final_shape[-1]
            gpu_result = _process_y_band_gpu(
                y0, y1, total_x, y_tiles, tile_cache, final_shape,
                dtype_val, use_edt, tile_weights, return_gpu=True,
                profile=profile,
            )
            if profile:
                norm_gpu, gpu_timings = gpu_result
            else:
                norm_gpu = gpu_result
            t_gpu_elapsed = time.time() - t_gpu

            # ---- D2H transfer ----
            t_d2h = time.time()
            norm_cpu = norm_gpu.get()
            del norm_gpu, tile_cache
            xp.get_default_memory_pool().free_all_blocks()
            t_d2h_elapsed = time.time() - t_d2h

            # ---- Wait for previous band's writes ----
            t_wait = time.time()
            _wait_band_writes()
            t_wait_elapsed = time.time() - t_wait

            # One-time blosc benchmark
            _run_blosc_benchmark(
                np.ascontiguousarray(norm_cpu[0, 0, 0, :, :tx])
            )

            # ---- Submit writes to persistent pool (non-blocking) ----
            global _band_write_submit_time
            _band_write_submit_time = time.time()
            pool = _init_band_write_pool()
            n_blocks = 0
            for x0 in range(0, total_x, tx):
                x1 = min(total_x, x0 + tx)
                _band_pending_futures.append(pool.submit(
                    _write_zarr_block, arr_out, final_shape, y0, y1, x0, x1,
                    norm_cpu[:, :, :, :, x0:x1],
                ))
                n_blocks += 1
            _band_pending_data.append(norm_cpu)
            print(
                f"[Band Worker] Submitted {n_blocks} write blocks, "
                f"{norm_cpu.nbytes/1e9:.2f}GB ({norm_cpu.dtype})"
            )

            # Flush on last band of a well
            if is_last:
                t_flush = time.time()
                _wait_band_writes()
                t_flush_elapsed = time.time() - t_flush
                print(
                    f"[Band Worker] Final flush well={well_id}: "
                    f"{t_flush_elapsed:.1f}s"
                )

            t_total = time.time() - t_start
            ts = datetime.now().strftime("%H:%M:%S.%f")[:12]
            print(
                f"[{ts}] [Band Worker] Done well={well_id} band={band_idx+1} "
                f"load={t_load_elapsed:.1f}s({load_src}) "
                f"gpu={t_gpu_elapsed:.1f}s d2h={t_d2h_elapsed:.1f}s "
                f"wait_prev={t_wait_elapsed:.1f}s total={t_total:.1f}s"
            )
            if profile and gpu_timings:
                gt = gpu_timings
                print(
                    f"  [GPU detail] alloc={gt.get('alloc',0):.2f}s "
                    f"accum={gt.get('accum',0):.2f}s "
                    f"({gt.get('n_tiles_hit',0)} tiles: "
                    f"slice_cpu={gt.get('slice_cpu',0):.2f}s "
                    f"h2d={gt.get('h2d',0):.2f}s "
                    f"gpu_kernel={gt.get('gpu_kernel',0):.2f}s) "
                    f"norm={gt.get('normalize',0):.2f}s "
                    f"gpu_mem: alloc={gt.get('alloc_gpu_mb',0):.0f}MB "
                    f"post_norm={gt.get('post_norm_gpu_mb',0):.0f}MB"
                )
            results.append((well_id, band_idx, True))

        except Exception as e:
            import traceback
            print(f"[Band Worker] Failed well={well_id} band={band_idx+1}: {e}")
            traceback.print_exc()
            results.append((well_id, band_idx, False))

    # Final flush: ensure all pending writes complete before returning
    _wait_band_writes()

    return results


def stitch(
    config_path: str,
    input_store_path: str,
    output_store_path: str,
    flipud: bool = False,
    fliplr: bool = False,
    rot90: int = 0,
    blending_method: Literal["average", "edt"] = "edt",
    parallel_mode: Literal["auto", "wells", "wells_threads", "y_bands", "sequential"] = "auto",
    **kwargs,
):
    """Mimic of biahub stitch function

    Args:
        config_path: Path to the YAML config file with shift estimates
        input_store_path: Path to the input OME-Zarr store
        output_store_path: Path for the output stitched OME-Zarr store
        flipud: Flip tiles vertically
        fliplr: Flip tiles horizontally
        rot90: Number of 90-degree rotations to apply
        blending_method: 'average' or 'edt' for blending overlapping tiles
        parallel_mode: Parallelization strategy:
            - "auto" (default): GPU + Dask uses multiprocess wells, GPU without Dask uses threaded wells, CPU uses parallel Y-bands
            - "wells": Force parallel well processing (Dask multiprocessing if available, else threads)
            - "wells_threads": Force parallel well processing via ThreadPoolExecutor (legacy)
            - "y_bands": Force parallel Y-band processing (joblib)
            - "sequential": No parallelization
        **kwargs: Additional arguments passed to assemble_streaming()
    """

    # get the shifts and split into a list of lists per well
    all_shifts = read_shifts_biahub(config_path)

    def get_group(key):
        return key.split("/")[1]

    grouped_shifts = defaultdict(dict)
    for key, value in all_shifts.items():
        group = get_group(key)
        grouped_shifts[group][key] = value

    # Read metadata from a single position via direct JSON access.
    # open_ome_zarr() parses the full HCS hierarchy (all 7035 positions) which
    # takes minutes; reading one position's JSON files is instant.
    input_store_p = Path(input_store_path)
    first_pos_key = next(iter(all_shifts.keys()))  # e.g. "A/1/002026"
    with open(input_store_p / first_pos_key / ".zattrs") as f:
        pos_attrs = json.load(f)
    channel_names = [c["label"] for c in pos_attrs["omero"]["channels"]]
    scale_transforms = pos_attrs.get("multiscales", [{}])[0].get("datasets", [{}])[0].get("coordinateTransformations", [])
    scale = tuple(scale_transforms[0]["scale"]) if scale_transforms and scale_transforms[0].get("type") == "scale" else None
    with open(input_store_p / first_pos_key / "0" / ".zarray") as f:
        arr_meta = json.load(f)
    tile_shape = tuple(arr_meta["shape"])
    chunks_size = tuple(arr_meta["chunks"])
    # Optional override for output chunking to improve viewer (dask/napari) responsiveness
    # Accept either full 5D "target_chunks" or YX-only via "target_chunks_yx"
    target_chunks = kwargs.get("target_chunks")
    target_chunks_yx = kwargs.get("target_chunks_yx")
    if target_chunks is not None:
        try:
            chunks_size = tuple(int(x) for x in target_chunks)
        except Exception:
            pass
    elif target_chunks_yx is not None:
        try:
            ty, tx = int(target_chunks_yx[0]), int(target_chunks_yx[1])
            # Use singleton chunks for T,C,Z for snappier per-channel/plane reads
            chunks_size = (1, 1, 1, ty, tx)
        except Exception:
            pass

    # initialize output zarr store
    output_store = open_ome_zarr(
        output_store_path, layout="hcs", mode="w-", channel_names=channel_names
    )
    print("output store created")

    # Determine parallelization strategy based on parallel_mode and hardware
    use_dask_wells = False
    use_thread_wells = False
    use_parallel_y_bands = False
    n_workers = None

    if parallel_mode == "auto":
        if _USING_CUPY and _DASK_DISTRIBUTED_AVAILABLE:
            use_dask_wells = True
            print("[stitch] Auto mode: GPU + Dask detected, using Dask multiprocess wells strategy")
        elif _USING_CUPY:
            use_thread_wells = True
            print("[stitch] Auto mode: GPU detected (no Dask), using threaded wells strategy")
        else:
            use_parallel_y_bands = True
            n_workers = _get_optimal_workers(use_gpu=False, verbose=True)
            print(f"[stitch] Auto mode: CPU detected, using parallel Y-bands strategy ({n_workers} workers)")
    elif parallel_mode == "wells":
        if _DASK_DISTRIBUTED_AVAILABLE:
            use_dask_wells = True
            print("[stitch] Forced Dask multiprocess wells strategy")
        else:
            use_thread_wells = True
            print("[stitch] Forced threaded wells strategy (Dask unavailable)")
    elif parallel_mode == "wells_threads":
        use_thread_wells = True
        print("[stitch] Forced threaded wells strategy (legacy)")
    elif parallel_mode == "y_bands":
        use_parallel_y_bands = True
        n_workers = _get_optimal_workers(use_gpu=False, verbose=True)
        print(f"[stitch] Forced parallel Y-bands strategy ({n_workers} workers)")
    else:  # "sequential"
        print("[stitch] Sequential processing (no parallelization)")

    well_ids = list(grouped_shifts.keys())
    num_wells = len(well_ids)

    if use_dask_wells:
        # DASK BAND-LEVEL PARALLEL PROCESSING
        # Break work into (well, Y-band) pairs for fine-grained parallelism.
        # Workers naturally desynchronize — some load tiles while others use GPU.
        t_phase1 = time.time()
        print(f"[Dask Bands] Processing {num_wells} wells with band-level Dask parallelism")
        print(f"[Dask Bands] Wells: {well_ids}")

        # Resolve divide_tile_size for Y-band computation and write chunking
        divide_tile_yx = kwargs.get("target_chunks_yx", (2048, 2048))
        ty_band, tx_write = int(divide_tile_yx[0]), int(divide_tile_yx[1])
        blending_exponent = kwargs.get("blending_exponent", 1.0)
        value_precision_bits = kwargs.get("value_precision_bits", 32)
        dtype_val = _resolve_value_dtype(value_precision_bits)

        # Phase 1: Pre-create wells, compute Y-bands, build interleaved work queue
        work_queue = []
        fov_store_p = Path(input_store_path)

        for well_id in well_ids:
            well_shifts = grouped_shifts[well_id]
            final_shape_xy = get_output_shape(well_shifts, tile_shape[-2:])

            # Determine T,C,Z from tile metadata
            first_tile_key = next(iter(well_shifts.keys()))
            with open(fov_store_p / first_tile_key / "0" / ".zarray") as f:
                meta = json.load(f)
            first_tile_shape = tuple(meta["shape"])
            final_shape = first_tile_shape[:3] + final_shape_xy

            # Create output position + zeros via iohub
            stitched_pos = output_store.create_position("A", well_id, "0")
            stitched_pos.create_zeros(
                "0",
                shape=final_shape,
                chunks=chunks_size,
                dtype=dtype_val,
                transform=(
                    [TransformationMeta(type="scale", scale=scale)]
                    if scale is not None
                    else None
                ),
            )

            # Override compressor: lz4 is 3x faster than zstd on dense
            # float16 image data (5ms vs 16ms per 8MB chunk) with the same
            # compression ratio (~2.1x). iohub hardcodes zstd; patch .zarray.
            zarray_path = (
                Path(output_store_path) / "A" / well_id / "0" / "0" / ".zarray"
            )
            if zarray_path.exists():
                with open(zarray_path) as f:
                    zmeta = json.load(f)
                old_cname = zmeta.get("compressor", {}).get("cname", "?")
                zmeta["compressor"] = {
                    "id": "blosc",
                    "cname": "lz4",
                    "clevel": 1,
                    "shuffle": 2,  # BITSHUFFLE
                    "blocksize": 0,
                }
                with open(zarray_path, "w") as f:
                    json.dump(zmeta, f)
                if well_id == well_ids[0]:
                    print(f"[Dask Bands] Compressor: {old_cname} -> lz4 (3x faster on dense float16)")

            # Pre-compute tile metadata (same logic as assemble_streaming)
            dtype_idx = _resolve_shift_dtype(well_shifts, tile_shape[-2:], base_bits=32)
            tile_shapes = {}
            for tname in well_shifts.keys():
                with open(fov_store_p / tname / "0" / ".zarray") as f:
                    tmeta = json.load(f)
                tile_shapes[tname] = tuple(tmeta["shape"])

            tile_meta = []
            for tile_name, shift in well_shifts.items():
                ts = tile_shapes[tile_name]
                t_end = min(ts[0], final_shape[0])
                c_end = min(ts[1], final_shape[1])
                z_end = min(ts[2], final_shape[2])
                ys = int(shift[0])
                ye = int(shift[0] + tile_shape[-2])
                xs = int(shift[1])
                xe = int(shift[1] + tile_shape[-1])
                tile_meta.append((tile_name, t_end, c_end, z_end, ys, ye, xs, xe))

            # Pre-identify Y-bands and their contributing tiles
            total_y = final_shape[-2]
            n_bands = len(range(0, total_y, ty_band))
            for band_idx, y0 in enumerate(range(0, total_y, ty_band)):
                y1 = min(total_y, y0 + ty_band)
                y_tiles = [
                    (nm, t_end, c_end, z_end, ys, ye, xs, xe)
                    for (nm, t_end, c_end, z_end, ys, ye, xs, xe) in tile_meta
                    if not (ye <= y0 or ys >= y1)
                ]
                is_last = (band_idx == n_bands - 1)
                work_queue.append((well_id, band_idx, y0, y1, y_tiles, final_shape, is_last))

            print(f"[Dask Bands] Well {well_id}: shape={final_shape}, {n_bands} bands")

        # Interleave work queue: w1b1, w2b1, w3b1, w1b2, w2b2, w3b2, ...
        # Keeps consecutive tasks on different wells → less filesystem cache thrashing
        work_queue.sort(key=lambda x: (x[1], well_ids.index(x[0])))
        t_phase1_elapsed = time.time() - t_phase1
        print(f"[Dask Bands] Total work units: {len(work_queue)} (interleaved)")
        print(f"[Dask Bands] Phase 1 (store creation + metadata): {t_phase1_elapsed:.1f}s")

        # Capture CUDA_PATH for worker processes (needed for CuPy JIT compilation)
        cuda_path = os.environ.get('CUDA_PATH', os.environ.get('CUDA_HOME', ''))

        # Close the output store before spawning workers
        output_store.close()
        del output_store

        # Phase 2: Launch Dask LocalCluster with band-level workers
        t_phase2 = time.time()

        # Detect available GPUs and scale workers across all devices.
        # Peak ~15GB GPU per worker (numer+denom+tile temps in float16).
        # Use 18GB divisor per device: A100-80GB→4/gpu, H200-141GB→7/gpu, A40-48GB→2/gpu.
        from ops_utils.hpc.gpu_utils import _setup_gpu_environment
        from ops_utils.hpc.parallel_utils import GPUWorkerPlugin
        available_gpus = _setup_gpu_environment()
        n_gpus = len(available_gpus)

        if n_gpus > 0 and _USING_CUPY:
            # Query per-device VRAM from the first visible GPU
            per_gpu_mb = xp.cuda.Device(0).mem_info[1]
            per_gpu_gb = per_gpu_mb / 1e9
            workers_per_gpu = max(1, int(per_gpu_gb // 18))
            n_dask_workers = min(workers_per_gpu * n_gpus, len(work_queue))
            print(f"[Dask Bands] {n_gpus} GPU(s), {per_gpu_gb:.0f}GB each → "
                  f"{workers_per_gpu}/gpu × {n_gpus} = {n_dask_workers} workers")
        else:
            n_dask_workers = min(4, len(work_queue))
            print(f"[Dask Bands] No GPU detected, using {n_dask_workers} CPU workers")

        # threads_per_worker=1: each worker runs one long-lived task (loop
        # over its assigned bands). Write pipelining + prefetch use module-
        # level state that is NOT thread-safe.
        if cuda_path:
            os.environ['CUDA_PATH'] = cuda_path  # workers inherit parent env

        # Clear parent CUDA_VISIBLE_DEVICES so workers don't all bind to the
        # same device. GPUWorkerPlugin will set it per-worker via round-robin.
        parent_cuda_devices = os.environ.pop("CUDA_VISIBLE_DEVICES", None)

        # Disable Dask's per-worker memory limit — workers hold ~28-42GB each
        # (current tile_cache + prefetched tile_cache + norm_cpu + prev norm_cpu).
        cluster = LocalCluster(n_workers=n_dask_workers, threads_per_worker=1,
                               memory_limit=0)
        client = Client(cluster)

        # Assign GPUs round-robin: worker 0→GPU0, worker 1→GPU1, ...
        if n_gpus > 0:
            client.register_worker_plugin(GPUWorkerPlugin(available_gpus))

        cpu_monitor_stop = None
        try:
            t_phase2_elapsed = time.time() - t_phase2
            print(f"[Dask Bands] LocalCluster started: {n_dask_workers} workers, 1 thread each ({t_phase2_elapsed:.1f}s)")

            # Start CPU/IO monitor (reads /proc for per-worker stats every 5s)
            n_cores = int(os.environ.get("SLURM_CPUS_PER_TASK", 32))
            cpu_monitor_stop = _start_cpu_monitor(interval=5.0, n_cores=n_cores)

            # Distribute work round-robin so each worker gets interleaved bands
            worker_chunks = [[] for _ in range(n_dask_workers)]
            for i, item in enumerate(work_queue):
                worker_chunks[i % n_dask_workers].append(item)

            chunk_sizes = [len(c) for c in worker_chunks]
            print(f"[Dask Bands] Chunks per worker: {chunk_sizes} (prefetch enabled)")

            futures = []
            for chunk in worker_chunks:
                if not chunk:
                    continue
                future = client.submit(
                    _stitch_bands_loop_worker,
                    band_list=chunk,
                    output_store_path=output_store_path,
                    input_store_path=input_store_path,
                    tile_size=tile_shape[-2:],
                    tx=tx_write,
                    flipud=flipud,
                    fliplr=fliplr,
                    rot90=rot90,
                    blending_method=blending_method,
                    blending_exponent=blending_exponent,
                    value_precision_bits=value_precision_bits,
                    cuda_path=cuda_path,
                )
                futures.append(future)

            # Gather results from all worker loops
            completed = []
            failed = []
            for future in futures:
                try:
                    results_list = future.result()
                    for well_id, band_idx, success in results_list:
                        if success:
                            completed.append((well_id, band_idx))
                        else:
                            failed.append((well_id, band_idx))
                except Exception as e:
                    print(f"[Dask Bands] Worker exception: {e}")
                    import traceback
                    traceback.print_exc()

        finally:
            if cpu_monitor_stop is not None:
                cpu_monitor_stop.set()
            try:
                client.close()
                cluster.close(timeout=120)
            except Exception:
                pass
            # Restore parent CUDA_VISIBLE_DEVICES
            if parent_cuda_devices is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = parent_cuda_devices

        print(f"[Dask Bands] Completed: {len(completed)}/{len(work_queue)} bands")
        if failed:
            print(f"[Dask Bands] Failed bands: {failed}")
            raise RuntimeError(f"Failed to process {len(failed)} bands: {failed}")

        print(f"[Dask Bands] All {len(work_queue)} bands processed successfully!")

    elif use_thread_wells:
        # THREADED WELL PROCESSING (legacy fallback) — GIL limits true parallelism
        well_lock = threading.Lock()
        max_workers = min(4, num_wells)

        print(f"[Thread Wells] Processing {num_wells} wells with {max_workers} thread workers")
        print(f"[Thread Wells] Wells: {well_ids}")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for well_id in well_ids:
                shifts = grouped_shifts[well_id]
                future = executor.submit(
                    _process_single_well,
                    well_id, shifts, output_store, input_store_path, tile_shape,
                    flipud, fliplr, rot90, kwargs, blending_method, chunks_size, scale,
                    well_lock, parallel_y_bands=False, n_workers=None
                )
                futures.append((well_id, future))

            completed_wells = []
            failed_wells = []
            for well_id, future in futures:
                try:
                    result_well_id, success = future.result()
                    if success:
                        completed_wells.append(result_well_id)
                    else:
                        failed_wells.append(result_well_id)
                except Exception as e:
                    print(f"[Thread Wells] Exception in well {well_id}: {e}")
                    failed_wells.append(well_id)

        print(f"[Thread Wells] Completed: {len(completed_wells)} wells: {completed_wells}")
        if failed_wells:
            print(f"[Thread Wells] Failed: {len(failed_wells)} wells: {failed_wells}")
            raise RuntimeError(f"Failed to process {len(failed_wells)} wells: {failed_wells}")

        print(f"[Thread Wells] All {num_wells} wells processed successfully!")

    else:
        # SEQUENTIAL WELL PROCESSING with optional parallel Y-bands
        well_lock = threading.Lock()
        print(f"[Sequential Wells] Processing {num_wells} wells: {well_ids}")

        completed_wells = []
        failed_wells = []

        for well_id in well_ids:
            shifts = grouped_shifts[well_id]
            result_well_id, success = _process_single_well(
                well_id, shifts, output_store, input_store_path, tile_shape,
                flipud, fliplr, rot90, kwargs, blending_method, chunks_size, scale,
                well_lock, parallel_y_bands=use_parallel_y_bands, n_workers=n_workers
            )
            if success:
                completed_wells.append(result_well_id)
            else:
                failed_wells.append(result_well_id)

        # Report final results
        print(f"[Sequential Wells] Completed: {len(completed_wells)} wells: {completed_wells}")
        if failed_wells:
            print(f"[Sequential Wells] Failed: {len(failed_wells)} wells: {failed_wells}")
            raise RuntimeError(f"Failed to process {len(failed_wells)} wells: {failed_wells}")

        print(f"[Sequential Wells] All {num_wells} wells processed successfully!")

    return


def estimate_stitch(
    input_store_path: str,
    output_config_path: Path,
    flipud: bool,
    fliplr: bool,
    rot90: int,
    tile_size: tuple = (2048, 2048),
    overlap: int = 150,
    x_guess: Optional[dict] = None,
    limit_positions: Optional[int] = None,
    channel: int = 0,
    timepoint: int = 0,
    timepoint_per_well: Optional[dict] = None,
    use_clahe: bool = False,
    clahe_clip_limit: float = 0.02,
    verbose: bool = False,
):
    """Mimic of Biahub estimate stitch function

    Args:
        channel: Channel index to use for registration (default: 0)
        timepoint: Timepoint index to use for registration (default: 0)
        timepoint_per_well: Optional dict mapping well names (e.g., "A/2") to timepoint indices.
                           Overrides the default timepoint for specific wells.
        use_clahe: Apply CLAHE preprocessing for better registration (default: False)
        clahe_clip_limit: CLAHE contrast limit (default: 0.02)
        verbose: Print confidence scores as edges are computed (default: False)
    """

    if verbose:
        print(f"[assemble.estimate_stitch] Configuration:")
        print(f"  input_store_path  = {input_store_path}")
        print(f"  output_config_path = {output_config_path}")
        print(f"  flipud={flipud}, fliplr={fliplr}, rot90={rot90}")
        print(f"  tile_size={tile_size}, overlap={overlap}")
        print(f"  channel={channel}, timepoint={timepoint}")
        print(f"  timepoint_per_well={timepoint_per_well}")
        print(f"  use_clahe={use_clahe}, clahe_clip_limit={clahe_clip_limit}")
        print(f"  limit_positions={limit_positions}")
        print(f"  x_guess={x_guess}")

    store = open_ome_zarr(input_store_path)
    if limit_positions is not None and int(limit_positions) > 0:
        print(f"[assemble.estimate_stitch] DEBUG MODE: Limiting to {limit_positions} positions")
    else:
        print(f"[assemble.estimate_stitch] Processing ALL positions (full stitching)")
    # Discover positions with an optional centered selection to avoid scanning entire store in debug
    if limit_positions is not None and int(limit_positions) > 0:
        # Build a true centered m×m block for EACH well from the filesystem (no fallback)
        root = Path(input_store_path)
        rows = sorted([d.name for d in root.iterdir() if d.is_dir()])
        if not rows:
            raise RuntimeError("No row directories found for centered grid selection")
        k = int(limit_positions)
        side = int(max(1, int(k**0.5)))
        target = min(k, side * side)

        def _rc_from_name(name: str):
            digits = "".join(ch for ch in name if ch.isdigit())
            if len(digits) >= 6:
                return int(digits[:3]), int(digits[3:6])
            half = len(digits) // 2
            return int(digits[:half] or 0), int(digits[half:] or 0)

        position_list = []
        for row in rows:
            cols = sorted([d.name for d in (root / row).iterdir() if d.is_dir()])
            for col in cols:
                well_dir = root / row / col
                tiles = sorted([d.name for d in well_dir.iterdir() if d.is_dir()])
                if not tiles:
                    continue
                rc = [_rc_from_name(t) for t in tiles]
                rs = sorted(set(r for r, _ in rc))
                cs = sorted(set(c for _, c in rc))
                if not rs or not cs:
                    continue
                rmid = rs[len(rs) // 2]
                cmid = cs[len(cs) // 2]
                selected = []
                radius = 0
                while len(selected) < target and radius <= max(len(rs), len(cs)):
                    rmin, rmax = rmid - radius, rmid + radius
                    cmin, cmax = cmid - radius, cmid + radius
                    for name, (rr, cc) in zip(tiles, rc):
                        if (
                            rmin <= rr <= rmax
                            and cmin <= cc <= cmax
                            and name not in selected
                        ):
                            selected.append(name)
                            if len(selected) >= target:
                                break
                    radius += 1
                tiles_sel = selected[:target]
                # Append positions for this well
                position_list.extend([f"{row}/{col}/{t}" for t in tiles_sel])
        print(
            f"[assemble.estimate_stitch] Using centered grid {side}x{side} per well; total tiles={len(position_list)}"
        )
    else:
        # Full list (may be large) - try fast discovery first
        position_list = _discover_positions_fast(Path(input_store_path))

        if position_list is not None:
            print(f"[assemble.estimate_stitch] Fast discovery found {len(position_list)} positions")
        else:
            # Not HCS layout or fast discovery failed - use slower iterator
            print(f"[assemble.estimate_stitch] Not HCS layout, falling back to iterator-based discovery")
            position_list = [
                p for p, _ in tqdm(store.positions(), desc="Getting positions")
            ]

    grouped_positions = defaultdict(list)
    for a in position_list:
        group = a[:3]
        grouped_positions[group].append(a)

    def _process_well(well_id):
        """Worker function to process a single well."""
        well_positions = grouped_positions[well_id]
        tile_lut = {t[4:]: i for i, t in enumerate(well_positions)}

        # Determine timepoint for this well
        well_timepoint = timepoint
        if timepoint_per_well is not None and well_id in timepoint_per_well:
            well_timepoint = timepoint_per_well[well_id]
            print(f"[assemble.estimate_stitch] Using timepoint {well_timepoint} for well {well_id}")

        edge_list, confidence_dict = pairwise_shifts(
            well_positions,
            input_store_path,
            well=well_id,
            flipud=flipud,
            fliplr=fliplr,
            rot90=rot90,
            overlap=overlap,
            channel=channel,
            timepoint=well_timepoint,
            use_clahe=use_clahe,
            clahe_clip_limit=clahe_clip_limit,
            verbose=verbose,
        )

        opt_shift_dict = optimal_positions(edge_list, tile_lut, well_id, tile_size, x_guess)

        return well_id, opt_shift_dict, confidence_dict

    running_opt_shift_dict = {}
    running_confidence_dict = {}

    # Parallel processing of wells using ThreadPoolExecutor
    num_wells = len(grouped_positions)
    max_workers = min(4, num_wells)  # Use up to 4 parallel wells
    print(f"[estimate_stitch] Processing {num_wells} wells with {max_workers} parallel workers")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all well computations
        futures = {
            executor.submit(_process_well, well_id): well_id
            for well_id in grouped_positions.keys()
        }

        # Collect results
        for future in tqdm(as_completed(futures), total=num_wells, desc="Processing wells"):
            well_id, opt_shift_dict, confidence_dict = future.result()
            running_opt_shift_dict = running_opt_shift_dict | opt_shift_dict
            running_confidence_dict[well_id] = confidence_dict

    to_write = {
        "total_translation": running_opt_shift_dict,
        "confidence": running_confidence_dict,
    }
    # Ensure parent directory exists before writing
    try:
        Path(output_config_path).parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    with open(output_config_path, "w") as f:
        yaml.dump(to_write, f)

    return opt_shift_dict


def array_apply(
    *in_arrays: ArrayLike,
    func: Callable,
    out_array: ArrayLike,
    axis: Union[Tuple[int], int] = 0,
    **kwargs,
) -> ArrayLike:
    """Apply a function over a given dimension of an array.

    adapted from Ultrack.apply_array
    """
    name = func.__name__ if hasattr(func, "__name__") else type(func).__name__

    try:
        in_shape = [arr.shape for arr in in_arrays]
        xp.broadcast_shapes(out_array.shape, *in_shape)
    except ValueError as e:
        print(
            f"Warning: if you are not using multichannel operations, "
            f"this can be an error. {e}."
        )

    if isinstance(axis, int):
        axis = (axis,)

    stub_slicing = [slice(None) for _ in range(out_array.ndim)]
    multi_indices = list(itertools.product(*[range(out_array.shape[i]) for i in axis]))
    for indices in tqdm(multi_indices, f"Applying {name} ..."):
        for a, i in zip(axis, indices):
            stub_slicing[a] = i
        indexing = tuple(stub_slicing)

        func_result = func(*[a[indexing] for a in in_arrays], **kwargs)
        output_shape = out_array[indexing].shape
        out_array[indexing] = xp.broadcast_to(func_result, output_shape)

    return out_array


def divide_tile(
    *in_arrays: ArrayLike,
    func: Callable,
    out_array: ArrayLike,
    tile: tuple,
    overlap: tuple = (0, 0),
):

    final_shape = out_array.shape[-2:]

    tiling_start = list(
        product(
            *[
                range(o, size + 2 * o, t + o)  # t + o step, because of blending map
                for size, t, o in zip(final_shape, tile, overlap)
            ]
        )
    )
    for start_indices in tqdm(tiling_start, "Applying function to tiles"):
        slicing = (...,) + tuple(
            slice(start - o, start + t + o)
            for start, t, o in zip(start_indices, tile, overlap)
        )
        out_array[slicing] = func(*[a[slicing] for a in in_arrays])

    return out_array