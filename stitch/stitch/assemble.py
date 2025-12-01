from iohub.ngff import open_ome_zarr
import os
import math
import zarr
import dask.array as da
from tqdm import tqdm
from stitch.stitch.tile import augment_tile, pairwise_shifts, optimal_positions
from stitch.connect import read_shifts_biahub
from collections import defaultdict
from iohub.ngff import TransformationMeta
from typing import Any, Callable, Dict, Literal, Optional, Tuple, Type, Union
import itertools
from numpy.typing import ArrayLike
from itertools import product
import yaml
from pathlib import Path
import numpy as np

# Try to use CuPy for GPU acceleration, fall back to NumPy
try:
    import cupy as xp
    from cupyx.scipy import ndimage as cundi
    _USING_CUPY = True
    print("[assemble.py] Using CuPy (GPU) for array operations")
except (ModuleNotFoundError, ImportError):
    import numpy as xp
    from scipy import ndimage as cundi
    _USING_CUPY = False
    print("[assemble.py] Using NumPy (CPU) for array operations")

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

# Cache for EDT-based blending weight maps to avoid recomputation per tile size/exponent
_EDT_WEIGHT_CACHE: Dict[Tuple[int, int, int], xp.ndarray] = {}


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


def _process_x_block_gpu(x0, x1, y0, y1, y_tiles, tile_cache, final_shape, dtype_val, use_edt, tile_weights, stream=None):
    """
    Process a single X block on GPU, optionally using a specific CUDA stream.

    Returns (numer, denom, norm_cpu) tuple ready for writing.
    """
    # Create buffers on the specified stream (or default)
    with xp.cuda.Stream(stream) if stream is not None else xp.cuda.Stream.null:
        numer = xp.zeros(
            (final_shape[0], final_shape[1], final_shape[2], y1 - y0, x1 - x0),
            dtype=dtype_val,
        )
        denom = xp.zeros_like(numer, dtype=dtype_val)

        for tile_name, t_end, c_end, z_end, ys, ye, xs, xe in y_tiles:
            # X intersection within this block
            ix0 = max(x0, xs)
            ix1 = min(x1, xe)
            if ix0 >= ix1:
                continue
            iy0 = max(y0, ys)
            iy1 = min(y1, ye)

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
            block = tile_full[tile_sl].astype(dtype_val, copy=False)
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

        # Normalize
        norm = xp.nan_to_num(numer / xp.maximum(denom, 1e-12))

        # Transfer to CPU for writing (synchronous on this stream)
        norm_cpu = _to_numpy(norm)

        # Clean up
        del numer, denom, norm

        return norm_cpu


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
                import numpy as _np

                _mask = _np.zeros((ty, tx), dtype=bool)
                _mask[1:-1, 1:-1] = True
                _dist = cundi.distance_transform_edt(_mask).astype(_np.float32)
                _dist += 1e-6
                _weights = _np.power(_dist, float(blending_exponent), where=(_dist > 0))
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
    stitched_pos,
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
):
    """Streamed assembly that avoids saving auxiliary arrays.

    For each output YX block (divide_tile_size), accumulates numerator and denominator in RAM,
    normalizes, and writes directly to the on-disk output image '0'. No auxiliary arrays are saved.

    Args:
        use_adaptive_blocks: If True, automatically determine optimal block size based on GPU memory
        parallel_x_blocks: If True, process X blocks in parallel using CUDA streams (GPU only)
    """
    print(f"[assemble.streaming] Using blending_method: {blending_method}")
    print(f"[assemble.streaming] Adaptive blocks: {use_adaptive_blocks}, Parallel X-blocks: {parallel_x_blocks}")

    fov_store = open_ome_zarr(fov_store_path)

    # Determine target T,C,Z
    tcz_list = [fov_store[tname].data.shape[:3] for tname in shifts.keys()]
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

    # Adaptive block sizing based on GPU memory
    if use_adaptive_blocks:
        divide_tile_size = _get_optimal_block_size(final_shape, tile_size, divide_tile_size)

    if chunks_size is None:
        ty, tx = divide_tile_size if divide_tile_size is not None else (1024, 1024)
        chunks_size = (1, 1, 1, int(ty), int(tx))

    # Create output array on disk only (no auxiliary arrays)
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
    if use_edt:
        ty, tx = int(tile_size[0]), int(tile_size[1])
        import numpy as _np

        _mask = _np.zeros((ty, tx), dtype=bool)
        if ty > 2 and tx > 2:
            _mask[1:-1, 1:-1] = True
        _dist = cundi.distance_transform_edt(_mask).astype(_np.float32)
        _dist += 1e-6
        _weights = _np.power(_dist, float(blending_exponent), where=(_dist > 0))
        tile_weights = xp.asarray(_weights, dtype=dtype_val)
    else:
        tile_weights = None

    arr_out = stitched_pos["0"]
    dtype_idx = _resolve_shift_dtype(shifts, tile_size, base_bits=32)

    # Precompute tile metadata for fast intersection checks
    tile_meta = []
    for tile_name, shift in shifts.items():
        tile_ref = fov_store[tile_name].data
        shift_array = xp.asarray(shift, dtype=dtype_idx)
        t_end = min(tile_ref.shape[0], final_shape[0])
        c_end = min(tile_ref.shape[1], final_shape[1])
        z_end = min(tile_ref.shape[2], final_shape[2])
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

    for band_idx, (y0, y1, y_tiles) in enumerate(tqdm(y_bands, desc="Stitching Y")):
        # Load all tiles for this Y-band to GPU in one batch
        tile_cache = {}
        for tile_name, t_end, c_end, z_end, ys, ye, xs, xe in y_tiles:
            tile_full = fov_store[tile_name].data
            # Transfer to GPU immediately after loading
            tile_cache[tile_name] = (
                augment_tile(xp.asarray(tile_full), flipud, fliplr, rot90),
                t_end,
                c_end,
                z_end,
                ys,
                ye,
                xs,
                xe,
            )

        # Optionally synchronize GPU before processing if using CuPy
        if _USING_CUPY:
            xp.cuda.Stream.null.synchronize()

        # Process all X blocks for this Y band
        x_blocks = []
        for x0 in range(0, total_x, tx):
            x1 = min(total_x, x0 + tx)
            x_blocks.append((x0, x1))

        # Parallel X-block processing using CUDA streams (GPU only)
        if parallel_x_blocks and _USING_CUPY and len(x_blocks) > 1:
            print(f"[Parallel] Processing {len(x_blocks)} X-blocks in parallel for Y-band {band_idx+1}/{len(y_bands)}")

            # Create streams for parallel processing (limit to 4 concurrent streams)
            num_streams = min(4, len(x_blocks))
            streams = [xp.cuda.Stream(non_blocking=True) for _ in range(num_streams)]

            # Process blocks in batches
            for batch_start in range(0, len(x_blocks), num_streams):
                batch_end = min(batch_start + num_streams, len(x_blocks))
                batch_blocks = x_blocks[batch_start:batch_end]

                # Launch processing on streams
                results = []
                for idx, (x0, x1) in enumerate(batch_blocks):
                    stream = streams[idx % num_streams]
                    norm_cpu = _process_x_block_gpu(
                        x0, x1, y0, y1, y_tiles, tile_cache, final_shape,
                        dtype_val, use_edt, tile_weights, stream
                    )
                    results.append((x0, x1, norm_cpu))

                # Synchronize streams before writing
                for stream in streams[:len(batch_blocks)]:
                    stream.synchronize()

                # Write results sequentially (I/O is sequential anyway)
                for x0, x1, norm_cpu in results:
                    arr_out[
                        (
                            slice(0, final_shape[0]),
                            slice(0, final_shape[1]),
                            slice(0, final_shape[2]),
                            slice(y0, y1),
                            slice(x0, x1),
                        )
                    ] = norm_cpu
                    del norm_cpu

                # Free GPU memory after batch
                if _USING_CUPY:
                    xp.get_default_memory_pool().free_all_blocks()

        else:
            # Sequential processing (fallback or CPU)
            for x0, x1 in x_blocks:
                numer = xp.zeros(
                    (final_shape[0], final_shape[1], final_shape[2], y1 - y0, x1 - x0),
                    dtype=dtype_val,
                )
                denom = xp.zeros_like(numer, dtype=dtype_val)

                for tile_name, t_end, c_end, z_end, ys, ye, xs, xe in y_tiles:
                    # X intersection within this block
                    ix0 = max(x0, xs)
                    ix1 = min(x1, xe)
                    if ix0 >= ix1:
                        continue
                    iy0 = max(y0, ys)
                    iy1 = min(y1, ye)
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
                    block = tile_full[tile_sl].astype(dtype_val, copy=False)
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

                # Normalize and write this block
                norm = xp.nan_to_num(numer / xp.maximum(denom, 1e-12))
                # Convert to NumPy for Zarr (transfers from GPU to CPU if using CuPy)
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

                # Free GPU memory immediately after writing
                del numer, denom, norm, norm_cpu
                if _USING_CUPY:
                    xp.get_default_memory_pool().free_all_blocks()

        # Free cache for this Y band
        tile_cache.clear()
        if _USING_CUPY:
            xp.get_default_memory_pool().free_all_blocks()

    return stitched_pos["0"]


def stitch(
    config_path: str,
    input_store_path: str,
    output_store_path: str,
    flipud: bool = False,
    fliplr: bool = False,
    rot90: int = 0,
    blending_method: Literal["average", "edt"] = "edt",
    **kwargs,
):
    """Mimic of biahub stitch function"""

    # get the shifts and split into a list of lists per well
    all_shifts = read_shifts_biahub(config_path)

    def get_group(key):
        return key.split("/")[1]

    grouped_shifts = defaultdict(dict)
    for key, value in all_shifts.items():
        group = get_group(key)
        grouped_shifts[group][key] = value

    input_store = open_ome_zarr(input_store_path)
    channel_names = input_store.channel_names
    temp_pos = next(input_store.positions())[0]
    tile_shape = input_store[temp_pos].data.shape
    chunks_size = input_store[temp_pos].data.chunks
    scale = input_store[temp_pos].scale
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
    # call assemble for each well
    for g in grouped_shifts.keys():
        shifts = grouped_shifts[g]
        # Streamed assembly into output zarr without allocating full canvas in RAM
        stitched_pos = output_store.create_position("A", g, "0")
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
        )
        del stitched_pos  # free reference
        print("-" * 30)
        print("finished a well")
        print("-" * 30)

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
):
    """Mimic of Biahub estimate stitch function"""

    store = open_ome_zarr(input_store_path)
    print(f"[assemble.estimate_stitch] Using {limit_positions} positions")
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
        # Full list (may be large)
        position_list = [
            p for p, _ in tqdm(store.positions(), desc="Getting positions")
        ]

    grouped_positions = defaultdict(list)
    for a in position_list:
        group = a[:3]
        grouped_positions[group].append(a)

    running_opt_shift_dict = {}
    running_confidence_dict = {}
    for g in grouped_positions.keys():
        well_positions = grouped_positions[g]
        tile_lut = {t[4:]: i for i, t in enumerate(well_positions)}

        edge_list, confidence_dict = pairwise_shifts(
            well_positions,
            input_store_path,
            well=g,
            flipud=flipud,
            fliplr=fliplr,
            rot90=rot90,
            overlap=overlap,
        )

        opt_shift_dict = optimal_positions(edge_list, tile_lut, g, tile_size, x_guess)

        running_opt_shift_dict = running_opt_shift_dict | opt_shift_dict
        running_confidence_dict[g] = confidence_dict

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
