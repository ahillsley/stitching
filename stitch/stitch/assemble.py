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
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
from joblib import Parallel, delayed

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
        debug_zero_mask: If True, print debug info about zero mask statistics
        per_channel_edt: If True, compute EDT weights per-channel based on nonzero mask (data-driven)
        parallel_y_bands: If True, process Y-bands in parallel using joblib (CPU only, ignored for GPU)
        n_workers: Number of workers for parallel Y-band processing. If None, uses _get_optimal_workers()
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

    # Inner function to process a single Y-band (can be called in parallel)
    def _process_y_band(band_idx: int, y0: int, y1: int, y_tiles: list):
        """Process a single Y-band - loads tiles, processes X-blocks, writes to output."""
        t_band_start = time.time()

        # Load all tiles for this Y-band
        t_load_start = time.time()
        tile_cache = {}

        def load_single_tile(tile_meta_item):
            """Load and augment a single tile."""
            tile_name, t_end, c_end, z_end, ys, ye, xs, xe = tile_meta_item
            tile_full = fov_store[tile_name].data
            tile_arr = augment_tile(xp.asarray(tile_full), flipud, fliplr, rot90)
            return tile_name, (tile_arr, t_end, c_end, z_end, ys, ye, xs, xe)

        # Load tiles in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(6, len(y_tiles))) as executor:
            futures = {executor.submit(load_single_tile, tm): tm[0] for tm in y_tiles}
            for future in as_completed(futures):
                tile_name, tile_data = future.result()
                tile_cache[tile_name] = tile_data

        t_load_elapsed = time.time() - t_load_start
        print(f"  [Y-band {band_idx+1}/{len(y_bands)}] Loaded {len(y_tiles)} tiles: {t_load_elapsed:.2f}s")

        # Synchronize GPU if using CuPy
        if _USING_CUPY:
            xp.cuda.Stream.null.synchronize()

        # Process all X blocks for this Y band
        x_blocks = [(x0, min(total_x, x0 + tx)) for x0 in range(0, total_x, tx)]
        t_process_start = time.time()

        # Sequential X-block processing (used for CPU and parallel Y-band mode)
        for x0, x1 in x_blocks:
            numer = xp.zeros(
                (final_shape[0], final_shape[1], final_shape[2], y1 - y0, x1 - x0),
                dtype=dtype_val,
            )
            denom = xp.zeros_like(numer, dtype=dtype_val)

            for tile_name, t_end_t, c_end_t, z_end_t, ys_t, ye_t, xs_t, xe_t in y_tiles:
                ix0 = max(x0, xs_t)
                ix1 = min(x1, xe_t)
                if ix0 >= ix1:
                    continue
                iy0 = max(y0, ys_t)
                iy1 = min(y1, ye_t)
                tile_full, t_end_c, c_end_c, z_end_c, ys_c, ye_c, xs_c, xe_c = tile_cache[tile_name]
                tile_sl = (
                    slice(0, t_end_c),
                    slice(0, c_end_c),
                    slice(0, z_end_c),
                    slice(iy0 - ys_c, iy1 - ys_c),
                    slice(ix0 - xs_c, ix1 - xs_c),
                )
                block = tile_full[tile_sl].astype(dtype_val, copy=False)

                if use_edt and per_channel_edt:
                    for c_idx in range(c_end_c):
                        ch_out_sl = (
                            slice(0, t_end_c),
                            slice(c_idx, c_idx + 1),
                            slice(0, z_end_c),
                            slice(iy0 - y0, iy1 - y0),
                            slice(ix0 - x0, ix1 - x0),
                        )
                        block_ch = block[:, c_idx : c_idx + 1, :, :, :]
                        nz2d = xp.any(block[:, c_idx, :, :, :] != 0, axis=(0, 1))

                        if debug_zero_mask:
                            total_px = nz2d.size
                            nonzero_px = int(xp.count_nonzero(nz2d))
                            print(f"    [debug] tile={tile_name} ch={c_idx}: {nonzero_px}/{total_px} nonzero")

                        if not xp.any(nz2d):
                            continue

                        if int(xp.count_nonzero(nz2d)) == nz2d.size:
                            wloc2d = tile_weights[iy0 - ys_c : iy1 - ys_c, ix0 - xs_c : ix1 - xs_c]
                        else:
                            _dist = cundi.distance_transform_edt(nz2d).astype(np.float32)
                            _dist += 1e-6
                            wloc2d = np.power(_dist, float(blending_exponent), where=(_dist > 0))
                            wloc2d = xp.asarray(wloc2d, dtype=dtype_val)

                        wloc5 = wloc2d[None, None, None, :, :]
                        numer[ch_out_sl] += (block_ch.astype(xp.float32) * wloc5).astype(dtype_val)
                        denom[ch_out_sl] += wloc5

                elif use_edt:
                    out_sl = (
                        slice(0, t_end_c),
                        slice(0, c_end_c),
                        slice(0, z_end_c),
                        slice(iy0 - y0, iy1 - y0),
                        slice(ix0 - x0, ix1 - x0),
                    )
                    wloc = tile_weights[(iy0 - ys_c) : (iy1 - ys_c), (ix0 - xs_c) : (ix1 - xs_c)]
                    wloc = xp.asarray(wloc, dtype=dtype_val)
                    nz = block != 0
                    wloc = wloc * nz
                    numer[out_sl] += block * wloc
                    denom[out_sl] += wloc
                else:
                    out_sl = (
                        slice(0, t_end_c),
                        slice(0, c_end_c),
                        slice(0, z_end_c),
                        slice(iy0 - y0, iy1 - y0),
                        slice(ix0 - x0, ix1 - x0),
                    )
                    numer[out_sl] += block
                    denom[out_sl] += (block != 0).astype(dtype_val)

            # Normalize and write this block
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
        tile_cache.clear()
        t_band_elapsed = time.time() - t_band_start
        print(f"  [Y-band {band_idx+1}/{len(y_bands)}] Total: {t_band_elapsed:.2f}s ({len(x_blocks)} X-blocks)")

    # Decide whether to use parallel Y-band processing
    use_parallel_y = parallel_y_bands and not _USING_CUPY
    if use_parallel_y:
        workers = n_workers if n_workers is not None else _get_optimal_workers(use_gpu=False, verbose=True)
        print(f"[assemble.streaming] Parallel Y-bands enabled with {workers} workers (CPU mode)")
        Parallel(n_jobs=workers, backend="threading")(
            delayed(_process_y_band)(band_idx, y0, y1, y_tiles)
            for band_idx, (y0, y1, y_tiles) in enumerate(y_bands)
        )
    else:
        # Sequential or GPU mode - use original loop with GPU optimizations
        for band_idx, (y0, y1, y_tiles) in enumerate(tqdm(y_bands, desc="Stitching Y")):
            t_band_start = time.time()

            # Load all tiles for this Y-band to GPU in parallel
            t_load_start = time.time()
            tile_cache = {}

            def load_single_tile(tile_meta):
                """Load and augment a single tile."""
                tile_name, t_end, c_end, z_end, ys, ye, xs, xe = tile_meta
                tile_full = fov_store[tile_name].data
                # Transfer to GPU immediately after loading
                tile_gpu = augment_tile(xp.asarray(tile_full), flipud, fliplr, rot90)
                return tile_name, (tile_gpu, t_end, c_end, z_end, ys, ye, xs, xe)

            # Load tiles in parallel using ThreadPoolExecutor
            # Zarr I/O releases GIL, so threads work well for parallel loading
            # Limited to 6 workers to balance speed and GPU memory (requires 80GB+ GPU for 3 parallel wells)
            with ThreadPoolExecutor(max_workers=min(6, len(y_tiles))) as executor:
                futures = {executor.submit(load_single_tile, tile_meta): tile_meta[0]
                          for tile_meta in y_tiles}
                for future in as_completed(futures):
                    tile_name, tile_data = future.result()
                    tile_cache[tile_name] = tile_data

            t_load_elapsed = time.time() - t_load_start
            print(f"  [Y-band {band_idx+1}/{len(y_bands)}] Loaded {len(y_tiles)} tiles in parallel: {t_load_elapsed:.2f}s")

            # Optionally synchronize GPU before processing if using CuPy
            if _USING_CUPY:
                xp.cuda.Stream.null.synchronize()

            # Process all X blocks for this Y band
            x_blocks = []
            for x0 in range(0, total_x, tx):
                x1 = min(total_x, x0 + tx)
                x_blocks.append((x0, x1))

            # Parallel X-block processing using CUDA streams (GPU only)
            t_process_start = time.time()
            if parallel_x_blocks and _USING_CUPY and len(x_blocks) > 1:
                print(f"  [Y-band {band_idx+1}/{len(y_bands)}] Processing {len(x_blocks)} X-blocks in parallel...")

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

                t_process_elapsed = time.time() - t_process_start
                print(f"  [Y-band {band_idx+1}/{len(y_bands)}] GPU processing: {t_process_elapsed:.2f}s")

            else:
                # Sequential processing (fallback or CPU)
                print(f"  [Y-band {band_idx+1}/{len(y_bands)}] Processing {len(x_blocks)} X-blocks sequentially...")
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
                        tile_sl = (
                            slice(0, t_end),
                            slice(0, c_end),
                            slice(0, z_end),
                            slice(iy0 - ys, iy1 - ys),
                            slice(ix0 - xs, ix1 - xs),
                        )
                        block = tile_full[tile_sl].astype(dtype_val, copy=False)

                        if use_edt and per_channel_edt:
                            # Per-channel EDT: compute weights based on each channel's nonzero mask
                            for c_idx in range(c_end):
                                ch_out_sl = (
                                    slice(0, t_end),
                                    slice(c_idx, c_idx + 1),
                                    slice(0, z_end),
                                    slice(iy0 - y0, iy1 - y0),
                                    slice(ix0 - x0, ix1 - x0),
                                )
                                block_ch = block[:, c_idx : c_idx + 1, :, :, :]
                                # Collapse T and Z for a 2D nonzero mask (Y, X)
                                nz2d = xp.any(block[:, c_idx, :, :, :] != 0, axis=(0, 1))

                                if debug_zero_mask:
                                    total_px = nz2d.size
                                    nonzero_px = int(xp.count_nonzero(nz2d))
                                    zero_px = int(total_px - nonzero_px)
                                    print(f"    [debug] tile={tile_name} ch={c_idx}: {nonzero_px}/{total_px} nonzero ({zero_px} zeros)")

                                if not xp.any(nz2d):
                                    # No signal for this channel in this intersection
                                    continue

                                # If mask is fully non-zero, use structural weights
                                if int(xp.count_nonzero(nz2d)) == nz2d.size:
                                    wloc2d = tile_weights[iy0 - ys : iy1 - ys, ix0 - xs : ix1 - xs]
                                else:
                                    # Compute EDT on actual nonzero mask (data-driven)
                                    _dist = cundi.distance_transform_edt(nz2d).astype(np.float32)
                                    _dist += 1e-6
                                    wloc2d = np.power(_dist, float(blending_exponent), where=(_dist > 0))
                                    wloc2d = xp.asarray(wloc2d, dtype=dtype_val)

                                # Expand to (T, 1, Z, Y, X) by broadcasting
                                wloc5 = wloc2d[None, None, None, :, :]
                                # Cast to float32 before multiplication to prevent overflow with uint16
                                numer[ch_out_sl] += (block_ch.astype(xp.float32) * wloc5).astype(dtype_val)
                                denom[ch_out_sl] += wloc5

                        elif use_edt:
                            # Standard EDT: use structural weights for all channels
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
                            # Average blending
                            out_sl = (
                                slice(0, t_end),
                                slice(0, c_end),
                                slice(0, z_end),
                                slice(iy0 - y0, iy1 - y0),
                                slice(ix0 - x0, ix1 - x0),
                            )
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

                t_process_elapsed = time.time() - t_process_start
                print(f"  [Y-band {band_idx+1}/{len(y_bands)}] Processing: {t_process_elapsed:.2f}s")

            # Free cache for this Y band
            tile_cache.clear()
            if _USING_CUPY:
                xp.get_default_memory_pool().free_all_blocks()

            t_band_elapsed = time.time() - t_band_start
            print(f"  [Y-band {band_idx+1}/{len(y_bands)}] Total: {t_band_elapsed:.2f}s ({len(x_blocks)} X-blocks)")

    return stitched_pos["0"]


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


def stitch(
    config_path: str,
    input_store_path: str,
    output_store_path: str,
    flipud: bool = False,
    fliplr: bool = False,
    rot90: int = 0,
    blending_method: Literal["average", "edt"] = "edt",
    parallel_mode: Literal["auto", "wells", "y_bands", "sequential"] = "auto",
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
            - "auto" (default): GPU uses parallel wells, CPU uses parallel Y-bands
            - "wells": Force parallel well processing (ThreadPoolExecutor)
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

    # Determine parallelization strategy based on parallel_mode and hardware
    use_parallel_wells = False
    use_parallel_y_bands = False
    n_workers = None

    if parallel_mode == "auto":
        if _USING_CUPY:
            use_parallel_wells = True  # GPU: parallel wells via ThreadPoolExecutor
            print("[stitch] Auto mode: GPU detected, using parallel wells strategy")
        else:
            use_parallel_y_bands = True  # CPU: parallel Y-bands via joblib
            n_workers = _get_optimal_workers(use_gpu=False, verbose=True)
            print(f"[stitch] Auto mode: CPU detected, using parallel Y-bands strategy ({n_workers} workers)")
    elif parallel_mode == "wells":
        use_parallel_wells = True
        print("[stitch] Forced parallel wells strategy")
    elif parallel_mode == "y_bands":
        use_parallel_y_bands = True
        n_workers = _get_optimal_workers(use_gpu=False, verbose=True)
        print(f"[stitch] Forced parallel Y-bands strategy ({n_workers} workers)")
    else:  # "sequential"
        print("[stitch] Sequential processing (no parallelization)")

    well_ids = list(grouped_shifts.keys())
    num_wells = len(well_ids)

    # Thread lock for thread-safe zarr store operations
    well_lock = threading.Lock()

    if use_parallel_wells:
        # PARALLEL WELL PROCESSING - Process wells in parallel using ThreadPoolExecutor
        max_workers = min(4, num_wells)  # Limit to 4 parallel wells to avoid resource contention

        print(f"[Parallel Wells] Processing {num_wells} wells with {max_workers} parallel workers")
        print(f"[Parallel Wells] Wells: {well_ids}")

        # Use ThreadPoolExecutor for parallel well processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all well processing tasks
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

            # Wait for all wells to complete and collect results
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
                    print(f"[Parallel Wells] Exception in well {well_id}: {e}")
                    failed_wells.append(well_id)

        # Report final results
        print(f"[Parallel Wells] Completed: {len(completed_wells)} wells: {completed_wells}")
        if failed_wells:
            print(f"[Parallel Wells] Failed: {len(failed_wells)} wells: {failed_wells}")
            raise RuntimeError(f"Failed to process {len(failed_wells)} wells: {failed_wells}")

        print(f"[Parallel Wells] All {num_wells} wells processed successfully!")

    else:
        # SEQUENTIAL WELL PROCESSING with optional parallel Y-bands
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

    running_opt_shift_dict = {}
    running_confidence_dict = {}
    for g in grouped_positions.keys():
        well_positions = grouped_positions[g]
        tile_lut = {t[4:]: i for i, t in enumerate(well_positions)}

        # Determine timepoint for this well
        well_timepoint = timepoint
        if timepoint_per_well is not None and g in timepoint_per_well:
            well_timepoint = timepoint_per_well[g]
            print(f"[assemble.estimate_stitch] Using timepoint {well_timepoint} for well {g}")

        edge_list, confidence_dict = pairwise_shifts(
            well_positions,
            input_store_path,
            well=g,
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
