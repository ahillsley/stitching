from iohub.ngff import open_ome_zarr
import os
import math
import zarr
import dask.array as da
from tqdm import tqdm
from joblib import Parallel, delayed
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

import numpy as xp
from scipy import ndimage as cundi

# Cache for EDT-based blending weight maps to avoid recomputation per tile size/exponent
_EDT_WEIGHT_CACHE: Dict[Tuple[int, int, int], xp.ndarray] = {}


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
    """Assemble a stitched mosaic in memory using provided tile shifts.

    Overview of the in-memory path:
    - Determine the target (T, C, Z) across tiles based on `tcz_policy`.
    - Allocate a full output canvas in RAM for accumulation.
    - If `blending_method=="edt"`, precompute a centered distance-transform weight map
      for a single tile (YX) and reuse it for every tile placement. Otherwise use
      legacy average-blending via per-pixel counts.
    - For each tile, place the tile into the output canvas at its `(y, x)` shift.
      When using EDT, accumulate `numerator += tile * weights` and `denominator += weights`.
      For average, accumulate `sum += tile` and `count += (tile != 0)`.
    - Normalize by dividing the accumulated numerator by the accumulated denominator.

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
    # We compute the per-tile shapes and choose either the min or max
    # along each dimension, to ensure all tiles fit in the final canvas.
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
    # We keep accumulation in floating types and shift indices in integer types
    # to avoid precision issues or index overflow on very large mosaics.
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
        # Denominator (sum of weights) accumulated alongside the numerator.
        weight_sum = xp.zeros(final_shape, dtype=dtype_val)
        # Precompute or reuse a tile-local weight map based on distance to edges.
        # This is a single centered 2D EDT map (YX) broadcast to T, C, Z when placed.
        ty, tx = int(tile_size[0]), int(tile_size[1])
        cache_key = (ty, tx, int(float(blending_exponent) * 1e6))
        tile_weights = _EDT_WEIGHT_CACHE.get(cache_key)
        if tile_weights is None:
            # Build a boolean mask with interior True, edges False.
            # The EDT of this mask provides distances that increase away from the edges,
            # which we exponentiate to control the sharpness of blend at overlaps.
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
            # Legacy simple averaging by counts: track how many non-zero contributions
            # each pixel received, to divide the sum at the end.
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

    # Allocate array for the normalized output
    stitched = xp.zeros_like(output_image, dtype=dtype_val)

    def _divide(a, b):
        return xp.nan_to_num(a / b)

    if use_edt:
        # Normalize per block to keep memory usage bounded during division.
        # output = numerator / sum_of_weights
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
    debug_zero_mask: bool = True,
):
    """Streamed assembly that avoids saving auxiliary arrays.

    What this does (high-level):
    - Determines a target (T, C, Z) and overall output canvas size from input tiles.
    - Creates the output zarr array on disk, chunked for efficient streaming.
    - Processes the mosaic in small (Y, X) blocks to bound peak RAM usage.
    - For each block, accumulates a numerator and a denominator (weights):
        - EDT blending: numerator += tile * local_EDT_weights; denominator += local_EDT_weights
        - Average blending: numerator += tile; denominator += (tile != 0)
    - Normalizes the block (numerator / max(denominator, eps)) and writes directly to disk.

    This mirrors the logic in `biahub/stitch.py`: weights are computed from a centered
    2D distance-from-edge map per tile, sliced to the overlapped region, and the
    final pixel values are normalized by the sum of all contributing weights.
    """
    print(f"[assemble.streaming] Using blending_method: {blending_method}")

    # Open the input store that contains all tile FOVs
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

    # Choose default YX chunking for the output if none is provided; keep (T, C, Z)=1
    if chunks_size is None:
        ty, tx = divide_tile_size if divide_tile_size is not None else (1024, 1024)
        chunks_size = (1, 1, 1, int(ty), int(tx))

    # Create output array on disk only (no auxiliary arrays). We will stream blocks into it.
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

    # Decide whether to use distance-transform-based blending or legacy averaging
    use_edt = str(blending_method).lower() == "edt"

    # In streaming mode, we will compute per-block, per-channel distance maps from
    # the data-driven nonzero mask to exclude zero-padded regions before EDT.
    # (A precomputed structural map can be used as a fallback if needed.)
    tile_weights = None
    if use_edt:
        ty, tx = int(tile_size[0]), int(tile_size[1])
        cache_key = (ty, tx, int(float(blending_exponent) * 1e6))
        tile_weights = _EDT_WEIGHT_CACHE.get(cache_key)
        if tile_weights is None:
            if ty > 2 and tx > 2:
                # Build a centered interior mask and compute its EDT once per tile size.
                # We then exponentiate the distances to adjust blending sharpness.
                _mask = np.zeros((ty, tx), dtype=bool)
                _mask[1:-1, 1:-1] = True
                _dist = cundi.distance_transform_edt(_mask).astype(np.float32)
                _dist += 1e-6
                _weights = np.power(_dist, float(blending_exponent), where=(_dist > 0))
                tile_weights = xp.asarray(_weights, dtype=dtype_val)
            else:
                tile_weights = xp.ones((ty, tx), dtype=dtype_val)
            _EDT_WEIGHT_CACHE[cache_key] = tile_weights

    # The output array that will receive normalized blocks
    arr_out = stitched_pos["0"]
    dtype_idx = _resolve_shift_dtype(shifts, tile_size, base_bits=32)

    # Precompute tile metadata (shape bounds and YX extents) for fast intersection checks
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

    # Blockwise accumulation over YX: iterate over output in manageable strips/tiles
    ty, tx = divide_tile_size if divide_tile_size is not None else (1024, 1024)
    total_y, total_x = final_shape[-2], final_shape[-1]

    def _process_y_band(y0: int, y1: int):
        # Preload tiles intersecting this Y band once; reuse across all X blocks
        y_tiles = [
            (nm, t_end, c_end, z_end, ys, ye, xs, xe)
            for (nm, t_end, c_end, z_end, ys, ye, xs, xe) in tile_meta
            if not (ye <= y0 or ys >= y1)
        ]
        tile_cache = {}
        for tile_name, t_end, c_end, z_end, ys, ye, xs, xe in y_tiles:
            tile_full = fov_store[tile_name].data
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

        # Dict for quick contributor filtering per X block
        y_extents = {
            nm: (ys, ye, xs, xe) for (nm, _t, _c, _z, ys, ye, xs, xe) in y_tiles
        }

        for x0 in range(0, total_x, tx):
            x1 = min(total_x, x0 + tx)
            # Filter contributors for this specific (Y, X) block
            contrib = set(
                find_contributing_fovs_yx((slice(y0, y1), slice(x0, x1)), y_extents)
            )
            # Numerator and denominator for this output block.
            # They are 5D: (T, C, Z, Y_block, X_block)
            numer = xp.zeros(
                (final_shape[0], final_shape[1], final_shape[2], y1 - y0, x1 - x0),
                dtype=dtype_val,
            )
            denom = xp.zeros_like(numer, dtype=dtype_val)

            for tile_name, t_end, c_end, z_end, ys, ye, xs, xe in y_tiles:
                if tile_name not in contrib:
                    continue
                # Compute the intersection of the current output block with the tile's footprint
                ix0 = max(x0, xs)
                ix1 = min(x1, xe)
                if ix0 >= ix1:
                    continue
                iy0 = max(y0, ys)
                iy1 = min(y1, ye)
                tile_full, t_end, c_end, z_end, ys, ye, xs, xe = tile_cache[tile_name]
                # Slices into the output block
                out_sl = (
                    slice(0, t_end),
                    slice(0, c_end),
                    slice(0, z_end),
                    slice(iy0 - y0, iy1 - y0),
                    slice(ix0 - x0, ix1 - x0),
                )
                # Slices into the tile data
                tile_sl = (
                    slice(0, t_end),
                    slice(0, c_end),
                    slice(0, z_end),
                    slice(iy0 - ys, iy1 - ys),
                    slice(ix0 - xs, ix1 - xs),
                )
                block = tile_full[tile_sl].astype(dtype_val, copy=False)
                if use_edt:
                    # Build a 2D mask per channel from data (exclude zeros), compute EDT on it,
                    # and accumulate per-channel weighted sums.
                    for c_idx in range(c_end):
                        # Channel-specific out slice
                        ch_out_sl = (
                            slice(0, t_end),
                            slice(c_idx, c_idx + 1),
                            slice(0, z_end),
                            slice(iy0 - y0, iy1 - y0),
                            slice(ix0 - x0, ix1 - x0),
                        )
                        # Channel-specific data block (T, 1, Z, Y, X)
                        block_ch = block[:, c_idx : c_idx + 1, :, :, :]
                        # Collapse T and Z for a 2D nonzero mask (Y, X)
                        nz2d = xp.any(block[:, c_idx, :, :, :] != 0, axis=(0, 1))
                        if debug_zero_mask:
                            total_px = nz2d.size
                            nonzero_px = int(xp.count_nonzero(nz2d))
                            zero_px = int(total_px - nonzero_px)
                        if not xp.any(nz2d):
                            # No signal for this channel in this intersection
                            continue
                        # If mask is fully non-zero, fall back to structural weights
                        # to avoid altering weights for channels without zero padding.
                        if int(xp.count_nonzero(nz2d)) == nz2d.size:
                            wloc2d = tile_weights[
                                iy0 - ys : iy1 - ys, ix0 - xs : ix1 - xs
                            ]
                        else:
                            _dist = cundi.distance_transform_edt(nz2d).astype(
                                np.float32
                            )
                            _dist += 1e-6
                            wloc2d = np.power(
                                _dist, float(blending_exponent), where=(_dist > 0)
                            )
                            wloc2d = xp.asarray(wloc2d, dtype=dtype_val)
                        # Expand to (T, 1, Z, Y, X) by broadcasting
                        wloc5 = wloc2d[None, None, None, :, :]
                        numer[ch_out_sl] += block_ch * wloc5
                        denom[ch_out_sl] += wloc5
                else:
                    numer[out_sl] += block
                    denom[out_sl] += (block != 0).astype(dtype_val)

            # Normalize and write this block to disk.
            # Use a small epsilon via maximum() to avoid division-by-zero, and nan_to_num
            # to clamp any remaining NaNs or infs to 0.
            norm = xp.nan_to_num(numer / xp.maximum(denom, 1e-12))
            arr_out[
                (
                    slice(0, final_shape[0]),
                    slice(0, final_shape[1]),
                    slice(0, final_shape[2]),
                    slice(y0, y1),
                    slice(x0, x1),
                )
            ] = norm
        # Free cache for this Y band
        tile_cache.clear()

    # Parallelize across Y bands (disjoint writes in Y) using joblib (threading backend)
    bands = [(y0, min(total_y, y0 + ty)) for y0 in range(0, total_y, ty)]
    # if can import get_optimal_workers, use it
    try:
        from ops_analysis.utils.resource_manager import get_optimal_workers

        workers = max(1, int(get_optimal_workers(use_gpu=False, verbose=False)))
    except ImportError:
        workers = 1
    if workers > 1:
        Parallel(n_jobs=workers, backend="threading")(
            delayed(_process_y_band)(y0, y1)
            for (y0, y1) in tqdm(bands, desc="Stitching Y")
        )
    else:
        for y0, y1 in tqdm(bands, desc="Stitching Y"):
            _process_y_band(y0, y1)

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
