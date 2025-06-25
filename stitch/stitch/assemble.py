from iohub.ngff import open_ome_zarr
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

import numpy as xp
from scipy import ndimage as cundi


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
):
    """Assemble the stitched image give the total shift of all tiles
    - Assume that we have paired image / total shift to be applied
    args:
        - shifts: dict of tile_name: (x_shift, y_shift)
        - tile_size: tuple of the tile size, only x and y dims
        - fov_store_path: path to the zarr file with the tiles
    """

    final_shape_xy = get_output_shape(shifts, tile_size)
    fov_store = open_ome_zarr(fov_store_path)
    pos_1 = next(fov_store.positions())[0]

    tile_shape = fov_store[pos_1].data.shape
    final_shape = tile_shape[:3] + final_shape_xy
    output_image = xp.zeros(final_shape, dtype=xp.float16)  # check dtype
    divisor = xp.zeros(final_shape, dtype=xp.uint8)

    for tile_name, shift in tqdm(shifts.items()):

        tile = fov_store[tile_name].data  # 5D array OME (T, C, Z, Y, X)
        tile = augment_tile(xp.asarray(tile), flipud, fliplr, rot90)
        # ignore sub-pixel shifts (which biahub does too by order=0 interpolation)
        shift_array = xp.asarray(shift).astype(xp.uint16)
        # Future: add rotation / interpolation by first padding, then placing padded block into
        # final output

        output_image[
            :,
            :,
            :,
            shift_array[0] : shift_array[0] + tile_size[0],
            shift_array[1] : shift_array[1] + tile_size[1],
        ] += tile

        divi_tile = tile[:,:,:,:,:] > 0*1 #only add divisor where the tile is not zero
        divisor[
            :,
            :,
            :,
            shift_array[0] : shift_array[0] + tile_size[0],
            shift_array[1] : shift_array[1] + tile_size[1],
        ] += divi_tile.astype(xp.uint8)

    stitched = xp.zeros_like(output_image, dtype=xp.float16)

    def _divide(a, b):
        return xp.nan_to_num(a / b)

    out = divide_tile(
        output_image,
        divisor,
        func=_divide,
        out_array=stitched,
        tile=divide_tile_size,  # need to increase
    )
    del out  # free memory

    return stitched


def stitch(
    config_path: str,
    input_store_path: str,
    output_store_path: str,
    flipud: bool = False,
    fliplr: bool = False,
    rot90: int = 0,
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

    # initialize output zarr store
    output_store = open_ome_zarr(
        output_store_path, layout="hcs", mode="w-", channel_names=channel_names
    )
    print("output store created")
    # call assemble for each well
    for g in grouped_shifts.keys():
        shifts = grouped_shifts[g]
        output = assemble(
            shifts=shifts,
            tile_size=tile_shape[-2:],
            fov_store_path=input_store_path,
            flipud=flipud,
            fliplr=fliplr,
            rot90=rot90,
        )
        stitched_pos = output_store.create_position("A", g, "0")
        stitched_pos.create_image(
            "0",
            data=output,
            chunks=chunks_size,
            transform=[TransformationMeta(type="scale", scale=scale)],
        )
        del output  # free memory
        del stitched_pos  # free memory
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
    tile_size: tuple=(2048,2048),
    overlap: int = 150,
    x_guess: Optional[dict] = None,
):
    """Mimic of Biahub estimate stitch function"""

    store = open_ome_zarr(input_store_path)
    position_list = [a[0] for a in store.positions()]

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
