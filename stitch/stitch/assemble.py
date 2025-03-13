from iohub.ngff import open_ome_zarr
from tqdm import tqdm
from stitch.stitch.tile import augment_tile

# try:
#     import cupy as xp
#     from cupyx.scipy import ndimage as cundi

# except (ModuleNotFoundError, ImportError):
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
):
    """Assemble the stitched image give the total shift of all tiles
    - Assume that we have paired image / total shift to be applied
    args:
        - shifts: dict of tile_name: (x_shift, y_shift)
        - tile_size: tuple of the tile size, only x and y dims
        - fov_store_path: path to the zarr file with the tiles
    """

    # TODO: Add option for flips

    final_shape_xy = get_output_shape(shifts, tile_size)

    fov_store = open_ome_zarr(fov_store_path)
    pos_1 = next(fov_store.positions())[0]

    tile_shape = fov_store[pos_1].data.shape
    final_shape = tile_shape[:3] + final_shape_xy
    print(final_shape)
    output_image = xp.zeros(final_shape, dtype=xp.float32)  # check dtype
    divisor = xp.zeros(final_shape, dtype=xp.uint8)

    count = 0
    for tile_name, shift in tqdm(shifts.items()):
        if count == 196:
            break

        tile = fov_store[tile_name].data  # 5D array OME (T, C, Z, Y, X)
        tile = augment_tile(xp.asarray(tile), flipud, fliplr, rot90)

        # ignore sub-pixel shifts (which biahub does to by order=0 interpolation)
        shift_array = xp.asarray(shift).astype(xp.uint16)
        # shift_remain = xp.sum(shift_array - shift_array.astype(int))
        # Check if the shift is exact or if we need to interpolate
        # if shift_remain == 0.0:
        #     print(shift_remain)
        #     print(f'shift was even: {shift_array}')
        # else:
        #     print(f'shift was not even: {shift_array}')
        #     padded_tile = xp.pad(tile, pad_width=((0, 1), (0, 1)))

        # pad the image
        # shift the image
        # crop the image
        # Index the image into the output image
        output_image[
            :,
            :,
            :,
            shift_array[0] : shift_array[0] + tile_size[0],
            shift_array[1] : shift_array[1] + tile_size[1],
        ] += tile
        divisor[
            :,
            :,
            :,
            shift_array[0] : shift_array[0] + tile_size[0],
            shift_array[1] : shift_array[1] + tile_size[1],
        ] += 1
        # index an image of 1s into the "divisor" output image
        count += 1

    # divide the images
    stitched = output_image / divisor
    stitched = xp.nan_to_num(stitched)

    return stitched
