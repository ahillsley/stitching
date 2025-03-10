

# scripts to take the place of biahub-register

import dexpv2 

def apply_affine_ome_zarr(
        array,
        transform,
        output_path
):
    """
    Apply an affine transform to an OME-Zarr array and save as a new array
    """
    return