import pytest
from stitch.registration import register
import numpy as np

def test_register()->None:

    return

def test_apply_affine():
    arr = np.random((10, 10))
    output = register.apply_affine_ome_zarr(arr, np.eye(3), (10, 10))