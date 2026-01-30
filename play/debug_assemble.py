#%%
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

import numpy as xp
from scipy import ndimage as cundi
from ops_analysis.data.experiment import OpsDataset
# %%
experiment = 'ops0034_20250416'
dataset = OpsDataset(experiment)
config_path = dataset.config_paths["iss_stitch"]
fov_store_path = dataset.store_paths["iss_drift_corrected"]
flipud = True
fliplr = False
rot90 = 0
all_shifts = read_shifts_biahub(config_path)

def get_group(key):
    return key.split("/")[1]

grouped_shifts = defaultdict(dict)
for key, value in all_shifts.items():
    group = get_group(key)
    grouped_shifts[group][key] = value

shifts= grouped_shifts['1']
tile_size= (2048, 2048)
divide_tile_size=(4000, 4000)

#%%
final_shape_xy = get_output_shape(shifts, tile_size)
fov_store = open_ome_zarr(fov_store_path)
pos_1 = next(fov_store.positions())[0]

tile_shape = fov_store[pos_1].data.shape
final_shape = tile_shape[:3] + final_shape_xy
output_image = xp.zeros(final_shape, dtype=xp.float16)  # check dtype
divisor = xp.zeros(final_shape, dtype=xp.uint8)
# %%
