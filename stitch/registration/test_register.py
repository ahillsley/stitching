# %%
from register import read_transform_biahub
import yaml

yaml_path = "/hpc/projects/intracellular_dashboard/ops/ops0006_20250121/2-tracking/pheno_a1_register.yml"

b = read_transform_biahub(yaml_path)
# %%
import dask.array as da
from iohub import open_ome_zarr
from cupyx.scipy import ndimage as cundi
from dexpv2.registration import apply_affine_transform
import cupy as cp
from iohub import open_ome_zarr

zarr_path = "/hpc/projects/intracellular_dashboard/ops/ops0006_20250121/1-preprocess/live_imaging/segmentation/phenotyping_segmentation_stitched.zarr/A/1/0/0"
zarr_array = da.from_zarr(zarr_path)
cp_array = cp.asarray(zarr_array[0, 0, :, :, :])
a = apply_affine_transform(b, cp_array, (1, 1, 1))
# %%
zarr_path = "/hpc/projects/intracellular_dashboard/ops/ops0006_20250121/1-preprocess/live_imaging/segmentation/phenotyping_segmentation_stitched.zarr/A/1/0"
source_ds = open_ome_zarr(zarr_path)

# %%
from register import register

a = register(
    input_path="/hpc/projects/intracellular_dashboard/ops/ops0006_20250121/1-preprocess/live_imaging/segmentation/phenotyping_segmentation_stitched.zarr",
    target_path="/hpc/projects/intracellular_dashboard/ops/ops0006_20250121/1-preprocess/live_imaging/segmentation/tracking_segmentation_stitched.zarr",
    transform_path="/hpc/projects/intracellular_dashboard/ops/ops0006_20250121/2-tracking/pheno_a1_register.yml",
    output_path="/hpc/projects/intracellular_dashboard/ops/ops0006_20250121/2-tracking/test.zarr",
)
# %%
