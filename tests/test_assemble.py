#%%
from stitch.stitch import assemble
from stitch.connect import read_shifts_biahub
from stitch import generate
# %%
shifts_path = '/hpc/projects/intracellular_dashboard/ops/ops0006_20250121/1-preprocess/in_situ_sequencing/stitch/stitch_settings.yml'

shifts = read_shifts_biahub(shifts_path)
output = assemble.get_output_shape(shifts, (2048, 2048))
#%%

out = assemble.assemble(shifts, (2048, 2048), '/hpc/projects/intracellular_dashboard/ops/ops0006_20250121/0-convert/in_situ_sequencing/A1_1.zarr', flipud=True)
#%%
import matplotlib.pyplot as plt
plt.imshow(out[0,0,0,:,:], vmin=0, vmax=500)
# %%
import skimage
import scipy.ndimage as ndi
import numpy as np
data = skimage.data.astronaut()
tiles = generate.example_tiles(data, 5, 10)
temp = tiles[0][:,:,0]
a = np.pad(temp, pad_width=((0,2),(0,2)))
# %%
from collections import defaultdict
def get_group(key):
    return key.split('/')[1]

grouped_dict = defaultdict(dict)
for key, value in shifts.items():
    group = get_group(key)
    grouped_dict[group][key] = value

# %%
from stitch.stitch import assemble

assemble.stitch(
    '/hpc/projects/intracellular_dashboard/ops/ops0006_20250121/1-preprocess/in_situ_sequencing/stitch/stitch_settings.yml',
    '/hpc/projects/intracellular_dashboard/ops/ops0006_20250121/0-convert/in_situ_sequencing/bc_symlink.zarr',
    '/hpc/projects/intracellular_dashboard/ops/ops0006_20250121/1-preprocess/in_situ_sequencing/stitch/test.zarr',
    flipud=True
)
# %%
from stitch.stitch import assemble
import numpy as np

a = np.ones((2, 2, 2, 2, 2)) * 2
b = np.ones((2, 2, 2, 2, 2)) * 3
c = np.zeros_like(a)

assemble.array_apply(
    a,
    b,
    func=lambda a, b: a / b,
    out_array=c,
    axis=0
)

# %%
from itertools import product
from tqdm import tqdm
import numpy as np

# %%
# orig_shape = output

tile = (4000, 4000)
func=lambda a, b: a / b
a = np.ones((1, 1, 1, 20000, 20000), dtype=np.float16) * 2
b = np.ones((1, 1, 1, 20000, 20000), dtype=np.float16) * 3
c = np.zeros_like(a)

# %%
out = assemble.divide_tile(a, 
                    b, 
                    func=func,
                    out_array=c,
                    tile=tile,)
# %%
orig_shape = c.shape[-2:]

tiling_start = list(
    product(
        *[
            range(o, size + 2 * o, t + o)  # t + o step, because of blending map
            for size, t, o in zip(orig_shape, tile, overlap)
        ]
    )
)

    
# %%
