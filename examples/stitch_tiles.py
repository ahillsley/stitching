#%%
from iohub import open_ome_zarr
from stitch import connect
from stitch.stitch import graph, tile
import matplotlib.pyplot as plt
#%%
fovs_path = '/hpc/projects/intracellular_dashboard/ops/ops0006_20250121/0-convert/in_situ_sequencing/A1_1.zarr'
fov_store = open_ome_zarr(fovs_path)
positions = [p[0].rsplit('/', -1)[-1] for p in fov_store.positions()]

grid_pos = connect.parse_positions(positions)
grid_pos_hild = graph.hilbert_over_points(grid_pos)
cont = graph.connectivity(grid_pos_hild)
#%%
plt.plot(grid_pos_hild[:, 0], grid_pos_hild[:, 1])
# %%
fovs_path = '/hpc/projects/intracellular_dashboard/ops/ops0006_20250121/0-convert/in_situ_sequencing/A1_1.zarr'
tile_stash = tile.TileCache(fovs_path, 'A/1')
# %%
a = tile.Edge(cont['0'][0], cont['0'][1], tile_stash)
# %%
from stitch.stitch.align import offset
tile_a = tile_stash['004004']
tile_b = tile_stash['004005']
a = offset(tile_a, tile_b)

# %%
