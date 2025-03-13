import numpy as np
from stitch.stitch.align import offset
from stitch import connect
from collections import OrderedDict
from iohub import open_ome_zarr
import dask.array as da


class LimitedSizeDict(OrderedDict):
    def __init__(self, max_size):
        super().__init__()
        self.max_size = max_size

    def __setitem__(self, key, value):
        # Add new entry
        super().__setitem__(key, value)
        # Remove the oldest entry if size limit is exceeded
        if len(self) > self.max_size:
            self.popitem(last=False)  # Removes the first (oldest) item


class Edge:
    """
    tile_a_key: str
    tile_b_key: str
    tile_cache: TileCache
    """

    def __init__(self, tile_a_key, tile_b_key, tile_cache):
        self.tile_cache = tile_cache
        self.tile_a = connect.pos_to_name(tile_a_key)
        self.tile_b = connect.pos_to_name(tile_b_key)
        self.offset = self.get_offset()

    def get_offset(self):
        """
        get the offset between the two tiles
        """
        tile_a = self.tile_cache[self.tile_a]
        tile_b = self.tile_cache[self.tile_b]
        return offset(tile_a, tile_b)


class TileCache:
    """
    has attributes:
    - list of tiles with a maximum length
    - indexed by their name, containing the tile array in memory

    methods:
    - load tile
    - get_item
    """

    def __init__(self, store_path, well):
        self.cache = LimitedSizeDict(max_size=10)
        self.store = open_ome_zarr(store_path)
        self.well = well

    def add(self, obj):
        """add an object to the cache"""
        self.cache.append(obj)

    def __getitem__(self, key):
        """get an object from the cache"""
        # if isinstance(key, int):
        #     return self.cache[key]
        if key in self.cache:
            return self.cache[key]
        else:
            try:
                # load a new tile and add it to the cache
                a = self.load_tile(key)
                return a
            except KeyError:
                print("tile not found")
                return None

    def load_tile(self, key):
        """load a tile and add it to the cache"""
        print("loading tile")
        da_tile = da.from_array(self.store[f"{self.well}/{key}"].data)

        # hardcoded for now to only take first slice and time-point
        self.cache[key] = da_tile[0, 0, 0, :, :]
        return da_tile[0, 0, 0, :, :]


def augment_tile(tile: np.ndarray, flipud: bool, fliplr: bool, rot90: int) -> np.array:
    """Augment a tile with flips and rotations"""
    if flipud:
        tile = np.flip(tile, axis=-2)
    if fliplr:
        tile = np.flip(tile, axis=-1)
    if rot90:
        tile = np.rot90(tile, k=rot90)
    return tile
