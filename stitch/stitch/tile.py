import numpy as np
from stitch import connect
from collections import OrderedDict
from iohub import open_ome_zarr
import dask.array as da
from typing import List, Dict
from tqdm import tqdm
import scipy

from dexp.processing.registration.model.translation_registration_model import (
    TranslationRegistrationModel,
)
from dexp.processing.registration import translation_nd as dexp_reg
from dexp.processing.utils.linear_solver import linsolve
from stitch.connect import parse_positions, pos_to_name
from stitch.stitch.graph import connectivity, hilbert_over_points


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
        self.ux = int(self.tile_a[:3])
        self.uy = int(self.tile_a[3:])
        self.vx = int(self.tile_b[:3])
        self.vy = int(self.tile_b[3:])
        self.relation = (self.ux - self.vx, self.uy - self.vy)
        self.model = self.get_offset()

    def get_offset(self):
        """
        get the offset between the two tiles
        """
        tile_a = self.tile_cache[self.tile_a]
        tile_b = self.tile_cache[self.tile_b]
        return offset(tile_a, tile_b, self.relation, overlap=150)


class TileCache:
    """
    has attributes:
    - list of tiles with a maximum length
    - indexed by their name, containing the tile array in memory

    methods:
    - load tile
    - get_item
    """

    def __init__(self, store_path, well, flipud, fliplr, rot90):
        self.cache = LimitedSizeDict(max_size=20)
        self.store = open_ome_zarr(store_path)
        self.well = well
        self.flipud = flipud
        self.fliplr = fliplr
        self.rot90 = rot90

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
        da_tile = da.from_array(self.store[f"{self.well}/{key}"].data)

        aug_tile = augment_tile(
            da_tile[0, 0, 0, :, :].compute(),
            flipud=self.flipud,
            fliplr=self.fliplr,
            rot90=self.rot90,
        )
        # hardcoded for now to only take first slice and time-point
        self.cache[key] = aug_tile
        return aug_tile


def augment_tile(tile: np.ndarray, flipud: bool, fliplr: bool, rot90: int) -> np.array:
    """Augment a tile with flips and rotations"""
    if flipud:
        tile = np.flip(tile, axis=-2)
    if fliplr:
        tile = np.flip(tile, axis=-1)
    if rot90:
        tile = np.rot90(tile, k=rot90, axes=(-2, -1))
    return tile


def offset(
    image_a: np.array, image_b: np.array, relation: tuple, overlap: int
) -> TranslationRegistrationModel:
    """
    overlap: estimate of the # pixels seen in both images
    """
    shape = image_a.shape
    # TODO: have this load only a small fraction of the entire image
    if relation[0] == -1:
        # tile_b is to the right of tile_a
        roi_a = image_a[:, -overlap:]
        roi_b = image_b[:, :overlap]
        corr_x = shape[-2] - overlap
        corr_y = 0

    if relation[1] == -1:
        # tile_b is below tile_a
        roi_a = image_a[-overlap:, :]
        roi_b = image_b[:overlap, :]
        corr_x = 0
        corr_y = shape[-1] - overlap

    # phase images are centered at 0, shift to make them all positive
    roi_a_min = np.min(roi_a)
    roi_b_min = np.min(roi_b)
    if roi_a_min < 0:
        roi_a = roi_a - roi_a_min
    if roi_b_min < 0:
        roi_b = roi_b - roi_b_min

    model = dexp_reg.register_translation_nd(roi_a, roi_b)
    # print(f"model shift vector is: {model.shift_vector}")
    model.shift_vector += np.array([corr_y, corr_x])

    return model


def pairwise_shifts(
    positions: List, store_path: str, well: str, flipud: bool, fliplr: bool, rot90: bool
) -> List:
    """ """
    # get neighboring tiles
    grid_positions = parse_positions(positions)
    hilbert_order = hilbert_over_points(grid_positions)
    edges_hilbert = connectivity(hilbert_order)

    tile_cache = TileCache(
        store_path=store_path,
        well=well,
        flipud=flipud,
        fliplr=fliplr,
        rot90=rot90,
    )

    edge_list = []
    edge_list = []
    confidence_dict = {}
    for key, pos in tqdm(edges_hilbert.items()):
        edge_model = Edge(pos[0], pos[1], tile_cache)
        edge_list.append(edge_model)

        # positions need to be not np.types to save to yaml
        pos_a_nice = list(int(x) for x in pos[0])
        pos_b_nice = list(int(x) for x in pos[1])
        confidence_dict[key] = [
            pos_a_nice,
            pos_b_nice,
            float(edge_model.model.confidence),
        ]

    return edge_list, confidence_dict


def optimal_positions(
    edge_list: List, tile_lut: Dict, well: str, tile_size: tuple
) -> Dict:
    """ """
    y_i = np.zeros(len(edge_list) + 1, dtype=np.float32)
    y_j = np.zeros(len(edge_list) + 1, dtype=np.float32)

    x_guess = np.asarray(
        [int(a[:3]) * tile_size[0] for a in tile_lut.keys()]
    )  # assumes square tiles

    a = scipy.sparse.lil_matrix((len(tile_lut), len(edge_list) + 1), dtype=np.float32)

    for c, e in enumerate(edge_list):
        a[[tile_lut[e.tile_a], tile_lut[e.tile_b]], c] = [-1, 1]
        y_i[c] = e.model.shift_vector[0]
        y_j[c] = e.model.shift_vector[1]

    y_i[-1] = 0
    y_j[-1] = 0
    a[0, -1] = 1

    a = a.T.tocsr()
    tolerance = 1e-5
    order_error = 1
    order_reg = 1
    alpha_reg = 0
    maxiter = 1e8
    print("optimizing positions")
    opt_i = linsolve(
        a,
        y_i,
        tolerance=tolerance,
        order_error=order_error,
        order_reg=order_reg,
        alpha_reg=alpha_reg,
        x0=x_guess,
        maxiter=maxiter,
    )
    opt_j = linsolve(
        a,
        y_j,
        tolerance=tolerance,
        order_error=order_error,
        order_reg=order_reg,
        alpha_reg=alpha_reg,
        x0=x_guess,
        maxiter=maxiter,
    )
    opt_shifts = np.vstack((opt_i, opt_j)).T

    opt_shifts_zeroed = opt_shifts - np.min(opt_shifts, axis=0)

    # needs to be a list of python types for exporting to yaml
    opt_shifts_dict = {
        f"{well}/{a}": [int(a) for a in opt_shifts_zeroed[i]]
        for i, a in enumerate(tile_lut.keys())
    }

    return opt_shifts_dict
