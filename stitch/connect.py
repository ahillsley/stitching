import numpy as np
import matplotlib.pyplot as plt

<<<<<<< HEAD

def parse_positions(fovs: list) -> np.array:
=======
def parse_positions(fovs: list)->np.array:
>>>>>>> fc260ae (change name of module to stitch)
    """
    read FOV names and parse into X/Y coordinates assuming that each FOV is named XXXYYY convention
    """
    fov_positions = []
    for p in fovs:
        x_cord = int(p[:3])
        y_cord = int(p[3:])
        fov_positions.append((x_cord, y_cord))
<<<<<<< HEAD

    return np.asarray(fov_positions)


def connectivity(points: np.array) -> np.array:
    """
    provided a list of points, get the connectivity graph between neighboring points
    """
    point_set = set(map(tuple, points))
    edges = set()
    directions = np.array([(0, 1), (1, 0)])
    for x, y in points:
        for dx, dy in directions:
            neighbor = (x + dx, y + dy)
            if neighbor in point_set:
                edges.add(tuple(sorted([(x, y), neighbor])))
    return edges
=======
        
    return np.asarray(fov_positions)

def hilbert_index_to_xy(n, d):
    """Convert a 1D Hilbert index to 2D coordinates in an n x n grid."""
    x = y = 0
    t = d
    s = 1
    while s < n:
        rx = 1 & (t // 2)
        ry = 1 & (t ^ rx)
        if ry == 0:
            if rx == 1:
                x, y = s - 1 - y, s - 1 - x
            x, y = y, x
        x += s * rx
        y += s * ry
        t //= 4
        s *= 2
    return x, y
>>>>>>> fc260ae (change name of module to stitch)
