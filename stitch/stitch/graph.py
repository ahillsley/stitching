import numpy as np

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
    return np.asarray([x, y])

def generate_hilbert_curve(n):
    """Generate the Hilbert curve order for an n x n grid.
    Only fills the entire grid if n is a power of 2
    """
    order = []
    for i in range(n * n):
        order.append(hilbert_index_to_xy(n, i))
    return np.asarray(order)

