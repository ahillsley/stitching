import numpy as np
import matplotlib.pyplot as plt

def parse_positions(fovs: list)->np.array:
    """
    read FOV names and parse into X/Y coordinates assuming that each FOV is named XXXYYY convention
    """
    fov_positions = []
    for p in fovs:
        x_cord = int(p[:3])
        y_cord = int(p[3:])
        fov_positions.append((x_cord, y_cord))
        
    return np.asarray(fov_positions)


def connectivity(points: np.array)->np.array:
    """
    provided a list of points, get the connectivity graph between neighboring points
    """
    point_set = set(map(tuple,points))
    edges = set()
    directions = np.array([(0, 1), (1, 0)])
    for x, y in points:
        for dx, dy in directions:
            neighbor = (x + dx, y + dy)
            if neighbor in point_set:
                edges.add(tuple(sorted([(x, y), neighbor])))
    return edges
