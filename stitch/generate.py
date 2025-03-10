import matplotlib.pyplot as plt
import numpy as np


def example_tiles(
    img: np.array,
    n: int,
    overlap: float,
) -> list:
    tiles = []
    width, height = img.shape[:2]
    tile_height = height // n
    tile_width = width // n
    i_stride = tile_height - overlap
    j_stride = tile_width - overlap

    for i in range(n):
        for j in range(n):
            i_shift, j_shift = np.random.randint(-0.2 * overlap, 0.2 * overlap, 2)
            i_start = np.clip(i * i_stride + i_shift, a_min=0, a_max=height)
            i_end = np.clip(i_start + tile_height, a_min=0, a_max=height)
            j_start = np.clip(j * j_stride + j_shift, a_min=0, a_max=width)
            j_end = np.clip(j_start + tile_width, a_min=0, a_max=width)

            print(f"{i_start}, {i_end}, {j_start}, {j_end}")

            tiles.append(img[i_start:i_end, j_start:j_end])

    return tiles


def circle_points(center: tuple, radius: int) -> np.array:
    x_c, y_c = center
    points = []

    # Iterate over a bounding box of the circle
    for x in range(x_c - radius, x_c + radius + 1):
        for y in range(y_c - radius, y_c + radius + 1):
            if (x - x_c) ** 2 + (y - y_c) ** 2 <= radius**2:
                points.append((x, y))

    return np.array(points)
