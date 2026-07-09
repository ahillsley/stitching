import numpy as np
import yaml


def parse_positions(fovs: list) -> np.array:
    """
    read FOV names and parse into X/Y coordinates assuming that each FOV is named XXXYYY convention
    """
    fov_positions = []
    for p in fovs:
        p = p.split('/')[-1] # handle positions either with row/column info or without
        x_cord = int(p[:3])
        y_cord = int(p[3:])
        fov_positions.append((x_cord, y_cord))

    return np.asarray(fov_positions)


def pos_to_name(pos: tuple) -> str:
    """
    convert a position tuple to a name
    """
    return f"{pos[0]:03d}{pos[1]:03d}"


def read_shifts_biahub(shifts_path: str) -> dict:
    # Use C-backed loader when available (pyyaml compiled with libyaml).
    # For LiveScreen shifts (~35 MB, 167k entries) this cuts parse time
    # from ~30 s (pure Python) to ~2-3 s. Falls back to pure-Python if
    # libyaml isn't installed.
    try:
        Loader = yaml.CSafeLoader
    except AttributeError:
        Loader = yaml.SafeLoader
    with open(shifts_path, "r") as file:
        raw_settings = yaml.load(file, Loader=Loader)

    return raw_settings["total_translation"]
