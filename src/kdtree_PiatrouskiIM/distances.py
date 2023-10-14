import numpy as np


def side_distance_from_min_max(x: float, min_val: float, max_val: float) -> float:
    return np.maximum(np.maximum(x - max_val, 0.), min_val - x)


def side_distance_box(hb: float, fb: float, x: float, min_val: float, max_val: float
                      # , k: int
                      ) -> float:
    # fb = tree.raw_boxsize_data[k]
    # hb = tree.raw_boxsize_data[k + tree.m]

    if fb <= 0:
        return side_distance_from_min_max(x, min_val, max_val)

    if min_val < x < max_val:
        return 0.

    tmin, tmax = abs(min_val - x), abs(x - max_val)
    if tmin > tmax:
        tmin, tmax = tmax, tmin

    if tmin < hb:
        return tmin
    if tmin > hb:
        return fb - tmax
    return min(tmin, fb - tmax)
