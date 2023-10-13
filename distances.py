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


def minkowski_distance_p(x, y, p=2):
    x = np.asarray(x)
    y = np.asarray(y)

    # Find smallest common datatype with float64 (return type of this
    # function) - addresses #10262.
    # Don't just cast to float64 for complex input case.
    common_datatype = np.promote_types(np.promote_types(x.dtype, y.dtype),
                                       'float64')

    # Make sure x and y are NumPy arrays of correct datatype.
    x = x.astype(common_datatype)
    y = y.astype(common_datatype)

    if p == np.inf:
        return np.amax(np.abs(y - x), axis=-1)
    elif p == 1:
        return np.sum(np.abs(y - x), axis=-1)
    else:
        return np.sum(np.abs(y - x) ** p, axis=-1)


if __name__ == "__main__":
    x = np.array([1, 2, 3])
    print(side_distance_from_min_max(x, np.array([0, 0, 0]), np.array([10, 10, 2])))
