import numpy as np
from src.kdtree_PiatrouskiIM import KDTree

points = np.array([
    [5, 0, 0],
    [0, 0, 0],
    [1, 0, 0],
    [2, 0, 0],
    [3, 0, 0],

    [6, 0, 0],
    [7, 0, 0],
    [8, 0, 0],
    [9, 0, 0],
    [10., 10., 10.],
    [4, 1, 0],
])
tree = KDTree(points)
print(tree.query_single_point(x=np.array([5, 5, 5]), k=2))
