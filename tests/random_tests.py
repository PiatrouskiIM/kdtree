import copy

from tqdm import tqdm
import numpy as np


def benchmark_scipy(N=100000):
    from scipy.spatial import KDTree
    print("scipy")
    rng = np.random.RandomState(0)
    X = rng.random_sample((N, 3))  # 10 points in 3 dimensions
    tree = KDTree(X)
    for i in tqdm(range(N)):
        tree.query(x=rng.random_sample((3,)), k=2, workers=1)


def benchmark_random(N=100000):
    from src.kdtree_PiatrouskiIM import KDTree
    from scipy.spatial import KDTree as GoodTree
    import numpy as np
    from tqdm import tqdm

    rng = np.random.RandomState(0)
    X = rng.random_sample((N, 3))

    tree = KDTree(np.array(copy.deepcopy(X)))
    reference_tree = GoodTree(np.array(X))

    for _ in tqdm(range(N)):
        p = rng.random_sample((3,))
        vvvv = tree.query(x=p, k=2)
        dd, ii = reference_tree.query(p, k=2)
        # print(p)
        # print(X[vvvv[1]], X[vvvv[0]])
        # print(X[ii[0]], X[ii[1]])

        # X[vvvv]
        # X[ii]

        # diff = np.max(np.abs(X[ii[0]]- np.asarray(result.nearest)))
        # print(diff)
        assert np.allclose(X[vvvv[0]], X[ii[0]]) or np.allclose(X[vvvv[0]], X[ii[1]]), f"{X[vvvv[0]]}, {X[ii[0]]}, {X[vvvv[1]]}"
        # assert np.allclose(X[ii[0]], np.asarray(result[0]))
        # true_result =



# benchmark_scipy()
benchmark_random()
# benchmark_simple()
