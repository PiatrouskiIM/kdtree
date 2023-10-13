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


def benchmark_wiki(N=100000):
    from wiki import KdTree
    from random import seed, random

    print("wiki")

    P = lambda *coords: list(coords)

    def random_point(k):
        return [random() for _ in range(k)]

    def random_points(k, n):
        return [random_point(k) for _ in range(n)]

    X = np.array(random_points(3, N))
    kd2 = KdTree(np.array(copy.deepcopy(X)))
    from scipy.spatial import KDTree as GoodTree
    good_tree = GoodTree(np.array(X))


    for i in tqdm(range(N)):
        p = random_point(3)
        vvvv = kd2.query(x=p, k=2)
        dd, ii = good_tree.query(p, k=2)
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


def benchmark_simple(N=100000):
    from simple import KDTree
    print("simple")
    rng = np.random.RandomState(0)
    X = rng.random_sample((N, 3))  # 10 points in 3 dimensions
    tree = KDTree(list(X), dim=3)
    for i in tqdm(range(N)):
        tree.get_knn(point=rng.random_sample((3,)), k=2)

# benchmark_scipy()
benchmark_wiki()
# benchmark_simple()
