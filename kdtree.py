import copy
import numpy as np
import heapq


class Node:
    def __init__(self, axis: int, start: int, end: int, x: float = 0., left=None, right=None):
        self.x = x
        self.axis = axis
        self.start, self.end = start, end
        self.left, self.right = left, right


class NodeInfo:
    def __init__(self, node, side_distances):
        self.node = node
        self.side_distances = side_distances
        self.distance = self.side_distances.sum()

    def update_side_distances(self, squared_value, axis):
        self.distance += -self.side_distances[axis] + squared_value
        self.side_distances[axis] = squared_value

    def __le__(self, other):
        return other.distance > self.distance

    def __eq__(self, other):
        return self.distance == other.distance and self.node == other.node

    def __ge__(self, other):
        return self.distance > other.distance


def heappush(heap, heap_priorities, item_priority, item):
    for i in range(len(heap)):
        if heap_priorities[i] > item_priority:
            heap_priorities.insert(i, item_priority)
            heap.inser(i, item)
    else:
        heap_priorities.append(item_priority)
        heap.append(item)


def grow(start, end, points, indexes, leaf_size=10):
    if end - start <= leaf_size:
        return Node(axis=-1, start=start, end=end)

    points_slice = points[start:end]
    axis = np.argmax(points_slice.max(axis=0) - points_slice.min(axis=0))

    argsort = np.argsort(points[start:end, axis])
    points[start:end] = points[start:end][argsort]
    indexes[start:end] = indexes[start:end][argsort]
    length = end - start

    m = length // 2
    median_point = points[start + m]
    while m + 1 < length and points[start + m + 1][axis] == median_point[axis]:
        m += 1
    median_point = points[start + m]
    return Node(x=median_point[axis],
                axis=axis,
                start=start,
                end=end,
                left=grow(start=start, end=start + m + 1, points=points, indexes=indexes, leaf_size=leaf_size),
                right=grow(start=start + m + 1, end=end, points=points, indexes=indexes, leaf_size=leaf_size))


def build_tree(points, leaf_size=10):
    indexes = np.arange(len(points))
    node_stack = []
    intervals = [((0, len(points)), None, 0)]
    while len(intervals):
        (start, end), father, is_right = intervals[0]
        intervals = intervals[1:]

        axis = np.argmax(points.max(axis=0) - points.min(axis=0))
        argsort = np.argsort(points[start:end, axis])
        points[start:end] = points[start:end][argsort]
        indexes[start:end] = indexes[start:end][argsort]

        length = end - start
        m = length // 2
        median_point = points[start + m]
        while m + 1 < length and points[start + m + 1][axis] == median_point[axis]:
            m += 1
        x = points[start + m][axis]

        is_leaf_not_complete = end - start > leaf_size
        node_stack.append(Node(x=x, start=start, end=end, axis=axis if is_leaf_not_complete else -1))
        if father is not None:
            if is_right:
                node_stack[father].right = node_stack[-1]
            else:
                node_stack[father].left = node_stack[-1]
        if is_leaf_not_complete:
            intervals.extend([((start, m+1), len(node_stack)-1, 0), ((m+1, end), len(node_stack)-1, 1)])


class KdTree:
    def __init__(self, X, leaf_size=40):
        self.points = copy.deepcopy(np.asarray(X))
        self.indexes = np.arange(len(self.points))
        # self.node = grow(start=0, end=len(X), points=self.points, indexes=self.indexes, leaf_size=leaf_size)

        self.node_stack = []
        intervals = [((0, len(self.points)), None, 0)]
        while len(intervals):
            (start, end), father, is_right = intervals[0]
            intervals = intervals[1:]

            axis = np.argmax(self.points[start:end].max(axis=0) - self.points[start:end].min(axis=0))
            argsort = np.argsort(self.points[start:end, axis])
            self.points[start:end] = self.points[start:end][argsort]
            self.indexes[start:end] = self.indexes[start:end][argsort]

            length = end - start
            m = length // 2
            median_point = self.points[start + m]
            while m + 1 < length and self.points[start + m + 1][axis] == median_point[axis]:
                m += 1
            x = self.points[start + m, axis]

            is_leaf_not_complete = end - start > leaf_size
            self.node_stack.append(Node(x=x, start=start, end=end, axis=axis if is_leaf_not_complete else -1))
            if father is not None:
                if is_right:
                    self.node_stack[father].right = self.node_stack[-1]
                else:
                    self.node_stack[father].left = self.node_stack[-1]
            if is_leaf_not_complete:
                intervals.extend([((start, start + m + 1), len(self.node_stack) - 1, 0),
                                  ((start + m + 1, end), len(self.node_stack) - 1, 1)])
        self.node = self.node_stack[0]

    def query_single_point(self, x, k, squared_distance_upper_bound=np.inf):
        heap = []
        queue = []

        favorite = NodeInfo(node=self.node, side_distances=np.zeros_like(x))
        while True:
            if favorite.node.axis == -1:
                start, end = favorite.node.start, favorite.node.end
                squared_distances = np.sum(np.square(self.points[start:end] - x), axis=-1)
                for d, i in zip(squared_distances, self.indexes[start:end]):
                    if d < squared_distance_upper_bound:
                        if len(heap) < k:
                            heapq.heappush(heap, (-d, i))
                        else:
                            heapq.heappushpop(heap, (-d, i))
                            squared_distance_upper_bound = -heap[0][0]
                if len(queue) == 0:
                    break
                favorite = heapq.heappop(queue)[1]
            else:
                if favorite.distance > squared_distance_upper_bound:
                    break
                node = favorite.node
                axis = node.axis
                dx = x[node.axis] - node.x

                favorite.node = node.left
                worst = NodeInfo(node=node.right, side_distances=np.copy(favorite.side_distances))
                worst.update_side_distances(squared_value=dx * dx, axis=axis)
                if dx >= 0:
                    favorite.node, worst.node = worst.node, favorite.node
                if favorite.distance > worst.distance:
                    favorite, worst = worst, favorite
                if worst.distance <= squared_distance_upper_bound:
                    heapq.heappush(queue, (worst.distance, worst))
        order = np.argsort([-d[0] for d in heap])
        return np.array([d[1] for d in heap])[order]

    def query(self, x, k=2):
        # return get_knn(self.n, x, k=k, return_dist_sq=True, heap=[])

        return self.query_single_point(x=x, k=k)
        # return approach(self.n, x, distance_upper_bound=np.inf)


# def approach(node, target, distance_upper_bound=None):
#     if node is None:
#         return np.zeros_like(target), np.inf
#     if node.left is None and node.right is None:
#         return node.point, np.sum(np.square(node.point - target))
#
#     axis = node.axis
#
#     nearer_node, further_node = node.left, node.right
#     dx = target[axis] - node.point[axis]
#     if dx > 0:
#         nearer_node, further_node = further_node, nearer_node
#
#     nearest, record = approach(nearer_node, target, distance_upper_bound=distance_upper_bound)
#     distance_upper_bound = min(record, distance_upper_bound)
#
#     if dx * dx > distance_upper_bound:
#         return nearest, record
#
#     current_distance = np.sum(np.square(node.point - target))
#     if current_distance < record:
#         nearest = node.point
#         record = current_distance
#         distance_upper_bound = record
#
#     candidate, candidate_distance = approach(further_node, target, distance_upper_bound=distance_upper_bound)
#     if candidate_distance < record:
#         return candidate, candidate_distance
#     return nearest, record

# dist_sq_func = lambda a, b: sum((x - b[i]) ** 2 for i, x in enumerate(a))
# dim = 3
#
#
# def get_knn(node, point, k, return_dist_sq, heap, i=0, tiebreaker=1):
#     if node is not None:
#         dist_sq = dist_sq_func(point, node[2])
#         dx = node[2][i] - point[i]
#         if len(heap) < k:
#             heapq.heappush(heap, (-dist_sq, tiebreaker, node[2]))
#         elif dist_sq < -heap[0][0]:
#             heapq.heappushpop(heap, (-dist_sq, tiebreaker, node[2]))
#         i = (i + 1) % dim
#         # Goes into the left branch, then the right branch if needed
#         for b in (dx < 0, dx >= 0)[:1 + (dx * dx < -heap[0][0])]:
#             get_knn(node[b], point, k, return_dist_sq, heap, i, (tiebreaker << 1) | b)
#     if tiebreaker == 1:
#         return [(-h[0], h[2]) if return_dist_sq else h[2] for h in sorted(heap)][::-1]


if __name__ == "__main__":
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
    tree = KdTree(points)
    print(tree.query_single_point(x=np.array([5, 5, 5]), k=2))

    print(tree)
