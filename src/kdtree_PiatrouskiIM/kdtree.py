import heapq
import numpy as np


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


class KDTree:
    def __init__(self, X, leaf_size=40, copy_input=True):
        assert leaf_size > 0, "leaf size denote maximal number of points allowed in single tree node."
        self.points = np.copy(np.asarray(X)) if copy_input else np.asarray(X)
        self.indexes = np.arange(len(self.points))

        node_stack = []
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
            node_stack.append(Node(x=x, start=start, end=end, axis=axis if is_leaf_not_complete else -1))
            if father is not None:
                if is_right:
                    node_stack[father].right = node_stack[-1]
                else:
                    node_stack[father].left = node_stack[-1]
            if is_leaf_not_complete:
                intervals.extend([((start, start + m + 1), len(node_stack) - 1, 0),
                                  ((start + m + 1, end), len(node_stack) - 1, 1)])
        self.root = node_stack[0]

    def query_single_point(self, x, k, squared_distance_upper_bound=np.inf):
        heap = []
        queue = []

        favorite = NodeInfo(node=self.root, side_distances=np.zeros_like(x))
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
        return self.query_single_point(x=x, k=k)



