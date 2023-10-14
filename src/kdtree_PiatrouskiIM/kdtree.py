import numpy as np
from .heap_queue import HeapQueue


class Node:
    def __init__(self, start: int, end: int, axis: int = -1, x: float = 0., left=None, right=None):
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


class KDTree:
    def __init__(self, X, leaf_size=40, copy_input=True):
        assert leaf_size > 0, "leaf size denote maximal number of points allowed in single tree node."
        self.points = np.copy(np.asarray(X)) if copy_input else np.asarray(X)
        self.indexes = np.arange(len(self.points))

        self.root = Node(start=0, end=len(self.points))
        raf_nodes = [self.root]
        while len(raf_nodes):
            node = raf_nodes[0]
            raf_nodes = raf_nodes[1:]
            start, end = node.start, node.end
            length = end - start
            if length > leaf_size:
                axis = np.argmax(self.points[start:end].max(axis=0) - self.points[start:end].min(axis=0))
                order = np.argsort(self.points[start:end, axis])
                self.points[start:end] = self.points[start:end][order]
                self.indexes[start:end] = self.indexes[start:end][order]

                m = length // 2
                node.x = self.points[start + m, axis]
                node.axis = axis
                node.left, node.right = Node(start=start, end=start + m + 1), Node(start=start + m + 1, end=end)
                raf_nodes.extend([node.left, node.right])

    def query(self, x, k=1, distance_upper_bound=np.inf):
        squared_distance_upper_bound = np.square(distance_upper_bound)
        heap = HeapQueue()
        queue = HeapQueue()

        favorite = NodeInfo(node=self.root, side_distances=np.zeros_like(x))
        while True:
            if favorite.node.axis == -1:
                start, end = favorite.node.start, favorite.node.end
                squared_distances = np.sum(np.square(self.points[start:end] - x), axis=-1)
                for d, i in zip(squared_distances, self.indexes[start:end]):
                    if d < squared_distance_upper_bound:
                        if len(heap) < k:
                            heap.push(priority=-d, item=i)
                        else:
                            heap.push_and_pop(priority=-d, item=i)
                            squared_distance_upper_bound = -heap.smallest_priority()
                if len(queue) == 0:
                    break
                favorite = queue.pop()[1]
            else:
                if favorite.distance > squared_distance_upper_bound:
                    break
                node = favorite.node
                dx = x[node.axis] - node.x

                favorite.node = node.left
                worst = NodeInfo(node=node.right, side_distances=np.copy(favorite.side_distances))
                worst.update_side_distances(squared_value=dx * dx, axis=node.axis)
                if dx >= 0:
                    favorite.node, worst.node = worst.node, favorite.node
                if favorite.distance > worst.distance:
                    favorite, worst = worst, favorite
                if worst.distance <= squared_distance_upper_bound:
                    queue.push(priority=worst.distance, item=worst)
        return np.asarray(heap.heap)[np.argsort(heap.priorities)[::-1]]

