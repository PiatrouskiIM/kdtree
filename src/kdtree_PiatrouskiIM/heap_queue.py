class HeapQueue:
    def __init__(self):
        self.priorities = []
        self.heap = []

    def __len__(self):
        return len(self.priorities)

    def push(self, priority, item):
        for i in range(len(self.heap)):
            if self.priorities[i] > priority:
                self.priorities.insert(i, priority)
                self.heap.insert(i, item)
                break
        else:
            self.priorities.append(priority)
            self.heap.append(item)

    def pop(self):
        assert len(self)
        priority = self.priorities[0]
        item = self.heap[0]

        self.priorities = self.priorities[1:]
        self.heap = self.heap[1:]
        return priority, item

    def push_and_pop(self, priority, item):
        if len(self) == 0 or self.priorities[0] > priority:
            return priority, item
        self.push(priority, item)
        return self.pop()

    def smallest_priority(self):
        assert len(self)
        return self.priorities[0]

    def top_priority(self):
        assert len(self)
        return self.priorities[-1]


if __name__ == "__main__":
    import heapq
    heap = HeapQueue()

    heap.push(1, 1)
    heap.push(2, 2)
    heap.push(0, 3)

    print(heap.heap)
    print(heap.pop())
    print(heap.pop())
    print(heap.pop())

    alt = []
    heapq.heappush(alt, (1, 1))
    heapq.heappush(alt, (2, 2))
    heapq.heappush(alt, (0, 3))

    print(alt)
    print(heapq.heappop(alt))
    print(heapq.heappop(alt))
    print(heapq.heappop(alt))
