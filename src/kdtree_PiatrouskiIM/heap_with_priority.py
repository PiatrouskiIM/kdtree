def heappush(heap, heap_priorities, item_priority, item):
    for i in range(len(heap)):
        if heap_priorities[i] > item_priority:
            heap_priorities.insert(i, item_priority)
            heap.inser(i, item)
    else:
        heap_priorities.append(item_priority)
        heap.append(item)