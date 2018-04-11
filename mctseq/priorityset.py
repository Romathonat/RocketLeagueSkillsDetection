import heapq


class PrioritySetQuality(object):
    """
    This class is a priority queue, removing duplicates and using node wracc
    as the metric to order the priority queue
    """

    def __init__(self):
        self.heap = []
        self.set = set()

    def add(self, node):
        if node not in self.set:
            heapq.heappush(self.heap, (node.wracc, node))
            self.set.add(node)

    def get(self):
        wracc, node = heapq.heappop(self.heap)
        self.set.remove(node)
        return (wracc, node)

    def get_top_k(self, k):
        return heapq.nlargest(k, self.heap)


class PrioritySet(object):
    """
    This class is a priority queue, removing duplicates and using node wracc
    as the metric to order the priority queue
    """
    def __init__(self):
        self.heap = []
        self.set = set()

    def add(self, sequence, wracc):
        if sequence not in self.set:
            heapq.heappush(self.heap, (wracc, sequence))
            self.set.add(sequence)

    def get(self):
        wracc, sequence = heapq.heappop(self.heap)
        self.set.remove(sequence)
        return (wracc, sequence)

    def get_top_k(self, k):
        return heapq.nlargest(k, self.heap)
