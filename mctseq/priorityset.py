import heapq
from mctseq.utils import jaccard_measure


def filter_results(results, theta):
    """
    Filter redundant elements
    :param results: must be a node
    :param theta:
    :return: filtered list
    """
    results_list = list(results)
    results_list.sort(key=lambda x: x[0], reverse=True)

    filtered_elements = []

    for result in results_list:
        similar = False

        for filtered_element in filtered_elements:
            if jaccard_measure(result[1],
                               filtered_element) > theta:
                similar = True

        if not similar:
            filtered_elements.append(result[1])

    return map(lambda x: (x.quality, x),filtered_elements)


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

    def get_top_k_non_redundant(self, k):
        self.heap = filter_results(self.heap, 0.9)
        return self.get_top_k(k)

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
