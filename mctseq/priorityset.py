import heapq
from mctseq.utils import jaccard_measure, is_subsequence

THETA = 0.8

def filter_results(results, theta, k):
    """
    Filter redundant elements
    :param results: must be a node
    :param theta:
    :return: filtered list
    """
    results_list = list(results)
    results_list.sort(key=lambda x: x[0], reverse=True)

    filtered_elements = []

    for i, result in enumerate(results_list):
        similar = False

        for filtered_element in filtered_elements:
            if jaccard_measure(result[1],
                               filtered_element) > theta:
                similar = True

        if not similar:
            filtered_elements.append(result[1])

        if len(filtered_elements) >= k:
            break

    return list(map(lambda x: (x.wracc, x), filtered_elements))


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
        self.heap = filter_results(self.heap, THETA, k)
        return self.get_top_k(k)

def jaccard_measure_misere(sequence1, sequence2, data):
    intersection = 0
    union = 0
    for sequence in data:
        seq1 = False
        seq2 = False

        if is_subsequence(sequence1, sequence):
            seq1 = True
        if is_subsequence(sequence2, sequence):
            seq2 = True

        if seq1 or seq2:
            union += 1

        if seq1 and seq2:
            intersection += 1

    try:
        return intersection / union
    except ZeroDivisionError:
        return 0



def filter_results_misere(results, data, theta, k):
    """
    Filter redundant elements
    :param results: must be a node
    :param theta:
    :return: filtered list
    """

    results_list = list(results)
    results_list.sort(key=lambda x: x[0], reverse=True)

    filtered_elements = []


    for i, result in enumerate(results_list):
        similar = False

        for filtered_element in filtered_elements:
            if jaccard_measure_misere(result[1],
                                      filtered_element[1], data) > theta:
                similar = True

        if not similar:
            filtered_elements.append(result)

        if i > k:
            break

    return filtered_elements


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

    def get_top_k_non_redundant(self, data, k):
        self.heap = filter_results_misere(self.heap, data, THETA, k)
        return self.get_top_k(k)
