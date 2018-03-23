import math


def sequence_mutable_to_immutable(sequence):
    """
    :param sequence: form [{}, {}, ...]
    :return: the same sequence in its immutable form
    """
    return tuple([frozenset(i) for i in sequence])


def sequence_immutable_to_mutable(sequence):
    """

    :param sequence: form (frozenset(), frozenset(), ...)
    :return: the same sequence in its mutable form
    """
    return [set(i) for i in sequence]


def read_data(filename):
    sequences = []
    with open(filename) as f:
        for line in f:
            sequence = [set(i.replace(' ', '')) for i in line[:-3].split('-1')]
            sequences.append(sequence)
    return sequences


def extract_items(data):
    items = set()
    for sequence in data:
        for itemset in sequence:
            for item in itemset:
                items.add(item)
    return items


def uct(node, child_node):
    return child_node.quality + math.sqrt(
        2 * math.log(node.number_visit) / child_node.number_visit)
