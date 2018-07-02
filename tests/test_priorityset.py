from mctseq.sequencenode import SequenceNode
from mctseq.priorityset import PrioritySetQuality

data = [['+', {'A', 'B'}, {'C'}], ['-', {'A'}, {'B'}]]
first_zero_mask = int('0101', 2)
last_ones_mask = int('0101', 2)
bitset_slot_size = 2

kwargs = {'first_zero_mask': first_zero_mask, 'last_ones_mask': last_ones_mask,
          'bitset_slot_size': bitset_slot_size, 'node_hashmap': {}}


def test_priorityset():
    root = SequenceNode([{'A'}], None, {'A', 'B', 'C'}, data, '+', 1, {}, **kwargs)
    seq2 = SequenceNode([{'B'}], None, {'A', 'B', 'C'}, data, '+', 1, {}, **kwargs)
    seq3 = SequenceNode([{'C'}], None, {'A', 'B', 'C'}, data, '+', 1, {}, **kwargs)
    seq4 = SequenceNode([{'B', 'C'}], None, {'A', 'B', 'C'}, data, '+', 1, {}, **kwargs)

    child = SequenceNode([{'A'}, {'C'}], None, {'A', 'B', 'C'}, data, '+', 1, {}, **kwargs)
    child2 = SequenceNode([{'A'}, {'B'}], None, {'A', 'B', 'C'}, data, '+', 1, {}, **kwargs)
    child3 = SequenceNode([{'A'}, {'B', 'C'}], None, {'A', 'B', 'C'}, data,
                          '+', 1, {}, **kwargs)

    priority = PrioritySetQuality()
    priority.add(root)
    priority.add(child)
    priority.add(child2)
    priority.add(child3)

    # _, best_element = priority.get()
    _, best_element = priority.get_top_k(1)[0]

    assert best_element == child


def test_unique():
    root = SequenceNode([{'A'}], None, {'A', 'B', 'C'}, data, '+', 1, {}, **kwargs)
    child = SequenceNode([{'A'}], None, {'A', 'B', 'C'}, data, '+', 1, {}, **kwargs)

    priority = PrioritySetQuality()
    priority.add(root)
    priority.add(child)

    assert len(priority.heap) == 1
